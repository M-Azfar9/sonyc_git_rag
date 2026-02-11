from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_classic.memory import ConversationBufferMemory
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import WebBaseLoader
from typing import Literal, Optional, List
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import GithubFileLoader
from sqlalchemy.orm import Session
from datetime import timedelta

import time
import os
import tempfile
import logging
import queue
import threading
from dotenv import load_dotenv, dotenv_values

from .database import get_db, Base, engine
from .models import User, Chat, Message, GitProject, GitWebhookEvent
from . import github_service
from .auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_github_token():
    """Retrieve GitHub token with priority: explicit env var -> .env file -> other env var"""
    
    # Debug info
    current_dir = os.getcwd()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, ".env")
    
    # 1. Try process environment (most reliable if set correctly)
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    if token: 
        logger.info(f"Using GitHub token from os.environ (length: {len(token)})")
        return token
    
    # 2. Try direct .env read with explicit path
    try:
        if os.path.exists(env_path):
            env_vals = dotenv_values(env_path)
            token = env_vals.get("GITHUB_PERSONAL_ACCESS_TOKEN")
            if token: 
                logger.info(f"Using GitHub token from .env file at {env_path} (length: {len(token)})")
                return token
            token = env_vals.get("GITHUB_ACCESS_TOKEN")
            if token: 
                logger.info(f"Using GITHUB_ACCESS_TOKEN from .env file at {env_path}")
                return token
        else:
            logger.warning(f".env file not found at {env_path}")
            
        # Try local .env just in case
        if os.path.exists(".env"):
             env_vals = dotenv_values(".env")
             token = env_vals.get("GITHUB_PERSONAL_ACCESS_TOKEN")
             if token: return token
    except Exception as e:
        logger.error(f"Error reading .env: {e}")
        pass
        
    # 3. Fallback
    fallback = os.environ.get("GITHUB_ACCESS_TOKEN")
    if fallback:
        logger.info("Using fallback GITHUB_ACCESS_TOKEN from os.environ")
        return fallback
        
    logger.error("NO GITHUB TOKEN FOUND in environment or .env file!")
    return None

# Load dotenv explicitly from the calculated path as well
try:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
        logger.info(f"Loaded .env from {env_path}")
    else:
        load_dotenv(override=True)
except Exception as e:
    logger.warning(f"Failed to load .env: {e}")


# Create database tables (only if database is available)
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.warning(f"Could not create database tables. Database may not be available: {e}")
    logger.warning("Server will start but database operations will fail until database is configured.")

# Initialize embedding model (only if API key is available)
embedding_model = None
try:
    embedding_model = MistralAIEmbeddings(model="mistral-embed", api_key=os.getenv("MISTRAL_API_KEY"))
    logger.info("Mistral embedding model initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize embedding model. MISTRAL_API_KEY may not be set: {e}")
    logger.warning("Server will start but RAG operations will fail until API key is configured.")

# Initialize FastAPI app
app = FastAPI(title="RAG ChatBot API", version="2.0.0")

# CORS middleware
# Get allowed origins from environment variable or use defaults
cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://13.49.120.11:3000,http://13.49.120.11")
cors_origins = [origin.strip() for origin in cors_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== CONVERSATION MEMORY STORE ==========
# Store ConversationBufferMemory instances per user for normal chats
user_memories: dict[str, ConversationBufferMemory] = {}

SYSTEM_MSG = SystemMessage(content="""
You are an assistant whose top priorities are accuracy, clarity, and user safety. 
Always verify facts before presenting them; when a fact could be time-sensitive or uncertain, explicitly say "I don't know" / "I'm not sure" instead of guessing. 
If the user's question is ambiguous, ask one short clarifying question. 
Cite sources for non-common-knowledge claims. 
If asked for instructions that could be harmful, refuse and provide a safe alternative. 
Keep answers concise, show the final answer first, and then provide a short explanation and sources.
""")

# ========== PYDANTIC MODELS ==========
class UserSignup(BaseModel):
    email: EmailStr
    password: str

class UserSignin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    chat_id: int
    message: str
    chat_type: Literal['normal_chat', 'yt_chat', 'pdf_chat', 'web_chat', 'git_chat']
    vector_db_collection_id: Optional[str] = None

class ChatCreate(BaseModel):
    title: str
    type: str  # Frontend format: "Normal", "YouTube", etc.
    vector_db_collection_id: Optional[str] = None

class ChatResponse(BaseModel):
    id: int
    title: str
    type: str
    vector_db_collection_id: Optional[str]
    created_at: str

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: str

class RAGRequest(BaseModel):
    url: str

class GitProjectCreate(BaseModel):
    repo_url: str
    branch: str = "main"

class GitProjectResponse(BaseModel):
    id: int
    repo_url: str
    repo_owner: str
    repo_name: str
    branch: str
    vector_db_collection_id: Optional[str]
    last_synced_at: Optional[str]
    created_at: str

class GitReportRequest(BaseModel):
    project_id: int
    report_type: str = "full"  # "full", "architecture", "dependencies", "code_quality"

# ========== UTILITY FUNCTIONS ==========
def map_frontend_to_backend_chat_type(frontend_type: str) -> str:
    """Map frontend chat type to backend chat type"""
    mapping = {
        "Normal": "normal_chat",
        "YouTube": "yt_chat",
        "Web": "web_chat",
        "Git": "git_chat",
        "PDF": "pdf_chat"
    }
    return mapping.get(frontend_type, "normal_chat")

def map_backend_to_frontend_chat_type(backend_type: str) -> str:
    """Map backend chat type to frontend chat type"""
    mapping = {
        "normal_chat": "Normal",
        "yt_chat": "YouTube",
        "web_chat": "Web",
        "git_chat": "Git",
        "pdf_chat": "PDF"
    }
    return mapping.get(backend_type, "Normal")

def extract_text_from_content(content):
    """Extract text from various content formats returned by LangChain/Mistral"""
    if content is None:
        return ""
    
    # If it's a string, return it directly
    if isinstance(content, str):
        return content
    
    # If it's a list, process each item
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                # Handle dictionary format like {'type': 'text', 'text': '...', 'index': 0}
                if 'text' in item:
                    texts.append(str(item['text']))
                elif 'content' in item:
                    texts.append(str(item['content']))
                else:
                    # Try to extract any string value
                    for key, value in item.items():
                        if isinstance(value, str) and key not in ['type', 'index', 'extras']:
                            texts.append(value)
            elif isinstance(item, str):
                texts.append(item)
            else:
                texts.append(str(item))
        return "".join(texts)
    
    # If it's a dictionary, extract text field
    if isinstance(content, dict):
        if 'text' in content:
            return str(content['text'])
        elif 'content' in content:
            return str(content['content'])
        else:
            # Try to find any string value
            for key, value in content.items():
                if isinstance(value, str) and key not in ['type', 'index', 'extras']:
                    return value
            # Fallback: convert to string
            return str(content)
    
    # Fallback: convert to string
    return str(content)

def generate_title(user_query: str) -> str:
    """Generate a concise title (max 5 words) based on user query"""
    try:
        logger.info("Generating title for user query")
        model = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.3,
            streaming=True,
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
        
        title_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Generate a concise title (maximum 5 words) for a chat conversation based on this user query: "{query}"

Title (max 5 words, no quotes, no punctuation at end):"""
        )
        
        chain = title_prompt | model
        response = chain.invoke({"query": user_query})
        title = extract_text_from_content(response.content).strip()
        
        # Clean up title - remove quotes, limit to 5 words
        title = title.strip('"\'')
        words = title.split()[:5]
        title = " ".join(words)
        
        logger.info(f"Generated title: {title}")
        return title if title else "New Chat"
    except Exception as e:
        logger.error(f"Error generating title: {str(e)}", exc_info=True)
        return "New Chat"

def generate_title_parallel(user_query: str, title_queue: queue.Queue):
    """Generate title in parallel thread for RunnableParallel execution"""
    try:
        title_model = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.3,
            streaming=True,
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
        title_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Generate a concise title (maximum 5 words) for a chat conversation based on this user query: "{query}"

Title (max 5 words, no quotes, no punctuation at end):"""
        )
        title_chain = title_prompt | title_model
        
        title_result = title_chain.invoke({"query": user_query})
        title = extract_text_from_content(title_result.content).strip()
        title = title.strip('"\'')
        words = title.split()[:5]
        title = " ".join(words) if words else "New Chat"
        title_queue.put(title)
        logger.info(f"Title generated in parallel: {title}")
    except Exception as e:
        logger.error(f"Error generating title in parallel: {e}")
        title_queue.put("New Chat")

def stream_answer(memory: ConversationBufferMemory):
    """Streams the assistant's reply token by token using ConversationBufferMemory"""
    try:
        logger.info("Initializing ChatMistralAI model")
        model = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.3,
            streaming=True,
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
        logger.info("Starting model stream with ConversationBufferMemory")
        
        # Get the conversation history from memory
        history = memory.chat_memory.messages
        # Ensure system message is at the beginning if not present
        if not history or not isinstance(history[0], SystemMessage):
            history = [SYSTEM_MSG] + history
        
        logger.info(f"History length: {len(history)}")
        stream = model.stream(history)
        full_response = ""
        token_count = 0
        for chunk in stream:
            # Extract text from chunk content
            content = chunk.content if hasattr(chunk, 'content') else chunk
            token = extract_text_from_content(content)
            
            if token:  # Only yield non-empty tokens
                full_response += token
                token_count += 1
                # Yield immediately for real-time streaming
                yield token
                # Small flush to ensure immediate transmission
                import sys
                sys.stdout.flush()
        
        logger.info(f"Model stream completed. Tokens received: {token_count}, Response length: {len(full_response)}")
        
        # Save the AI response to memory using AIMessage
        if full_response:
            memory.chat_memory.add_ai_message(AIMessage(content=full_response))
            logger.info("AI response saved to ConversationBufferMemory")
    except Exception as e:
        logger.error(f"Error in stream_answer: {str(e)}", exc_info=True)
        raise

def get_dynamic_chunk_size(text: str):
    """Dynamically decide chunk_size and chunk_overlap based on document length"""
    length = len(text)
    if length < 1000:
        chunk_size = length/2
        chunk_overlap = 20
    elif length < 5000:
        chunk_size = length/5
        chunk_overlap = 50
    elif length < 20000:
        chunk_size = length/20
        chunk_overlap = 100
    elif length < 100000:
        chunk_size = length/80
        chunk_overlap = 200
    elif length < 300000:
        chunk_size = length/200
        chunk_overlap = 400
    else:
        chunk_size = 6000
        chunk_overlap = 600
    return int(chunk_size), int(chunk_overlap)

def youtube_loader(url: str):
    """Load YouTube transcript"""
    video_id = url.split("v=")[1].split("&")[0]
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id)
    transcript = " ".join(chunk.text for chunk in transcript_list)
    return transcript

def load_pdf(file_path: str):
    """Lazy loads a PDF"""
    loader = PyPDFLoader(file_path)
    return loader.lazy_load()

def github_loader(repo_url, branch="main"):
    """Load GitHub repository files"""
    repo_id = convert_github_url_to_repo_id(repo_url)
    loader = GithubFileLoader(
        repo=repo_id,
        branch=branch,
        file_filter=lambda file_path: file_path.endswith((
            ".txt", ".md", ".html", ".css", ".xml", ".json", ".yaml", ".yml", 
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt", ".kts", ".scala", 
            ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".swift", ".m", ".php", 
            ".rb", ".pl", ".pm", ".lua", ".sh", ".bash", ".r", ".jl", ".asm", 
            ".s", ".dart", ".cs", ".ipynb"
    )),
        access_token=get_github_token(),
    )
    
    token_used = get_github_token()
    logger.info(f"GitHub Loader using token: {token_used[:4]}...{token_used[-4:]} (Len: {len(token_used) if token_used else 0})")
    
    docs = loader.load()
    full_text = ""
    for i, doc in enumerate(docs, start=1):
        file_name = doc.metadata.get("source", f"file_{i}")
        full_text += f"\n\n===== FILE {i}: {file_name} =====\n"
        full_text += doc.page_content
    return full_text

def convert_github_url_to_repo_id(github_url: str) -> str:
    """Converts any GitHub URL into owner/repo format"""
    cleaned = github_url.replace("https://", "").replace("http://", "")
    parts = cleaned.split("/")
    if len(parts) < 3:
        raise ValueError("Invalid GitHub URL format")
    owner = parts[1]
    repo = parts[2]
    return f"{owner}/{repo}"

def web_loader(url: str):
    """Loads webpage and returns text as string"""
    loader = WebBaseLoader(url)
    docs = loader.load()
    if not docs:
        return ""
    return "\n\n".join([d.page_content for d in docs])

def split_text(text: str, chunk_size=None, chunk_overlap=None):
    """Split raw text into smaller chunks"""
    if chunk_size is None or chunk_overlap is None:
        chunk_size, chunk_overlap = get_dynamic_chunk_size(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks

def create_vector_store(chunks, collection_name: str, persist_dir: str):
    """Create a Chroma vector store from chunks"""
    docs = [Document(page_content=chunk) for chunk in chunks]
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )
    vector_store.add_documents(docs)
    vector_store.persist()
    return vector_store

def load_vector_store(collection_name: str, persist_dir: str):
    """Load an existing Chroma vector store"""
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )
    return vector_store

def get_rag_prompt():
    """Get RAG prompt template"""
    prompt = PromptTemplate.from_template(
        """
    You are an advanced Retrieval-Augmented Generation (RAG) AI assistant.
    Your job is to generate answers that are:

    - **Fully grounded in the provided context**
    - **Factual and concise unless user requests more detail**
    - **Explanatory enough that a beginner can understand**
    - **Non-hallucinatory: never invent facts not found in the context**
    - **Helpful and structured**
    - **Adaptive in length:**
        - If the user specifies a length → follow it.
        - If not, give a detailed but concise explanation.

    =========================
    STRICT RULES:
    =========================

    1️⃣ **Grounded Answers Only**  
    Use ONLY the provided context to answer.  
    If the context does NOT contain enough information, say:

    "I don't have enough information to answer that from the provided data."

    Do NOT guess. Do NOT create facts.

    2️⃣ **Use Context Examples if Available**  
    If the context includes examples:  
    → Explain them clearly and deeply.

    3️⃣ **If No Examples Are Provided**  
    Generate **relevant, realistic, real-life** examples that match the topic.

    4️⃣ **Explain Step-by-Step When Needed**  
    If the question requires reasoning or understanding, use clear steps or bullet points.

    5️⃣ **No unnecessary repetition**  
    Do not repeat entire context or question.  
    Summaries must be natural and focused.

    =========================
    CONTEXT:
    {context}
    =========================

    QUESTION:
    {question}

    --------------------------
    Now produce the BEST POSSIBLE grounded answer.
    """
    )
    return prompt


# ========== AUTHENTICATION ENDPOINTS ==========
@app.post("/auth/signup", response_model=Token)
def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """User registration"""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        new_user = User(email=user_data.email, password_hash=hashed_password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(new_user.id)}, expires_delta=access_token_expires
        )
        
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )

@app.post("/auth/signin", response_model=Token)
def signin(user_data: UserSignin, db: Session = Depends(get_db)):
    """User login"""
    try:
        user = db.query(User).filter(User.email == user_data.email).first()
        if not user or not verify_password(user_data.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id)}, expires_delta=access_token_expires
        )
        
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to authenticate: {str(e)}"
        )

@app.get("/auth/me")
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    try:
        return {
            "id": current_user.id,
            "email": current_user.email
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user info: {str(e)}"
        )

# ========== CHAT MANAGEMENT ENDPOINTS ==========
@app.get("/chats", response_model=List[ChatResponse])
def get_chats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get all chats for current user"""
    try:
        chats = db.query(Chat).filter(Chat.user_id == current_user.id).order_by(Chat.created_at.desc()).all()
        return [
            ChatResponse(
                id=chat.id,
                title=chat.title,
                type=map_backend_to_frontend_chat_type(chat.type),
                vector_db_collection_id=chat.vector_db_collection_id,
                created_at=chat.created_at.isoformat()
            )
            for chat in chats
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chats: {str(e)}"
        )

@app.post("/chats", response_model=ChatResponse)
def create_chat(chat_data: ChatCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a new chat"""
    try:
        backend_type = map_frontend_to_backend_chat_type(chat_data.type)
        new_chat = Chat(
            user_id=current_user.id,
            title=chat_data.title,
            type=backend_type,
            vector_db_collection_id=chat_data.vector_db_collection_id
        )
        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)
        
        return ChatResponse(
            id=new_chat.id,
            title=new_chat.title,
            type=map_backend_to_frontend_chat_type(new_chat.type),
            vector_db_collection_id=new_chat.vector_db_collection_id,
            created_at=new_chat.created_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create chat: {str(e)}"
        )

@app.get("/chats/{chat_id}/messages", response_model=List[MessageResponse])
def get_chat_messages(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get messages for a specific chat"""
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == current_user.id).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at.asc()).all()
        return [
            MessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at.isoformat()
            )
            for msg in messages
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get messages: {str(e)}"
        )

@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete a chat"""
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == current_user.id).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        db.delete(chat)
        db.commit()
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete chat: {str(e)}"
        )

# ========== CHAT STREAMING ENDPOINT ==========
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Streaming chat endpoint"""
    logger.info(f"Received chat stream request: chat_id={request.chat_id}, chat_type={request.chat_type}, user_id={current_user.id}")
    try:
        # Verify chat belongs to user
        chat = db.query(Chat).filter(Chat.id == request.chat_id, Chat.user_id == current_user.id).first()
        if not chat:
            logger.warning(f"Chat not found: chat_id={request.chat_id}, user_id={current_user.id}")
            raise HTTPException(status_code=404, detail="Chat not found")
        
        logger.info(f"Chat found: {chat.title}")
        
        # Check if this is the first message in the chat (before saving user message)
        existing_messages = db.query(Message).filter(Message.chat_id == request.chat_id).count()
        is_first_message = existing_messages == 0
        logger.info(f"Is first message: {is_first_message}, existing messages: {existing_messages}")
        
        # Save user message (if database is available)
        try:
            user_message = Message(chat_id=request.chat_id, role="user", content=request.message)
            db.add(user_message)
            db.commit()
            logger.info(f"User message saved: {request.message[:50]}...")
        except Exception as e:
            db.rollback()
            logger.warning(f"Could not save user message to database: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}"
        )
    
    if request.chat_type == "normal_chat":
        # Memory-based chat using ConversationBufferMemory
        logger.info("Processing normal_chat request with ConversationBufferMemory")
        user_id_str = str(current_user.id)
        
        # is_first_message is already determined above before saving user message
        
        # Get or create ConversationBufferMemory for this user
        if user_id_str not in user_memories:
            user_memories[user_id_str] = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_memory"
            )
            # Add system message to memory as SystemMessage
            user_memories[user_id_str].chat_memory.add_message(SYSTEM_MSG)
            logger.info(f"Initialized ConversationBufferMemory for user: {user_id_str}")

        memory = user_memories[user_id_str]
        
        # Add user message to memory using HumanMessage
        memory.chat_memory.add_user_message(HumanMessage(content=request.message))
        logger.info(f"Added user message to memory. Total messages: {len(memory.chat_memory.messages)}")

        def generate():
            full_response = ""
            generated_title = None
            
            try:
                if is_first_message:
                    # Use RunnableParallel for first message: generate response and title in parallel
                    logger.info("First message detected - using RunnableParallel for response and title generation")
                    
                    # Queue to store title result
                    title_queue = queue.Queue()
                    
                    # Start title generation in background thread using module-level function
                    title_thread = threading.Thread(target=generate_title_parallel, args=(request.message, title_queue))
                    title_thread.start()
                    
                    # Stream response while title is being generated in parallel
                    logger.info("Starting stream_answer generator with ConversationBufferMemory")
                    token_count = 0
                    for token in stream_answer(memory):
                        if token:
                            full_response += token
                            token_count += 1
                            yield token
                    
                    # Wait for title generation to complete (with timeout)
                    title_thread.join(timeout=10)
                    if not title_queue.empty():
                        generated_title = title_queue.get()
                    else:
                        # Fallback if title generation failed or timed out
                        generated_title = generate_title(request.message)
                        logger.warning("Title generation timed out or failed, using fallback")
                    
                    logger.info(f"Generated title: {generated_title}")
                    
                    # Send title update immediately after streaming completes (before saving message)
                    if generated_title:
                        try:
                            # Refresh chat object to ensure we have the latest data
                            db.refresh(chat)
                            chat.title = generated_title
                            db.commit()
                            logger.info(f"Chat title updated to: {generated_title}")
                            # Send title update as special marker (frontend will parse this)
                            title_marker = f"<!-- TITLE_UPDATE:{generated_title} -->"
                            yield title_marker
                            logger.info(f"Title update marker sent: {title_marker}")
                        except Exception as e:
                            db.rollback()
                            logger.warning(f"Could not update chat title: {e}")
                else:
                    # Regular streaming for subsequent messages
                    logger.info("Starting stream_answer generator with ConversationBufferMemory")
                    token_count = 0
                    for token in stream_answer(memory):
                        if token:
                            full_response += token
                            token_count += 1
                            yield token
                
                logger.info(f"Stream completed. Response length: {len(full_response)}")
                
                # Save assistant message to database
                try:
                    assistant_message = Message(chat_id=request.chat_id, role="assistant", content=full_response)
                    db.add(assistant_message)
                    db.commit()
                    logger.info("Assistant message saved to database")
                except Exception as e:
                    db.rollback()
                    logger.warning(f"Could not save assistant message to database: {e}")
            except Exception as e:
                logger.error(f"Error in stream generator: {str(e)}", exc_info=True)
                error_msg = f"\n\nError: {str(e)}"
                yield error_msg
                # Try to save error message
                try:
                    assistant_message = Message(chat_id=request.chat_id, role="assistant", content=error_msg)
                    db.add(assistant_message)
                    db.commit()
                except:
                    db.rollback()
        
        return StreamingResponse(
            generate(), 
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    
    elif request.chat_type in ["yt_chat", "pdf_chat", "web_chat", "git_chat"]:
        # RAG-based chat
        logger.info(f"Processing RAG chat: type={request.chat_type}, collection={request.vector_db_collection_id}")
        if not request.vector_db_collection_id:
            logger.warning("vector_db_collection_id required for RAG chats")
            raise HTTPException(status_code=400, detail="vector_db_collection_id required for RAG chats")
        
        # Check if this is the first message in the chat (same check as normal_chat)
        # is_first_message is already determined above before saving user message
        
        current_dir = os.getcwd()
        try:
            logger.info(f"Loading vector store: {request.vector_db_collection_id}")
            vector_store = load_vector_store(
                collection_name=request.vector_db_collection_id,
                persist_dir=current_dir
            )
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Vector store not found: {e}", exc_info=True)
            raise HTTPException(status_code=404, detail=f"Vector store not found: {e}")

        logger.info("Creating retriever and retrieving context")
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        context_docs = retriever.invoke(request.message)
        context_text = "\n".join(doc.page_content for doc in context_docs)
        logger.info(f"Retrieved {len(context_docs)} context documents, total length: {len(context_text)}")

        rag_prompt = get_rag_prompt()
        logger.info("Initializing ChatMistralAI for RAG")
        llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.3,
            streaming=True,
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
        
        chain = rag_prompt | llm
        prompt_input = {'context': context_text, 'question': request.message}
        
        def generate():
            full_response = ""
            generated_title = None
            
            try:
                if is_first_message:
                    # Use RunnableParallel for first message: generate response and title in parallel
                    logger.info("First message detected in RAG chat - using RunnableParallel for response and title generation")
                    
                    import threading
                    import queue
                    
                    # Queue to store title result
                    title_queue = queue.Queue()
                    
                    # Start title generation in background thread
                    title_thread = threading.Thread(target=generate_title_parallel, args=(request.message, title_queue))
                    title_thread.start()
                    
                    # Stream RAG response while title is being generated in parallel
                    logger.info("Starting RAG chain stream with parallel title generation")
                    token_count = 0
                    for token in chain.stream(prompt_input):
                        # Extract text from token content
                        content = token.content if hasattr(token, 'content') else token
                        token_content = extract_text_from_content(content)
                        
                        if token_content:
                            full_response += token_content
                            token_count += 1
                            yield token_content
                    
                    # Wait for title generation to complete (with timeout)
                    title_thread.join(timeout=10)
                    if not title_queue.empty():
                        generated_title = title_queue.get()
                    else:
                        # Fallback if title generation failed or timed out
                        generated_title = generate_title(request.message)
                        logger.warning("Title generation timed out or failed, using fallback")
                    
                    logger.info(f"Generated title for RAG chat: {generated_title}")
                    
                    # Send title update immediately after streaming completes (before saving message)
                    if generated_title:
                        try:
                            # Refresh chat object to ensure we have the latest data
                            db.refresh(chat)
                            chat.title = generated_title
                            db.commit()
                            logger.info(f"RAG chat title updated to: {generated_title}")
                            # Send title update as special marker (frontend will parse this)
                            title_marker = f"<!-- TITLE_UPDATE:{generated_title} -->"
                            yield title_marker
                            logger.info(f"Title update marker sent for RAG chat: {title_marker}")
                        except Exception as e:
                            db.rollback()
                            logger.warning(f"Could not update RAG chat title: {e}")
                else:
                    # Regular streaming for subsequent messages
                    logger.info("Starting RAG chain stream")
                    token_count = 0
                    for token in chain.stream(prompt_input):
                        # Extract text from token content
                        content = token.content if hasattr(token, 'content') else token
                        token_content = extract_text_from_content(content)
                        
                        if token_content:
                            full_response += token_content
                            token_count += 1
                            yield token_content
                
                logger.info(f"RAG stream completed. Response length: {len(full_response)}")
                
                # Save assistant message (if database is available)
                try:
                    assistant_message = Message(chat_id=request.chat_id, role="assistant", content=full_response)
                    db.add(assistant_message)
                    db.commit()
                    logger.info("RAG assistant message saved to database")
                except Exception as e:
                    db.rollback()
                    logger.warning(f"Could not save assistant message to database: {e}")
            except Exception as e:
                logger.error(f"Error in RAG stream generator: {str(e)}", exc_info=True)
                error_msg = f"\n\nError: {str(e)}"
                yield error_msg
                # Try to save error message
                try:
                    assistant_message = Message(chat_id=request.chat_id, role="assistant", content=error_msg)
                    db.add(assistant_message)
                    db.commit()
                except:
                    db.rollback()
        
        return StreamingResponse(
            generate(), 
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    else:
        raise HTTPException(status_code=400, detail="Invalid chat_type")

# ========== RAG ENDPOINTS ==========
@app.post("/yt_rag")
def create_youtube_rag(request: RAGRequest, current_user: User = Depends(get_current_user)):
    """Create RAG vector store from YouTube video"""
    try:
        logger.info(f"Creating YouTube RAG for user {current_user.id}, URL: {request.url}")
        transcript = youtube_loader(request.url)
        if not transcript or transcript.strip() == "":
            logger.warning(f"Empty transcript extracted from YouTube URL: {request.url}")
            raise HTTPException(status_code=400, detail="Could not extract transcript from YouTube video. Please check if the video has captions enabled.")
        
        chunk_size, chunk_overlap = get_dynamic_chunk_size(transcript)
        split_documents = split_text(transcript, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        current_millis = int(time.time() * 1000)
        collection_name = f"{current_user.id}_{current_millis}"
        current_dir = os.getcwd()
        create_vector_store(split_documents, collection_name=collection_name, persist_dir=current_dir)
        logger.info(f"Successfully created YouTube RAG collection: {collection_name}")
        return {"collection_name": collection_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating YouTube RAG: {str(e)}", exc_info=True)
        error_message = str(e)
        if "transcript" in error_message.lower() or "caption" in error_message.lower():
            raise HTTPException(status_code=400, detail="Could not extract transcript from YouTube video. Please ensure the video has captions enabled.")
        raise HTTPException(status_code=500, detail=f"Failed to process YouTube video: {error_message}")

@app.post("/git_rag")
def create_github_rag(request: RAGRequest, current_user: User = Depends(get_current_user)):
    """Create RAG vector store from GitHub repository with FULL context (code + issues + PRs + commits + branches)"""
    try:
        logger.info(f"Creating Git RAG for user {current_user.id}, URL: {request.url}")
        token = get_github_token()
        if not token:
            raise HTTPException(status_code=500, detail="GitHub token not configured")

        # Build full context: code + issues + PRs + commits + branches
        full_context = github_service.build_full_context(request.url, branch="main", token=token)
        if not full_context or len(full_context) == 0:
            logger.warning(f"No content found in Git repository: {request.url}")
            raise HTTPException(status_code=400, detail="Could not access Git repository or repository is empty.")
        
        chunk_size, chunk_overlap = get_dynamic_chunk_size(full_context)
        split_documents = split_text(full_context, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        current_millis = int(time.time() * 1000)
        collection_name = f"{current_user.id}_{current_millis}"
        current_dir = os.getcwd()
        create_vector_store(split_documents, collection_name=collection_name, persist_dir=current_dir)
        logger.info(f"Successfully created Git RAG collection: {collection_name}")
        return {"collection_name": collection_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating Git RAG: {str(e)}", exc_info=True)
        error_message = str(e)
        if "not found" in error_message.lower() or "404" in error_message.lower():
            raise HTTPException(status_code=404, detail="Git repository not found. Please check the URL and ensure the repository exists and is accessible.")
        if "private" in error_message.lower() or "access" in error_message.lower():
            raise HTTPException(status_code=403, detail="Cannot access private repository. Please ensure the repository is public or provide proper authentication.")
        raise HTTPException(status_code=500, detail=f"Failed to process Git repository: {error_message}")

@app.post("/pdf_rag")
async def create_pdf_rag(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    """Create RAG vector store from PDF file"""
    temp_path = None
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name

        pdf_docs = list(load_pdf(temp_path))
        if not pdf_docs:
            raise HTTPException(status_code=400, detail="Could not load or parse PDF")

        full_text = "\n".join([doc.page_content for doc in pdf_docs])
        chunk_size, chunk_overlap = get_dynamic_chunk_size(full_text)
        split_documents = split_text(
            full_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        current_millis = int(time.time() * 1000)
        collection_name = f"{current_user.id}_{current_millis}"
        current_dir = os.getcwd()
        create_vector_store(split_documents, collection_name=collection_name, persist_dir=current_dir)

        return {"collection_name": collection_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@app.post("/web_rag")
def create_web_rag(request: RAGRequest, current_user: User = Depends(get_current_user)):
    """Create RAG vector store from webpage"""
    try:
        logger.info(f"Creating Web RAG for user {current_user.id}, URL: {request.url}")
        webpage_text = web_loader(request.url)
        if not webpage_text or webpage_text.strip() == "":
            logger.warning(f"Empty content extracted from webpage: {request.url}")
            raise HTTPException(status_code=400, detail="Could not extract text from webpage. The page may be empty, require JavaScript, or be inaccessible.")

        chunk_size, chunk_overlap = get_dynamic_chunk_size(webpage_text)
        split_documents = split_text(
            webpage_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        current_millis = int(time.time() * 1000)
        collection_name = f"{current_user.id}_{current_millis}"
        current_dir = os.getcwd()
        create_vector_store(split_documents, collection_name=collection_name, persist_dir=current_dir)
        logger.info(f"Successfully created Web RAG collection: {collection_name}")
        return {"collection_name": collection_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating Web RAG: {str(e)}", exc_info=True)
        error_message = str(e)
        if "not found" in error_message.lower() or "404" in error_message.lower():
            raise HTTPException(status_code=404, detail="Webpage not found. Please check the URL and ensure it's accessible.")
        if "timeout" in error_message.lower() or "connection" in error_message.lower():
            raise HTTPException(status_code=408, detail="Connection timeout. The webpage may be slow or inaccessible.")
        raise HTTPException(status_code=500, detail=f"Failed to process webpage: {error_message}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== HOME ROUTE ==========
@app.get("/")
def home():
    return {
        "message": "Welcome to the Streaming ChatBot API!",
        "version": "2.0.0"
    }

# ========== DEBUG ENDPOINT ==========
@app.get("/debug_token")
def debug_token():
    """Debug endpoint to check GitHub token status"""
    token = get_github_token()
    
    # Check env vars directly
    env_vars = {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "Present" if os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") else "Missing",
        "GITHUB_ACCESS_TOKEN": "Present" if os.environ.get("GITHUB_ACCESS_TOKEN") else "Missing",
    }
    
    # Check .env file visibility
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, ".env")
    env_file_exists = os.path.exists(env_path)
    
    return {
        "has_token": bool(token),
        "token_prefix": token[:4] if token else None,
        "token_suffix": token[-4:] if token else None,
        "token_length": len(token) if token else 0,
        "env_file_path": env_path,
        "env_file_exists": env_file_exists,
        "os_environ_status": env_vars,
        "cwd": os.getcwd()
    }


# ========== GIT PROJECT MANAGEMENT ENDPOINTS ==========

@app.get("/projects", response_model=List[GitProjectResponse])
def list_projects(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all Git projects for current user"""
    try:
        projects = db.query(GitProject).filter(GitProject.user_id == current_user.id).order_by(GitProject.created_at.desc()).all()
        return [
            GitProjectResponse(
                id=p.id,
                repo_url=p.repo_url,
                repo_owner=p.repo_owner,
                repo_name=p.repo_name,
                branch=p.branch,
                vector_db_collection_id=p.vector_db_collection_id,
                last_synced_at=p.last_synced_at.isoformat() if p.last_synced_at else None,
                created_at=p.created_at.isoformat()
            )
            for p in projects
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")


@app.post("/projects", response_model=GitProjectResponse)
def create_project(
    project_data: GitProjectCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a new GitHub repo project — ingests full context and optionally registers a webhook"""
    try:
        token = get_github_token()
        if not token:
            raise HTTPException(status_code=500, detail="GitHub token not configured")

        # Parse owner/repo
        owner, repo_name = github_service._parse_owner_and_name(project_data.repo_url)

        # Check for duplicate
        existing = db.query(GitProject).filter(
            GitProject.user_id == current_user.id,
            GitProject.repo_owner == owner,
            GitProject.repo_name == repo_name
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="This repository is already added as a project")

        # Build full context and vectorize
        logger.info(f"Creating project for {project_data.repo_url} ({project_data.branch})")
        full_context = github_service.build_full_context(project_data.repo_url, project_data.branch, token)
        if not full_context or len(full_context) == 0:
            raise HTTPException(status_code=400, detail="Could not access repository or it is empty")

        chunk_size, chunk_overlap = get_dynamic_chunk_size(full_context)
        split_documents = split_text(full_context, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        current_millis = int(time.time() * 1000)
        collection_name = f"{current_user.id}_{current_millis}"
        current_dir = os.getcwd()
        create_vector_store(split_documents, collection_name=collection_name, persist_dir=current_dir)

        # Generate webhook secret
        webhook_secret = github_service.generate_webhook_secret()

        # Try registering webhook (non-blocking, may fail for repos user doesn't own)
        callback_url = os.getenv("WEBHOOK_CALLBACK_URL", "")
        if callback_url:
            try:
                github_service.register_webhook(project_data.repo_url, token, callback_url, webhook_secret)
            except Exception as e:
                logger.warning(f"Webhook registration failed (non-fatal): {e}")

        # Save project to DB
        now = datetime.utcnow()
        new_project = GitProject(
            user_id=current_user.id,
            repo_url=project_data.repo_url,
            repo_owner=owner,
            repo_name=repo_name,
            branch=project_data.branch,
            vector_db_collection_id=collection_name,
            webhook_secret=webhook_secret,
            last_synced_at=now,
        )
        db.add(new_project)
        db.commit()
        db.refresh(new_project)

        logger.info(f"Project created: {new_project.id} for {owner}/{repo_name}")

        return GitProjectResponse(
            id=new_project.id,
            repo_url=new_project.repo_url,
            repo_owner=new_project.repo_owner,
            repo_name=new_project.repo_name,
            branch=new_project.branch,
            vector_db_collection_id=new_project.vector_db_collection_id,
            last_synced_at=new_project.last_synced_at.isoformat() if new_project.last_synced_at else None,
            created_at=new_project.created_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating project: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")


@app.delete("/projects/{project_id}")
def delete_project(project_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete a Git project and its associated data"""
    try:
        project = db.query(GitProject).filter(
            GitProject.id == project_id, GitProject.user_id == current_user.id
        ).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Try to delete the vector store collection
        if project.vector_db_collection_id:
            try:
                current_dir = os.getcwd()
                vs = load_vector_store(project.vector_db_collection_id, current_dir)
                vs.delete_collection()
                logger.info(f"Deleted vector store collection: {project.vector_db_collection_id}")
            except Exception as e:
                logger.warning(f"Could not delete vector store: {e}")

        db.delete(project)
        db.commit()
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")


def _sync_project_background(project_id: int, db_url: str):
    """Background task to re-ingest a project's repo context"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    sync_engine = create_engine(db_url)
    SyncSession = sessionmaker(bind=sync_engine)
    db = SyncSession()

    try:
        project = db.query(GitProject).filter(GitProject.id == project_id).first()
        if not project:
            logger.error(f"Sync: project {project_id} not found")
            return

        token = get_github_token()
        if not token:
            logger.error("Sync: no GitHub token available")
            return

        logger.info(f"Syncing project {project_id}: {project.repo_owner}/{project.repo_name}")

        # Build fresh context
        full_context = github_service.build_full_context(project.repo_url, project.branch, token)
        if not full_context:
            logger.warning(f"Sync: empty context for project {project_id}")
            return

        # Delete old vector store
        if project.vector_db_collection_id:
            try:
                current_dir = os.getcwd()
                vs = load_vector_store(project.vector_db_collection_id, current_dir)
                vs.delete_collection()
            except Exception:
                pass

        # Create new vector store
        chunk_size, chunk_overlap = get_dynamic_chunk_size(full_context)
        split_documents = split_text(full_context, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        current_millis = int(time.time() * 1000)
        collection_name = f"{project.user_id}_{current_millis}"
        current_dir = os.getcwd()
        create_vector_store(split_documents, collection_name=collection_name, persist_dir=current_dir)

        # Update project
        project.vector_db_collection_id = collection_name
        project.last_synced_at = datetime.utcnow()
        db.commit()
        logger.info(f"Project {project_id} synced successfully: {collection_name}")
    except Exception as e:
        db.rollback()
        logger.error(f"Error syncing project {project_id}: {e}", exc_info=True)
    finally:
        db.close()


@app.post("/projects/{project_id}/sync")
def sync_project(
    project_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Manually re-sync a project (re-ingest full context)"""
    try:
        project = db.query(GitProject).filter(
            GitProject.id == project_id, GitProject.user_id == current_user.id
        ).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get the database URL for the background task
        from .database import DATABASE_URL
        background_tasks.add_task(_sync_project_background, project_id, DATABASE_URL)

        return {"status": "syncing", "message": "Project sync started in background"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start sync: {str(e)}")


# ========== GITHUB WEBHOOK ENDPOINT ==========

@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Receive GitHub webhook events. On push events, re-ingest the affected project.
    Verifies HMAC-SHA256 signature for security.
    """
    try:
        event_type = request.headers.get("X-GitHub-Event", "")
        payload_body = await request.body()

        if not event_type:
            raise HTTPException(status_code=400, detail="Missing X-GitHub-Event header")

        # Parse payload
        import json
        try:
            payload = json.loads(payload_body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        # Extract repo info from payload
        repo_info = payload.get("repository", {})
        repo_full_name = repo_info.get("full_name", "")
        if not repo_full_name:
            raise HTTPException(status_code=400, detail="Missing repository info in payload")

        owner, repo_name = repo_full_name.split("/")

        # Find matching project(s)
        projects = db.query(GitProject).filter(
            GitProject.repo_owner == owner,
            GitProject.repo_name == repo_name
        ).all()

        if not projects:
            logger.info(f"Webhook received for untracked repo: {repo_full_name}")
            return {"status": "ignored", "message": "No matching project found"}

        # Verify signature against each project's secret
        signature = request.headers.get("X-Hub-Signature-256", "")
        verified_project = None
        for project in projects:
            if project.webhook_secret and github_service.verify_webhook_signature(
                payload_body, signature, project.webhook_secret
            ):
                verified_project = project
                break

        # If no signature match, try global secret
        if not verified_project:
            global_secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
            if global_secret and github_service.verify_webhook_signature(payload_body, signature, global_secret):
                verified_project = projects[0]  # Use first matching project

        if not verified_project:
            # If webhook verification is not strictly enforced in dev, still allow
            logger.warning(f"Webhook signature verification failed for {repo_full_name}, processing anyway in dev mode")
            verified_project = projects[0]

        # Log the webhook event
        event_summary = ""
        if event_type == "push":
            ref = payload.get("ref", "")
            pusher = payload.get("pusher", {}).get("name", "unknown")
            commits_count = len(payload.get("commits", []))
            event_summary = f"Push to {ref} by {pusher} ({commits_count} commits)"
        elif event_type == "issues":
            action = payload.get("action", "")
            issue_title = payload.get("issue", {}).get("title", "")
            event_summary = f"Issue {action}: {issue_title}"
        elif event_type == "pull_request":
            action = payload.get("action", "")
            pr_title = payload.get("pull_request", {}).get("title", "")
            event_summary = f"PR {action}: {pr_title}"
        else:
            event_summary = f"Event: {event_type}"

        webhook_event = GitWebhookEvent(
            project_id=verified_project.id,
            event_type=event_type,
            payload_summary=event_summary,
            processed=False
        )
        db.add(webhook_event)
        db.commit()

        # On push, issues, or PR events — trigger re-ingest
        if event_type in ["push", "issues", "pull_request"]:
            from .database import DATABASE_URL
            background_tasks.add_task(_sync_project_background, verified_project.id, DATABASE_URL)

            # Mark event as processed
            webhook_event.processed = True
            db.commit()

            logger.info(f"Webhook {event_type} for {repo_full_name}: re-ingestion triggered")
            return {"status": "processing", "event": event_type, "summary": event_summary}

        return {"status": "received", "event": event_type, "summary": event_summary}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")


# ========== REPORT GENERATION ENDPOINT ==========

REPORT_PROMPTS = {
    "full": """You are a senior software architect analyzing a GitHub repository.
Based on the following repository context, generate a comprehensive report covering:

1. **Repository Overview** — What this project is about, its purpose, and tech stack
2. **Architecture** — How the codebase is structured, main modules/packages, design patterns
3. **Dependencies** — Key libraries and frameworks used
4. **Code Quality** — Code organization, naming conventions, potential improvements
5. **Activity Summary** — Recent commits, open issues, active PRs
6. **Recommendations** — Suggestions for improvement

Format the report in clean markdown with headers and bullet points.

CONTEXT:
{context}

Generate the repository report:""",

    "architecture": """You are a senior software architect.
Based on the following repository context, generate a detailed architecture analysis covering:

1. **System Architecture** — High-level overview, components, layers
2. **Module Structure** — How code is organized into modules/packages
3. **Design Patterns** — Patterns identified in the codebase
4. **Data Flow** — How data moves through the system
5. **API Design** — Endpoint structure and design choices

CONTEXT:
{context}

Generate the architecture analysis:""",

    "dependencies": """You are a dependency analysis expert.
Based on the following repository context, analyze all dependencies:

1. **Core Dependencies** — Essential libraries and their purposes
2. **Dev Dependencies** — Testing, linting, build tools
3. **Dependency Health** — Version freshness, known issues
4. **Security Considerations** — Any concerning patterns
5. **Optimization** — Unnecessary or duplicate dependencies

CONTEXT:
{context}

Generate the dependency analysis:""",

    "code_quality": """You are a code quality expert.
Based on the following repository context, assess code quality:

1. **Code Organization** — File structure, naming conventions
2. **Best Practices** — Adherence to language-specific best practices
3. **Error Handling** — How errors are managed
4. **Testing** — Test coverage indicators
5. **Documentation** — Inline docs, README quality
6. **Technical Debt** — Areas needing refactoring
7. **Actionable Improvements** — Specific suggestions with priority

CONTEXT:
{context}

Generate the code quality assessment:""",
}


@app.post("/git_report")
def generate_git_report(
    report_request: GitReportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate a structured report from a Git project's vectorized context"""
    try:
        # Find the project
        project = db.query(GitProject).filter(
            GitProject.id == report_request.project_id,
            GitProject.user_id == current_user.id
        ).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        if not project.vector_db_collection_id:
            raise HTTPException(status_code=400, detail="Project has no vectorized data. Please sync first.")

        # Load vector store and retrieve broad context
        current_dir = os.getcwd()
        vector_store = load_vector_store(project.vector_db_collection_id, current_dir)
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 20})

        # Use multiple queries to get diverse context
        queries = [
            "project overview architecture main functionality",
            "dependencies imports packages libraries",
            "recent issues bugs pull requests",
            "code structure classes functions",
            "commit history recent changes",
        ]

        all_contexts = set()
        for query in queries:
            docs = retriever.invoke(query)
            for doc in docs:
                all_contexts.add(doc.page_content)

        context_text = "\n\n---\n\n".join(all_contexts)
        logger.info(f"Report context: {len(all_contexts)} unique chunks, {len(context_text)} chars")

        # Get the appropriate prompt
        report_type = report_request.report_type
        if report_type not in REPORT_PROMPTS:
            report_type = "full"

        prompt_template = REPORT_PROMPTS[report_type]
        prompt = PromptTemplate.from_template(prompt_template)

        llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.3,
            api_key=os.getenv("MISTRAL_API_KEY"),
        )

        chain = prompt | llm
        response = chain.invoke({"context": context_text})
        report_content = extract_text_from_content(response.content)

        logger.info(f"Report generated for project {project.id}: {len(report_content)} chars")

        return {
            "report": report_content,
            "project": {
                "id": project.id,
                "repo_url": project.repo_url,
                "repo_name": f"{project.repo_owner}/{project.repo_name}",
                "branch": project.branch,
                "last_synced_at": project.last_synced_at.isoformat() if project.last_synced_at else None,
            },
            "report_type": report_type,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

