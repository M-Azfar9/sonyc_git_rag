from fastapi import FastAPI, UploadFile, File

from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import WebBaseLoader
from typing import Literal
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import GithubFileLoader
import time
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

app = FastAPI()
load_dotenv()

# ========== SIMPLE IN-MEMORY MEMORY STORE ==========
memory = {}

SYSTEM_MSG = SystemMessage(content="""
You are an assistant whose top priorities are accuracy, clarity, and user safety. 
Always verify facts before presenting them; when a fact could be time-sensitive or uncertain, explicitly say “I don't know” / “I’m not sure” instead of guessing. 
If the user’s question is ambiguous, ask one short clarifying question. 
Cite sources for non-common-knowledge claims. 
If asked for instructions that could be harmful, refuse and provide a safe alternative. 
Keep answers concise, show the final answer first, and then provide a short explanation and sources.
""")

# ========== MODELS ==========
class ChatRequest(BaseModel):
    user_id: str
    message: str
    chat_type: Literal['normal_chat', 'yt_chat', 'pdf_chat', 'web_chat', 'git_chat']
    vector_db_collection_id: str

class LLMResponse(BaseModel):
    answer: str
    source: str

# ========== STREAM RESPONSE GENERATOR ==========
async def stream_answer(history):
    """
    Streams the assistant's reply token by token.
    Uses a regular generator from ChatGoogleGenerativeAI.
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview",
        temperature=0
    )

    stream = model.stream(history)  # regular generator

    full_response = ""
    for chunk in stream:  # regular for, not async for
        token = chunk.content
        full_response += token
        yield token  # stream token to client

    # Save assistant final message to memory
    history.append(AIMessage(content=full_response))

def get_dynamic_chunk_size(text: str):
    """
    Dynamically decide chunk_size and chunk_overlap based on document length.
    Works for short documents, transcripts, and large academic books.
    """
    length = len(text)  # number of characters (or words if you prefer)
    
    # Basic adaptive scaling
    if length < 1000:
        chunk_size = length/2
        chunk_overlap = 20
    elif length < 5000:
        chunk_size = length/5
        chunk_overlap = 50
    elif length < 20000:
        chunk_size = length/20
        chunk_overlap = 100
    elif length < 100000:  # medium-length book
        chunk_size = length/80
        chunk_overlap = 200
    elif length < 300000:  # large textbook
        chunk_size = length/200
        chunk_overlap = 400
    else:  # very large book (e.g., 500k+ words)
        chunk_size = 6000
        chunk_overlap = 600

    return int(chunk_size), int(chunk_overlap)


def youtube_loader(url:str):
    video_id =  url.split("v=")[1].split("&")[0]
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id)
    transcript = " ".join(chunk.text for chunk in transcript_list)
    return transcript

def load_pdf(file_path: str):
    """Lazy loads a PDF"""
    loader = PyPDFLoader(file_path)
    return loader.lazy_load()  # generator of Document objects

def is_text_file(file_path: str) -> bool:
    """
    Return True for files that are likely text/code and skip binaries.
    Common extensions from GitHub repos.
    """
    # text_extensions = (
    #     # Programming languages
    #     ".py", ".cpp", ".c", ".h", ".hpp", ".java", ".js", ".ts", ".jsx", ".tsx", ".go",
    #     ".rs", ".cs", ".php", ".rb", ".swift", ".m", ".scala", ".kt", ".kts", ".dart",
    #     ".pl", ".pm", ".lua", ".r", ".jl", ".asm", ".s", ".sh", ".bash",

    #     # Markup and web
    #     ".html", ".htm", ".css", ".scss", ".sass", ".less", ".xml", ".xhtml", ".svg", ".json",
    #     ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".md", ".markdown",

    #     # Data and configs
    #     ".csv", ".tsv", ".log", ".txt", ".rst", ".bib",

    #     # SQL / Database
    #     ".sql", ".sqlite", ".db", ".prisma",

    #     # Notebook / AI / ML
    #     ".ipynb",

    #     # Others (scripts, templates)
    #     ".dockerfile", ".makefile", ".gradle", ".bat", ".ps1", ".env", ".gitignore", ".gitattributes",

    #     # Additional common text/code file extensions
    #     ".vue", ".svelte", ".elm", ".coffee", ".ejs", ".mustache", ".tpl", ".hcl", ".proto", ".thrift"
    # )
    text_extensions = (
        ".txt", ".md", ".html", ".css", ".xml", ".json", ".yaml", ".yml", ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt", ".kts", ".scala", ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".swift", ".m", ".php", ".rb", ".pl", ".pm", ".lua", ".sh", ".bash", ".r", ".jl", ".asm", ".s", ".dart", ".cs",
    )
    return text_extensions


def github_loader(repo_url, branch="main", file_filter=None, folder=None):
    repo_id = convert_github_url_to_repo_id(repo_url)
    loader = GithubFileLoader(
        repo=repo_id,
        branch=branch,
        file_filter=lambda file_path: file_path.endswith((
".txt", ".md", ".html", ".css", ".xml", ".json", ".yaml", ".yml", ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt", ".kts", ".scala", ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".swift", ".m", ".php", ".rb", ".pl", ".pm", ".lua", ".sh", ".bash", ".r", ".jl", ".asm", ".s", ".dart", ".cs", ".ipynb"
    )),
        access_token=os.environ.get("GITHUB_ACCESS_TOKEN"),
    )
    docs = loader.load()
    # Convert docs to a single text string
    full_text = ""
    for i, doc in enumerate(docs, start=1):
        file_name = doc.metadata.get("source", f"file_{i}")  # file path or name
        
        full_text += f"\n\n===== FILE {i}: {file_name} =====\n"
        full_text += doc.page_content
    return full_text

def convert_github_url_to_repo_id(github_url: str) -> str:
    """
    Converts any GitHub URL into owner/repo format.
    Supports URLs with trailing slashes, tree paths, etc.
    """

    import re

    # remove protocol
    cleaned = github_url.replace("https://", "").replace("http://", "")

    # expect: github.com/owner/repo/...
    parts = cleaned.split("/")

    if len(parts) < 3:
        raise ValueError("Invalid GitHub URL format")

    owner = parts[1]
    repo = parts[2]

    return f"{owner}/{repo}"


def web_loader(url: str):
    """Loads webpage and returns text as string."""
    loader = WebBaseLoader(url)
    docs = loader.load()

    if not docs:
        return ""

    return "\n\n".join([d.page_content for d in docs])

# def create_youtube_rag(url:str):
#     transcript = youtube_loader(url)
#     chunk_size, chunk_overlap = get_dynamic_chunk_size(transcript)
#     split_documents = split_text(transcript, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     vector_stores = create_vector_store(split_documents, collection_name=, persist_dir=)


# ========== Splitting Text ==========
def split_text(text: str, chunk_size=None, chunk_overlap=None):
    """
    Split raw text into smaller chunks using LangChain's text splitter.
    Automatically uses dynamic chunk size if not provided.
    Returns a list of strings.
    """
    if chunk_size is None or chunk_overlap is None:
        chunk_size, chunk_overlap = get_dynamic_chunk_size(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_text(text)
    return chunks


# ========== Splitting Documents ==========
def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split Document objects into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for doc in docs:
        all_chunks.extend(splitter.split_documents([doc]))
    return all_chunks


# ========== Create and Load Vector Stores ==========
def create_vector_store(chunks, collection_name: str, persist_dir: str):
    """Create a Chroma vector store from chunks"""
    docs = [Document(page_content=chunk) for chunk in chunks]
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )
    vector_store.add_documents(docs)  # ✅ pass Document objects
    vector_store.persist()            # ✅ persist to disk
    return vector_store


# ========== Load Vector Stores ==========
def load_vector_store(collection_name: str, persist_dir: str):
    """Load an existing Chroma vector store."""
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )
    return vector_store


# ========== Retriever ==========
def create_retriever(vector_store, k=7):
    """Returns a retriever for RAG queries"""
    print('Hello world')
    retriever = vector_store.as_retriever(serch_type='mmr',kwargs={"k": k})
    return retriever

# ========== Fetcher ==========
def fetch_documents_from_db(retriever, query):
    """Returns fetched documents for RAG queries"""
    return retriever.invoke(query)


# ========== General Prompt Template ==========
def get_rag_prompt():
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


# ========== HOME ROUTE ==========
@app.get("/")
def home():
    return {
        "message": "Welcome to the Streaming ChatBot API!",
        "endpoints": {
            "/chat/stream": "POST endpoint for streaming chat. Requires JSON: {user_id, message}",
            "/reset": "POST endpoint to reset a user's memory. Query param: user_id"
        }
    }

# ========== CHAT ENDPOINT (STREAMING) ==========
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    user_id = request.user_id
    chat_type = request.chat_type
    vector_db_collection_id = request.vector_db_collection_id

    if chat_type == "normal_chat":
        # Memory-based chat
        if user_id not in memory:
            memory[user_id] = [SYSTEM_MSG]

        history = memory[user_id]
        history.append(HumanMessage(content=request.message))

        return StreamingResponse(
            stream_answer(history),
            media_type="text/plain"
        )

    elif chat_type in ["yt_chat", "pdf_chat", "web_chat", "git_chat"]:
        # Simple RAG (no memory)
        current_dir = os.getcwd()
        try:
            vector_store = load_vector_store(
                collection_name=vector_db_collection_id,
                persist_dir=current_dir
            )
        except Exception as e:
            return {"error": f"Vector store not found: {e}"}
        print(vector_store)

        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        print("vector_store")
        context_docs = retriever.invoke(request.message)
        print(context_docs)
        context_text = "\n".join(doc.page_content for doc in context_docs)

        # Use RAG prompt
        rag_prompt = get_rag_prompt()
        print("rag_prompt = get_rag_prompt()")
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview",
            temperature=0.5,
            streaming=True  # Enable streaming
        )
        
        chain = rag_prompt | llm
        prompt_input = {'context': context_text, 'question': request.message}
        # return chain.invoke(prompt_input)
        # Wrap streaming logic in a generator
        def generate():
            for token in chain.stream(prompt_input):
                yield token.content

        return StreamingResponse(generate(), media_type="text/plain")

    else:
        return {"error": "Invalid chat_type"}


# ========== Youtube Rag ENDPOINT  ==========
@app.post("/yt_rag")
def create_youtube_rag(url:str, user_id:str):
    transcript = youtube_loader(url)
    chunk_size, chunk_overlap = get_dynamic_chunk_size(transcript)
    split_documents = split_text(transcript, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    current_millis = int(time.time() * 1000)
    collection_name = f"{user_id}_{current_millis}"
    current_dir = os.getcwd()
    vector_stores = create_vector_store(split_documents, collection_name=collection_name, persist_dir=current_dir)
    return collection_name


# ========== Youtube Rag ENDPOINT  ==========
@app.post("/git_rag")
def create_github_rag(url:str, user_id:str):
    file_list = github_loader(url)
    
    chunk_size, chunk_overlap = get_dynamic_chunk_size(file_list)
    split_documents = split_text(file_list, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    current_millis = int(time.time() * 1000)
    collection_name = f"{user_id}_{current_millis}"
    current_dir = os.getcwd()
    vector_stores = create_vector_store(split_documents, collection_name=collection_name, persist_dir=current_dir)
    return collection_name


# ========== PDF Rag ENDPOINT  ==========
@app.post("/pdf_rag")
def create_pdf_rag(file: UploadFile, user_id: str):
    try:
        # 1. Save UploadFile to a temporary .pdf file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.file.read())
            temp_path = tmp_file.name   # path to use with load_pdf()

        # 2. Lazy load PDF using your existing function
        pdf_docs = list(load_pdf(temp_path))  # generator → list of Document objects

        if not pdf_docs:
            return {"error": "Could not load or parse PDF."}

        # 3. Convert Document objects into text chunks
        full_text = "\n".join([doc.page_content for doc in pdf_docs])

        # 4. Dynamic chunk sizing
        chunk_size, chunk_overlap = get_dynamic_chunk_size(full_text)

        # 5. Split documents
        split_documents = split_text(
            full_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 6. Create unique collection name
        current_millis = int(time.time() * 1000)
        collection_name = f"{user_id}_{current_millis}"

        # 7. Create vector store
        current_dir = os.getcwd()
        vector_store = create_vector_store(
            split_documents,
            collection_name=collection_name,
            persist_dir=current_dir
        )

        return {"collection_name": collection_name}

    except Exception as e:
        return {"error": str(e)}

# ========== Web Rag ENDPOINT  ==========
@app.post("/web_rag")
def create_web_rag(url: str, user_id: str):
    try:
        # 1. Load the webpage content
        webpage_text = web_loader(url)
        if not webpage_text or webpage_text.strip() == "":
            return {"error": "Could not extract text from webpage."}

        # 2. Dynamic chunk sizing
        chunk_size, chunk_overlap = get_dynamic_chunk_size(webpage_text)

        # 3. Split into chunks
        split_documents = split_text(
            webpage_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 4. Create unique collection name
        current_millis = int(time.time() * 1000)
        collection_name = f"{user_id}_{current_millis}"

        # 5. Persist vector DB
        current_dir = os.getcwd()
        vector_store = create_vector_store(
            split_documents,
            collection_name=collection_name,
            persist_dir=current_dir
        )

        return {"collection_name": collection_name}

    except Exception as e:
        return {"error": str(e)}


# ========== RESET MEMORY ==========
@app.post("/reset")
def reset_memory(user_id: str):
    if user_id in memory:
        del memory[user_id]
        return {"status": "memory cleared for user", "user_id": user_id}
    return {"status": "user not found", "user_id": user_id}