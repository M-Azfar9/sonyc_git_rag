"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import type { Chat, ChatType, Message } from "@/lib/types";
import { AppSidebar } from "@/components/chat/app-sidebar";
import { NewChatDialog } from "@/components/chat/new-chat-dialog";
import { ChatView } from "@/components/chat/chat-view";
import {
  Sidebar,
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { SourceInputDialog } from "@/components/chat/source-input-dialog";
import { api } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

// Map frontend chat types to backend chat types
const mapToBackendType = (type: ChatType): string => {
  const mapping: Record<ChatType, string> = {
    Normal: "normal_chat",
    YouTube: "yt_chat",
    Web: "web_chat",
    Git: "git_chat",
    PDF: "pdf_chat",
  };
  return mapping[type];
};

export default function ChatPage() {
  const router = useRouter();
  const { toast } = useToast();
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isNewChatDialogOpen, setNewChatDialogOpen] = useState(false);
  const [isSourceDialogOpen, setSourceDialogOpen] = useState(false);
  const [sourceType, setSourceType] = useState<ChatType | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);

  const activeChat = chats.find((chat) => chat.id === activeChatId);

  const handleLogout = () => {
    api.logout();
    toast({
      title: "Logged out",
      description: "You have been successfully logged out.",
    });
    router.push("/");
  };

  // Check authentication and load chats on mount
  useEffect(() => {
    // Check if user is authenticated
    const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
    if (!token) {
      router.push('/');
      return;
    }
    
    // Verify token is valid by checking current user
    api.getCurrentUser()
      .then(() => {
        // Token is valid, load chats
        loadChats();
      })
      .catch(() => {
        // Token is invalid, redirect to login
        router.push('/');
      });
  }, [router]);

  // Load messages when chat is selected
  useEffect(() => {
    if (activeChatId) {
      loadMessages(activeChatId);
    } else {
      setMessages([]);
    }
  }, [activeChatId]);

  const loadChats = async () => {
    try {
      setIsLoading(true);
      const backendChats = await api.getChats();
      const frontendChats: Chat[] = backendChats.map((chat) => ({
        id: chat.id.toString(),
        title: chat.title,
        type: chat.type as ChatType,
        createdAt: new Date(chat.created_at),
        vector_db_collection_id: chat.vector_db_collection_id,
      }));
      setChats(frontendChats);
      if (frontendChats.length > 0 && !activeChatId) {
        setActiveChatId(frontendChats[0].id);
      }
    } catch (error: any) {
      console.error("Error loading chats:", error);
      if (error.message === "Unauthorized") {
        router.push("/");
      } else {
        toast({
          title: "Error",
          description: "Failed to load chats",
          variant: "destructive",
        });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const loadMessages = async (chatId: string) => {
    try {
      setIsLoadingMessages(true);
      const backendMessages = await api.getChatMessages(parseInt(chatId));
      const frontendMessages: Message[] = backendMessages.map((msg) => ({
        id: msg.id.toString(),
        role: msg.role as "user" | "assistant",
        content: msg.content,
      }));
      setMessages(frontendMessages);
    } catch (error: any) {
      console.error("Error loading messages:", error);
      toast({
        title: "Error",
        description: "Failed to load messages",
        variant: "destructive",
      });
    } finally {
      setIsLoadingMessages(false);
    }
  };

  const handleNewChatSelect = (type: ChatType) => {
    setNewChatDialogOpen(false);
    if (type === "Normal") {
      createNewChat(type);
    } else {
      setSourceType(type);
      setSourceDialogOpen(true);
    }
  };

  const createNewChat = async (type: ChatType, source?: string, collectionName?: string) => {
    try {
      const title = source ? `${type}: ${source}` : `${type} Chat`;
      const backendChat = await api.createChat({
        title,
        type,
        vector_db_collection_id: collectionName || undefined,
      });

    const newChat: Chat = {
        id: backendChat.id.toString(),
        title: backendChat.title,
        type: backendChat.type as ChatType,
        createdAt: new Date(backendChat.created_at),
        vector_db_collection_id: backendChat.vector_db_collection_id,
    };

    setChats([newChat, ...chats]);
    setActiveChatId(newChat.id);
    setMessages([]);
    setSourceDialogOpen(false);
    } catch (error: any) {
      console.error("Error creating chat:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to create chat",
        variant: "destructive",
      });
    }
  };

  const handleSelectChat = (chatId: string) => {
    setActiveChatId(chatId);
  };

  const handleSendMessage = async (content: string) => {
    if (!activeChat) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
    };
    setMessages((prev) => [...prev, userMessage]);

    const assistantId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
    };
    setMessages((prev) => [...prev, assistantMessage]);
    setStreamingMessageId(assistantId);

    try {
      const backendType = mapToBackendType(activeChat.type);
      console.log("Sending message:", {
        chatId: activeChat.id,
        chatType: activeChat.type,
        backendType,
        vectorDbCollectionId: activeChat.vector_db_collection_id,
        message: content.substring(0, 50) + "..."
      });
      
      let fullResponse = "";
      let chunkCount = 0;
      let lastUpdateLength = 0;
      let lastUpdateTime = Date.now();
      let rafId: number | null = null;

      const updateMessage = () => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId ? { ...msg, content: fullResponse } : msg
          )
        );
        lastUpdateLength = fullResponse.length;
        lastUpdateTime = Date.now();
        rafId = null;
      };

      await api.streamChat(
        parseInt(activeChat.id),
        content,
        backendType,
        activeChat.vector_db_collection_id || null,
        (chunk: string) => {
          chunkCount++;
          fullResponse = chunk;
          if (chunkCount === 1 || chunkCount % 20 === 0) {
            console.log(`Received chunk ${chunkCount}, length: ${chunk.length}`);
          }
          
          // More aggressive updates for smoother experience
          const contentChanged = fullResponse.length - lastUpdateLength > 10;
          const timeSinceLastUpdate = Date.now() - lastUpdateTime;
          
          // Update immediately on first chunk or significant changes
          if (chunkCount === 1 || contentChanged || timeSinceLastUpdate > 8) {
            // Cancel any pending RAF
            if (rafId) {
              cancelAnimationFrame(rafId);
            }
            
            // Use requestAnimationFrame for smooth, synchronized updates
            rafId = requestAnimationFrame(updateMessage);
          }
        },
        (newTitle: string) => {
          // Handle title update
          console.log('Title update received:', newTitle);
          setChats((prevChats) => {
            const updatedChats = prevChats.map((chat) =>
              chat.id === activeChat.id ? { ...chat, title: newTitle } : chat
            );
            console.log('Updated chats:', updatedChats);
            return updatedChats;
          });
          // Also reload chats to ensure consistency with backend
          loadChats().catch((err) => {
            console.error('Error reloading chats after title update:', err);
          });
        }
      );
      
      // Ensure final update
      if (rafId) {
        cancelAnimationFrame(rafId);
      }
      updateMessage();

      console.log(`Streaming completed. Total chunks: ${chunkCount}, Final length: ${fullResponse.length}`);

      // Final update to ensure all content is displayed
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId ? { ...msg, content: fullResponse } : msg
        )
      );

      // Clear streaming state
      setStreamingMessageId(null);

      // Reload messages to get the saved ones from backend
      await loadMessages(activeChat.id);
      
      // Reload chats to ensure title is updated in the sidebar
      await loadChats();
    } catch (error: any) {
      console.error("Error streaming chat:", error);
      console.error("Error details:", {
        message: error.message,
        stack: error.stack
      });
      setStreamingMessageId(null);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                content:
                  error.message ||
                  "Sorry, something went wrong. Please check the connection to the backend.",
              }
            : msg
        )
      );
      toast({
        title: "Error",
        description: error.message || "Failed to send message",
        variant: "destructive",
      });
    }
  };

  const handleRAGSourceCreation = async (type: ChatType, source: string | File) => {
    try {
      setSourceDialogOpen(false);
      toast({
        title: "Processing...",
        description: `Creating ${type} RAG source. This may take a moment.`,
      });

      let collectionName: string;
      const sourceUrl = source instanceof File ? source.name : source;

      try {
        if (type === "YouTube") {
          const result = await api.createYouTubeRAG(source as string);
          collectionName = result.collection_name;
        } else if (type === "Web") {
          const result = await api.createWebRAG(source as string);
          collectionName = result.collection_name;
        } else if (type === "Git") {
          const result = await api.createGitRAG(source as string);
          collectionName = result.collection_name;
        } else if (type === "PDF") {
          const result = await api.createPDFRAG(source as File);
          collectionName = result.collection_name;
        } else {
          throw new Error("Invalid source type");
        }
      } catch (ragError: any) {
        console.error(`Error creating ${type} RAG source:`, ragError);
        
        // Provide user-friendly error messages based on error content
        let errorMessage = ragError.message || `Failed to create ${type} RAG source`;
        
        if (errorMessage.includes("transcript") || errorMessage.includes("caption")) {
          errorMessage = "Could not extract transcript from YouTube video. Please ensure the video has captions enabled.";
        } else if (errorMessage.includes("not found") || errorMessage.includes("404")) {
          errorMessage = `The ${type === "Git" ? "repository" : type === "Web" ? "webpage" : "video"} was not found. Please check the URL.`;
        } else if (errorMessage.includes("private") || errorMessage.includes("access")) {
          errorMessage = `Cannot access ${type === "Git" ? "private repository" : "resource"}. Please ensure it's publicly accessible.`;
        } else if (errorMessage.includes("timeout") || errorMessage.includes("connection")) {
          errorMessage = `Connection timeout. The ${type === "Web" ? "webpage" : "resource"} may be slow or inaccessible.`;
        } else if (errorMessage.includes("extract text") || errorMessage.includes("empty")) {
          errorMessage = `Could not extract content from ${type === "Web" ? "webpage" : "source"}. The page may be empty or require JavaScript.`;
        }
        
        toast({
          title: "Error",
          description: errorMessage,
          variant: "destructive",
        });
        setSourceDialogOpen(true); // Reopen dialog so user can try again
        return;
      }

      await createNewChat(type, sourceUrl, collectionName);
      toast({
        title: "Success!",
        description: `${type} RAG source created successfully.`,
      });
    } catch (error: any) {
      console.error("Error in RAG source creation flow:", error);
      toast({
        title: "Error",
        description: error.message || `Failed to create ${type} RAG source`,
        variant: "destructive",
      });
      setSourceDialogOpen(true); // Reopen dialog so user can try again
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-[100dvh] w-full items-center justify-center">
        <div className="text-muted-foreground">Loading chats...</div>
      </div>
    );
  }

  return (
    <SidebarProvider>
      <main className="flex h-[100dvh] w-full text-foreground overflow-hidden">
        <Sidebar
          variant="inset"
          collapsible="icon"
          className="p-0 border-0 bg-transparent data-[variant=inset]:w-[19rem] data-[collapsible=icon]:data-[variant=inset]:w-[calc(var(--sidebar-width-icon)_+2rem)]"
        >
          <div className="h-full p-2">
            <AppSidebar
              chats={chats}
              activeChatId={activeChatId}
              onNewChatClick={() => setNewChatDialogOpen(true)}
              onSelectChat={handleSelectChat}
              onLogout={handleLogout}
              isMobile={false}
            />
          </div>
        </Sidebar>
        <SidebarInset>
          <NewChatDialog
            open={isNewChatDialogOpen}
            onOpenChange={setNewChatDialogOpen}
            onSelectChatType={handleNewChatSelect}
          />
           {sourceType && (
            <SourceInputDialog
              open={isSourceDialogOpen}
              onOpenChange={setSourceDialogOpen}
              chatType={sourceType}
              onSubmit={(source) => handleRAGSourceCreation(sourceType, source)}
            />
          )}
          <div className="flex-1 flex flex-col h-full bg-background/50 backdrop-blur-xl border shadow-sm rounded-lg chat-gradient">
            <ChatView
              chat={activeChat}
              messages={messages}
              onSendMessage={handleSendMessage}
              onNewChatSelect={handleNewChatSelect}
              mobileMenu={<SidebarTrigger className="md:hidden" />}
              streamingMessageId={streamingMessageId}
            />
          </div>
        </SidebarInset>
      </main>
    </SidebarProvider>
  );
}
