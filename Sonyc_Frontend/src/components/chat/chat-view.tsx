
"use client";

import type { Chat, ChatType, Message } from "@/lib/types";
import { ChatMessages } from "./chat-messages";
import { ChatInput } from "./chat-input";
import { Button } from "@/components/ui/button";
import { ChatIcons } from "@/components/chat-icons";
import { Sparkles } from "lucide-react";
import { SidebarTrigger } from "@/components/ui/sidebar";

interface ChatViewProps {
  chat?: Chat | null;
  messages: Message[];
  onSendMessage: (content: string) => void;
  onNewChatSelect: (type: ChatType) => void;
  mobileMenu: React.ReactNode;
  streamingMessageId?: string | null;
}

const chatTypes: { type: ChatType; description: string }[] = [
    { type: "Normal", description: "General conversation" },
    { type: "YouTube", description: "From a YouTube video" },
    { type: "Web", description: "From a webpage" },
    { type: "Git", description: "From a Git repository" },
  ];

const chatPrompts: Record<ChatType, string[]> = {
    Normal: ["Explain quantum computing in simple terms", "What are the best places to visit in Japan?", "Write a short story about a robot who discovers music"],
    YouTube: ["Summarize this video", "What are the main points?", "What is the tone of this video?"],
    Web: ["What is this page about?", "Summarize the content of this website", "Extract the key information from this article"],
    Git: ["Explain this repository", "What is the purpose of this file?", "How does this function work?"],
    PDF: ["Summarize this document", "What are the key findings?", "Explain the main concepts in this PDF"],
}

export function ChatView({
  chat,
  messages,
  onSendMessage,
  onNewChatSelect,
  mobileMenu,
  streamingMessageId
}: ChatViewProps) {
  const MainContent = () => {
    if (!chat) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center p-4">
          <div className="absolute top-4 left-4">
            {mobileMenu}
          </div>
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="h-8 w-8 text-primary"/>
            <h1 className="text-4xl font-headline uppercase">Sonyc</h1>
          </div>
          <p className="text-muted-foreground mb-8">How can I help you today?</p>
          <div className="grid grid-cols-2 gap-4 max-w-md w-full">
              {chatTypes.map(({type, description}) => {
                  const Icon = ChatIcons[type];
                  return (
                      <Button key={type} variant="outline" className="h-20 flex-col items-center justify-center p-4 gap-2 bg-background/50 backdrop-blur-xl border hover:bg-primary/10 hover:border-primary/30 hover:text-primary whitespace-normal break-words overflow-wrap-anywhere min-w-0 w-full text-center" onClick={() => onNewChatSelect(type)}>
                          <Icon className="h-5 w-5 text-primary flex-shrink-0"/>
                          <span className="font-semibold break-words text-center">{type}</span>
                          <span className="text-xs text-muted-foreground break-words text-center">{description}</span>
                      </Button>
                  )
              })}
          </div>
        </div>
      );
    }

    const Icon = ChatIcons[chat.type];
    const prompts = chatPrompts[chat.type] || chatPrompts.Normal;

    return (
      <>
        <header className="flex items-center gap-3 p-4 border-b">
            {mobileMenu}
            <SidebarTrigger className="hidden md:flex"/>
            <Icon className="h-6 w-6 text-primary"/>
            <h2 className="text-lg font-headline uppercase font-semibold">Sonyc</h2>
        </header>
        {messages.length > 0 ? (
          <ChatMessages messages={messages} streamingMessageId={streamingMessageId} />
        ) : (
            <div className="flex-1 flex flex-col items-center justify-center p-4 text-center">
                <div className="bg-primary/10 p-4 rounded-full mb-4">
                    <Icon className="h-10 w-10 text-primary" />
                </div>
                <h2 className="text-2xl font-bold mb-2">Start your {chat.type} chat</h2>
                <p className="text-muted-foreground mb-8">Ask me anything, or try one of these prompts.</p>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 max-w-3xl w-full">
                    {prompts.map((prompt, index) => (
                        <Button
                          key={index}
                          variant="outline"
                          className="h-auto text-center justify-center p-3 bg-background/50 backdrop-blur-xl border hover:bg-primary/10 hover:border-primary/30 whitespace-normal break-words overflow-wrap-anywhere min-w-0 w-full hover:text-primary"
                          onClick={() => onSendMessage(prompt)}
                        >
                            <span className="break-words text-center">{prompt}</span>
                        </Button>
                    ))}
                </div>
            </div>
        )}
      </>
    );
  }

  return (
    <>
      <MainContent />
      { chat && <ChatInput onSendMessage={onSendMessage} /> }
    </>
  );
}
