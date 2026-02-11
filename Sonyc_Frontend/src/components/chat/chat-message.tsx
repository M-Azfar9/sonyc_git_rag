"use client";

import type { Message } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { User } from "lucide-react";
import { ChatAvatar } from "./chat-avatar";
import { Prose } from "./prose";
import { parseMarkdown } from "@/lib/markdown";
import { CodeBlock } from "./code-block";

// Function to split markdown content into text and code parts
const splitMarkdown = (markdown: string): { type: 'text' | 'code'; content: string, lang?: string }[] => {
  if (!markdown) return [];
  const codeBlockRegex = /(```(\w*)\n[\s\S]*?```)/g;
  const parts = markdown.split(codeBlockRegex);
  const result: { type: 'text' | 'code'; content: string, lang?: string }[] = [];
  
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    if (part.match(codeBlockRegex)) {
      const langMatch = part.match(/```(\w*)/);
      const language = langMatch ? langMatch[1] : 'plaintext';
      const codeContent = part.replace(/```(\w*)\n|```/g, '');
      result.push({ type: 'code', content: codeContent, lang: language });
      // The regex split includes the delimiter, so we skip the next two parts which are the full match and the language capture group
      i += 2; 
    } else if (part.trim() !== '') {
      result.push({ type: 'text', content: part });
    }
  }
  return result;
};

interface ChatMessageProps {
  message: Message;
  isStreaming?: boolean;
}

export function ChatMessage({ message, isStreaming = false }: ChatMessageProps) {
  const isUser = message.role === "user";

  if (isUser) {
    const htmlContent = parseMarkdown(message.content);
    return (
      <div className="flex items-start gap-4 animate-enter-from-bottom justify-end">
        <div className="bg-primary/20 rounded-br-none rounded-lg px-4 py-3 max-w-3xl break-words overflow-wrap-anywhere min-w-0">
          <Prose html={htmlContent} />
        </div>
        <Avatar className="h-8 w-8 flex-shrink-0">
          <AvatarFallback className="bg-accent text-accent-foreground">
            <User className="h-5 w-5" />
          </AvatarFallback>
        </Avatar>
      </div>
    );
  }

  // Handle assistant messages
  const displayContent = isStreaming && !message.content ? "..." : message.content;
  const contentParts = splitMarkdown(displayContent);

  return (
    <div
      className={cn(
        "flex items-start gap-4",
        "justify-start",
        // Only animate on initial render, not during streaming updates
        !isStreaming && "animate-enter-from-bottom"
      )}
      style={{
        // Aggressive containment for smooth rendering
        contain: 'layout style paint',
        contentVisibility: isStreaming ? 'auto' : 'visible',
        // GPU acceleration
        transform: 'translateZ(0)',
        willChange: isStreaming ? 'contents' : 'auto',
        // Prevent layout shifts
        minHeight: isStreaming && message.content ? '1px' : 'auto',
      }}
    >
      <Avatar className="h-8 w-8 flex-shrink-0">
        <AvatarImage src="/logo.png" alt="Chatbot" />
        <AvatarFallback className="bg-accent text-accent-foreground">
          <ChatAvatar />
        </AvatarFallback>
      </Avatar>
      <div 
        className="flex flex-col gap-4 max-w-3xl w-full min-w-0"
        style={{
          // Optimize for smooth updates
          contain: 'layout style',
        }}
      >
        {isStreaming && !message.content ? (
          <div className="flex items-center gap-1 text-muted-foreground px-4 py-2">
            <span className="inline-block w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
            <span className="inline-block w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
            <span className="inline-block w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
          </div>
        ) : (
          contentParts.map((part, index) => {
            if (part.type === 'code') {
              return (
                <CodeBlock 
                  key={index}
                  language={part.lang || 'plaintext'}
                  code={part.content}
                />
              );
            } else {
              const htmlContent = parseMarkdown(part.content);
              return (
                <div 
                  key={index} 
                  className="break-words min-w-0"
                  style={{
                    // Prevent reflow during streaming
                    wordBreak: 'break-word',
                    overflowWrap: 'break-word',
                    // Optimize rendering
                    transform: 'translateZ(0)',
                  }}
                >
                   <Prose html={htmlContent} />
                </div>
              );
            }
          })
        )}
      </div>
    </div>
  );
}
