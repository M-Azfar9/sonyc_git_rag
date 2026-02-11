"use client";

import { useState, useRef, useEffect } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Send } from "lucide-react";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSendMessage: (content: string) => void;
}

export function ChatInput({ onSendMessage }: ChatInputProps) {
  const [content, setContent] = useState("");
  const [isExpanded, setIsExpanded] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const singleLineHeight = useRef<number | null>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      if (singleLineHeight.current === null) {
        // Temporarily set to auto to measure the initial scroll height for a single line.
        textarea.style.height = "auto";
        singleLineHeight.current = textarea.scrollHeight;
      }
      
      textarea.style.height = "auto"; // Reset height to recalculate
      const scrollHeight = textarea.scrollHeight;

      // Determine if expanded
      const expanded = scrollHeight > (singleLineHeight.current ?? 40);
      if (expanded !== isExpanded) {
        setIsExpanded(expanded);
      }

      // Set height based on content, up to a max of 256px (max-h-64)
      textarea.style.height = `${Math.min(scrollHeight, 256)}px`;
    }
  }, [content, isExpanded]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (content.trim()) {
      onSendMessage(content);
      setContent("");
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto"; // Reset height after sending
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };
  
  const handleContentChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setContent(e.target.value);
  };

  return (
    <div className="px-4 pb-4 bg-transparent">
      <div className="relative w-full max-w-2xl mx-auto">
        <form
          onSubmit={handleSubmit}
          className={cn(
            "relative bg-background/70 backdrop-blur-xl border shadow-lg transition-all duration-200",
            isExpanded ? "rounded-t-2xl rounded-b-lg" : "rounded-full"
          )}
        >
          <Textarea
            ref={textareaRef}
            value={content}
            onChange={handleContentChange}
            onKeyDown={handleKeyDown}
            placeholder="Message Sonyc..."
            className={cn(
              "w-full bg-transparent border-none pr-14 py-3 pl-4 resize-none focus-visible:ring-0 focus-visible:ring-offset-0 max-h-64 overflow-y-auto [&::-webkit-scrollbar]:hidden transition-all duration-200",
              isExpanded ? "rounded-t-2xl rounded-b-lg" : "rounded-full"
            )}
            rows={1}
          />
          <Button
            type="submit"
            size="icon"
            className="absolute right-2 bottom-2 h-8 w-8 rounded-full bg-primary text-primary-foreground hover:bg-primary/90 disabled:bg-muted disabled:text-muted-foreground"
            disabled={!content.trim()}
          >
            <Send className="h-5 w-5" />
          </Button>
        </form>
        <p className="text-xs text-center text-muted-foreground mt-2">
          Sonyc can make mistakes. Consider checking important information.
        </p>
      </div>
    </div>
  );
}
