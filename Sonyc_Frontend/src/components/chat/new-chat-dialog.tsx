"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ChatIcons } from "@/components/chat-icons";
import type { ChatType } from "@/lib/types";

interface NewChatDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelectChatType: (type: ChatType) => void;
}

const chatTypes: { type: ChatType; description: string }[] = [
  { type: "Normal", description: "Start a general conversation" },
  { type: "YouTube", description: "Chat about a YouTube video" },
  { type: "Web", description: "Discuss content from a webpage" },
  { type: "Git", description: "Analyze a Git repository" },
  { type: "PDF", description: "Ask questions about a PDF file" },
];

export function NewChatDialog({
  open,
  onOpenChange,
  onSelectChatType,
}: NewChatDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md bg-background/80 backdrop-blur-xl">
        <DialogHeader>
          <DialogTitle>Start a New Chat</DialogTitle>
          <DialogDescription>
            Choose a source to begin your conversation.
          </DialogDescription>
        </DialogHeader>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 py-4">
          {chatTypes.map(({ type, description }) => {
            const Icon = ChatIcons[type];
            return (
              <Button
                key={type}
                variant="outline"
                className="h-auto p-4 flex flex-col items-center justify-center gap-2 text-center bg-transparent hover:bg-primary/10 hover:text-primary"
                onClick={() => onSelectChatType(type)}
              >
                <Icon className="h-8 w-8 text-primary" />
                <div className="flex flex-col items-center">
                  <span className="font-semibold">{type}</span>
                  <span className="text-sm text-muted-foreground text-balance">
                    {description}
                  </span>
                </div>
              </Button>
            );
          })}
        </div>
      </DialogContent>
    </Dialog>
  );
}
