
"use client";

import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { PanelLeft } from "lucide-react";
import type { Chat } from "@/lib/types";
import { AppSidebar } from "./app-sidebar";

interface MobileMenuProps {
  chats: Chat[];
  activeChatId: string | null;
  onNewChatClick: () => void;
  onSelectChat: (id: string) => void;
}

export function MobileMenu({
  chats,
  activeChatId,
  onNewChatClick,
  onSelectChat,
}: MobileMenuProps) {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" className="md:hidden">
          <PanelLeft />
          <span className="sr-only">Toggle Menu</span>
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="w-[300px] sm:w-[350px] p-0">
        <AppSidebar
          chats={chats}
          activeChatId={activeChatId}
          onNewChatClick={onNewChatClick}
          onSelectChat={onSelectChat}
          isMobile
        />
      </SheetContent>
    </Sheet>
  );
}
