
"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Plus, LogOut, GitBranch } from "lucide-react";
import type { Chat } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatIcons } from "@/components/chat-icons";
import { formatDistanceToNow } from 'date-fns';
import { ThemeSwitcher } from "../theme-switcher";
import { SidebarFooter, SidebarGroup, SidebarMenu, SidebarMenuItem, SidebarMenuButton } from "@/components/ui/sidebar";
import { GitProjectsPanel } from "./git-projects-panel";

interface AppSidebarProps {
  chats: Chat[];
  activeChatId: string | null;
  onNewChatClick: () => void;
  onSelectChat: (id: string) => void;
  onLogout?: () => void;
  isMobile?: boolean;
}

export function AppSidebar({
  chats,
  activeChatId,
  onNewChatClick,
  onSelectChat,
  onLogout,
  isMobile = false,
}: AppSidebarProps) {
  const [projectsPanelOpen, setProjectsPanelOpen] = useState(false);

  const containerClasses = cn(
    "flex flex-col h-full",
    "rounded-lg bg-background/50 backdrop-blur-xl border shadow-sm"
  );

  return (
    <div className={containerClasses}>
      <div className="p-2">
        <Button
          variant="ghost"
          className="w-full justify-start gap-2 hover:bg-primary/10 hover:backdrop-blur-lg transition-colors duration-200 group-data-[collapsible=icon]:h-10 group-data-[collapsible=icon]:w-10 group-data-[collapsible=icon]:p-0 group-data-[collapsible=icon]:justify-center"
          onClick={onNewChatClick}
        >
          <Plus className="h-5 w-5 text-accent group-data-[collapsible=icon]:m-0" />
          <span className="group-data-[collapsible=icon]:hidden">New Chat</span>
        </Button>
        <Button
          variant="ghost"
          className="w-full justify-start gap-2 hover:bg-primary/10 hover:backdrop-blur-lg transition-colors duration-200 group-data-[collapsible=icon]:h-10 group-data-[collapsible=icon]:w-10 group-data-[collapsible=icon]:p-0 group-data-[collapsible=icon]:justify-center"
          onClick={() => setProjectsPanelOpen(true)}
        >
          <GitBranch className="h-5 w-5 text-accent group-data-[collapsible=icon]:m-0" />
          <span className="group-data-[collapsible=icon]:hidden">Git Projects</span>
        </Button>
      </div>
      <ScrollArea className="flex-1">
        <SidebarGroup className="p-2 pt-0">
          <SidebarMenu>
            {chats.map((chat) => {
              const Icon = ChatIcons[chat.type];
              return (
                <SidebarMenuItem key={chat.id}>
                  <SidebarMenuButton
                    onClick={() => onSelectChat(chat.id)}
                    isActive={activeChatId === chat.id}
                    className={cn(
                      "h-auto py-2 group-data-[collapsible=icon]:h-10 group-data-[collapsible=icon]:w-10",
                      activeChatId === chat.id && "bg-primary/10 backdrop-blur-lg shadow-sm border border-primary/20",
                    )}
                  >
                    <Icon className="h-4 w-4 text-primary" />
                    <div className="flex flex-col flex-1 truncate text-left group-data-[collapsible=icon]:hidden">
                      <span className="font-medium truncate">{chat.title}</span>
                      <span className="text-xs text-muted-foreground">
                        {formatDistanceToNow(chat.createdAt, { addSuffix: true })}
                      </span>
                    </div>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              );
            })}
          </SidebarMenu>
        </SidebarGroup>
      </ScrollArea>
      <SidebarFooter className="p-2 border-t flex-row items-center justify-between group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:gap-2">
        <p className="text-xs text-muted-foreground group-data-[collapsible=icon]:hidden">Sonyc</p>
        <div className="flex items-center gap-2">
          <ThemeSwitcher />
          {onLogout && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onLogout}
              className="h-9 w-9"
              title="Logout"
            >
              <LogOut className="h-4 w-4" />
              <span className="sr-only">Logout</span>
            </Button>
          )}
        </div>
      </SidebarFooter>
      <GitProjectsPanel open={projectsPanelOpen} onOpenChange={setProjectsPanelOpen} />
    </div>
  );
}
