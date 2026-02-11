"use client";

import { useEffect, useRef, useCallback, useLayoutEffect } from "react";
import type { Message } from "@/lib/types";
import { ChatMessage } from "./chat-message";

interface ChatMessagesProps {
  messages: Message[];
  streamingMessageId?: string | null;
}

export function ChatMessages({ messages, streamingMessageId }: ChatMessagesProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const lastContentHeightRef = useRef<number>(0);
  const shouldAutoScrollRef = useRef(true);
  const scrollRafRef = useRef<number | null>(null);
  const isStreamingRef = useRef(false);
  const scrollLockRef = useRef(false);

  // Track if we're streaming
  useEffect(() => {
    isStreamingRef.current = !!streamingMessageId;
  }, [streamingMessageId]);

  // Check if user is near bottom before allowing auto-scroll
  const checkShouldAutoScroll = useCallback(() => {
    if (!containerRef.current) return false;
    const container = containerRef.current;
    const threshold = 100; // pixels from bottom
    const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
    return distanceFromBottom < threshold;
  }, []);

  // Ultra-smooth scroll to bottom
  const scrollToBottom = useCallback((force: boolean = false) => {
    if (!containerRef.current || scrollLockRef.current) return;

    // Cancel any pending scroll
    if (scrollRafRef.current) {
      cancelAnimationFrame(scrollRafRef.current);
    }

    const container = containerRef.current;
    
    // Check if we should auto-scroll (unless forced during streaming)
    if (!force && !checkShouldAutoScroll() && !isStreamingRef.current) {
      shouldAutoScrollRef.current = false;
      return;
    }

    shouldAutoScrollRef.current = true;

    // Use requestAnimationFrame for smooth, non-blocking scroll
    scrollRafRef.current = requestAnimationFrame(() => {
      if (container && !scrollLockRef.current) {
        // Use scrollTop directly for instant, smooth updates
        const targetScroll = container.scrollHeight;
        container.scrollTop = targetScroll;
        lastContentHeightRef.current = container.scrollHeight;
      }
    });
  }, [checkShouldAutoScroll]);

  // Handle scroll events to detect user scrolling (throttled)
  const handleScroll = useCallback(() => {
    if (!containerRef.current || isStreamingRef.current) return;
    const container = containerRef.current;
    
    // If user scrolls up, disable auto-scroll
    const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
    if (distanceFromBottom > 150) {
      shouldAutoScrollRef.current = false;
    } else if (distanceFromBottom < 30) {
      // User scrolled back to bottom, re-enable auto-scroll
      shouldAutoScrollRef.current = true;
    }
  }, []);

  // Use useLayoutEffect for synchronous scroll updates during streaming
  useLayoutEffect(() => {
    if (streamingMessageId && containerRef.current && shouldAutoScrollRef.current) {
      const container = containerRef.current;
      const currentHeight = container.scrollHeight;
      
      // Only scroll if content height changed
      if (currentHeight !== lastContentHeightRef.current) {
        // Synchronous scroll for immediate update
        container.scrollTop = currentHeight;
        lastContentHeightRef.current = currentHeight;
      }
    }
  }, [messages, streamingMessageId]);

  // Scroll when messages change (but only if auto-scroll is enabled)
  useEffect(() => {
    if (messages.length > 0 && shouldAutoScrollRef.current && !isStreamingRef.current) {
      scrollToBottom(false);
    }
  }, [messages.length, scrollToBottom]);

  // Continuous smooth scroll during streaming
  useEffect(() => {
    if (!streamingMessageId) return;

    // Use a faster interval for smoother updates
    const intervalId = setInterval(() => {
      if (shouldAutoScrollRef.current && containerRef.current) {
        const container = containerRef.current;
        const currentHeight = container.scrollHeight;
        
        // Only scroll if content height changed
        if (currentHeight !== lastContentHeightRef.current) {
          // Direct scroll for instant update
          container.scrollTop = currentHeight;
          lastContentHeightRef.current = currentHeight;
        }
      }
    }, 16); // ~60fps for ultra-smooth scrolling

    return () => clearInterval(intervalId);
  }, [streamingMessageId]);

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className="flex-1 overflow-y-auto overscroll-contain"
      style={{
        scrollBehavior: 'auto',
        willChange: streamingMessageId ? 'scroll-position' : 'auto',
        // Optimize for smooth scrolling
        transform: 'translateZ(0)', // Force GPU acceleration
        backfaceVisibility: 'hidden',
      }}
    >
      <div 
        className="p-4 md:p-6 space-y-6"
        style={{
          // Prevent layout shifts
          minHeight: '100%',
        }}
      >
        {messages.map((message) => (
          <ChatMessage 
            key={message.id} 
            message={message} 
            isStreaming={streamingMessageId === message.id && !message.content}
          />
        ))}
      </div>
    </div>
  );
}
