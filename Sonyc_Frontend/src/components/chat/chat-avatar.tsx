export function ChatAvatar() {
  // This is now used as a fallback only
  // The actual logo is displayed via AvatarImage in chat-message.tsx
  return (
    <div className="h-full w-full bg-primary/20 rounded-full flex items-center justify-center text-xs font-semibold text-primary">
      AI
    </div>
  );
}
