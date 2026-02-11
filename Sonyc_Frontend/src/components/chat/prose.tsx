"use client";

import { useEffect, useRef } from 'react';

// A custom component to render HTML and apply syntax highlighting
export function Prose({ html }: { html: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // This is a simple client-side only way to highlight code.
    // In a real app, you'd likely use a library like highlight.js
    // and integrate it with your build process for server-side rendering.
    ref.current?.querySelectorAll('pre code').forEach((block) => {
        // This is a placeholder for a real syntax highlighting library
        // For now, it just ensures the code is styled for readability
        block.parentElement!.style.whiteSpace = 'pre-wrap';
        block.parentElement!.style.wordWrap = 'break-word';
        block.parentElement!.style.overflowX = 'auto';
    });
  }, [html]);

  return (
    <div
      ref={ref}
      className="prose prose-sm dark:prose-invert text-foreground max-w-none break-words overflow-wrap-anywhere"
      style={{
        wordBreak: 'break-word',
        overflowWrap: 'anywhere',
      }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}