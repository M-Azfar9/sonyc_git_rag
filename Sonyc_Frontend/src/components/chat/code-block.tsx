"use client";

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Check, Copy } from 'lucide-react';
import { Prose } from './prose';
import { parseMarkdown } from '@/lib/markdown';

interface CodeBlockProps {
  language: string;
  code: string;
}

export function CodeBlock({ language, code }: CodeBlockProps) {
  const [isCopied, setIsCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    });
  };

  const codeHtml = parseMarkdown('```' + language + '\n' + code + '\n```');

  return (
    <div className="bg-background/50 backdrop-blur-md border rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-1 bg-muted/50">
        <span className="text-xs font-sans text-muted-foreground">{language}</span>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
          onClick={handleCopy}
        >
          {isCopied ? <Check className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
        </Button>
      </div>
      <div className="px-4 py-3">
        <Prose html={codeHtml} />
      </div>
    </div>
  );
}
