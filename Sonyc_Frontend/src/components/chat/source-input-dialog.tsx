
"use client";

import { useState, useRef } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ChatIcons } from "@/components/chat-icons";
import type { ChatType } from "@/lib/types";
import { Upload } from "lucide-react";

interface SourceInputDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  chatType: ChatType;
  onSubmit: (source: string | File, branch?: string) => void;
}

const sourceDetails: Record<
  ChatType,
  { title: string; description: string; inputLabel: string; inputType: string; placeholder: string }
> = {
  YouTube: {
    title: "YouTube Video",
    description: "Enter the URL of the YouTube video you want to chat about.",
    inputLabel: "YouTube URL",
    inputType: "url",
    placeholder: "https://www.youtube.com/watch?v=...",
  },
  Web: {
    title: "Website",
    description: "Enter the URL of the website you want to discuss.",
    inputLabel: "Website URL",
    inputType: "url",
    placeholder: "https://example.com",
  },
  Git: {
    title: "Git Repository",
    description: "Enter the URL of the Git repository you want to analyze. All code, issues, PRs, and commits will be ingested.",
    inputLabel: "Git Repository URL",
    inputType: "url",
    placeholder: "https://github.com/username/repo",
  },
  PDF: {
    title: "PDF Document",
    description: "Upload a PDF document to start chatting.",
    inputLabel: "PDF File",
    inputType: "file",
    placeholder: "",
  },
  Normal: {
    title: "",
    description: "",
    inputLabel: "",
    inputType: "",
    placeholder: "",
  },
};

export function SourceInputDialog({
  open,
  onOpenChange,
  chatType,
  onSubmit,
}: SourceInputDialogProps) {
  const [source, setSource] = useState<string>("");
  const [branch, setBranch] = useState<string>("main");
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const details = sourceDetails[chatType];
  const Icon = ChatIcons[chatType];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (chatType === "PDF" && file) {
      onSubmit(file);
      setSource("");
      setFile(null);
    } else if (source.trim()) {
      onSubmit(source, chatType === "Git" ? branch : undefined);
      setSource("");
      setFile(null);
      setBranch("main");
    }
  };

  const isFileInput = details.inputType === 'file';
  const isGitChat = chatType === "Git";

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setSource(selectedFile.name);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md bg-background/80 backdrop-blur-xl">
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Icon className="h-6 w-6 text-primary" />
              New {details.title} Chat
            </DialogTitle>
            <DialogDescription>{details.description}</DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            {isFileInput ? (
              <>
                <Label htmlFor="source-input" className="sr-only">
                  {details.inputLabel}
                </Label>
                <Label
                  htmlFor="source-input"
                  className="flex flex-col items-center justify-center h-24 border-2 border-dashed rounded-lg cursor-pointer hover:border-primary hover:text-primary bg-transparent"
                >
                  <div className="flex flex-col items-center justify-center gap-2 text-muted-foreground">
                    <Upload className="h-6 w-6" />
                    {source ? (
                      <span className="text-sm font-medium text-foreground">{source}</span>
                    ) : (
                      <span>Click to upload or drag & drop</span>
                    )}
                  </div>
                </Label>
                <Input
                  id="source-input"
                  type="file"
                  className="sr-only"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept=".pdf"
                />
              </>
            ) : (
              <>
                <Label htmlFor="source-input" className="sr-only">
                  {details.inputLabel}
                </Label>
                <Input
                  id="source-input"
                  type={details.inputType}
                  value={source}
                  onChange={(e) => setSource(e.target.value)}
                  placeholder={details.placeholder}
                  required
                />
              </>
            )}
            {isGitChat && (
              <div className="grid gap-2">
                <Label htmlFor="branch-input" className="text-sm font-medium">
                  Branch
                </Label>
                <Input
                  id="branch-input"
                  type="text"
                  value={branch}
                  onChange={(e) => setBranch(e.target.value)}
                  placeholder="main"
                  className="h-9"
                />
                <p className="text-xs text-muted-foreground">
                  Specify the branch to analyze (default: main)
                </p>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button type="button" variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={isFileInput ? !file : !source.trim()}>
              Start Chat
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
