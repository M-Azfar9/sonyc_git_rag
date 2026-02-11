import {
  MessageSquare,
  Youtube,
  Globe,
  Github,
  FileText,
  type LucideProps,
} from "lucide-react";

export const ChatIcons = {
  Normal: (props: LucideProps) => <MessageSquare {...props} />,
  YouTube: (props: LucideProps) => <Youtube {...props} />,
  Web: (props: LucideProps) => <Globe {...props} />,
  Git: (props: LucideProps) => <Github {...props} />,
  PDF: (props: LucideProps) => <FileText {...props} />,
};
