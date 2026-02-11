export type ChatType = "Normal" | "YouTube" | "Web" | "Git" | "PDF";

export interface Chat {
  id: string;
  title: string;
  type: ChatType;
  createdAt: Date;
  vector_db_collection_id?: string;
}

export interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
}

export interface GitProject {
  id: number;
  repo_url: string;
  repo_owner: string;
  repo_name: string;
  branch: string;
  vector_db_collection_id?: string;
  last_synced_at?: string;
  created_at: string;
}
