const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiClient {
  private baseURL: string;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }

  private getToken(): string | null {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem('token');
  }

  private setToken(token: string): void {
    if (typeof window === 'undefined') return;
    localStorage.setItem('token', token);
  }

  private removeToken(): void {
    if (typeof window === 'undefined') return;
    localStorage.removeItem('token');
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const token = this.getToken();
    // Convert HeadersInit to Record<string, string> to allow bracket notation
    const headers: Record<string, string> = {
      ...(options.headers instanceof Headers
        ? Object.fromEntries(options.headers.entries())
        : Array.isArray(options.headers)
          ? Object.fromEntries(options.headers)
          : (options.headers as Record<string, string>) || {}),
    };

    // Only add Content-Type for requests with body
    if (options.body && !headers['Content-Type']) {
      headers['Content-Type'] = 'application/json';
    }

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${this.baseURL}${endpoint}`, {
      ...options,
      headers,
    });

    if (response.status === 401) {
      // Unauthorized - remove token and redirect to login
      console.error(`Authentication failed for ${endpoint}:`, {
        status: response.status,
        hasToken: !!token,
        tokenLength: token?.length || 0,
      });
      this.removeToken();
      if (typeof window !== 'undefined') {
        // Use router if available, otherwise use window.location
        window.location.href = '/';
      }
      const errorData = await response.json().catch(() => ({ detail: 'Unauthorized' }));
      throw new Error(errorData.detail || 'Unauthorized');
    }

    if (!response.ok) {
      let errorMessage = 'Request failed';
      try {
        const error = await response.json();
        // Handle different error response formats
        if (typeof error === 'string') {
          errorMessage = error;
        } else if (error && typeof error === 'object') {
          errorMessage = error.detail || error.error || error.message || JSON.stringify(error);
        }
      } catch (parseError) {
        // If JSON parsing fails, try to get text
        try {
          const text = await response.text();
          errorMessage = text || `HTTP ${response.status}: ${response.statusText}`;
        } catch {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
      }
      throw new Error(errorMessage);
    }

    // Handle empty responses
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      return response.json();
    }
    return response.text() as unknown as T;
  }

  // Authentication
  async signup(email: string, password: string): Promise<{ access_token: string; token_type: string }> {
    const response = await this.request<{ access_token: string; token_type: string }>('/auth/signup', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    this.setToken(response.access_token);
    return response;
  }

  async signin(email: string, password: string): Promise<{ access_token: string; token_type: string }> {
    const response = await this.request<{ access_token: string; token_type: string }>('/auth/signin', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    this.setToken(response.access_token);
    return response;
  }

  async getCurrentUser(): Promise<{ id: number; email: string }> {
    return this.request<{ id: number; email: string }>('/auth/me');
  }

  logout(): void {
    this.removeToken();
  }

  // Chats
  async getChats(): Promise<Array<{
    id: number;
    title: string;
    type: string;
    vector_db_collection_id?: string;
    created_at: string;
  }>> {
    return this.request('/chats');
  }

  async createChat(data: {
    title: string;
    type: string;
    vector_db_collection_id?: string;
  }): Promise<{
    id: number;
    title: string;
    type: string;
    vector_db_collection_id?: string;
    created_at: string;
  }> {
    return this.request('/chats', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getChatMessages(chatId: number): Promise<Array<{
    id: number;
    role: string;
    content: string;
    created_at: string;
  }>> {
    return this.request(`/chats/${chatId}/messages`);
  }

  async deleteChat(chatId: number): Promise<{ status: string }> {
    return this.request(`/chats/${chatId}`, {
      method: 'DELETE',
    });
  }

  // Chat streaming
  async streamChat(
    chatId: number,
    message: string,
    chatType: string,
    vectorDbCollectionId: string | null,
    onChunk: (chunk: string) => void,
    onTitleUpdate?: (title: string) => void
  ): Promise<void> {
    const token = this.getToken();
    const requestBody = {
      chat_id: chatId,
      message,
      chat_type: chatType,
      vector_db_collection_id: vectorDbCollectionId,
    };

    console.log('Streaming request:', {
      url: `${this.baseURL}/chat/stream`,
      method: 'POST',
      hasToken: !!token,
      body: requestBody
    });

    const response = await fetch(`${this.baseURL}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(requestBody),
    });

    console.log('Stream response status:', response.status, response.statusText);

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'An error occurred' }));
      console.error('Stream request failed:', {
        status: response.status,
        statusText: response.statusText,
        error
      });
      throw new Error(error.detail || error.error || 'Request failed');
    }

    if (!response.body) {
      console.error('No response body received');
      throw new Error('No response body');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullResponse = '';
    let readCount = 0;

    try {
      while (true) {
        const { value, done } = await reader.read();
        readCount++;

        if (done) {
          console.log(`Stream completed. Reads: ${readCount}, Total length: ${fullResponse.length}`);
          if (fullResponse.length === 0) {
            console.warn('WARNING: Stream completed but no content was received!');
          }
          break;
        }

        // Decode chunk immediately and update UI in real-time
        const chunk = decoder.decode(value, { stream: true });
        if (chunk) {
          fullResponse += chunk;

          // Check for title update marker (handle multiline and special characters)
          // Try multiple regex patterns to catch the marker
          let titleMatch = fullResponse.match(/<!-- TITLE_UPDATE:([^>]+) -->/);
          if (!titleMatch) {
            // Try without spaces
            titleMatch = fullResponse.match(/<!--TITLE_UPDATE:([^>]+)-->/);
          }
          if (!titleMatch) {
            // Try with newlines
            titleMatch = fullResponse.match(/<!--\s*TITLE_UPDATE:\s*([^>]+)\s*-->/);
          }

          if (titleMatch && onTitleUpdate) {
            const newTitle = titleMatch[1].trim();
            console.log('Title update detected in stream:', newTitle);
            console.log('Full response before cleanup:', fullResponse.substring(0, 200));
            onTitleUpdate(newTitle);
            // Remove the title marker from the response (try all patterns)
            fullResponse = fullResponse.replace(/<!--\s*TITLE_UPDATE:[^>]+\s*-->/g, '');
            fullResponse = fullResponse.replace(/<!--TITLE_UPDATE:[^>]+-->/g, '');
          }

          // Update UI immediately with each chunk for real-time streaming (without title markers)
          const cleanResponse = fullResponse.replace(/<!-- TITLE_UPDATE:.+? -->/g, '');
          onChunk(cleanResponse);

          if (readCount <= 3 || readCount % 20 === 0) {
            console.log(`Read ${readCount}: chunk length=${chunk.length}, total=${fullResponse.length}`);
          }
        } else {
          console.warn(`Read ${readCount}: empty chunk received`);
        }
      }
    } catch (error) {
      console.error('Error reading stream:', error);
      console.error('Stream state:', {
        readCount,
        fullResponseLength: fullResponse.length,
        errorMessage: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  // RAG endpoints
  async createYouTubeRAG(url: string): Promise<{ collection_name: string }> {
    return this.request('/yt_rag', {
      method: 'POST',
      body: JSON.stringify({ url }),
    });
  }

  async createWebRAG(url: string): Promise<{ collection_name: string }> {
    return this.request('/web_rag', {
      method: 'POST',
      body: JSON.stringify({ url }),
    });
  }

  async createGitRAG(url: string): Promise<{ collection_name: string }> {
    return this.request('/git_rag', {
      method: 'POST',
      body: JSON.stringify({ url }),
    });
  }

  async createPDFRAG(file: File): Promise<{ collection_name: string }> {
    const token = this.getToken();
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseURL}/pdf_rag`, {
      method: 'POST',
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'An error occurred' }));
      throw new Error(error.detail || error.error || 'Request failed');
    }

    return response.json();
  }

  // Git Project Management
  async getProjects(): Promise<Array<{
    id: number;
    repo_url: string;
    repo_owner: string;
    repo_name: string;
    branch: string;
    vector_db_collection_id?: string;
    last_synced_at?: string;
    created_at: string;
  }>> {
    return this.request('/projects');
  }

  async createProject(data: { repo_url: string; branch?: string }): Promise<{
    id: number;
    repo_url: string;
    repo_owner: string;
    repo_name: string;
    branch: string;
    vector_db_collection_id?: string;
    last_synced_at?: string;
    created_at: string;
  }> {
    return this.request('/projects', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async deleteProject(projectId: number): Promise<{ status: string }> {
    return this.request(`/projects/${projectId}`, {
      method: 'DELETE',
    });
  }

  async syncProject(projectId: number): Promise<{ status: string; message: string }> {
    return this.request(`/projects/${projectId}/sync`, {
      method: 'POST',
    });
  }

  // Report Generation
  async generateGitReport(projectId: number, reportType: string = 'full'): Promise<{
    report: string;
    project: {
      id: number;
      repo_url: string;
      repo_name: string;
      branch: string;
      last_synced_at?: string;
    };
    report_type: string;
  }> {
    return this.request('/git_report', {
      method: 'POST',
      body: JSON.stringify({ project_id: projectId, report_type: reportType }),
    });
  }
}

export const api = new ApiClient(API_URL);

