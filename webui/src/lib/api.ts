// lib/api.ts

import type {
    PipelineType, BasePipelineSchema, ChatConfig,
    ChatModel, ChatMessage, DocumentInfo,
    CompletionConfig, CompletionResponse,
    Persona, ModelCategory,
} from './types';

declare const window: Window;

class Api {
    private baseUrl: string;
    private apiKey: string | null;

    constructor() {
        // Safe default in case window is not available (e.g., SSR or tests)
        let protocol = 'http:';
        let hostname = 'localhost';

        // Check if `window` is available
        if (typeof window !== 'undefined') {
            protocol = window.location.protocol;
            hostname = window.location.hostname;
        }

        // Dynamically construct the base URL
        this.baseUrl = `${protocol}//${hostname}:8000`;

        // Set the API key with a fallback
        this.apiKey = import.meta.env.VITE_API_KEY || '123';
    }

    private async fetch(endpoint: string, options: RequestInit = {}): Promise<Response> {
        const url = `${this.baseUrl}${endpoint}`;
        const headers = {
            ...(options.headers ? {} : { 'Content-Type': 'application/json' }),
            ...(this.apiKey && { 'Authorization': `Bearer ${this.apiKey}` }),
            ...options.headers,
        };

        const response = await fetch(url, { ...options, headers });

        if (!response.ok) {
            throw new Error(`API call failed: ${response.statusText}`);
        }

        return response;
    }

    async getPipelineTasks(): Promise<any> {
        const response = await this.fetch('/api/pipeline/task');
        return response.json();
    }

    async createPipelineTask(pipelineType: PipelineType, config: BasePipelineSchema): Promise<any> {
        try {
            const response = await this.fetch('/api/pipeline/task', {
                method: 'POST',
                body: JSON.stringify({ pipeline_type: pipelineType, config }),
            });
            if (!response.ok) {
                const data = await response.json();
                throw new Error(`API call failed: ${data}`);
            }
            return response.json();
        } catch (error) {
            console.error('Error creating pipeline task:', error);
            alert(error);
            throw error;
        }
    }

    async retryPipelineTask(taskId: number): Promise<any> {
        try {
            const response = await this.fetch(`/api/pipeline/task/${taskId}/retry`, {
                method: 'POST',
            });
            if (!response.ok) {
                const data = await response.json();
                throw new Error(`API call failed: ${data}`);
            }
            return response.json();
        } catch (error) {
            console.error('Error retrying pipeline task:', error);
            alert(error);
            throw error;
        }
    }

    async removePipelineTask(taskId: number): Promise<any> {
        try {
            const response = await this.fetch(`/api/pipeline/task/${taskId}/remove`, {
                method: 'POST',
            });
            if (!response.ok) {
                const data = await response.json();
                throw new Error(`API call failed: ${data}`);
            }
            return response.json();
        } catch (error) {
            console.error('Error removing pipeline task:', error);
            alert(error);
            throw error;
        }
    }

    async getCompletion(config: CompletionConfig): Promise<CompletionResponse> {
        const response = await this.fetch('/v1/completions', {
            method: 'POST',
            body: JSON.stringify(config)
        });
        return response.json();
    }

    async getChatCompletions(messages: any[], config: ChatConfig): Promise<any> {
        const response = await this.fetch('/v1/chat/completions', {
            method: 'POST',
            body: JSON.stringify({
                messages,
                model: config.model || 'default',
                system_message: config.systemMessage,
                location: config.location,
                max_tokens: config.maxTokens,
                temperature: config.temperature,
                pinned_messages: config.pinnedMessages.map(message => message.doc_id),
                stream: true
            }),
        });
        return response.json();
    }

    async getChatModels(): Promise<{ models: ChatModel[], categories: ModelCategory[] }> {
        const response = await this.fetch('/v1/chat/models');
        return response.json();
    }

    async saveConversation(conversationId: string, messages: any[]): Promise<any> {
        const response = await this.fetch('/api/conversation', {
            method: 'POST',
            body: JSON.stringify({
                conversation_id: conversationId,
                messages,
            }),
        });

        if (!response.ok) {
            throw new Error('Failed to save conversation');
        }

        return response.json();
    }

    async deleteConversation(conversationId: string): Promise<any> {
        const response = await this.fetch(`/api/conversation/${conversationId}/remove`, {
            method: 'POST',
        });
        return response.json();
    }

    async getChatMatrix(): Promise<any> {
        const response = await this.fetch('/api/report/conversation_matrix');
        return response.json();
    }

    async getConversation(conversationId: string): Promise<any> {
        const response = await this.fetch(`/api/conversation/${conversationId}`);
        return response.json();
    }

    async searchConversations(query: string, top_n: number = 5, document_type: string = 'all'): Promise<any> {
        const params = new URLSearchParams({
            query: query,
            top_n: top_n.toString(),
            document_type: document_type
        });
        const response = await this.fetch(`/api/memory/search?${params.toString()}`);
        return response.json();
    }

    async createMessage(message: Partial<ChatMessage>): Promise<any> {
        const response = await this.fetch('/api/memory', {
            method: 'POST',
            body: JSON.stringify({ message })
        });
        return response.json();
    }

    async updateMessage(conversationId: string, docId: string, content: string): Promise<any> {
        const response = await this.fetch(`/api/memory/${conversationId}/${docId}`, {
            method: 'PUT',
            body: JSON.stringify({
                data: { content }
            })
        });
        if (!response.ok) {
            const data = await response.json();
            alert(`Error updating message: ${data}`);
            return data;
        } else {
            return response.json();
        }
    }

    async deleteMessage(conversationId: string, docId: string): Promise<{ status: string; message: string }> {
        const response = await this.fetch(`/api/memory/${conversationId}/${docId}/remove`, {
            method: 'POST'
        });

        if (!response.ok) {
            const data = await response.json();
            throw new Error(`Failed to delete document: ${data.message || 'Unknown error'}`);
        }

        return response.json();
    }

    async getDocuments(): Promise<{ documents: DocumentInfo[] }> {
        const response = await this.fetch('/api/document/list');
        return response.json();
    }

    async uploadDocument(file: File): Promise<{ status: string; message: string; filename: string }> {
        const formData = new FormData();
        formData.append('file', file);

        const response = await this.fetch('/api/document/upload', {
            method: 'POST',
            body: formData,
            headers: {
                ...(this.apiKey && { 'Authorization': `Bearer ${this.apiKey}` })
            }
        });
        return response.json();
    }

    async deleteDocument(documentName: string): Promise<{ status: string; message: string }> {
        const response = await this.fetch(`/api/document/${documentName}/remove`, {
            method: 'POST'
        });
        return response.json();
    }

    async downloadDocument(documentName: string): Promise<Blob> {
        const response = await this.fetch(`/api/document/${documentName}`);
        return response.blob();
    }

    async getDocumentContents(documentName: string): Promise<string> {
        const response = await this.fetch(`/api/document/${documentName}/contents`);
        const blob = await response.blob();
        return await blob.text();
    }
        
    async sendChatCompletion(body: string, handleResponse: (response: string) => void): Promise<any> {
        const API_URL = this.baseUrl + '/v1/chat/completions';
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`
            },
            body,
        });

        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let assistantResponse = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data:')) {
                    const data = line.slice(5).trim();
                    if (data === '[DONE]') {
                        break;
                    }
                    try {
                        const parsed = JSON.parse(data);
                        if (parsed.choices && parsed.choices[0].delta && parsed.choices[0].delta.content) {
                            assistantResponse += parsed.choices[0].delta.content;
                            if (handleResponse) {
                                handleResponse(assistantResponse);
                            }
                        }
                    } catch (error) {
                        console.error('Error parsing JSON:', error);
                    }
                }
            }
        }
        return assistantResponse;
    }

    async getRoster(): Promise<{ personas: Persona[] }> {
        const response = await this.fetch('/api/roster');
        return response.json();
    }

    async getPersona(personaId: string): Promise<Persona> {
        const response = await this.fetch(`/api/roster/${personaId}`);
        return response.json();
    }

    async createPersona(persona: Omit<Persona, 'persona_id'>): Promise<{ status: string; message: string; data: Persona }> {
        const response = await this.fetch('/api/roster', {
            method: 'POST',
            body: JSON.stringify(persona)
        });
        return response.json();
    }

    async updatePersona(personaId: string, updates: Partial<Persona>): Promise<{ status: string; message: string; data: Persona }> {
        const response = await this.fetch(`/api/roster/${personaId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json; charset=utf-8'
            },
            body: JSON.stringify(updates)
        });
        return response.json();
    }

    async deletePersona(personaId: string): Promise<{ status: string; message: string }> {
        const response = await this.fetch(`/api/roster/${personaId}`, {
            method: 'DELETE'
        });
        return response.json();
    }
}

export const api = new Api();
