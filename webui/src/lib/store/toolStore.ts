// lib/store/toolStore.ts
import { writable, get } from 'svelte/store';
import { browser } from '$app/environment';
import { api } from '$lib';
import type { ToolResponse, ChatConfig, ChatMessage, DocumentInfo, CompletionMessage } from '$lib/types';

interface ToolStore {
    tools: ToolResponse[];
    selectedTools: ToolResponse[];
    loading: boolean;
    error: string | null;
    generatedOutput: string;
    isGenerating: boolean;
}

const STORAGE_KEY = 'toolStore';

function createToolStore() {
    // Initialize from localStorage if in browser environment
    let initialTools: ToolStore;

    if (browser) {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            const parsed = stored ? JSON.parse(stored) : null;
            initialTools = {
                tools: Array.isArray(parsed?.tools) ? parsed.tools : [],
                selectedTools: Array.isArray(parsed?.selectedTools) ? parsed.selectedTools : [],
                loading: false,
                error: null,
                generatedOutput: parsed?.generatedOutput || "",
                isGenerating: false
            };
        } catch (e) {
            console.error('Error parsing stored tool state:', e);
            initialTools = {
                tools: [],
                selectedTools: [],
                loading: false,
                error: null,
                generatedOutput: "",
                isGenerating: false
            };
        }
    } else {
        initialTools = {
            tools: [],
            selectedTools: [],
            loading: false,
            error: null,
            generatedOutput: "",
            isGenerating: false
        };
    }

    const { subscribe, set, update } = writable<ToolStore>(initialTools);

    // Save to localStorage when store updates (browser only)
    if (browser) {
        subscribe(state => {
            try {
                localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
            } catch (e) {
                console.error('Error saving tool state:', e);
            }
        });
    }

    async function fetchTools() {
        update(store => ({ ...store, loading: true, error: null }));

        try {
            const results = await api.getTools();
            update(store => ({
                ...store,
                tools: results.tools,
                loading: false,
                error: null
            }));
        } catch (error) {
            update(store => ({
                ...store,
                loading: false,
                error: 'Failed to fetch tools'
            }));
            console.error('Error fetching tools:', error);
        }
    }

    async function generateWithTools(
        messages: CompletionMessage[],
        config: ChatConfig,
        options: {
            workspaceContent?: string | null,
            thoughtContent?: string | null,
            systemMessage?: string | null,
            currentLocation?: string | null,
            pinnedMessages?: ChatMessage[] | null,
            activeDocument?: DocumentInfo | null,
            temperature?: number | null,
            maxTokens?: number | null,
            frequencyPenalty?: number | null,
            presencePenalty?: number | null,
            repetitionPenalty?: number | null,
            minP?: number | null,
            topP?: number | null,
            topK?: number | null,
            disableGuidance?: boolean | null,
            disablePif?: boolean | null,
        } = {}
    ) {
        const store = get({ subscribe });
        if (store.selectedTools.length === 0) {
            console.error("No tools selected");
            return;
        }

        if (config.toolModel === undefined) {
            console.error("Tool model is undefined");
            return;
        }

        update(store => ({ ...store, isGenerating: true }));

        const messagesCopy = [...messages];
        messagesCopy.push({
            role: "user",
            content: "Please select from the avaiable <Tools> and use them to perform a task.",
            timestamp: Date.now()
        });

        try {
            const requestBody = JSON.stringify({
                messages: messagesCopy,
                metadata: {
                    user_id: config.user_id || "user",
                    persona_id: config.persona_id || "default",
                    pinned_messages: options.pinnedMessages ? options.pinnedMessages.map(message => message.doc_id) : undefined,
                    active_document: options.activeDocument ? options.activeDocument.name : undefined,
                    workspace_content: options.workspaceContent || undefined,
                    thought_content: options.thoughtContent || undefined,
                    disable_guidance: options.disableGuidance || undefined,
                    disable_pif: options.disablePif || undefined,
                },
                model: config.toolModel,
                system_message: options.systemMessage || config.systemMessage || undefined,
                location: options.currentLocation || config.location || undefined,
                max_tokens: options.maxTokens || config.maxTokens || undefined,
                temperature: options.temperature || config.temperature || undefined,
                frequency_penalty: options.frequencyPenalty || config.frequencyPenalty || undefined,
                presence_penalty: options.presencePenalty || config.presencePenalty || undefined,
                repetition_penalty: options.repetitionPenalty || config.repetitionPenalty || undefined,
                min_p: options.minP || config.minP || undefined,
                top_p: options.topP || config.topP || undefined,
                top_k: options.topK || config.topK || undefined,
                tools: store.selectedTools,
                stream: true
            });

            let output = "";
            await api.sendChatCompletion(requestBody, (chunk) => {
                try {
                    output = chunk;
                    update(store => ({ ...store, generatedOutput: output }));
                } catch (e) {
                    console.error('Error handling tool response:', e);
                }
            });
        } catch (error) {
            console.error('Error generating tool response:', error);
            update(store => ({
                ...store,
                error: 'Failed to generate response'
            }));
        } finally {
            update(store => ({ ...store, isGenerating: false }));
        }
    }

    return {
        subscribe,
        refresh: fetchTools,
        initialize: async () => {
            const state = get({ subscribe });
            if (state.tools.length === 0) {
                await fetchTools();
            }
        },
        selectTool: (tool: ToolResponse) => {
            update(store => ({
                ...store,
                selectedTools: [...store.selectedTools, tool]
            }));
        },
        unselectTool: (tool: ToolResponse) => {
            update(store => ({
                ...store,
                selectedTools: store.selectedTools.filter(t =>
                    !(t.type === tool.type && t.function.name === tool.function.name)
                )
            }));
        },
        clearSelection: () => {
            update(store => ({ ...store, selectedTools: [] }));
        },
        generate: generateWithTools,
        clearOutput: () => {
            update(store => ({ ...store, generatedOutput: "" }));
        }
    };
}

export const toolStore = createToolStore(); 