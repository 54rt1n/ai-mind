import { writable, get } from 'svelte/store';
import { browser } from '$app/environment';
import { api } from '$lib/api';
import type { Tool, ToolResponse, CompletionMessage } from '$lib/types';

export interface ActionCard {
    id: string;
    type: 'simple' | 'tool';
    position: number;
    title: string;
    thoughtTurns: number;
    config: {
        thoughts?: string;
        title?: string;  // Add title to config
        prompt: string;
        system_message: string;
        timestamp?: number; // The last time the card was run
        tool?: Tool;
        pattern?: string;  // For tool cards
        chatConfig: {
            metadata: {
                user_id: string;
                persona_id: string;
            };
            model?: string;
            temperature?: number;
            max_tokens?: number;
            frequency_penalty?: number;
            presence_penalty?: number;
            repetition_penalty?: number;
            min_p?: number;
            top_p?: number;
            top_k?: number;
        };
    };
    result?: {
        content: string;
        timestamp: number;
        error?: string;
    };
    status: 'idle' | 'running' | 'completed' | 'error';
}

interface ToolboxState {
    cards: ActionCard[];
    tools: ToolResponse[];
    loading: boolean;
    error: string | null;
}

function createToolboxStore() {
    const STORAGE_KEY = 'toolboxStore';

    // Initialize from localStorage if in browser
    const initialState: ToolboxState = browser
        ? JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null') || {
            cards: [],
            tools: [],
            loading: false,
            error: null
        }
        : {
            cards: [],
            tools: [],
            loading: false,
            error: null
        };

    const { subscribe, set, update } = writable<ToolboxState>(initialState);

    // Save to localStorage when store updates (browser only)
    if (browser) {
        subscribe(state => {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        });
    }

    return {
        subscribe,
        addCard: (type: 'simple' | 'tool',
            card_id: string | null = null,
            user_id: string = '',
            persona_id: string = '',
            model: string = '',
            system_message: string = '',
            pattern: string = '',
            temperature: number = 0.7,
            thoughtTurns: number = 0,
            max_tokens: number = 1000
        ) => update(store => {
            const newCard: ActionCard = {
                id: card_id || `card_${Date.now()}`,
                type,
                position: store.cards.length,
                title: `New ${type === 'simple' ? 'Simple Action' : 'Tool Action'}`,
                thoughtTurns,
                config: {
                    prompt: '',
                    system_message,
                    thoughts: undefined,
                    pattern,
                    chatConfig: {
                        metadata: {
                            user_id,
                            persona_id,
                        },
                        model,
                        temperature,
                        max_tokens: max_tokens
                    }
                },
                status: 'idle'
            };
            return {
                ...store,
                cards: [...store.cards, newCard]
            };
        }),
        removeCard: (id: string) => update(store => ({
            ...store,
            cards: store.cards.filter(card => card.id !== id)
        })),
        updateCardPosition: (id: string, newPosition: number) => update(store => {
            const cards = [...store.cards];
            const cardIndex = cards.findIndex(card => card.id === id);
            if (cardIndex === -1) return store;

            const [card] = cards.splice(cardIndex, 1);
            cards.splice(newPosition, 0, card);

            return {
                ...store,
                cards: cards.map((c, i) => ({ ...c, position: i }))
            };
        }),
        updateCardConfig: (id: string, config: Partial<ActionCard['config']>) =>
            update(store => ({
                ...store,
                cards: store.cards.map(card =>
                    card.id === id
                        ? { ...card, config: { ...card.config, ...config } }
                        : card
                )
            })),
        executeCard: async (card: ActionCard, messages: CompletionMessage[] = []) => {
            update(store => ({
                ...store,
                cards: store.cards.map(c =>
                    c.id === card.id
                        ? { ...c, status: 'running', config: { ...c.config, timestamp: Date.now() } }
                        : c
                )
            }));

            try {
                let result: string;
                result = await new Promise((resolve, reject) => {
                    let response = '';
                    const tools = card.type === 'tool' ? [card.config.tool] : [];
                    const thoughts = card.config.thoughts ? card.config.thoughts : undefined;

                    api.sendChatCompletion(
                        JSON.stringify({
                            messages,
                            ...card.config.chatConfig,
                            metadata: {
                                user_id: card.config.chatConfig.metadata.user_id,
                                persona_id: card.config.chatConfig.metadata.persona_id,
                                thought_content: thoughts,
                            },
                            tools: tools,
                            stream: true
                        }),
                        (chunk) => {
                            response = chunk;
                            // Update the card's result in real-time
                            update(store => ({
                                ...store,
                                cards: store.cards.map(c =>
                                    c.id === card.id
                                        ? {
                                            ...c,
                                            result: {
                                                content: response,
                                                timestamp: Date.now()
                                            }
                                        }
                                        : c
                                )
                            }));
                        }
                    ).then(() => resolve(response))
                        .catch(reject);
                });

                update(store => ({
                    ...store,
                    cards: store.cards.map(c =>
                        c.id === card.id
                            ? {
                                ...c,
                                status: 'completed',
                                result: {
                                    content: result,
                                    timestamp: Date.now()
                                }
                            }
                            : c
                    )
                }));
            } catch (error) {
                update(store => ({
                    ...store,
                    cards: store.cards.map(c =>
                        c.id === card.id
                            ? {
                                ...c,
                                status: 'error',
                                result: {
                                    content: error instanceof Error ? error.message : 'An unknown error occurred',
                                    timestamp: Date.now(),
                                    error: error instanceof Error ? error.message : 'An unknown error occurred'
                                }
                            }
                            : c
                    )
                }));
            }
        },
        executePipeline: async () => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const store = get({ subscribe });
                for (const card of store.cards) {
                    if (card.config.thoughts) {
                        // Skip execution if thoughts failed
                        if (!card.config.thoughts) {
                            update(store => ({
                                ...store,
                                error: `Card "${card.title}" has thoughts enabled but no thought content. Please execute the card individually first.`
                            }));
                            return;
                        }
                    }
                    await toolboxStore.executeCardCascade(card.id);
                }
            } catch (error) {
                update(store => ({ ...store, loading: false, error: 'Pipeline execution failed' }));
            } finally {
                update(store => ({ ...store, loading: false }));
            }
        },
        clearAll: () => update(store => ({
            ...store,
            cards: [],
            error: null
        })),
        resetAllCards: () => update(store => ({
            ...store,
            cards: store.cards.map(card => ({
                ...card,
                status: 'idle'
            })),
            loading: false,
            error: null
        })),
        updateCardTitle: (id: string, title: string) => update(store => ({
            ...store,
            cards: store.cards.map(card =>
                card.id === id
                    ? { ...card, title }
                    : card
            )
        })),
        updateCardStatus: (id: string, status: ActionCard['status'], error?: string) => update(store => ({
            ...store,
            cards: store.cards.map(card =>
                card.id === id
                    ? {
                        ...card,
                        status,
                        result: error
                            ? {
                                content: error,
                                timestamp: Date.now(),
                                error
                            }
                            : card.result
                    }
                    : card
            )
        })),
        getConversationHistory: (id: string): CompletionMessage[] => {
            const store = get({ subscribe });
            const cardIndex = store.cards.findIndex(c => c.id === id);
            if (cardIndex === -1) return [];

            // Collect messages from previous cards
            const messages: CompletionMessage[] = [];
            for (let i = 0; i < cardIndex; i++) {
                const card = store.cards[i];
                if (card.result) {
                    messages.push(
                        { role: 'user', content: card.config.prompt, timestamp: card.config.timestamp ?? Date.now() },
                        { role: 'assistant', content: card.result.content, timestamp: card.result.timestamp }
                    );
                }
            }
            messages.push({ role: 'user', content: store.cards[cardIndex].config.prompt, timestamp: Date.now() });

            return messages;
        },
        executeCardCascade: async (id: string) => {
            const messages = toolboxStore.getConversationHistory(id);
            // Add the current card's prompt
            const store = get({ subscribe });
            const cardIndex = store.cards.findIndex(c => c.id === id);
            if (cardIndex === -1) return;
            const currentCard = store.cards[cardIndex];

            // Execute the card with the collected context
            await toolboxStore.executeCard(currentCard, messages);
        },
        loadTools: async () => {
            update(store => ({ ...store, loading: true }));
            try {
                const response = await api.getTools();
                update(store => ({
                    ...store,
                    tools: response.tools,
                    loading: false
                }));
            } catch (error) {
                update(store => ({
                    ...store,
                    error: error instanceof Error ? error.message : 'Failed to load tools',
                    loading: false
                }));
            }
        },
        addTool: async (tool: Tool) => {
            update(store => ({ ...store, loading: true }));
            try {
                const response = await api.createTool(tool);
                update(store => ({
                    ...store,
                    tools: [...store.tools, response.data],
                    loading: false
                }));
            } catch (error) {
                update(store => ({
                    ...store,
                    error: error instanceof Error ? error.message : 'Failed to add tool',
                    loading: false
                }));
            }
        },
        updateTool: async (toolType: string, updates: Partial<Tool>) => {
            update(store => ({ ...store, loading: true }));
            try {
                const response = await api.updateTool(toolType, updates);
                update(store => ({
                    ...store,
                    tools: store.tools.map(tool =>
                        tool.type === toolType ? response.data : tool
                    ),
                    loading: false
                }));
            } catch (error) {
                update(store => ({
                    ...store,
                    error: error instanceof Error ? error.message : 'Failed to update tool',
                    loading: false
                }));
            }
        },
        removeTool: async (toolType: string) => {
            update(store => ({ ...store, loading: true }));
            try {
                await api.deleteTool(toolType);
                update(store => ({
                    ...store,
                    tools: store.tools.filter(tool => tool.type !== toolType),
                    loading: false
                }));
            } catch (error) {
                update(store => ({
                    ...store,
                    error: error instanceof Error ? error.message : 'Failed to remove tool',
                    loading: false
                }));
            }
        },
    };
}

export const toolboxStore = createToolboxStore(); 