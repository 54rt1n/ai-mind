import { writable, get } from 'svelte/store';
import { browser } from '$app/environment';
import { api } from '$lib/api';
import type { CompletionMessage, ChatConfig, ChatMessage, DocumentInfo } from '$lib/types';

interface ThoughtState {
    thoughtContent: string;
    contentStream: string;
    thoughtXml: string;
    iteration: number;
    loading: boolean;
    error: string | null;
    thoughtUserContext: string;
    thoughtPrompt: string;
    thoughtSystemMessage: string;
    thoughtDefaultContent: string;
}

function createThoughtStore() {
    const STORAGE_KEY = 'thoughtStore';

    // Initialize from localStorage if in browser environment
    const initialState: ThoughtState = browser
        ? JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null') || {
            thoughtContent: '',
            contentStream: '',
            thoughtXml: 'think',
            iteration: 1,
            loading: false,
            error: null,
            thoughtUserContext: '',
            thoughtPrompt: '',
            thoughtSystemMessage: '',
            thoughtDefaultContent: ''
        }
        : {
            thoughtContent: '',
            contentStream: '',
            thoughtXml: 'think',
            iteration: 1,
            loading: false,
            error: null,
            thoughtUserContext: '',
            thoughtPrompt: '',
            thoughtSystemMessage: '',
            thoughtDefaultContent: ''
        };

    const { subscribe, set, update } = writable<ThoughtState>(initialState);

    // Save to localStorage when store updates (browser only)
    if (browser) {
        subscribe(state => {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        });
    }

    function getDefaultThought(entityName: string, iter: string): string {
        let thoughts = [
            "I will follow a chain of thought, reasoning through my ideas.",
            "These are the most important things I should consider:",
        ];
        return `<${entityName} iter="${iter}">\n\t${thoughts.map((thought) => `<thought>${thought}</thought>`).join("\n\t")}\n</${entityName}>`;
    }

    function getDefaultSystemMessage(): string {
        return `<format_override>\n\t<override>You are in your thought processes. You are only to output a thought turn.</override>\n\t<output_mode>xml</output_mode>\n\t<description>All Thought Output Is In XML Format</description>\n</format_override>`;
    }

    function getThoughtPrompt(entityName: string, iter: number, userContext: string = ""): string {
        const defaultThoughts = getDefaultThought(entityName, "n");
        return `Thought Turn Format:\n\n${defaultThoughts}\n\n<directive>Your next turn is a thought turn. Please update your thought block appropriately, enhancing and improving your current thoughts and reasoning${userContext}. Please output the next thoughts document. This should be an xml document.</directive>\n\n[~~ Begin XML Output "<${entityName} iter="${iter}">" ~~]`;
    }

    return {
        set,
        subscribe,
        reset: () => {
            const state = get({ subscribe });
            update(store => ({
                ...store,
                thoughtContent: store.thoughtDefaultContent,
                contentStream: '',
                iteration: 1,
                thoughtPrompt: getThoughtPrompt(state.thoughtXml, 1, state.thoughtUserContext)
            }));
        },
        setDefaults: () => {
            const state = get({ subscribe });
            update(store => ({
                ...store,
                thoughtDefaultContent: getDefaultThought(state.thoughtXml, "0"),
                thoughtSystemMessage: getDefaultSystemMessage(),
                thoughtPrompt: getThoughtPrompt(state.thoughtXml, state.iteration, store.thoughtUserContext)
            }));
        },
        updateThoughtContent: (content: string) => {
            update(store => ({ ...store, thoughtContent: content }));
        },
        updateContentStream: (content: string) => {
            update(store => ({ ...store, contentStream: content }));
        },
        updateIteration: () => {
            update(store => {
                const newIteration = store.iteration + 1;
                return {
                    ...store,
                    iteration: newIteration,
                    thoughtPrompt: getThoughtPrompt(store.thoughtXml, newIteration, store.thoughtUserContext)
                };
            });
        },
        setThoughtXml: (xml: string) => {
            update(store => ({ ...store, thoughtXml: xml }));
        },
        setUserContext: (context: string) => {
            update(store => ({
                ...store,
                thoughtUserContext: context,
                thoughtPrompt: getThoughtPrompt(store.thoughtXml, store.iteration, context)
            }));
        },
        updateThoughtPrompt: (prompt: string) => {
            update(store => ({ ...store, thoughtPrompt: prompt }));
        },
        updateSystemMessage: (message: string) => {
            update(store => ({ ...store, thoughtSystemMessage: message }));
        },
        updateDefaultContent: (content: string) => {
            update(store => ({ ...store, thoughtDefaultContent: content }));
        },
        generateThought: async (
            thoughtModel: string,
            messages: CompletionMessage[],
            options: {
                user_id?: string,
                persona_id?: string,
                workspaceContent?: string | null,
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
        ) => {
            const state = get({ subscribe });
            update(store => ({ ...store, loading: true, error: null }));

            const conversationCopy = messages.map(message => ({
                role: message.role,
                content: message.content
            }));

            const lastMessage = conversationCopy[conversationCopy.length - 1];
            if (lastMessage === null || lastMessage === undefined || lastMessage.content === null || lastMessage.content === undefined) {
                throw new Error("Missing conversation history");
            }
            lastMessage.content = lastMessage.content + "\n\n[~~ End of User turn ~~]\n\n" + state.thoughtPrompt;

            conversationCopy[conversationCopy.length - 1] = lastMessage;

            const temperature = options.temperature || 0.9;
            const maxTokens = options.maxTokens || 2048;
            const frequencyPenalty = options.frequencyPenalty || undefined;
            const presencePenalty = options.presencePenalty || undefined;
            const repetitionPenalty = options.repetitionPenalty || undefined;
            const minP = options.minP || 0.05;
            const topP = options.topP || undefined;
            const topK = options.topK || undefined;

            try {
                if (!options.user_id || !options.persona_id) {
                    throw new Error("User ID and Persona ID are required");
                }
                if (!thoughtModel) {
                    throw new Error("Thought Model is required");
                }
                let response = '';
                await api.sendChatCompletion(
                    JSON.stringify({
                        metadata: {
                            user_id: options.user_id,
                            persona_id: options.persona_id,
                            thought_content: state.thoughtContent,
                            workspace_content: options.workspaceContent,
                            pinned_messages: options.pinnedMessages ? options.pinnedMessages.map(message => message.doc_id) : undefined,
                            active_document: options.activeDocument ? options.activeDocument.name : undefined,
                            disable_guidance: options.disableGuidance || undefined,
                            disable_pif: options.disablePif || undefined,
                        },
                        messages: conversationCopy,
                        model: thoughtModel,
                        system_message: state.thoughtSystemMessage,
                        temperature: temperature,
                        min_p: minP,
                        max_tokens: maxTokens,
                        frequency_penalty: frequencyPenalty,
                        presence_penalty: presencePenalty,
                        repetition_penalty: repetitionPenalty,
                        top_p: topP,
                        top_k: topK,
                        stream: true
                    }),
                    (chunk) => {
                        response = chunk;
                        update(store => ({ ...store, contentStream: response }));
                    }
                );

                update(store => ({ ...store, loading: false }));
                return true;
            } catch (error) {
                update(store => ({
                    ...store,
                    loading: false,
                    error: error instanceof Error ? error.message : 'Failed to generate thought'
                }));
                return false;
            }
        }
    };
}

export const thoughtStore = createThoughtStore(); 