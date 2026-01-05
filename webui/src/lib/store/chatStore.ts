// lib/store/chatStore.ts
import { writable, derived, get } from 'svelte/store';
import { browser } from '$app/environment';
import type { ChatConfig, ChatMessage, DocumentInfo, CompletionMessage } from '$lib';
import { api } from '$lib';


function createChatStore() {
    const { subscribe, set, update } = writable<{
        conversationId: string;
        conversationHistory: CompletionMessage[];
        contentStream: string;
        loading: boolean;
    }>({
        conversationId: '',
        conversationHistory: [],
        contentStream: '',
        loading: false,
    });

    function generateConversationId(): string {
        return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    function loadConversationHistory() {
        const savedData = localStorage.getItem('chatData');
        // console.log(savedData)
        if (savedData) {
            const parsed = JSON.parse(savedData);
            update(store => ({
                ...store,
                conversationHistory: parsed.messages || [],
                conversationId: parsed.conversationId,
            }));
        }

        update(store => ({
            ...store,
            conversationId: store.conversationId || generateConversationId()
        }));
    }

    function saveConversationData() {
        const store = get({ subscribe });
        localStorage.setItem('chatData', JSON.stringify({
            conversationId: store.conversationId,
            messages: store.conversationHistory,
        }));
        return store;
    }

    if (browser) {
        subscribe(state => {
            if (state.conversationId) {
                // console.log('ChatStore state:', state);
                saveConversationData();
            }
        });
    }

    async function _sendMessage(
        model: string,
        messages: CompletionMessage[],
        handleResponse: (response: string) => void,
        config?: ChatConfig,
        options: {
            workspaceContent?: string | null,
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
            onComplete?: () => void,
        } = {}
    ) {
        update(store => ({ ...store, loading: true }));
        const body = JSON.stringify({
            messages,
            stream: true,
            metadata: {
                user_id: config?.user_id || undefined,
                persona_id: config?.persona_id || undefined,
                pinned_messages: (options.pinnedMessages || config?.pinnedMessages)?.map(message => message.doc_id) || undefined,
                active_document: options.activeDocument ? options.activeDocument.name : undefined,
                workspace_content: options.workspaceContent || undefined,
                thought_content: config?.thoughtContent || undefined,
                disable_guidance: options.disableGuidance || undefined,
                disable_pif: options.disablePif || undefined,
            },
            model: model || config?.chatModel || undefined,
            system_message: options.systemMessage || config?.systemMessage || undefined,
            location: options.currentLocation || config?.location || undefined,
            max_tokens: options.maxTokens || config?.maxTokens || undefined,
            temperature: options.temperature || config?.temperature || undefined,
            frequency_penalty: options.frequencyPenalty || config?.frequencyPenalty || undefined,
            presence_penalty: options.presencePenalty || config?.presencePenalty || undefined,
            repetition_penalty: options.repetitionPenalty || config?.repetitionPenalty || undefined,
            min_p: options.minP || config?.minP || undefined,
            top_p: options.topP || config?.topP || undefined,
            top_k: options.topK || config?.topK || undefined,
        });

        try {
            await api.sendChatCompletion(body, handleResponse);
            saveConversationData();
            if (options.onComplete) {
                options.onComplete();
            }
        } catch (error) {
            console.error('Error:', error);
        } finally {
            update(store => ({ ...store, loading: false }));
        }
    }

    // Define our type for the sendMessage callback - which is either:
    // { status: 'success', message: string } or { status: 'error', message: string }

    type SendMessageCallback = { status: 'success', message: string } | { status: 'error', message: string };

    async function sendMessage(userInput: string, config: ChatConfig,
        options: {
            workspaceContent?: string,
            skipAppend?: boolean,
            systemMessage?: string | null;
            currentLocation?: string | null;
            pinnedMessages?: ChatMessage[] | null;
            activeDocument?: DocumentInfo | null;
            temperature?: number | null;
            maxTokens?: number | null;
            frequencyPenalty?: number | null;
            presencePenalty?: number | null;
            repetitionPenalty?: number | null;
            minP?: number | null;
            topP?: number | null;
            topK?: number | null;
            disableGuidance?: boolean | null;
            disablePif?: boolean | null;
        } = {}
    ): Promise<SendMessageCallback> {
        const model = config.chatModel;
        if (!model) {
            return { status: 'error', message: "No model specified in config" };
        }
        if (config.persona_id == '') {
            return { status: 'error', message: "No persona specified in config" };
        }
        if (config.user_id == '') {
            return { status: 'error', message: "No user specified in config" };
        }
        try {
            update(store => ({ ...store, loading: true }));
            if (!options.skipAppend) {
                appendMessage('user', userInput);
            }

            const conversation = get({ subscribe }).conversationHistory;
            // Filter to remove all role: system messages
            const messages = conversation.filter(msg => msg.role !== 'system');

            update(store => ({
                ...store,
                conversationHistory: [...store.conversationHistory, { timestamp: Math.floor(Date.now() / 1000), role: 'assistant', content: '' }]
            }));

            const handleResponse = (response: string) => {
                update(store => {
                    const newHistory = [...store.conversationHistory];
                    newHistory[newHistory.length - 1].content = response;
                    return { ...store, conversationHistory: newHistory };
                });
            }

            // Handle final response with think extraction when complete
            const handleComplete = () => {
                update(store => {
                    const newHistory = [...store.conversationHistory];
                    const lastMessage = newHistory[newHistory.length - 1];

                    // Process the think content when streaming is complete
                    if (lastMessage && lastMessage.content) {
                        const { content, think } = extractThinkContent(lastMessage.content);
                        if (think) {
                            // If think tags were found, preserve the original content but add think metadata
                            lastMessage.think = think;
                        }
                    }

                    return { ...store, conversationHistory: newHistory };
                });
            }

            await _sendMessage(model, messages, handleResponse, config, {
                workspaceContent: options.workspaceContent,
                onComplete: handleComplete,
                systemMessage: options.systemMessage,
                currentLocation: options.currentLocation,
                pinnedMessages: options.pinnedMessages,
                activeDocument: options.activeDocument,
                temperature: options.temperature,
                maxTokens: options.maxTokens,
                frequencyPenalty: options.frequencyPenalty,
                presencePenalty: options.presencePenalty,
                repetitionPenalty: options.repetitionPenalty,
                minP: options.minP,
                topP: options.topP,
                topK: options.topK,
                disableGuidance: options.disableGuidance,
                disablePif: options.disablePif,
            });

            saveConversationData();
            return { status: 'success', message: 'Message sent successfully' };
        } catch (error) {
            update(store => ({
                ...store,
                conversationHistory: [...store.conversationHistory, { timestamp: Math.floor(Date.now() / 1000), role: 'system', content: 'An error occurred while fetching the response.' }]
            }));
            if (error instanceof Error) {
                return { status: 'error', message: error.message };
            } else {
                return { status: 'error', message: 'An unknown error occurred' };
            }
        } finally {
            update(store => ({ ...store, loading: false }));
        }
    }

    function clearChat() {
        update(store => ({
            ...store,
            conversationHistory: [],
            thoughtContent: undefined,
            conversationId: generateConversationId()
        }));
        saveConversationData();
    }

    async function saveConversation(persona_id: string) {
        if (!persona_id) {
            alert('No persona selected. Please select a persona from the roster.');
            return;
        }

        update(store => ({ ...store, loading: true }));

        try {
            // Process each message before saving
            const processedHistory = get({ subscribe }).conversationHistory.map(message => {
                // Only process if not already processed
                if (message.content) {
                    const { content, think } = extractThinkContent(message.content);

                    // If think tags were found (including empty ones), return modified message
                    if (think !== null) {
                        return {
                            ...message,
                            content, // Clean content without think tags
                            think,   // Extracted think content
                        };
                    }
                }

                // Return message unchanged if no think tags or already processed
                return message;
            });

            const response = await api.saveConversation(
                persona_id,
                get({ subscribe }).conversationId,
                processedHistory
            )

            if (!response.ok) {
                throw new Error('Failed to save conversation');
            }

            const result = await response.json();
            //console.log('Conversation saved:', result);
            alert('Conversation saved successfully!');
        } catch (error) {
            // TODO why does this fail even on success?
            console.error('Error saving conversation:', error);
            alert('Failed to save conversation. Please try again.');
        } finally {
            update(store => ({ ...store, loading: false }));
        }
    }

    function goBack() {
        update(store => {
            // console.log('Before:', store.conversationHistory);
            const history = store.conversationHistory;
            let newHistory = [...history];

            // First check if we have a system message at the end
            if (history[history.length - 1].role === 'system') {
                // console.log('Removing system message');
                newHistory.pop();
            }

            if (newHistory[newHistory.length - 1].role === 'assistant') {
                // console.log('Removing assistant message');
                newHistory.pop();
            }

            if (newHistory[newHistory.length - 1].role === 'user') {
                // console.log('Removing user message');
                newHistory.pop();
            }

            // console.log('After:', newHistory);
            return {
                ...store,
                conversationHistory: newHistory
            };
        });
        saveConversationData();
    }


    function retry(config: ChatConfig, workspaceContent?: string) {
        update(store => {
            const history = store.conversationHistory;
            let newHistory = [...history];
            let lastUserMessage;

            // Find the last user message
            for (let i = newHistory.length - 1; i >= 0; i--) {
                if (newHistory[i].role === 'user') {
                    lastUserMessage = newHistory[i];
                    break;
                }
            }

            // If we found a user message, trim the history up to that point
            if (lastUserMessage) {
                newHistory = newHistory.slice(0, newHistory.indexOf(lastUserMessage) + 1);
                return {
                    ...store,
                    conversationHistory: newHistory
                };
            }

            // If no user message was found, just return the original state
            return store;
        });
        saveConversationData();

        // Resubmit the last user message
        const currentState = get({ subscribe });
        const lastMessage = currentState.conversationHistory[currentState.conversationHistory.length - 1];
        // console.log('Retrying last user message:', lastMessage);
        if (lastMessage && lastMessage.role === 'user') {
            // Remove the message from the history
            update(store => ({
                ...store,
                conversationHistory: store.conversationHistory.filter(msg => msg !== lastMessage)
            }));
            sendMessage(lastMessage.content, config, {
                workspaceContent,
                systemMessage: config.systemMessage,
                currentLocation: config.location,
                pinnedMessages: config.pinnedMessages,
                activeDocument: config.selectedDocument,
                temperature: config.temperature,
                maxTokens: config.maxTokens,
                frequencyPenalty: config.frequencyPenalty,
                presencePenalty: config.presencePenalty,
                repetitionPenalty: config.repetitionPenalty,
                minP: config.minP,
                topP: config.topP,
                topK: config.topK,
                disableGuidance: false,
                disablePif: false,
            });
        }
    }

    function updateMessage(index: number, newContent: string) {
        update(store => {
            const newHistory = [...store.conversationHistory];
            newHistory[index] = {
                ...newHistory[index],
                content: newContent
            };

            // Save to localStorage
            localStorage.setItem('chatData', JSON.stringify({
                conversationId: store.conversationId,
                messages: newHistory
            }));

            return {
                ...store,
                conversationHistory: newHistory
            };
        });
    }

    function appendMessage(role: 'user' | 'assistant', content: string) {
        const message = {
            role,
            content,
            timestamp: Math.floor(Date.now() / 1000)
        };
        update(store => {
            const newHistory = [...store.conversationHistory, message];
            return {
                ...store,
                conversationHistory: newHistory
            };
        });
        saveConversationData();
    }

    function popMessage(): CompletionMessage | undefined {
        const newHistory = [...get({ subscribe }).conversationHistory];
        const result = newHistory.pop();
        update(store => {
            return {
                ...store,
                conversationHistory: newHistory
            };
        });
        saveConversationData();
        return result;
    }

    function swapRoles() {
        update(store => {
            const newHistory = store.conversationHistory.map(message => ({
                ...message,
                role: message.role === 'user' ? 'assistant' :
                    message.role === 'assistant' ? 'user' : message.role
            }));

            return {
                ...store,
                conversationHistory: newHistory
            };
        });
        saveConversationData();
    }

    // Extract think content from message text
    function extractThinkContent(text: string): { content: string, think: string | null } {
        // First check for complete think tags
        const completeThinkRegex = /<think>([\s\S]*?)<\/think>/;
        const completeMatch = completeThinkRegex.exec(text);

        if (completeMatch) {
            // Complete think tag found - extract content
            const thinkContent = completeMatch[1].trim();
            // Remove think tags for display
            const cleanContent = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
            return { content: cleanContent, think: thinkContent };
        }

        // If no complete tag, check for partial think tag (open without close)
        const partialThinkRegex = /<think>([\s\S]*?)$/;
        const partialMatch = partialThinkRegex.exec(text);

        if (partialMatch) {
            // Partial think tag found - extract content
            const thinkContent = partialMatch[1].trim();
            // Remove partial think tag for display
            const cleanContent = text.replace(/<think>[\s\S]*?$/, "").trim();
            return { content: cleanContent, think: thinkContent };
        }

        // No think tags found
        return { content: text, think: null };
    }

    return {
        subscribe,
        loadConversationHistory,
        sendMessage,
        clearChat,
        saveConversation,
        goBack,
        popMessage,
        retry,
        updateMessage,
        swapRoles,
        appendMessage,
        get,
        set,
        setConversationId: (id: string) => {
            // console.log('Setting conversationId to:', id);
            update(store => ({ ...store, conversationId: id }))
        },
    };
}

export const chatStore = createChatStore();

export const canGoBack = derived(chatStore, $store => {
    const history = $store.conversationHistory;
    return history.length >= 2;
});

export const canRetry = derived(chatStore, $store => {
    const history = $store.conversationHistory;
    return history.length >= 1;
});