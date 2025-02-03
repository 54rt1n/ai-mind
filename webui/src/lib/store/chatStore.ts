// lib/store/chatStore.ts
import { writable, derived, get } from 'svelte/store';
import { browser } from '$app/environment';
import type { ChatConfig, ChatMessage, DocumentInfo } from '$lib';
import { api } from '$lib';

interface Message {
    timestamp: number;
    role: string;
    content: string;
}


function createChatStore() {
    const { subscribe, set, update } = writable<{
        conversationId: string;
        conversationHistory: Message[];
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
        messages: Message[],
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
        } = {}
    ) {
        update(store => ({ ...store, loading: true }));
        const body = JSON.stringify({
            messages,
            stream: true,
            user_id: config?.user_id || undefined,
            persona_id: config?.persona_id || undefined,
            model: model || config?.model || undefined,
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
            pinned_messages: options.pinnedMessages ? options.pinnedMessages.map(message => message.doc_id) : undefined,
            active_document: options.activeDocument ? options.activeDocument.name : undefined,
            workspace_content: options.workspaceContent || undefined,
            thought_content: config?.thoughtContent || undefined,
            disable_guidance: options.disableGuidance || undefined,
            disable_pif: options.disablePif || undefined,
        });

        try {
            await api.sendChatCompletion(body, handleResponse);
            saveConversationData();
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
        } = {}
    ): Promise<SendMessageCallback> {
        const model = config.model;
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
            update(store => ({...store, loading: true}));
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

            await _sendMessage(model, messages, handleResponse, config, {workspaceContent: options.workspaceContent});

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

    async function generateThought(config: ChatConfig, workspaceContent?: string) : Promise<SendMessageCallback> {
        const model = config.thoughtModel;
        if (!model) {
            return { status: 'error', message: "No model specified in config" };
        }
        // Thought generation is 
        try {
            update(store => ({...store, loading: true, contentStream: ''}));
            const conversation = get({ subscribe }).conversationHistory;
            // Filter to remove all role: system messages
            const messages = conversation.map( msg => { return {...msg} } );
            const thoughtPrompt = config.thoughtPrompt;
            const lastMessage = messages[messages.length - 1];
            if (lastMessage.role == 'user') {
                messages[messages.length - 1]['content'] += `\n\n${thoughtPrompt}`;

            } else {
                messages.push({
                    role: 'user',
                    content: thoughtPrompt || 'Output your thoughts.',
                    timestamp: Math.floor(Date.now() / 1000)
                })
            }

            const handleResponse = (response: string) => {
                update(store => {
                    return { ...store, contentStream: response };
                });
            }

            await _sendMessage(model, messages, handleResponse, config, {workspaceContent, systemMessage: config.thoughtSystemMessage, disableGuidance: true});

            saveConversationData();
            return { status: 'success', message: 'Thought generated successfully' };
        } catch (error) {
            update(store => ({
                ...store,
                thoughtContent: 'An error occurred while fetching the response.'
            }));
            if (error instanceof Error) {
                return { status: 'error', message: error.message };
            } else {
                console.error('Error:', error);
                return { status: 'error', message: 'An unknown error occurred.' };
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

    async function saveConversation() {
        update(store => ({ ...store, loading: true }));

        try {
            const response = await api.saveConversation(
                get({ subscribe }).conversationId,
                get({ subscribe }).conversationHistory
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
            sendMessage(lastMessage.content, config, {workspaceContent});
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

    function popMessage() : Message | undefined {
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

    return {
        subscribe,
        loadConversationHistory,
        sendMessage,
        clearChat,
        saveConversation,
        goBack,
        popMessage,
        retry,
        generateThought,
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