// lib/store/workspaceStore.ts
import { writable, derived, get } from 'svelte/store';
import { api } from '$lib';
import { browser } from '$app/environment';
import type { ChatConfig, WorkspaceItem } from '$lib';
import { WorkspaceCategory } from '$lib';

const DEFAULT_WORKSPACE: WorkspaceItem = {
    name: WorkspaceCategory.None,
    content: '',
    contentStream: '',
    wordCount: 0,
};

function createWorkspaceStore() {
    const STORAGE_KEY_BASE = 'workspaceStore';
    const STORAGE_KEY_CATEGORY = 'category';

    function getCategoryKey(category: WorkspaceCategory): string {
        return `${STORAGE_KEY_BASE}::${category}`;
    }

    function loadCategory(categoryName: string | WorkspaceCategory): WorkspaceItem {
        const category = (typeof categoryName === 'string') ? Object.values(WorkspaceCategory).find(c => c.toLowerCase() === categoryName.toLowerCase()) : categoryName;
        if (!category) {
            console.warn(`Invalid category name: ${categoryName}`);
            return DEFAULT_WORKSPACE;
        }
        const storageKey = getCategoryKey(category);
        const savedData = localStorage.getItem(storageKey);
        const wordCount = savedData ? savedData.split(/\s+/).length : 0;
        const newState = {
            name: category,
            content: savedData || '',
            contentStream: '',
            wordCount: wordCount,
        };
        return newState;
    }

    const initialState: WorkspaceItem = (() => {
        if (browser) {
            const savedCategory = localStorage.getItem(STORAGE_KEY_CATEGORY);
            if (savedCategory) {
                return loadCategory(savedCategory);
            }
            return DEFAULT_WORKSPACE;
        } else {
            return DEFAULT_WORKSPACE;
        }
    })();

    const { subscribe, set, update } = writable<WorkspaceItem>(initialState);

    function changeCategory(newCategory: WorkspaceCategory) {
        const storageKey = getCategoryKey(newCategory);
        const savedData = localStorage.getItem(storageKey);
        const wordCount = savedData ? savedData.split(/\s+/).length : 0;
        const newState = {
            name: newCategory,
            content: savedData || '',
            contentStream: '',
            wordCount: wordCount,
        };
        localStorage.setItem(STORAGE_KEY_CATEGORY, newCategory);
        set(newState);
    }

    function setContent(content: string) {
        const storageKey = getCategoryKey(get(workspaceStore).name);
        localStorage.setItem(storageKey, content);
        const wordCount = content.split(/\s+/).length;
        update(state => ({ ...state, content, wordCount }));
    }

    function updateContentStream(content: string) {
        update(state => ({ ...state, contentStream: content }));
    }

    function commitStreamUpdate() {
        update(state => {
            const newContent = state.contentStream;
            const storageKey = getCategoryKey(state.name);
            localStorage.setItem(storageKey, newContent);
            const wordCount = newContent.split(/\s+/).length;
            return { ...state, content: newContent, contentStream: '', wordCount };
        });
    }

    async function generateWorkspaceUpdate(
        config: ChatConfig,
        conversationHistory: any[],
        systemMessage: string) {
        update(state => ({ ...state, contentStream: '' }));

        if (!config.workspaceModel) {
            console.error('No workspace model provided');
            return false;
        }

        const currentState = get(workspaceStore);

        // Create a copy of the conversation history and add workspace context
        const conversationWithContext = [...conversationHistory];

        // Add a user message requesting the workspace update
        conversationWithContext.push({
            role: 'user',
            content: `Please update the following workspace content based on our conversation:\n\n${currentState.content}\n\nProvide the complete updated content.`,
            timestamp: Math.floor(Date.now() / 1000)
        });

        try {
            await api.sendChatCompletion(
                JSON.stringify({
                    metadata: {
                        user_id: config.user_id,
                        persona_id: config.persona_id,
                        workspace_content: currentState.content,
                        active_document: config.selectedDocument?.name,
                        thought_content: config.thoughtContent ?? undefined,
                    },
                    messages: conversationWithContext,
                    model: config.workspaceModel,
                    system_message: systemMessage,
                    max_tokens: config.maxTokens,
                    temperature: config.temperature,
                    top_p: config.topP,
                    top_k: config.topK,
                    min_p: config.minP,
                    frequency_penalty: config.frequencyPenalty,
                    presence_penalty: config.presencePenalty,
                    repetition_penalty: config.repetitionPenalty,
                    stream: true
                }),
                (chunk) => {
                    updateContentStream(chunk);
                }
            );
            return true;
        } catch (error) {
            console.error('Failed to generate workspace update:', error);
            return false;
        }
    }

    async function saveRemoteCopy(fileName: string) {
        // We are going to use our api to save a copy of the current workspace as a remote document

        // Lets grab the current minimally short timestamp. Isostring isn't acceptable due to spaces
        // We just want the ymd
        const fileContent = get(workspaceStore).content;

        // Call the api to save the file
        // Create a File object from the content
        const file = new File([fileContent], fileName, {
            type: 'text/plain'
        });

        // Use the API client to upload the document
        const response = await api.uploadDocument(file);
        // console.log(`Workspace saved as document: ${response.filename}`);
    }

    function deleteCategory(category: WorkspaceCategory) {
        const storageKey = getCategoryKey(category);
        localStorage.removeItem(storageKey);
        set(DEFAULT_WORKSPACE);
    }

    function deleteCurrent() {
        const storageKey = getCategoryKey(get(workspaceStore).name);
        localStorage.removeItem(storageKey);
        set(DEFAULT_WORKSPACE);
    }

    function clear() {
        Object.values(WorkspaceCategory).forEach(category => {
            const storageKey = getCategoryKey(category);
            localStorage.removeItem(storageKey);
        });
        set(DEFAULT_WORKSPACE);
    }

    return {
        subscribe,
        get,
        set,
        update,
        changeCategory,
        setContent,
        updateContentStream,
        commitStreamUpdate,
        generateWorkspaceUpdate,
        getCategories: () => Object.values(WorkspaceCategory),
        deleteCategory,
        deleteCurrent,
        saveRemoteCopy,
        clear,
    };
}

export const workspaceStore = createWorkspaceStore();
