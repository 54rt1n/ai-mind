// lib/store/modelStore.ts
import { writable, get } from 'svelte/store';
import { browser } from '$app/environment';
import { api } from '$lib';
import type { ChatModel, ModelCategory } from '$lib';

interface ModelStore {
    models: ChatModel[];
    categories: ModelCategory[];
    loading: boolean;
    error: string | null;
}

const STORAGE_KEY = 'chatModels';

function createModelStore() {
    // Initialize from localStorage if in browser environment
    const initialModels: ModelStore = browser
        ? JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null') || { models: [], categories: [], loading: false, error: null }
        : { models: [], categories: [], loading: false, error: null };

    const { subscribe, set, update } = writable<ModelStore>(initialModels);

    // Save to localStorage when store updates (browser only)
    if (browser) {
        subscribe(state => {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        });
    }

    async function fetchModels() {
        update(store => ({ ...store, loading: true, error: null }));
        
        try {
            const results = await api.getChatModels();
            update(store => ({
                models: results.models,
                categories: results.categories,
                loading: false,
                error: null
            }));
        } catch (error) {
            update(store => ({
                ...store,
                loading: false,
                error: 'Failed to fetch models'
            }));
            console.error('Error fetching models:', error);
        }
    }

    return {
        subscribe,
        refresh: fetchModels,
        initialize: async () => {
            const state = get({ subscribe });
            if (state.models.length === 0) {
                await fetchModels();
            }
        }
    };
}

export const modelStore = createModelStore();