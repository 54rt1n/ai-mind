// lib/store/toolStore.ts
import { writable, get } from 'svelte/store';
import { browser } from '$app/environment';
import { api } from '$lib';
import type { ToolResponse } from '$lib/types';

interface ToolStore {
    tools: ToolResponse[];
    loading: boolean;
    error: string | null;
}

const STORAGE_KEY = 'toolStore';

function createToolStore() {
    // Initialize from localStorage if in browser environment
    const initialTools: ToolStore = browser
        ? JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null') || { tools: [], loading: false, error: null }
        : { tools: [], loading: false, error: null };

    const { subscribe, set, update } = writable<ToolStore>(initialTools);

    // Save to localStorage when store updates (browser only)
    if (browser) {
        subscribe(state => {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        });
    }

    async function fetchTools() {
        update(store => ({ ...store, loading: true, error: null }));

        try {
            const results = await api.getTools();
            update(store => ({
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

    return {
        subscribe,
        refresh: fetchTools,
        initialize: async () => {
            const state = get({ subscribe });
            if (state.tools.length === 0) {
                await fetchTools();
            }
        }
    };
}

export const toolStore = createToolStore(); 