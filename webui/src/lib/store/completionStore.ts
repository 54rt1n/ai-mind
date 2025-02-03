// lib/store/completionStore.ts
import { writable, get } from 'svelte/store';
import { browser } from '$app/environment';
import type { CompletionConfig } from '$lib';

interface CompletionState {
    config: CompletionConfig;
    input: string;
    output: string;
    loading: boolean;
    error: string | null;
}

function createCompletionStore() {
    const STORAGE_KEY = 'completionStore';
    const API_URL = 'http://10.100.0.57:8000/v1/completions';
    const API_KEY = '123';
    
    const initialState: CompletionState = browser 
        ? JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null') || {
            config: {},
            input: '',
            output: '',
            loading: false,
            error: null
        }
        : {
            config: {},
            input: '',
            output: '',
            loading: false,
            error: null
        };

    const { subscribe, set, update } = writable<CompletionState>(initialState);

    if (browser) {
        subscribe(state => {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        });
    }

    async function sendCompletion(config: CompletionConfig) {
        update(store => ({ 
            ...store, 
            output: '',
            loading: true,
            error: null 
        }));

        config.stream = true;
        config.include_stop_str_in_output = true;

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_KEY}`
                },
                body: JSON.stringify({...config})
            });

            const reader = response.body!.getReader();
            const decoder = new TextDecoder();
            let result = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data:')) {
                        const data = line.slice(5).trim();
                        if (data === '[DONE]') break;

                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.choices && parsed.choices[0].text) {
                                result += parsed.choices[0].text;
                                update(store => ({
                                    ...store,
                                    output: result
                                }));
                            }
                        } catch (error) {
                            console.error('Error parsing chunk:', error);
                        }
                    }
                }
            }
        } catch (error) {
            update(store => ({
                ...store,
                error: 'Failed to get completion'
            }));
        } finally {
            update(store => ({ ...store, loading: false }));
        }
    }

    function appendToInput() {
        update(store => ({ ...store, input: store.input + store.output, output: '' }));
    }

    return {
        subscribe,
        get,
        set,
        clear: () => set({ input: '', output: '', loading: false, error: null, config: {} }),
        appendToInput,
        sendCompletion,
    };
}

export const completionStore = createCompletionStore();