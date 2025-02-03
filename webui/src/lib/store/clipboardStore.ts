// lib/store/clipboardStore.ts
import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';

interface ClipboardState {
    pattern: string;
    content: string;
    workspace: string;
    includeInMessages: boolean;  // Added new property
}

const STORAGE_KEY = 'clipboardStore';

function createClipboardStore() {
    // Load initial state from localStorage if in browser
    const initialState: ClipboardState = browser 
        ? JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null') || { 
            pattern: '', 
            content: '', 
            workspace: '',
            includeInMessages: false 
        }
        : { 
            pattern: '', 
            content: '', 
            workspace: '',
            includeInMessages: false 
        };

    const { subscribe, set, update } = writable<ClipboardState>(initialState);

    // Subscribe to changes and save to localStorage when in browser
    if (browser) {
        subscribe(state => {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        });
    }

    const extract = (state: ClipboardState) => {
        try {
            const regex = new RegExp(state.pattern, 'gs');
            const matches = [...state.content.matchAll(regex)];
            
            // Check if pattern has capture groups by looking at first match
            if (matches.length > 0 && matches[0].length > 1) {
                // Has capture groups - get the results
                return matches
                    .map(match => match.slice(1))
                    .join('\n');
            }
            
            // No capture groups - return full matches
            return matches
                .map(match => match[0])
                .join('\n');
        } catch (e) {
            console.error('Regex error:', e);
            return state.content; // Return original content if regex fails
        }
    }

    return {
        subscribe,
        extract,
        set,
        setPattern: (pattern: string) => update(state => ({ ...state, pattern })),
        setContent: (content: string) => update(state => ({ ...state, content })),
        setWorkspace: (workspace: string) => update(state => ({ ...state, workspace })),
        setIncludeInMessages: (includeInMessages: boolean) => update(state => ({ ...state, includeInMessages })),
        clear: () => set({ pattern: '', content: '', workspace: '', includeInMessages: false })
    };
}

export const clipboardStore = createClipboardStore();