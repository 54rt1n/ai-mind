// lib/store/workspaceStore.ts
import { writable, derived, get } from 'svelte/store';
import { api } from '$lib';
import { browser } from '$app/environment';

// Define our workspace categories
export enum WorkspaceCategory {
    None = 'None',
    Schedule = 'Schedule',
    Plan = 'Plan',
    Tasks = 'Tasks',
    Notes = 'Notes',
    Fitness = 'Fitness',
    Persona = 'Persona',
    Fun = 'Fun'
}

interface WorkspaceItem {
    name: WorkspaceCategory;
    content: string;
    wordCount: number;
}

const DEFAULT_WORKSPACE: WorkspaceItem = {
    name: WorkspaceCategory.None,
    content: '',
    wordCount: 0,
};

function createWorkspaceStore() {
    const STORAGE_KEY_BASE = 'workspaceStore';
    const STORAGE_KEY_CATEGORY = 'category';

    function getCategoryKey(category: WorkspaceCategory): string {
        return `${STORAGE_KEY_BASE}::${category}`;
    }

    function loadCategory(categoryName: string | WorkspaceCategory) : WorkspaceItem {
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
        getCategories: () => Object.values(WorkspaceCategory),
        deleteCategory,
        deleteCurrent,
        saveRemoteCopy,
        clear,
    };
}

export const workspaceStore = createWorkspaceStore();
