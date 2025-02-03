// lib/store/documentStore.ts
import { writable, get } from 'svelte/store';
import { browser } from '$app/environment';
import { api } from '$lib';
import type { DocumentInfo } from '$lib';

interface DocumentStore {
    documents: DocumentInfo[];
    loading: boolean;
    error: string | null;
}

const STORAGE_KEY = 'documentStore';

function createDocumentStore() {
    // Initialize from localStorage if in browser environment
    const initialState: DocumentStore = browser
        ? JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null') || {
            documents: [],
            selectedDocument: null,
            loading: false,
            error: null
        }
        : {
            documents: [],
            selectedDocument: null,
            loading: false,
            error: null
        };

    const { subscribe, set, update } = writable<DocumentStore>(initialState);

    // Save to localStorage when store updates (browser only)
    if (browser) {
        subscribe(state => {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        });
    }

    return {
        subscribe,
        
        // Fetch all documents
        fetchDocuments: async () => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const response = await api.getDocuments();
                update(store => ({
                    ...store,
                    documents: response.documents,
                    loading: false
                }));
            } catch (error) {
                update(store => ({
                    ...store,
                    loading: false,
                    error: 'Failed to fetch documents'
                }));
            }
        },

        // Upload a new document
        uploadDocument: async (file: File) => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                await api.uploadDocument(file);
                // Refresh document list after upload
                const response = await api.getDocuments();
                update(store => ({
                    ...store,
                    documents: response.documents,
                    loading: false
                }));
            } catch (error) {
                update(store => ({
                    ...store,
                    loading: false,
                    error: 'Failed to upload document'
                }));
                throw error;
            }
        },

        // Delete a document
        deleteDocument: async (documentName: string) => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                await api.deleteDocument(documentName);
                // Refresh document list after deletion
                const response = await api.getDocuments();
                update(store => ({
                    ...store,
                    documents: response.documents,
                    selectedDocument: null,
                    loading: false
                }));
            } catch (error) {
                update(store => ({
                    ...store,
                    loading: false,
                    error: 'Failed to delete document'
                }));
                throw error;
            }
        },

        // Download a document
        downloadDocument: async (documentName: string) => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const blob = await api.downloadDocument(documentName);
                // Create download link
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = documentName;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                update(store => ({ ...store, loading: false }));
            } catch (error) {
                update(store => ({
                    ...store,
                    loading: false,
                    error: 'Failed to download document'
                }));
                throw error;
            }
        },

    };
}

export const documentStore = createDocumentStore();