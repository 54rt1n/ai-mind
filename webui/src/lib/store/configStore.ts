// lib/store/configStore.ts
import { writable, derived, get } from 'svelte/store';
import { browser } from '$app/environment';
import type { ChatConfig, DocumentInfo } from '$lib';

const DEFAULT_CONFIG: ChatConfig = {
    emotionalStateEnabled: false,
    user_id: '',
    persona: '',
    persona_id: '',
    state1: '',
    state2: '',
    state3: '',
    sentimentMeterEnabled: false,
    selectedSentiment: 'neutral',
    sentimentName: '',
    sentimentGuidance: '',
    sentimentLevel: 0,
    selectedFooter: '',
    systemMessage: '',
    location: '',
    mood: 'Delighted',
    chatModel: undefined,
    completionModel: undefined,
    thoughtModel: undefined,
    workspaceModel: undefined,
    pipelineModel: undefined,
    toolModel: undefined,
    maxTokens: 512,
    temperature: 0.7,
    frequencyPenalty: 0.0,
    presencePenalty: 0.0,
    repetitionPenalty: 1.05,
    minP: 0.05,
    topP: 1.0,
    topK: 40,
    selectedDocument: null,
    pinnedMessages: [],
    showAdvanced: false,
    showControls: false,
    showHeader: true,
    showClipboard: false,
    showWorkspace: false,
    showThought: false,
    showTools: false,
    includeWorkspaceInMessages: false,
    autoScroll: true,
    autoThink: false,
    thoughtContent: "",
};


function getDefaultPrompt() {
}
function createConfigStore() {
    // Load initial state from localStorage only in browser environment
    const initialConfig: ChatConfig = browser
        ? JSON.parse(localStorage.getItem('chatConfig') || 'null') || DEFAULT_CONFIG
        : DEFAULT_CONFIG;

    const { subscribe, set, update } = writable<ChatConfig>(initialConfig);

    // Subscribe to changes and save to localStorage only in browser environment
    if (browser) {
        subscribe(config => {
            localStorage.setItem('chatConfig', JSON.stringify(config));
        });
    }

    const formatters = {
        getEmotionalStateHeader: (config: ChatConfig): string => {
            if (!config.emotionalStateEnabled || !config.state1) return '';

            return `[== ${config.user_id ? `${config.user_id}'s ` : ''}Emotional State: +${config.state1}+${config.state2 ? ` with a sense of +${config.state2}+` : ''
                }${config.state3 ? ` and +${config.state3}+` : ''} ==]`;
        },

        getSentimentMeter: (config: ChatConfig): string => {
            if (!config.sentimentMeterEnabled) return '';

            const personaName = config.persona ? `${config.persona}'s ` : '';
            const currentLevel = config.sentimentLevel || 0;
            const guidance = config.sentimentGuidance ? ` | ${config.sentimentGuidance}` : '';
            const meterAndLevel = config.sentimentName ? ` | ${config.sentimentName} Meter: ${currentLevel}%` : '';

            return `[~~ ${personaName}Sentiment: ${config.selectedSentiment}${guidance}${meterAndLevel} ~~]`;
        }
    };

    return {
        get,
        subscribe,
        set,
        update,
        formatters,
        reset: () => {
            set(DEFAULT_CONFIG);
        },
        updateField: <K extends keyof ChatConfig>(field: K, value: ChatConfig[K]) => {
            update(config => ({
                ...config,
                [field]: value
            }));
        },
        // Select/deselect a document
        documentSelect: (document: DocumentInfo | null) => {
            update(store => ({
                ...store,
                selectedDocument: (!document || (store.selectedDocument?.name === document.name)) ? null : document
            }));
        },
    };
}

export const configStore = createConfigStore();