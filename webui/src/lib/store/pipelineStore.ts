// lib/store/pipelineStore.ts
import { writable } from 'svelte/store';
import { browser } from '$app/environment';
import type { PipelineType, BasePipelineSchema } from '$lib';

interface PipelineConfig {
    pipelineType: PipelineType;
    formData: BasePipelineSchema;
    isExpanded: boolean;
}

interface PipelineInfo {
    type: PipelineType;
    name: string;
    description: string;
    icon: string;
}
// Pipeline metadata for UI display
const PIPELINE_INFO: PipelineInfo[] = [
    {
        type: 'analyst',
        name: 'Analyst',
        description: 'Analyze conversation patterns and insights',
        icon: 'brain'
    },
    {
        type: 'summarizer',
        name: 'Summarizer',
        description: 'Generate a concise summary of the conversation',
        icon: 'file-text'
    },
    {
        type: 'journaler',
        name: 'Journaler',
        description: 'Create reflective journal entries',
        icon: 'book'
    },
    {
        type: 'philosopher',
        name: 'Philosopher',
        description: 'Deep contemplation and analysis',
        icon: 'thought-bubble'
    },
    {
        type: 'dreamer',
        name: 'Dreamer',
        description: 'Explore imaginative scenarios',
        icon: 'dream'
    }
];

// Type guard to check if a string is a valid PipelineType
function isPipelineType(value: string): value is PipelineType {
    return PIPELINE_INFO.map(p => p.type).includes(value as PipelineType);
}

const DEFAULT_CONFIG: PipelineConfig = {
    pipelineType: 'analyst',
    formData: {
        user_id: undefined,
        persona_id: undefined,
        conversation_id: undefined,
        mood: undefined,
        model: undefined,
        guidance: undefined,
        top_n: undefined,
        query_text: undefined,
    },
    isExpanded: true
};

function createPipelineStore() {
    const initialConfig: PipelineConfig = browser
        ? JSON.parse(localStorage.getItem('pipelineConfig') || 'null') || DEFAULT_CONFIG
        : DEFAULT_CONFIG;

    const { subscribe, set, update } = writable<PipelineConfig>(initialConfig);

    if (browser) {
        subscribe(config => {
            localStorage.setItem('pipelineConfig', JSON.stringify(config));
        });
    }

    return {
        subscribe,
        set,
        update,
        setPipelineType: (type: string) => {
            if (isPipelineType(type)) {
                update(config => ({
                    ...config,
                    pipelineType: type
                }));
            } else {
                console.error(`Invalid pipeline type: ${type}`);
            }
        },
        updateFormData: (data: Partial<BasePipelineSchema>) => {
            update(config => ({
                ...config,
                formData: {
                    ...config.formData,
                    ...data
                }
            }));
        },
        toggleExpanded: () => {
            update(config => ({
                ...config,
                isExpanded: !config.isExpanded
            }));
        },
        reset: () => {
            set(DEFAULT_CONFIG);
        },
        // Add getter for pipeline info
        getPipelineInfo: () => PIPELINE_INFO
    };
}

export const pipelineStore = createPipelineStore();