// lib/types.ts

export type PipelineType = 'analysis' | 'journal' | 'ponder' | 'summary' | 'daydream';
export type NotificationType = 'success' | 'loading' | 'copy' | 'info' | 'warning' | 'error';

export interface NotificationParams {
    show: boolean;
    notificationType: NotificationType;
    message: string;
    loading: boolean;
    duration?: number;
}

export interface BasePipelineSchema {
    user_id?: string;
    persona_id?: string;
    conversation_id?: string;
    mood?: string;
    guidance?: string;
    top_n?: number;
    query_text?: string;
    model?: string;
}

export interface ChatMessage {
    doc_id: string;
    document_type: string;
    user_id: string;
    persona_id: string;
    conversation_id: string;
    date: string;
    role: string;
    content: string;
    branch?: number;
    sequence_no?: number;
    speaker?: string;
    score?: number;
    emotion_a?: string;
    emotion_b?: string;
    emotion_c?: string;
    emotion_d?: string;
}

export interface ConversationMessage {
    doc_id: string;
    document_type: string;
    user_id: string;
    persona_id: string;
    conversation_id: string;
    timestamp: number;
    role: string;
    content: string;
    speaker_id: string;
    listener_id: string;
    branch?: number;
    sequence_no?: number;
    speaker?: string;
    score?: number;
    emotion_a?: string;
    emotion_b?: string;
    emotion_c?: string;
    emotion_d?: string;
}

export interface ChatConfig {
    user_id: string;
    persona: string;
    persona_id: string;

    // Emotional State Settings
    emotionalStateEnabled: boolean;
    state1: string;
    state2: string;
    state3: string;

    // Sentiment Meter Settings
    sentimentMeterEnabled: boolean;
    selectedSentiment: string;
    sentimentName: string;
    sentimentGuidance: string;
    sentimentLevel: number;

    // Footer Settings
    selectedFooter: string;

    // Advanced Settings
    model: string | null | undefined;
    mood: string | null | undefined;
    systemMessage: string | null | undefined;
    location: string | null | undefined;
    maxTokens: number | null | undefined;
    temperature: number | null | undefined;
    frequencyPenalty: number | null | undefined;
    presencePenalty: number | null | undefined;
    repetitionPenalty: number | null | undefined;
    minP: number | null | undefined;
    topP: number | null | undefined;
    topK: number | null | undefined;

    // Pinned Messages
    selectedDocument: DocumentInfo | null;
    pinnedMessages: ChatMessage[];

    // UI State
    showAdvanced: boolean;
    showControls: boolean;
    showHeader: boolean;
    showClipboard: boolean;
    showThought: boolean;
        
    // Thought Settings
    thoughtModel: string | undefined;
    thoughtDefaultContent: string | undefined;
    thoughtPrompt: string | undefined;
    thoughtContent: string | undefined;
    thoughtSystemMessage: string | undefined;
    thoughtXml: string | undefined;
    thoughtUserContext: string | undefined;
}

export interface ChatModel {
    name: string;
    provider: string;
    architecture: string;
    category: string[];
    size: string;
    default_for: string[];
    sampler?: {
        temperature?: number;
        frequency_penalty?: number;
        presence_penalty?: number;
        repetition_penalty?: number;
        min_p?: number;
        top_p?: number;
        top_k?: number;
    };
}

export interface ModelCategory {
    analysis: string[];
    conversation: string[];
    thought: string[];
    functions: string[];
    completion: string[];
}

export interface DocumentInfo {
    name: string;
    modified_time: number;
    size: number;
}

export interface CompletionConfig {
    model?: string | null;
    prompt?: string;
    presence_penalty?: number;
    frequency_penalty?: number;
    repetition_penalty?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    min_p?: number;
    seed?: number;
    stop?: string[];
    stop_token_ids?: number[];
    stream?: boolean;
    include_stop_str_in_output?: boolean;
    max_tokens?: number;
}

export interface CompletionResponse {
    id: string;
    object: string;
    created: number;
    model: string;
    choices: {
        text: string;
        index: number;
        logprobs: null;
        finish_reason: string;
    }[];
    usage: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
}

export interface PersonaSection {
    [key: string]: string;
}

export interface PersonaNestedSection {
    [key: string]: {
        [key: string]: string;
    };
}

export interface Persona {
    persona_id: string;
    chat_strategy: string;
    name: string;
    birthday: string;
    full_name: string;
    attributes: PersonaSection;
    features: PersonaSection;
    wakeup: string[];
    base_thoughts: string[];
    pif: PersonaSection;
    nshot: PersonaNestedSection;
    default_location: string;
    wardrobe: PersonaNestedSection;
    current_outfit: string;
    persona_version?: string;
    system_header?: string;
    include_date?: boolean;
}
