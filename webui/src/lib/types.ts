// lib/types.ts

export type PipelineType = 'analyst' | 'coder' | 'dreamer' | 'journaler' | 'philosopher' | 'reporter' | 'summarizer';
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
    user_id: string | null | undefined;
    persona: string | null | undefined;
    persona_id: string | null | undefined;

    // Emotional State Settings
    emotionalStateEnabled: boolean;
    state1: string | null | undefined;
    state2: string | null | undefined;
    state3: string | null | undefined;

    // Sentiment Meter Settings
    sentimentMeterEnabled: boolean;
    selectedSentiment: string | null | undefined;
    sentimentName: string | null | undefined;
    sentimentGuidance: string | null | undefined;
    sentimentLevel: number | null | undefined;

    // Footer Settings
    selectedFooter: string | null | undefined;

    // Advanced Settings
    chatModel: string | null | undefined;
    completionModel: string | null | undefined;
    thoughtModel: string | null | undefined;
    workspaceModel: string | null | undefined;
    pipelineModel: string | null | undefined;
    toolModel: string | null | undefined;

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

    // Auto-scroll and Auto-think
    autoScroll: boolean;
    autoThink: boolean;

    // Thought Content
    thoughtContent: string | null | undefined;

    // UI State
    showAdvanced: boolean;
    showControls: boolean;
    showHeader: boolean;
    showClipboard: boolean;
    showWorkspace: boolean;
    showThought: boolean;
    showTools: boolean;
    includeWorkspaceInMessages: boolean;
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

export interface CompletionMessage {
    role: string;
    content: string;
    timestamp: number;
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

export interface ToolFunctionParameters {
    type: string;
    properties: Record<string, any>;
    required: string[];
    examples?: Record<string, any>[];
}

export interface ToolFunction {
    name: string;
    description: string;
    parameters: ToolFunctionParameters;
}

export interface Tool {
    type: string;
    function: ToolFunction;
}

export interface CreateToolRequest extends Tool { }

export interface UpdateToolRequest {
    type?: string;
    function?: ToolFunction;
}

export interface ToolResponse extends Tool { }

export interface ToolListResponse {
    tools: ToolResponse[];
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

// Define our workspace categories - we should make these dynamic someday
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

export interface WorkspaceItem {
    name: WorkspaceCategory;
    content: string;
    contentStream: string;
    wordCount: number;
}