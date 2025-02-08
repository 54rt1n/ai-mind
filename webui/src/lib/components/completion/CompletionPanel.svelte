<!-- src/lib/components/CompletionPanel.svelte -->
<script lang="ts">
    import { completionStore } from "$lib/store/completionStore";
    import { configStore } from "$lib/store/configStore";
    import { modelStore } from "$lib/store/modelStore";
    import type { CompletionConfig, ChatConfig } from "$lib/types";
    import { workspaceStore } from "$lib/store/workspaceStore";
    import { Copy, Send, Edit2 } from "lucide-svelte";
    import type { ChangeEventHandler } from "svelte/elements";
    import { ChevronDown, ChevronRight } from 'lucide-svelte';
    import ModelSelect from "../model/ModelSelect.svelte";
    
    let showAdvancedSettings = false;
    
    function toggleAdvancedSettings() {
        showAdvancedSettings = !showAdvancedSettings;
    }

    async function handleSubmit() {
        if (!$completionStore.input.trim()) return;

        const config = $configStore;

        const request: CompletionConfig = {
            model: config.completionModel,
            prompt: $completionStore.input,
            ...$completionStore.config,
        };

        await completionStore.sendCompletion(request);
    }

    function copyFromWorkspace() {
        $completionStore.input = $workspaceStore.content;
    }

    function copyToWorkspace() {
        $workspaceStore.content = $completionStore.output;
    }

    function appendToInput() {
        completionStore.appendToInput();
    }

    let isEditing = false;
    let editedOutput = "";

    function startEditing() {
        editedOutput = $completionStore.output;
        isEditing = true;
    }

    function saveEdit() {
        $completionStore.output = editedOutput;
        isEditing = false;
    }

    $: inputWordCount = $completionStore.input.split(/\s+/).length;
    $: inputTotalLength = $completionStore.input.length;
</script>

<div class="meta-panel">
    <div class="meta-panel-header">Completion</div>
    <div class="meta-panel-content">
        <div class="completion-input">
            <button
                class="workspace-button from-workspace"
                on:click={copyFromWorkspace}
                disabled={!$workspaceStore.content}
            >
                <Copy size={16} />
                From Workspace
            </button>
        </div>
        <div class="completion-input">
            <textarea
                bind:value={$completionStore.input}
                placeholder="Enter your prompt here..."
                rows="5"
            />
        </div>

        <div class="word-count">
            Word Count: {inputWordCount} / Total Length: {inputTotalLength}
        </div>

        <button
            class="submit-button"
            on:click={handleSubmit}
            disabled={$completionStore.loading ||
                !$completionStore.input.trim()}
        >
            <Send size={16} />
            Generate
        </button>

        <div class="filter-group">
            <button class="filter-toggle" on:click={toggleAdvancedSettings}>
                {#if showAdvancedSettings}
                    <ChevronDown size={16} />
                {:else}
                    <ChevronRight size={16} />
                {/if}
                Advanced Settings
            </button>
            
            {#if showAdvancedSettings}
                <div class="filter-content">
                    <ModelSelect bind:value={$completionStore.config.model} category="completion" />
                    <div class="filter-grid">
                        <label>
                            Temperature
                            <input type="number" bind:value={$completionStore.config.temperature} min="0" max="2" step="0.1" />
                        </label>
                        
                        <label>
                            Presence Penalty
                            <input type="number" bind:value={$completionStore.config.presence_penalty} min="-2" max="2" step="0.1" />
                        </label>
                        
                        <label>
                            Frequency Penalty
                            <input type="number" bind:value={$completionStore.config.frequency_penalty} min="-2" max="2" step="0.1" />
                        </label>
                        
                        <label>
                            Top P
                            <input type="number" bind:value={$completionStore.config.top_p} min="0" max="1" step="0.05" />
                        </label>
                        
                        <label>
                            Top K
                            <input type="number" bind:value={$completionStore.config.top_k} min="1" step="1" />
                        </label>
                        
                        <label>
                            Min P
                            <input type="number" bind:value={$completionStore.config.min_p} min="0" max="1" step="0.05" />
                        </label>
                        
                        <label>
                            Max Tokens
                            <input type="number" bind:value={$completionStore.config.max_tokens} min="1" step="1" />
                        </label>
                        
                        <label>
                            Stop Token
                            <input 
                                type="text" 
                                value={$completionStore.config.stop?.[0] || ''} 
                                on:input={(e) => {
                                    $completionStore.config.stop = e.currentTarget.value ? [e.currentTarget.value] : undefined;
                                }}
                                placeholder="Enter stop token" 
                            />
                        </label>
                    </div>
                </div>
            {/if}
        </div>

        {#if $completionStore.output}
            <div class="completion-output">
                <div class="output-header">
                    <h3>Output:</h3>
                    <button
                        class="edit-button"
                        on:click={startEditing}
                        disabled={isEditing}
                    >
                        <Edit2 size={16} />
                    </button>
                </div>
                {#if isEditing}
                    <textarea
                        bind:value={editedOutput}
                        rows="5"
                        class="edit-textarea"
                    />
                    <button class="save-button" on:click={saveEdit}>Save</button
                    >
                {:else}
                    <pre>{$completionStore.output}</pre>
                {/if}
            </div>
        {:else if $completionStore.error}
            <div class="error">{$completionStore.error}</div>
        {:else if $completionStore.loading}
            <div class="loading">Generating...</div>
        {/if}
        <div class="completion-controls">
            <button
                class="workspace-button to-workspace"
                on:click={copyToWorkspace}
                disabled={!$completionStore.output}
            >
                <Copy size={16} />
                To Workspace
            </button>
            <button
                class="workspace-button to-input"
                on:click={appendToInput}
                disabled={!$completionStore.output}
            >
                <Copy size={16} />
                Append to Input
            </button>
        </div>
    </div>
</div>

<style>
    .completion-controls {
        margin-bottom: 1rem;
    }

    .completion-input {
        margin-bottom: 1rem;
    }

    textarea {
        width: 100%;
        padding: 0.75rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        resize: vertical;
        font-family: inherit;
        line-height: 1.5;
    }

    .submit-button {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: #4caf50;
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 0.375rem;
        cursor: pointer;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    .submit-button:hover:not(:disabled) {
        background: #43a047;
    }

    .submit-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .workspace-button {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: #2196f3;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        cursor: pointer;
        font-weight: 500;
    }

    .workspace-button:hover:not(:disabled) {
        background: #1976d2;
    }

    .workspace-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .completion-output {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        padding: 1rem;
    }

    .completion-output h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        color: #374151;
    }

    .completion-output pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: inherit;
        line-height: 1.5;
    }

    .loading {
        color: #4b5563;
        font-style: italic;
        margin: 1rem 0;
    }

    .error {
        color: #dc2626;
        margin: 1rem 0;
    }

    .output-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .edit-button {
        background: none;
        border: none;
        cursor: pointer;
        padding: 0.25rem;
        border-radius: 0.25rem;
        color: #666;
    }

    .edit-button:hover:not(:disabled) {
        background-color: rgba(0, 0, 0, 0.1);
        color: #000;
    }

    .edit-textarea {
        width: 100%;
        margin: 0.5rem 0;
    }

    .save-button {
        background: #4caf50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        cursor: pointer;
    }

    .filter-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        padding: 1rem;
        background: #f9fafb;
        border-radius: 0.375rem;
    }

    .filter-grid label {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .filter-grid input[type="number"],
    .filter-grid input[type="text"] {
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.25rem;
    }

</style>
