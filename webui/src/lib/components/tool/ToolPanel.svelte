<!-- src/lib/components/tool/ToolPanel.svelte -->
<script lang="ts">
    import { configStore } from "$lib/store/configStore";
    import { toolStore } from "$lib/store/toolStore";
    import { thoughtStore } from "$lib/store/thoughtStore";
    import { workspaceStore } from "$lib/store/workspaceStore";
    import { clipboardStore } from "$lib/store/clipboardStore";
    import { chatStore } from "$lib/store/chatStore";
    import ModelSelect from "../model/ModelSelect.svelte";
    import { RefreshCw } from "lucide-svelte";
    import "$lib/../styles/meta-panels.css";
    import type { ToolResponse } from "$lib/types";

    let tools: ToolResponse[] = [];
    let isLoading = false;
    let error: string | null = null;
    let isPanelVisible = false;

    // Reactive statements to handle store updates
    $: {
        tools = $toolStore?.tools || [];
        isLoading = $toolStore?.loading || false;
        error = $toolStore?.error || null;
        isPanelVisible = $configStore?.showTools || false;
    }

    // Initialize tools on component mount
    $: {
        if (!isLoading && tools.length === 0 && !error) {
            toolStore.initialize();
        }
    }

    function handleGenerate() {
        const workspaceContent = $clipboardStore.includeInMessages
            ? $workspaceStore.content
            : undefined;
        toolStore.generate($chatStore.conversationHistory, $configStore, {
            workspaceContent,
            systemMessage: $configStore.systemMessage,
            currentLocation: $configStore.location,
            pinnedMessages: $configStore.pinnedMessages,
            activeDocument: $configStore.selectedDocument,
            temperature: $configStore.temperature,
            maxTokens: $configStore.maxTokens,
            frequencyPenalty: $configStore.frequencyPenalty,
            presencePenalty: $configStore.presencePenalty,
            repetitionPenalty: $configStore.repetitionPenalty,
            minP: $configStore.minP,
            topP: $configStore.topP,
            topK: $configStore.topK,
            thoughtContent: $thoughtStore.thoughtContent,
        });
    }

    function handleToolSelect(tool: ToolResponse): void {
        const isSelected = isToolSelected(tool);
        if (isSelected) {
            toolStore.unselectTool(tool);
        } else {
            toolStore.selectTool(tool);
        }
    }

    function isToolSelected(tool: ToolResponse): boolean {
        const selectedTools = $toolStore?.selectedTools || [];
        return selectedTools.some(
            (t) =>
                t.type === tool.type && t.function.name === tool.function.name,
        );
    }
</script>

<div class="meta-panel">
    <button
        class="meta-panel-header collapsible"
        on:click={() =>
            configStore.updateField("showTools", !$configStore.showTools)}
    >
        <span>Tools</span>
        <span class="toggle-icon">{$configStore.showTools ? "▼" : "▶"}</span>
    </button>

    {#if $configStore.showTools}
        <div class="meta-panel-content">
            <div class="tool-section">
                <div class="tool-content">
                    <div class="tool-list">
                        <div class="tool-header">
                            <h3>Available Tools</h3>
                            <button
                                class="refresh-button"
                                on:click={() => toolStore.refresh()}
                                disabled={isLoading}
                                title="Refresh tools list"
                            >
                                <span class:loading={isLoading}>
                                    <RefreshCw size={16} />
                                </span>
                            </button>
                        </div>
                        {#if isLoading}
                            <p>Loading tools...</p>
                        {:else if error}
                            <p class="error">{error}</p>
                        {:else}
                            <div class="tool-select">
                                {#each tools as tool}
                                    <label class="tool-item">
                                        <input
                                            type="checkbox"
                                            checked={isToolSelected(tool)}
                                            on:change={() =>
                                                handleToolSelect(tool)}
                                        />
                                        <div class="tool-info">
                                            <span class="tool-name"
                                                >{tool.function.name}</span
                                            >
                                            <span class="tool-description"
                                                >{tool.function
                                                    .description}</span
                                            >
                                        </div>
                                    </label>
                                {/each}
                            </div>
                        {/if}
                    </div>

                    <!-- Generated Output Column -->
                    <div class="tool-output">
                        <h3>Generated Output</h3>
                        <textarea
                            class="output-textarea"
                            value={$toolStore.generatedOutput}
                            readonly
                            placeholder="Tool output will appear here..."
                        />
                    </div>
                </div>

                <!-- Controls Row -->
                <div class="tool-controls">
                    <ModelSelect
                        bind:value={$configStore.toolModel}
                        category="functions"
                    />
                    <button
                        class="submit-button"
                        on:click={handleGenerate}
                        disabled={$toolStore.selectedTools?.length === 0 ||
                            $toolStore.isGenerating}
                    >
                        {$toolStore.isGenerating ? "Generating..." : "Generate"}
                    </button>
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
    .tool-list {
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        padding: 1rem;
        background-color: #f9fafb;
    }

    .tool-select {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 300px;
        overflow-y: auto;
    }

    .tool-item {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 0.5rem;
        cursor: pointer;
        border-radius: 0.25rem;
        transition: background-color 0.2s;
    }

    .tool-item:hover {
        background-color: #f3f4f6;
    }

    .tool-info {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .tool-name {
        font-size: 0.875rem;
        font-weight: 500;
        color: #374151;
    }

    .tool-description {
        font-size: 0.75rem;
        color: #6b7280;
        line-height: 1.25;
    }

    .tool-section {
        width: 100%;
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .tool-content {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .tool-output {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .output-textarea {
        flex: 1;
        min-height: 300px;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        padding: 0.5rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
            monospace;
        font-size: 0.875rem;
        resize: none;
        background-color: #f9fafb;
    }

    .tool-controls {
        display: flex;
        gap: 1rem;
        align-items: center;
    }

    .submit-button {
        padding: 0.5rem 1rem;
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 0.375rem;
        cursor: pointer;
        font-size: 0.875rem;
        transition: background-color 0.2s;
    }

    .submit-button:hover:not(:disabled) {
        background-color: #2563eb;
    }

    .submit-button:disabled {
        background-color: #9ca3af;
        cursor: not-allowed;
    }

    .error {
        color: #dc2626;
        font-size: 0.875rem;
    }

    h3 {
        margin: 0;
        font-size: 0.875rem;
        font-weight: 600;
        color: #374151;
    }

    .tool-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .refresh-button {
        background: none;
        border: none;
        padding: 0.25rem;
        cursor: pointer;
        color: #6b7280;
        border-radius: 0.25rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .refresh-button:hover:not(:disabled) {
        background-color: #f3f4f6;
        color: #374151;
    }

    .refresh-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .loading {
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
</style>
