<!-- src/lib/components/toolbox/ActionCard.svelte -->
<script lang="ts">
    import {
        Edit2,
        Trash2,
        GripVertical,
        ChevronDown,
        ChevronUp,
    } from "lucide-svelte";
    import { toolboxStore, type ActionCard } from "$lib/store/toolboxStore";
    import { configStore } from "$lib/store/configStore";
    import ModelSelect from "$lib/components/model/ModelSelect.svelte";
    import ToolSelect from "$lib/components/tool/ToolSelect.svelte";
    import "$lib/../styles/meta-panels.css";
    import PersonaSelect from "$lib/components/persona/PersonaSelect.svelte";
    import type { Tool } from "$lib";
    import { createEventDispatcher } from "svelte";

    export let card: ActionCard;

    const dispatch = createEventDispatcher<{
        executeCard: { card: ActionCard };
    }>();

    let expanded = false;
    let editingTitle = false;

    function toggleExpanded() {
        expanded = !expanded;
    }

    function handleTitleClick() {
        editingTitle = true;
    }

    function handleTitleChange(event: Event) {
        const target = event.target as HTMLInputElement;
        const newTitle = target.value.trim();
        if (newTitle) {
            toolboxStore.updateCardTitle(card.id, newTitle);
        }
        editingTitle = false;
    }

    function handleTitleKeydown(event: KeyboardEvent) {
        if (event.key === "Enter") {
            const target = event.target as HTMLInputElement;
            const newTitle = target.value.trim();
            if (newTitle) {
                toolboxStore.updateCardTitle(card.id, newTitle);
            }
            editingTitle = false;
        } else if (event.key === "Escape") {
            editingTitle = false;
        }
    }

    function handleTitleBlur() {
        editingTitle = false;
    }

    function handlePromptChange(event: Event) {
        const target = event.target as HTMLTextAreaElement;
        toolboxStore.updateCardConfig(card.id, {
            prompt: target.value,
        });
    }

    function handleToolChange(event: CustomEvent<Tool>) {
        toolboxStore.updateCardConfig(card.id, {
            tool: event.detail,
        });
    }

    function handleModelChange(event: CustomEvent<string>) {
        toolboxStore.updateCardConfig(card.id, {
            chatConfig: {
                ...card.config.chatConfig,
                model: event.detail,
            },
        });
    }

    function handleTemperatureChange(event: Event) {
        const target = event.target as HTMLInputElement;
        toolboxStore.updateCardConfig(card.id, {
            chatConfig: {
                ...card.config.chatConfig,
                temperature: parseFloat(target.value),
            },
        });
    }

    function handleMaxTokensChange(event: Event) {
        const target = event.target as HTMLInputElement;
        toolboxStore.updateCardConfig(card.id, {
            chatConfig: {
                ...card.config.chatConfig,
                max_tokens: parseInt(target.value),
            },
        });
    }

    function handlePersonaChange(event: CustomEvent<string>) {
        toolboxStore.updateCardConfig(card.id, {
            chatConfig: {
                ...card.config.chatConfig,
                metadata: {
                    ...card.config.chatConfig.metadata,
                    persona_id: event.detail,
                },
            },
        });
    }

    function handleDelete() {
        toolboxStore.removeCard(card.id);
    }

    async function handleExecute() {
        dispatch("executeCard", { card });
    }

    async function handleExecuteDirect() {
        const originalTurns = card.thoughtTurns;
        card.thoughtTurns = 0;
        await dispatch("executeCard", { card });
        card.thoughtTurns = originalTurns;
    }

    function formatTimestamp(timestamp: number): string {
        return new Date(timestamp).toLocaleString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
            month: "short",
            day: "numeric",
        });
    }

    // Add computed properties for button state
    $: canExecute = Boolean(
        card.config.chatConfig.model && // Has model selected
            card.config.prompt.trim() && // Has non-empty prompt
            card.status !== "running" && // Not currently running
            (card.type !== "tool" || !!card.config.tool), // If tool type, must have pattern
    );

    $: buttonState = getButtonState(card, canExecute);

    $: isEditable = card.status !== "running";

    function getButtonState(
        card: ActionCard,
        canExecute: boolean,
    ): {
        disabled: boolean;
        label: string;
        tooltip: string;
    } {
        if (card.status === "running") {
            return {
                disabled: true,
                label: "Running...",
                tooltip: "Action is currently executing",
            };
        }

        if (!card.config.chatConfig.model) {
            return {
                disabled: true,
                label: "Execute",
                tooltip: "Please select a model",
            };
        }

        if (!card.config.prompt.trim()) {
            return {
                disabled: true,
                label: "Execute",
                tooltip: "Please enter a prompt",
            };
        }

        if (card.type === "tool" && !card.config.tool) {
            return {
                disabled: true,
                label: "Execute",
                tooltip: "Please select a tool",
            };
        }

        return {
            disabled: false,
            label: card.status === "error" ? "Retry" : "Execute",
            tooltip:
                card.status === "error"
                    ? "Click to retry the last execution"
                    : "Execute this action",
        };
    }

    function handleThoughtTurnsChange(event: Event) {
        const target = event.target as HTMLInputElement;
        toolboxStore.updateCardConfig(card.id, {
            thoughts: undefined,
        });
        card.thoughtTurns = Math.max(
            0,
            Math.min(4, parseInt(target.value) || 0),
        );
    }
</script>

<div class="meta-panel" data-card-id={card.id}>
    <div class="meta-panel-header">
        <div class="drag-handle">
            <GripVertical size={16} />
        </div>
        {#if editingTitle}
            <input
                type="text"
                class="title-input"
                value={card.title}
                on:change={handleTitleChange}
                on:keydown={handleTitleKeydown}
                on:blur={handleTitleBlur}
                autofocus
                disabled={!isEditable}
            />
        {:else}
            <span class="card-title" on:click={handleTitleClick}
                >{card.title}</span
            >
        {/if}
        <div class="card-controls">
            <button
                class="control-button"
                on:click={toggleExpanded}
                disabled={!isEditable}
            >
                {#if expanded}
                    <ChevronUp size={16} />
                {:else}
                    <ChevronDown size={16} />
                {/if}
            </button>
            <button
                class="control-button delete"
                on:click={handleDelete}
                disabled={!isEditable}
            >
                <Trash2 size={16} />
            </button>
        </div>
    </div>

    <div class="meta-panel-content">
        {#if expanded}
            <div class="config-section">
                <h3>Prompt</h3>
                <textarea
                    value={card.config.prompt}
                    rows="3"
                    placeholder="Enter prompt..."
                    on:input={handlePromptChange}
                    disabled={!isEditable}
                />

                <div class="thoughts-control">
                    <label>
                        Thought Turns
                        <input
                            type="number"
                            min="0"
                            max="4"
                            value={card.thoughtTurns}
                            on:change={handleThoughtTurnsChange}
                            disabled={!isEditable}
                        />
                    </label>
                </div>

                {#if card.config.thoughts !== undefined}
                    <div class="thoughts-display">
                        <h4>Current Thoughts</h4>
                        <div class="thoughts-content">
                            {#if card.config.thoughts}
                                <pre>{card.config.thoughts}</pre>
                            {:else}
                                <em>No thoughts generated yet</em>
                            {/if}
                        </div>
                    </div>
                {/if}

                {#if card.type === "tool"}
                    <div class="tool-input">
                        <h3>Tool</h3>
                        <ToolSelect
                            bind:value={card.config.tool}
                            on:change={handleToolChange}
                            disabled={!isEditable}
                        />
                    </div>
                {/if}
            </div>

            <div class="model-section">
                <h3>Model Configuration</h3>
                <div class="model-select">
                    <ModelSelect
                        bind:value={card.config.chatConfig.model}
                        category={card.type === "simple"
                            ? "conversation"
                            : "functions"}
                        on:change={handleModelChange}
                        disabled={!isEditable}
                    />
                </div>
                <div class="persona-select">
                    <PersonaSelect
                        bind:value={card.config.chatConfig.metadata.persona_id}
                        on:change={handlePersonaChange}
                        disabled={!isEditable}
                    />
                </div>
                <div class="config-grid">
                    <div class="config-item">
                        <label for="temperature">Temperature</label>
                        <div class="range-with-value">
                            <input
                                type="range"
                                id="temperature"
                                value={card.config.chatConfig.temperature}
                                min="0"
                                max="2"
                                step="0.1"
                                on:input={handleTemperatureChange}
                                disabled={!isEditable}
                            />
                            <span class="value"
                                >{card.config.chatConfig.temperature}</span
                            >
                        </div>
                    </div>

                    <div class="config-item">
                        <label for="maxTokens">Max Tokens</label>
                        <input
                            type="number"
                            id="maxTokens"
                            value={card.config.chatConfig.max_tokens}
                            min="1"
                            step="1"
                            on:input={handleMaxTokensChange}
                            disabled={!isEditable}
                        />
                    </div>
                </div>
            </div>
        {/if}

        <div class="execution-section">
            <div class="status-bar">
                <span class="status-indicator {card.status}">
                    {card.status}
                </span>
                <div class="button-group">
                    <button
                        class="execute-button-small"
                        on:click={handleExecuteDirect}
                        disabled={!canExecute}
                        title="Execute without thoughts"
                    >
                        Direct
                    </button>
                    <button
                        class="execute-button"
                        on:click={handleExecute}
                        disabled={!canExecute}
                        title={buttonState.tooltip}
                    >
                        {buttonState.label}
                    </button>
                </div>
            </div>

            {#if card.result}
                <div class="result-panel">
                    <div class="result-header">
                        <h4>Result</h4>
                        <span class="timestamp">
                            {formatTimestamp(card.result.timestamp)}
                        </span>
                    </div>
                    <div class="result-content" class:error={card.result.error}>
                        {#if card.result.error}
                            <div class="error-message">
                                {card.result.error}
                            </div>
                        {:else}
                            <div class="response-content">
                                {card.result.content}
                            </div>
                        {/if}
                    </div>
                </div>
            {/if}
        </div>
    </div>
</div>

<style>
    .meta-panel {
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background: white;
    }

    .meta-panel-header {
        padding: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        border-bottom: 1px solid #e5e7eb;
    }

    .drag-handle {
        cursor: move;
        color: #9ca3af;
    }

    .card-controls {
        margin-left: auto;
        display: flex;
        gap: 0.5rem;
    }

    .control-button {
        background: none;
        border: none;
        padding: 0.25rem;
        cursor: pointer;
        color: #6b7280;
        border-radius: 0.25rem;
    }

    .control-button:hover {
        background: #f3f4f6;
    }

    .control-button.delete:hover {
        color: #dc2626;
        background: #fee2e2;
    }

    .config-section,
    .model-section,
    .execution-section {
        padding: 1rem;
        border-bottom: 1px solid #e5e7eb;
    }

    .config-section h3,
    .model-section h3 {
        margin: 0 0 1rem 0;
        font-size: 0.875rem;
        font-weight: 600;
        color: #374151;
    }

    textarea,
    input[type="text"] {
        width: 100%;
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        line-height: 1.5;
    }

    .model-select {
        margin-bottom: 1rem;
    }

    .persona-select {
        margin-bottom: 1rem;
    }

    .config-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }

    .config-item {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .range-with-value {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .range-with-value input[type="range"] {
        flex: 1;
    }

    .range-with-value .value {
        min-width: 2.5rem;
        text-align: right;
        font-size: 0.875rem;
        color: #6b7280;
    }

    .tool-input {
        margin-top: 1rem;
    }

    .status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .status-indicator {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .status-indicator.idle {
        background: #f3f4f6;
        color: #6b7280;
    }

    .status-indicator.running {
        background: #dbeafe;
        color: #2563eb;
    }

    .status-indicator.completed {
        background: #d1fae5;
        color: #059669;
    }

    .status-indicator.error {
        background: #fee2e2;
        color: #dc2626;
    }

    .button-group {
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }

    .execute-button-small {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: #6b7280;
        color: white;
        border: none;
        padding: 0.5rem;
        border-radius: 0.375rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.875rem;
    }

    .execute-button-small:not(:disabled):hover {
        background: #4b5563;
        transform: translateY(-1px);
    }

    .execute-button-small:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        background: #9ca3af;
    }

    .execute-button {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: #4caf50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        min-width: 100px;
        justify-content: center;
    }

    .execute-button:not(:disabled):hover {
        background: #388e3c;
        transform: translateY(-1px);
    }

    .execute-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        background: #9ca3af;
    }

    .execute-button.error {
        background: #ef4444;
    }

    .execute-button.error:not(:disabled):hover {
        background: #dc2626;
    }

    .execute-button.success {
        background: #10b981;
    }

    .execute-button.success:not(:disabled):hover {
        background: #059669;
    }

    .spinner {
        width: 16px;
        height: 16px;
        border: 2px solid transparent;
        border-top-color: currentColor;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }

    /* Tooltip styles */
    button[title] {
        position: relative;
    }

    button[title]:hover::after {
        content: attr(title);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        padding: 0.5rem;
        background: #374151;
        color: white;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        white-space: nowrap;
        z-index: 10;
        margin-bottom: 0.5rem;
    }

    button[title]:hover::before {
        content: "";
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        border: 6px solid transparent;
        border-top-color: #374151;
        margin-bottom: -0.25rem;
    }

    .result-panel {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        overflow: hidden;
    }

    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem;
        background: white;
        border-bottom: 1px solid #e5e7eb;
    }

    .result-header h4 {
        margin: 0;
        font-size: 0.875rem;
        font-weight: 600;
    }

    .timestamp {
        font-size: 0.75rem;
        color: #6b7280;
    }

    .result-content {
        padding: 0.75rem;
    }

    .result-content.error {
        background: #fee2e2;
    }

    .error-message {
        color: #dc2626;
        font-size: 0.875rem;
    }

    .response-content {
        white-space: pre-wrap;
        word-break: break-word;
        font-family: ui-monospace, monospace;
        font-size: 0.875rem;
        line-height: 1.5;
        color: #374151;
    }

    .card-title {
        font-weight: 500;
        color: #374151;
        cursor: pointer;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        transition: background-color 0.2s;
    }

    .card-title:hover {
        background-color: #f3f4f6;
    }

    .title-input {
        font-size: 1rem;
        font-weight: 500;
        color: #374151;
        border: 1px solid #e5e7eb;
        border-radius: 0.25rem;
        padding: 0.25rem 0.5rem;
        margin: 0;
        width: auto;
        min-width: 200px;
        background: white;
    }

    .title-input:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    .meta-panel-content {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .thoughts-control {
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .thoughts-control label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #374151;
        font-size: 0.875rem;
    }

    .thoughts-control input[type="number"] {
        width: 4rem;
        padding: 0.25rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.25rem;
    }

    .thoughts-display {
        margin-top: 1rem;
        padding: 1rem;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
    }

    .thoughts-display h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.875rem;
        font-weight: 600;
        color: #374151;
    }

    .thoughts-content {
        font-family: ui-monospace, monospace;
        font-size: 0.875rem;
        line-height: 1.5;
        white-space: pre-wrap;
        word-break: break-word;
        color: #374151;
        max-height: 200px;
        overflow-y: auto;
    }

    .thoughts-content em {
        color: #6b7280;
        font-style: italic;
    }

    .thoughts-content pre {
        margin: 0;
        font-family: inherit;
    }
</style>
