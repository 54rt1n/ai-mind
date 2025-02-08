<svelte:options accessors={true} />

<!-- src/lib/components/ThoughtPanel.svelte -->
<script lang="ts">
    import { Send, Trash2, RotateCw, Check } from "lucide-svelte";
    import { get } from "svelte/store";
    import { clipboardStore } from "$lib/store/clipboardStore";
    import { workspaceStore } from "$lib/store/workspaceStore";
    import { configStore } from "$lib/store/configStore";
    import { thoughtStore } from "$lib/store/thoughtStore";
    import ModelSelect from "../model/ModelSelect.svelte";
    import Thinking from "../ui/Thinking.svelte";
    import { createEventDispatcher } from "svelte";
    import "$lib/../styles/meta-panels.css";
    import type { CompletionMessage } from "$lib/types";

    const dispatch = createEventDispatcher();

    let lastConversationHistory: CompletionMessage[] = [];
    let lastWorkspaceContent: string | undefined = undefined;

    $: thoughtContent = $thoughtStore.thoughtContent;
    $: contentStream = $thoughtStore.contentStream;
    $: isValid = validateThought(contentStream);

    export async function clearAndGenerate(
        conversationHistory: CompletionMessage[],
    ): Promise<boolean> {
        const iters = 1;
        if (
            !$thoughtStore.thoughtPrompt ||
            $thoughtStore.thoughtPrompt.length === 0
        ) {
            thoughtStore.setDefaults();
            thoughtStore.reset();
        }
        //thoughtStore.reset();
        if (conversationHistory.length === 0) {
            console.log("clearAndGenerate failure", conversationHistory);
            throw new Error("No conversation history");
        }
        const workspaceContent = $clipboardStore.includeInMessages
            ? $workspaceStore.content
            : undefined;

        lastConversationHistory = conversationHistory;
        lastWorkspaceContent = workspaceContent;

        return await thoughtStore.generateThought(
            $configStore,
            conversationHistory,
            workspaceContent,
        );
    }

    export async function advanceThought(
        conversationHistory: CompletionMessage[],
    ): Promise<boolean> {
        if (
            !$thoughtStore.thoughtPrompt ||
            $thoughtStore.thoughtPrompt.length === 0
        ) {
            thoughtStore.setDefaults();
            thoughtStore.reset();
        }
        const workspaceContent = $clipboardStore.includeInMessages
            ? $workspaceStore.content
            : undefined;

        lastConversationHistory = conversationHistory;
        lastWorkspaceContent = workspaceContent;
        
        return await thoughtStore.generateThought(
            $configStore,
            conversationHistory,
            workspaceContent,
        );
    }

    export async function validateAndAcceptThought(): Promise<boolean> {
        const isValid = validateThought(contentStream);
        if (isValid > 0) {
            acceptThought();
        }
        return isValid > 0;
    }

    function validateThought(content: string): number {
        // check the current contentStream, and identify the beginning and end of the content within our <{thoughtXml}> tags
        const pattern = `<${$thoughtStore.thoughtXml}(.*?)</${$thoughtStore.thoughtXml}>`;
        const matcher = new RegExp(pattern, "s");
        const match = content.match(matcher);
        if (match) {
            return 1;
        }
        const secondPattern = `<${$thoughtStore.thoughtXml}(.*?)`;
        const secondMatcher = new RegExp(secondPattern, "s");
        const secondMatch = content.match(secondMatcher);
        if (secondMatch) {
            return 2;
        }
        const thirdPattern = `</thought>`;
        const thirdMatcher = new RegExp(thirdPattern, "s");
        const thirdMatch = content.match(thirdMatcher);
        if (thirdMatch) {
            return 3;
        }
        return 0;
    }

    function handleThoughtChange(
        event: Event & { currentTarget: HTMLTextAreaElement },
    ) {
        thoughtStore.updateThoughtContent(event.currentTarget.value);
    }

    function handleStreamChange(
        event: Event & { currentTarget: HTMLTextAreaElement },
    ) {
        const newValue = event.currentTarget.value;
        thoughtStore.updateContentStream(newValue);
        dispatch("streamChange", { value: newValue });
    }

    async function handleGenerate(conversationHistory: CompletionMessage[]) {
        const workspaceContent = $clipboardStore.includeInMessages
            ? $workspaceStore.content
            : undefined;
        await thoughtStore.generateThought(
            $configStore,
            conversationHistory,
            workspaceContent,
        );
    }

    function handleClear() {
        thoughtStore.reset();
    }

    function handleResetThought() {
        thoughtStore.setDefaults();
    }

    function acceptThought() {
        $thoughtStore.thoughtContent = $thoughtStore.contentStream || "";
        $thoughtStore.contentStream = "";
        $thoughtStore.iteration = 1;
    }
</script>

<div class="meta-panel">
    <button
        class="meta-panel-header collapsible"
        on:click={() =>
            configStore.updateField("showThought", !$configStore.showThought)}
    >
        <span>Thought</span>
        <span class="toggle-icon">{$configStore.showThought ? "▼" : "▶"}</span>
    </button>

    {#if $configStore.showThought}
        <div class="meta-panel-content">
            <div class="grid-container">
                <div class="thought-section">
                    <div class="content-left">
                        <textarea
                            class="thought-textarea"
                            value={thoughtContent || ""}
                            placeholder="Content will appear here..."
                            rows="5"
                            on:keyup={handleThoughtChange}
                        />
                    </div>
                    <div class="content-right">
                        <textarea
                            class="thought-textarea"
                            disabled={$thoughtStore.loading}
                            value={contentStream}
                            placeholder="Content will appear here..."
                            rows="5"
                            on:keyup={handleStreamChange}
                        />
                    </div>
                </div>
                <div class="button-group">
                    <button
                        class="control-button generate"
                        disabled={$thoughtStore.loading}
                        on:click={() => handleGenerate(lastConversationHistory)}
                        title="Generate"
                    >
                        <Send size={16} />
                    </button>
                    <input
                        type="number"
                        class="control-input iteration"
                        value={$thoughtStore.iteration}
                        on:input={(event) => {
                            thoughtStore.updateIteration();
                        }}
                        min="1"
                        max="100"
                        step="1"
                    />
                    <input
                        type="text"
                        class="control-input xml"
                        value={$thoughtStore.thoughtXml}
                        on:input={(event) => {
                            thoughtStore.setThoughtXml(
                                event.currentTarget.value,
                            );
                        }}
                    />
                    <button
                        class="control-button clear"
                        disabled={$thoughtStore.loading}
                        on:click={handleClear}
                        title="Clear"
                    >
                        <Trash2 size={16} />
                    </button>
                    {#if isValid > 0}
                        <span class="validity-indicator valid-{isValid}"
                            >Valid</span
                        >
                    {:else}
                        <span class="validity-indicator invalid">Not Valid</span
                        >
                    {/if}
                    <button
                        class="control-button accept"
                        disabled={$thoughtStore.loading || !isValid}
                        on:click={acceptThought}
                        title="Accept"
                    >
                        <Check size={16} />
                    </button>

                    {#if $thoughtStore.loading}
                        <Thinking loading={$thoughtStore.loading} />
                    {/if}

                    <button
                        class="control-button reset"
                        disabled={!$thoughtStore.loading}
                        on:click={() => ($thoughtStore.loading = false)}
                        title="Reset"
                        style="margin-left: auto;"
                    >
                        <RotateCw size={16} />
                    </button>
                </div>

                <ModelSelect
                    bind:value={$configStore.thoughtModel}
                    category="thought"
                />
            </div>
            <label for="thought-user-context">Thought User Context</label>
            <input
                type="text"
                class="thought-textinput"
                value={$thoughtStore.thoughtUserContext || ""}
                placeholder="Thought additional prompt context..."
                on:keyup={(event) =>
                    thoughtStore.setUserContext(event.currentTarget.value)}
            />
            <label for="thought-prompt">Thought Prompt</label>
            <textarea
                class="thought-textarea"
                value={$thoughtStore.thoughtPrompt || ""}
                placeholder="Thought prompt will appear here..."
                rows="3"
                on:keyup={(event) =>
                    thoughtStore.updateThoughtPrompt(event.currentTarget.value)}
            />
            <label for="thought-system-message">Thought System Message</label>
            <textarea
                class="thought-textarea"
                value={$thoughtStore.thoughtSystemMessage || ""}
                placeholder="Thought system message will appear here..."
                rows="3"
                on:keyup={(event) =>
                    thoughtStore.updateSystemMessage(event.currentTarget.value)}
            />
            <label for="thought-default-content">Default Thought Content</label>
            <textarea
                class="thought-textarea"
                value={$thoughtStore.thoughtDefaultContent || ""}
                placeholder="Set your default thought content here..."
                rows="3"
                on:keyup={(event) =>
                    thoughtStore.updateDefaultContent(
                        event.currentTarget.value,
                    )}
            />
        </div>
    {/if}
</div>

<style>
    .thought-section {
        width: 100%;
        display: grid;
        grid-template-columns: 1fr 1fr; /* Creates two equal columns */
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .content-left,
    .content-right {
        width: 100%;
    }

    .thought-textarea {
        width: 100%;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
            monospace;
        font-size: 0.875rem;
        resize: none;
        background-color: #f9fafb;
    }
    .thought-textarea:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        background-color: white;
    }

    @media (max-width: 768px) {
        .grid-container {
            grid-template-columns: 1fr;
        }
    }

    .button-group {
        display: flex;
        gap: 0.25rem;
        height: 32px;
    }

    .control-button {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 32px;
        width: 32px;
        padding: 0;
        border: none;
        border-radius: 0.375rem;
        cursor: pointer;
        transition: all 0.2s;
        color: white;
        flex-shrink: 0;
    }

    .control-button.accept {
        background-color: #10b981;
        margin-right: 0.5rem;
    }

    .control-button.generate {
        background-color: #10b981;
    }

    .control-button.generate:hover:not(:disabled) {
        background-color: #059669;
    }

    .control-button.clear:hover:not(:disabled) {
        background-color: #dc2626;
    }

    .control-button.accept:hover:not(:disabled) {
        background-color: #059669;
    }

    .control-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .control-input {
        box-sizing: border-box;
        height: 32px;
        padding: 0.25rem 0.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        background-color: #f9fafb;
        transition: all 0.2s;
    }

    .control-input:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        background-color: white;
    }

    .control-input.iteration {
        width: 60px;
    }

    .control-input.xml {
        width: 140px;
    }

    .validity-indicator {
        display: flex;
        align-items: center;
        height: 32px;
        padding: 0 0.5rem;
        background-color: #f3f4f6;
        border-radius: 0.375rem;
        font-size: 0.875rem;
    }

    .validity-indicator.invalid {
        background-color: #fce5e5;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }

    .validity-indicator.valid-1 {
        background-color: rgb(115, 192, 115);
        color: #ffffff;
    }

    .validity-indicator.valid-2 {
        background-color: #60a5fa;
        color: #ffffff;
    }

    .validity-indicator.valid-3 {
        background-color: #facc15;
        color: #6b7280;
    }

    .control-button.reset {
        background-color: #ef4444;
    }

    .control-button.reset:hover:not(:disabled) {
        background-color: #dc2626;
    }
</style>
