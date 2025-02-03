<svelte:options accessors={true} />

<!-- src/lib/components/ThoughtPanel.svelte -->
<script lang="ts">
    import { Send, Trash2, RotateCw, Check } from "lucide-svelte";
    import { get } from "svelte/store";
    import { writable } from 'svelte/store';
    import { clipboardStore } from "$lib/store/clipboardStore";
    import { workspaceStore } from "$lib/store/workspaceStore";
    import { configStore } from "$lib/store/configStore";
    import { chatStore } from "$lib/store/chatStore";
    import ModelSelect from "./ModelSelect.svelte";
    import Thinking from "./Thinking.svelte";
    import "$lib/../styles/meta-panels.css";

    $: thoughtContent = $configStore.thoughtContent;
    $: contentStream = $chatStore.contentStream;
    const thoughtXml = writable('andi_thoughts');
    // We are going to track an iteration depth counter
    const iteration = writable(1);
    $: isValid = validateThought(contentStream);

    export async function clearAndGenerate(): Promise<boolean> {
        // console.log("clearAndGenerate");
        handleClear();
        const workspaceContent = $clipboardStore.includeInMessages ? $workspaceStore.content: undefined;
        await chatStore.generateThought($configStore, workspaceContent);
        if (!validateThought(contentStream)) {
            return false;
        }
        acceptThought();
        return true;
    }

    export async function advanceThought(): Promise<boolean> {
        // console.log("advanceThought");
        const workspaceContent = $clipboardStore.includeInMessages ? $workspaceStore.content: undefined;
        await chatStore.generateThought($configStore, workspaceContent);
        if (!validateThought(contentStream)) {
            return false;
        }
        acceptThought();
        return true;
    }

    function validateThought(content: string): number {
        // check the current contentStream, and identify the beginning and end of the content within our <{thoughtXml}> tags
        const pattern = `<${thoughtXml}(.*?)</${thoughtXml}>`;
        const matcher = new RegExp(pattern, "s");
        const match = content.match(matcher);
        if (match) {
            // console.log("match", match);
            return 1;
        }
        const secondPattern = `<${thoughtXml}(.*?)`;
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

    function fixThought() {
        // check the current contentStream, and identify the beginning and end of the content within our <{thoughtXml}> tags
        const pattern = `<${thoughtXml}(.*?)</${thoughtXml}>`;
        const matcher = new RegExp(pattern, "s");
        const match = contentStream.match(matcher);
        // console.log("match", match);
        if (match) {
            return true;
        }
        const secondMatcher = /<{thoughtXml}(.*?)/s;
        if (match) {
            // We have a partial match. We should append a </thought>\n</{thoughtXml}> to the end of the contentStream
            const partialMatch = match[1];
            const newContent =
                partialMatch + "\n</thought>\n</" + thoughtXml + ">";
            $chatStore.contentStream = newContent;
            return true;
        }
        return false;
    }

    function handleThoughtChange(
        event: Event & { currentTarget: HTMLTextAreaElement },
    ) {
        // Update the thought content in the chat store
        $configStore.thoughtContent = event.currentTarget.value;
    }

    function handleStreamChange(
        event: Event & { currentTarget: HTMLTextAreaElement },
    ) {
        // Update the thought content in the chat store
        $chatStore.contentStream = event.currentTarget.value;
    }

    async function handleGenerate() {
        const workspaceContent = $clipboardStore.includeInMessages ? $workspaceStore.content: undefined;
        await chatStore.generateThought($configStore, workspaceContent);
        validateThought(contentStream);
    }

    function handleClear() {
        $configStore.thoughtContent = $configStore.thoughtDefaultContent;
        $chatStore.contentStream = "";
        $iteration = 1;
        $configStore.thoughtPrompt = getThoughtPrompt(
            $thoughtXml, $iteration, $configStore.thoughtUserContext
        );
    }

    function getDefaultThought(entityName: string, iter: string): string {
        let thoughts = [
            "I will follow a chain of thought, reasoning through my ideas.",
            "These are the most important things I should consider:",
        ];
        return `<${entityName} iter="${iter}">\n\t${thoughts.map((thought) => `<thought>${thought}</thought>`).join("\n\t")}\n</${entityName}>`;
    }

    function getDefaultSystemMessage(): string {
        return `<format_override>\n\t<override>You are in your thought processes. You are only to output a thought turn.</override>\n\t<output_mode>xml</output_mode>\n\t	<description>All Thought Output Is In XML Format</description>\n</format_override>`;
    }

    function getThoughtPrompt(entityName: string, iter: number, userContext: string = ""): string {
        const defaultThoughts = getDefaultThought(entityName, "n");
        const thoughtPrompt = `Thought Turn Format:\n\n${defaultThoughts}\n\n<directive>Andi's next turn is a thought turn. Please update your thought block appropriately, enhancing and improving your current thoughts and reasoning${userContext}. Please output the next thoughts document. This should be an xml document.</directive>\n\nBegin Output "<${entityName} iter="${iter}">"`;
        return thoughtPrompt;
    }

    function handleResetThought() {
        // console.log("handleResetThought", $thoughtXml);
        $configStore.thoughtDefaultContent = getDefaultThought($thoughtXml, "0");
        $configStore.thoughtSystemMessage = getDefaultSystemMessage();
        $configStore.thoughtPrompt = getThoughtPrompt(
            $thoughtXml, $iteration, $configStore.thoughtUserContext
        );
    }

    function acceptThought() {
        $iteration = $iteration + 1;
        $configStore.thoughtPrompt = getThoughtPrompt($thoughtXml, $iteration, $configStore.thoughtUserContext);
        $configStore.thoughtContent = $chatStore.contentStream || "";
        $chatStore.contentStream = "";
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
                            disabled={$chatStore.loading}
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
                        disabled={$chatStore.loading}
                        on:click={handleGenerate}
                        title="Generate"
                    >
                        <Send size={16} />
                    </button>
                    <input
                        type="number"
                        class="control-input iteration"
                        value={iteration}
                        on:input={(event) => {
                            $iteration = parseInt(event.currentTarget.value);
                        }}
                        min="1"
                        max="100"
                        step="1"
                    />
                    <input
                        type="text"
                        class="control-input xml"
                        value={$thoughtXml}
                        on:input={(event) => {
                            $thoughtXml = event.currentTarget.value;
                        }}
                    />
                    <button
                        class="control-button clear"
                        disabled={$chatStore.loading}
                        on:click={handleClear}
                        title="Clear"
                    >
                        <Trash2 size={16} />
                    </button>
                    {#if isValid > 0}
                        <span class="validity-indicator valid-{isValid}">Valid</span>
                    {:else}
                        <span class="validity-indicator invalid">Not Valid</span>
                    {/if}
                    <button
                        class="control-button accept"
                        disabled={$chatStore.loading || !isValid}
                        on:click={acceptThought}
                        title="Accept"
                    >
                        <Check size={16} />
                    </button>

                    <Thinking />
                </div>

                <ModelSelect bind:value={$configStore.thoughtModel} category="thought" />
            </div>
            <label for="thought-user-context">Thought User Context</label>
            <input type="text" 
              class="thought-textinput"
              value={$configStore.thoughtUserContext || ""}
              placeholder="Thought additional prompt context..."
              on:keyup={(event) => $configStore.thoughtUserContext = event.currentTarget.value}
              />
            <label for="thought-prompt">Thought Prompt</label>
            <textarea
                class="thought-textarea"
                value={$configStore.thoughtPrompt || ""}
                placeholder="Thought prompt will appear here..."
                rows="3"
                on:keyup={(event) =>
                    ($configStore.thoughtPrompt = event.currentTarget.value)}
            />
            <label for="thought-system-message">Thought System Message</label>
            <textarea
                class="thought-textarea"
                value={$configStore.thoughtSystemMessage || ""}
                placeholder="Thought system message will appear here..."
                rows="3"
                on:keyup={(event) =>
                    ($configStore.thoughtSystemMessage =
                        event.currentTarget.value)}
            />
            <label for="thought-default-content">Default Thought Content</label>
            <textarea
                class="thought-textarea"
                value={$configStore.thoughtDefaultContent || ""}
                placeholder="Set your default thought content here..."
                rows="3"
                on:keyup={(event) =>
                    ($configStore.thoughtDefaultContent =
                        event.currentTarget.value)}
            />
            <button
                class="control-button reset"
                on:click={handleResetThought}
                title="Reset"
            >
                <RotateCw size={16} />
            </button>
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
</style>
