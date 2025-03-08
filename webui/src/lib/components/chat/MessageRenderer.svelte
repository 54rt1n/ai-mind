<script lang="ts">
    import { onMount } from "svelte";
    import { slide } from "svelte/transition";

    export let content: string = "";
    // We're removing the isActive prop since we're removing timer functionality

    let displayContent: string = "";
    let thinkContent: string | null = null;
    let isThinkExpanded: boolean = false;

    // Process content on initial load and when it changes
    $: {
        processContent(content);
    }

    function processContent(text: string) {
        // First check for complete think tags
        const completeThinkRegex = /<think>([\s\S]*?)<\/think>/;
        const completeMatch = completeThinkRegex.exec(text);

        if (completeMatch && completeMatch[1]) {
            // Complete think tag found - extract content
            thinkContent = completeMatch[1].trim();
            // Remove think tags for display
            displayContent = text
                .replace(/<think>[\s\S]*?<\/think>/g, "")
                .trim();
            return;
        }

        // If no complete tag, check for partial think tag (open without close)
        const partialThinkRegex = /<think>([\s\S]*?)$/;
        const partialMatch = partialThinkRegex.exec(text);

        if (partialMatch && partialMatch[1]) {
            // Partial think tag found - extract content
            thinkContent = partialMatch[1].trim();
            // Remove partial think tag for display
            displayContent = text.replace(/<think>[\s\S]*?$/, "").trim();
            return;
        }

        // No think tags found
        displayContent = text;
        thinkContent = null;
    }

    function toggleThinkExpansion() {
        isThinkExpanded = !isThinkExpanded;
    }

    // When the component is mounted, process the content
    onMount(() => {
        processContent(content);
    });
</script>

<!-- Think block rendering -->
{#if thinkContent}
    <div class="think-block">
        <button class="think-header" on:click={toggleThinkExpansion}>
            <div class="think-indicator">
                <span class="think-icon">💡</span>
                <span>Thought</span>
            </div>
            <span class="chevron-icon">
                {#if isThinkExpanded}
                    <span>▲</span>
                {:else}
                    <span>▼</span>
                {/if}
            </span>
        </button>

        {#if isThinkExpanded}
            <div class="think-content" transition:slide>
                {thinkContent}
            </div>
        {/if}
    </div>
{/if}

<!-- Main content rendering -->
<div class="message-content">
    {displayContent}
</div>

<style>
    .think-block {
        background-color: #1e262f;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        color: #c9d1d9;
        overflow: hidden;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
            Helvetica, Arial, sans-serif;
    }

    .think-header {
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1rem;
        background: none;
        border: none;
        cursor: pointer;
        color: inherit;
        text-align: left;
    }

    .think-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .think-icon {
        opacity: 0.8;
    }

    .chevron-icon {
        font-size: 0.75rem;
        opacity: 0.7;
    }

    .think-content {
        padding: 0 1rem 1rem;
        white-space: pre-wrap;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .message-content {
        white-space: pre-wrap;
        word-break: break-word;
    }
</style>
