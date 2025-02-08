<script lang="ts">
    import { RefreshCw } from "lucide-svelte";
    import { toolStore } from "$lib/store/toolStore";
    import "$lib/../styles/meta-panels.css";
    import { onMount } from "svelte";
    import type { Tool } from "$lib";

    export let value: Tool | null | undefined = null;
    export let disabled: boolean = false;

    async function handleRefreshTools() {
        await toolStore.refresh();
    }

    // Initialize the store when the component is mounted
    onMount(() => {
        toolStore.initialize();
    });

    function formatToolName(tool: Tool) {
        return tool.type + "/" + tool.function.name;
    }

    function handleToolChange(event: Event) {
        const target = event.target as HTMLSelectElement;
        const selectedTool = $toolStore.tools.find(
            (t) => formatToolName(t) === target.value,
        );
        console.log("selectedTool", selectedTool);
        if (selectedTool) {
            value = selectedTool;
        }
    }

    // Check if the current value exists in the tools list
    $: isUnknownTool =
        value &&
        value !== null &&
        value !== undefined &&
        $toolStore.tools.length > 0 &&
        !$toolStore.tools.some(
            (t) =>
                t.type === value!.type &&
                t.function.name === value!.function.name,
        );

    $: sortedTools = $toolStore.tools.sort((a, b) => {
        return (
            a.type.localeCompare(b.type) ||
            a.function.name.localeCompare(b.function.name)
        );
    });
</script>

<div class="meta-input">
    <div class="tool-select-container">
        {#if $toolStore.tools && Array.isArray($toolStore.tools)}
            <select
                id="tool"
                bind:value
                disabled={$toolStore.loading || disabled}
            >
                <option value="">Select a tool</option>
                {#if isUnknownTool}
                    <option {value}>{value} (Unknown)</option>
                {/if}
                {#each sortedTools as tool}
                    <option value={formatToolName(tool)}>
                        {formatToolName(tool)}
                    </option>
                {/each}
            </select>
        {:else}
            <span>No Tools Found</span>
        {/if}
        <button
            class="refresh-button"
            on:click={handleRefreshTools}
            disabled={$toolStore.loading}
            title="Refresh tool list"
        >
            <span class:loading={$toolStore.loading}>
                <RefreshCw size={16} />
            </span>
        </button>
    </div>
    {#if $toolStore.error}
        <span class="error-message">{$toolStore.error}</span>
    {/if}
</div>

<style>
    .tool-select-container {
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }

    select {
        flex: 1;
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        background-color: white;
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

    .error-message {
        color: #dc2626;
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }
</style>
