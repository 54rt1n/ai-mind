<script lang="ts">
    import { modelStore } from "$lib/store/modelStore";
    import { RefreshCw } from "lucide-svelte";
    import "$lib/../styles/meta-panels.css";

    export let value: string | null | undefined = undefined;
    export let category: string | null | undefined = undefined;
    export let disabled: boolean = false;

    async function handleRefreshModels() {
        await modelStore.refresh();
    }

    $: filteredModels = category
        ? $modelStore.models.filter((model) =>
              model.category?.includes(category),
          )
        : $modelStore.models;
</script>

<div class="meta-input">
    <label for="model">Model:</label>
    <div class="model-select-container">
        {#if filteredModels && Array.isArray(filteredModels)}
            <select
                id="model"
                bind:value
                disabled={$modelStore.loading || disabled}
            >
                <option value="">Select a model</option>
                {#each filteredModels as model}
                    <option value={model.name}
                        >{model.name} ({model.provider})</option
                    >
                {/each}
            </select>
        {:else}
            <span>No Models Found</span>
        {/if}
        <button
            class="refresh-button"
            on:click={handleRefreshModels}
            disabled={$modelStore.loading}
            title="Refresh model list"
        >
            <span class:loading={$modelStore.loading}>
                <RefreshCw size={16} />
            </span>
        </button>
    </div>
    {#if $modelStore.error}
        <span class="error-message">{$modelStore.error}</span>
    {/if}
</div>

<style>
    .model-select-container {
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
