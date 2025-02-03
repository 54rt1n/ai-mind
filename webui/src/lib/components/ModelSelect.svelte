<script lang="ts">
    import { configStore } from "$lib/store/configStore";
    import { modelStore } from "$lib/store/modelStore";
    import { RefreshCw } from "lucide-svelte";
    import "$lib/../styles/meta-panels.css";

    export let value: string | null | undefined;
    export let category: string | null | undefined;

    async function handleRefreshModels() {
        await modelStore.refresh();
    }

    $: filteredModels = category ? $modelStore.models.filter(model => model.category?.includes(category)) : $modelStore.models;

</script>

<div class="meta-input">
    <label for="model">Model:</label>
    <div class="model-select-container">
            {#if filteredModels && Array.isArray(filteredModels)}
        <select
            id="model"
            bind:value={value}
            disabled={$modelStore.loading}
        >
                {#each filteredModels as model}
                    <option value={model.name}>{model.name}</option>
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