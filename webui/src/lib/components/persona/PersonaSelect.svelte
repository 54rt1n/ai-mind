<script lang="ts">
    import { RefreshCw } from "lucide-svelte";
    import { rosterStore } from "$lib/store/rosterStore";
    import "$lib/../styles/meta-panels.css";
    import { onMount } from "svelte";

    export let value: string | null | undefined = undefined;
    export let disabled: boolean = false;

    async function handleRefreshPersonas() {
        await rosterStore.fetchPersonas();
    }

    // Initialize the store when the component is mounted
    onMount(() => {
        rosterStore.initialize();
    });

    // Check if the current value exists in the personas list
    $: isUnknownPersona =
        value &&
        $rosterStore.personas.length > 0 &&
        !$rosterStore.personas.some((p) => p.persona_id === value);
</script>

<div class="meta-input">
    <label for="persona">Persona:</label>
    <div class="model-select-container">
        {#if $rosterStore.personas && Array.isArray($rosterStore.personas)}
            <select
                id="persona"
                bind:value
                disabled={$rosterStore.loading || disabled}
            >
                <option value="">Select a persona</option>
                {#if isUnknownPersona}
                    <option {value}>{value} (Unknown)</option>
                {/if}
                {#each $rosterStore.personas as persona}
                    <option value={persona.persona_id}>{persona.name}</option>
                {/each}
            </select>
        {:else}
            <span>No Personas Found</span>
        {/if}
        <button
            class="refresh-button"
            on:click={handleRefreshPersonas}
            disabled={$rosterStore.loading}
            title="Refresh persona list"
        >
            <span class:loading={$rosterStore.loading}>
                <RefreshCw size={16} />
            </span>
        </button>
    </div>
    {#if $rosterStore.error}
        <span class="error-message">{$rosterStore.error}</span>
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
