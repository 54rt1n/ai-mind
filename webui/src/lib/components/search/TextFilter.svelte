<!-- src/lib/components/TextFilter.svelte -->
<script lang="ts">
    import { createEventDispatcher } from 'svelte';
    import type { ChatMessage } from '$lib/types';
    import { Search, X } from 'lucide-svelte';
    
    const dispatch = createEventDispatcher<{
        filterChange: string
    }>();
    
    export let messages: ChatMessage[] = [];
    export let activeFilter = '';

    let searchTimeout: ReturnType<typeof setTimeout>;

    function handleInput(event: Event) {
        const input = event.target as HTMLInputElement;
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            dispatch('filterChange', input.value);
        }, 300);
    }

    function clearSearch() {
        activeFilter = '';
        dispatch('filterChange', '');
    }

    $: matchCount = activeFilter ? 
        messages.filter(message => 
            message.content.toLowerCase().includes(activeFilter.toLowerCase())
        ).length : messages.length;
</script>

<div class="filter-container">
    <div class="search-box">
        <Search size={16} class="search-icon" />
        <input
            type="text"
            bind:value={activeFilter}
            on:input={handleInput}
            placeholder="Filter messages..."
            aria-label="Filter messages"
        />
        {#if activeFilter}
            <button class="clear-button" on:click={clearSearch} aria-label="Clear search">
                <X size={16} />
            </button>
        {/if}
    </div>
    <div class="results-count">
        {#if activeFilter}
            <span>Found {matchCount} matching messages</span>
        {:else}
            <span>All Messages: {messages.length}</span>
        {/if}
    </div>
</div>

<style>
    .filter-container {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding: 0.75rem;
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    .search-box {
        flex: 1;
        position: relative;
        display: flex;
        align-items: center;
        min-width: 200px;
        height: 38px; /* Fixed height to match other filters */
    }

    input {
        width: 100%;
        height: 100%;
        padding: 0 2.5rem;
        margin-left: 8px;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        color: #374151;
        background-color: #f9fafb;
        transition: all 0.2s;
    }

    input:focus {
        outline: none;
        border-color: #3b82f6;
        background-color: white;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    .clear-button {
        position: absolute;
        right: 0.75rem;
        top: 50%;
        transform: translateY(-50%);
        background: none;
        border: none;
        padding: 0.25rem;
        color: #6b7280;
        cursor: pointer;
        border-radius: 0.25rem;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
    }

    .clear-button:hover {
        background-color: #f3f4f6;
        color: #374151;
    }

    .results-count {
        display: flex;
        align-items: center;
        height: 38px; /* Match height of search box */
        padding: 0 0.75rem;
        background-color: #f3f4f6;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        color: #374151;
        font-weight: 500;
    }
</style>