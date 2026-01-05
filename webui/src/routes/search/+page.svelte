<!-- src/routes/search/+page.svelte -->
<script lang="ts">
    import { api } from '$lib/api';
    import { goto } from '$app/navigation';
    import { configStore } from '$lib/store/configStore';
    import { type ChatMessage } from '$lib/types';
    import ChatMessageCard from '$lib/components/conversation/ChatMessageCard.svelte';

    let searchQuery = '';
    let searchResults: any[] = [];
    let loading = false;
    let error: string | null = null;
    let topN = 5;
    let documentType = 'all';

    const documentTypes = [
        { value: 'all', label: 'All Types' },
        { value: 'conversation', label: 'Conversation' },
        { value: 'analysis', label: 'Analysis' },
        { value: 'journal', label: 'Journal' },
        { value: 'chore', label: 'Chore' },
        { value: 'report', label: 'Report' },
    ];

    async function handleSearch() {
        if (!searchQuery.trim()) return;

        const persona_id = $configStore.persona_id;
        if (!persona_id) {
            error = "No persona selected. Please select a persona from the roster.";
            return;
        }

        loading = true;
        error = null;

        try {
            const response = await api.searchConversations(persona_id, searchQuery, topN, documentType);
            searchResults = response.results;
        } catch (e) {
            error = "Failed to perform search";
            searchResults = [];
        } finally {
            loading = false;
        }
    }

    function handleJumpTo(conversationId: string, docId: string) {
        goto(`/conversation/${conversationId}#${docId}`);
    }

    function handleKeydown(event: KeyboardEvent) {
        if (event.key === 'Enter') {
            handleSearch();
        }
    }

    $: pinnedMessages = $configStore?.pinnedMessages || [];

function isMessagePinned(docId: string): boolean {
    return pinnedMessages.some(msg => msg.doc_id === docId);
}

function handleTogglePin(message: ChatMessage) {
    const currentPinned = $configStore?.pinnedMessages || [];
    if (isMessagePinned(message.doc_id)) {
        configStore.updateField('pinnedMessages', 
            currentPinned.filter(msg => msg.doc_id !== message.doc_id)
        );
    } else {
        configStore.updateField('pinnedMessages', 
            [...currentPinned, message]
        );
    }
}
</script>

<svelte:head>
    <title>Search Conversations | MindAI</title>
</svelte:head>

<main>
    <h1>Search Conversations</h1>

    {#if $configStore.persona_id}
        <div class="persona-info">
            Searching in: <strong>{$configStore.persona}</strong> ({$configStore.persona_id})
        </div>
    {:else}
        <div class="persona-warning">
            No persona selected. Please select a persona from the roster.
        </div>
    {/if}

    <div class="filter-ribbon">
        <input
            type="number"
            bind:value={topN}
            min="1"
            max="100"
            placeholder="Top N"
            aria-label="Top N results"
        />
        <select bind:value={documentType} aria-label="Document type filter">
            {#each documentTypes as type}
                <option value={type.value}>{type.label}</option>
            {/each}
        </select>
    </div>

    <div class="search-container">
        <input
            type="text"
            bind:value={searchQuery}
            on:keydown={handleKeydown}
            placeholder="Enter your search query"
            aria-label="Search query"
        />
        <button on:click={handleSearch} disabled={loading}>Search</button>
    </div>

    {#if loading}
        <p>Searching...</p>
    {:else if error}
        <p class="error">{error}</p>
    {:else if searchResults.length > 0}
        <h2>Search Results</h2>
        <div class="results-container">
            {#each searchResults as result}
                <ChatMessageCard
                    message={result}
                    showActions={true}
                    onJumpTo={handleJumpTo}
                    onTogglePin={handleTogglePin}
                />
            {/each}
        </div>
    {:else if searchQuery}
        <p>No results found for "{searchQuery}"</p>
    {/if}
</main>

<style>
    main {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }

    .filter-ribbon {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 20px;
    }

    .filter-ribbon input {
        width: 80px;
        padding: 5px;
        font-size: 14px;
    }

    .search-container {
        display: flex;
        margin-bottom: 20px;
    }

    .search-container input {
        flex-grow: 1;
        padding: 10px;
        font-size: 16px;
    }

    button {
        padding: 10px 20px;
        font-size: 16px;
        background-color: #4CAF50;
        color: white;
        border: none;
        cursor: pointer;
    }

    button:hover {
        background-color: #45a049;
    }

    button:disabled {
        background-color: #cccccc;
        cursor: not-allowed;
    }

    .results-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-top: 1rem;
    }

    .error {
        color: red;
        font-weight: bold;
    }

    .filter-ribbon {
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        margin-bottom: 20px;
    }

    .filter-ribbon input,
    .filter-ribbon select {
        padding: 5px;
        font-size: 14px;
    }

    .filter-ribbon select {
        min-width: 120px;
    }

    .persona-info {
        background-color: #e8f4f8;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin-bottom: 20px;
        border-radius: 4px;
    }

    .persona-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin-bottom: 20px;
        border-radius: 4px;
        color: #856404;
    }
</style>

