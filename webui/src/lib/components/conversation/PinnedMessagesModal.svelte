<!-- src/lib/components/PinnedMessagesModal.svelte -->
<script lang="ts">
    import { fade } from 'svelte/transition';
    import { X, MessageCircle, Clock } from 'lucide-svelte';
    import { configStore } from '$lib/store/configStore';

    export let show = false;
    export let onClose: () => void;

    $: messages = $configStore?.pinnedMessages || [];

    function removePinnedMessage(docId: string) {
        const currentMessages = $configStore?.pinnedMessages || [];
        configStore.updateField('pinnedMessages', 
            currentMessages.filter(msg => msg.doc_id !== docId)
        );
    }

    function formatDate(timestamp: number | undefined): string {
        if (!timestamp) return 'Unknown date';
        return new Date(timestamp).toLocaleString();
    }
</script>

{#if show}
    <div class="modal-backdrop" transition:fade on:click|self={onClose}>
        <div class="modal-content">
            <div class="modal-header">
                <h2>Pinned Messages ({messages.length})</h2>
                <button class="close-button" on:click={onClose} aria-label="Close">
                    <X size={20} />
                </button>
            </div>
            <div class="modal-body">
                {#if messages.length === 0}
                    <p class="empty-state">No pinned messages yet</p>
                {:else}
                    <div class="pinned-messages">
                        {#each messages as message}
                            <div class="message-card">
                                <div class="card-header">
                                    <div class="metadata">
                                        <span class="role-badge" class:assistant={message.role === 'assistant'} class:user={message.role === 'user'}>
                                            <MessageCircle size={14} />
                                            {message.role}
                                        </span>
                                        {#if message.conversation_id}
                                            <a href="/conversation/{message.conversation_id}" class="conversation-link">
                                                View Conversation
                                            </a>
                                        {/if}
                                    </div>
                                    <button 
                                        class="remove-button" 
                                        on:click={() => removePinnedMessage(message.doc_id)}
                                        aria-label="Remove pinned message"
                                    >
                                        <X size={16} />
                                    </button>
                                </div>
                                <div class="card-content">
                                    {message.content}
                                </div>
                                <div class="card-footer">
                                    <span class="timestamp">
                                        <Clock size={14} />
                                        {message.date}
                                    </span>
                                    <span class="message-id">ID: {message.doc_id}</span>
                                </div>
                            </div>
                        {/each}
                    </div>
                {/if}
            </div>
        </div>
    </div>
{/if}

<style>
    .modal-backdrop {
        position: fixed;
        inset: 0;
        background-color: rgba(0, 0, 0, 0.5);
        display: grid;
        place-items: center;
        z-index: 50;
    }

    .modal-content {
        background-color: white;
        border-radius: 0.5rem;
        width: 90%;
        max-width: 800px;
        max-height: 80vh;
        display: flex;
        flex-direction: column;
    }

    .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        border-bottom: 1px solid #e5e7eb;
    }

    .modal-header h2 {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 600;
    }

    .close-button {
        background: none;
        border: none;
        cursor: pointer;
        padding: 0.5rem;
        color: #666;
        transition: color 0.2s;
    }

    .modal-body {
        padding: 1rem;
        overflow-y: auto;
        max-height: calc(80vh - 4rem);
    }

    .empty-state {
        text-align: center;
        color: #666;
        font-style: italic;
        padding: 2rem;
    }

    .pinned-messages {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .message-card {
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        overflow: hidden;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .card-header {
        background-color: #f9fafb;
        padding: 0.75rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #e5e7eb;
    }

    .metadata {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .role-badge {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .role-badge.assistant {
        background-color: #e6ffe6;
        color: #16a34a;
    }

    .role-badge.user {
        background-color: #e6f3ff;
        color: #2563eb;
    }

    .conversation-link {
        color: #4CAF50;
        text-decoration: none;
        font-size: 0.875rem;
    }

    .conversation-link:hover {
        text-decoration: underline;
    }

    .card-content {
        padding: 1rem;
        white-space: pre-wrap;
        word-break: break-word;
        line-height: 1.5;
    }

    .card-footer {
        padding: 0.75rem;
        background-color: #f9fafb;
        border-top: 1px solid #e5e7eb;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.875rem;
        color: #666;
    }

    .timestamp {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }

    .message-id {
        font-family: monospace;
    }

    .remove-button {
        background: none;
        border: none;
        cursor: pointer;
        color: #ef4444;
        padding: 0.25rem;
        transition: color 0.2s;
        display: flex;
        align-items: center;
    }

    .remove-button:hover {
        color: #dc2626;
    }
</style>