<!-- src/lib/components/ChatMessageCard.svelte -->
<script lang="ts">
    import {
        MessageCircle,
        Clock,
        User,
        Bot,
        Hash,
        GitBranch,
        Type,
        Edit2,
        Plus,
        Minus,
        ArrowUpRight,
        Trash2,
    } from "lucide-svelte";
    import { configStore } from "$lib/store/configStore";
    import { type ChatMessage } from "../types";

    export let message: ChatMessage;
    // Feature flags for different actions
    export let showActions = true;

    // Callback functions for different actions
    export let onEdit: ((docId: string, content: string) => void) | undefined =
        undefined;
    export let onTogglePin: ((message: ChatMessage) => void) | undefined =
        undefined;
    export let onJumpTo:
        | ((conversationId: string, docId: string) => void)
        | undefined = undefined;
    export let onDelete: ((docId: string) => void) | undefined = undefined;

    let isHovered = false;

    $: pinnedMessages = $configStore?.pinnedMessages || [];
    $: isPinned = pinnedMessages.some((msg) => msg.doc_id === message.doc_id);

    function handleMouseEnter() {
        isHovered = true;
    }

    function handleMouseLeave() {
        isHovered = false;
    }
</script>

<div
    class="message-card"
    role="article"
    aria-label="{message.role} message"
    on:mouseenter={handleMouseEnter}
    on:mouseleave={handleMouseLeave}
    on:focus={handleMouseEnter}
    on:blur={handleMouseLeave}
>
    <div class="card-header">
        <div class="metadata">
            <span
                class="role-badge"
                class:assistant={message.role === "assistant"}
                class:user={message.role === "user"}
            >
                {#if message.role === "assistant"}
                    <Bot size={14} />
                {:else}
                    <User size={14} />
                {/if}
                {message.role}
            </span>
            <span class="entity-badge">
                <MessageCircle size={14} />
                {message.persona_id || message.user_id}
            </span>
            <span class="doc-type-badge">
                <Type size={14} />
                {message.document_type}
            </span>
            <span class="branch-badge">
                <GitBranch size={14} />
                Branch {message.branch} #{message.sequence_no}
            </span>
            <span class="timestamp">
                <Clock size={14} />
                {message.date}
            </span>
            <span class="message-id">
                <Hash size={14} />
                {message.conversation_id}/{message.doc_id}
            </span>
        </div>
        <div class="message-actions">
            {#if showActions && isHovered}
                {#if onDelete}
                    <button
                        class="action-button delete-button"
                        aria-label="Delete document"
                        title="Delete document"
                        on:click={() => onDelete(message.doc_id)}
                    >
                        <Trash2 size={16} />
                    </button>
                {/if}
                {#if onJumpTo}
                    <button
                        class="action-button jump-button"
                        aria-label="Jump to conversation"
                        title="Jump to conversation"
                        on:click={() =>
                            onJumpTo(message.conversation_id, message.doc_id)}
                    >
                        <ArrowUpRight size={16} />
                    </button>
                {/if}
                {#if onEdit}
                    <button
                        class="action-button edit-button"
                        aria-label="Edit message"
                        title="Edit message"
                        on:click={() => onEdit(message.doc_id, message.content)}
                    >
                        <Edit2 size={16} />
                    </button>
                {/if}

                {#if onTogglePin}
                    <button
                        class="action-button {isPinned
                            ? 'unpin-button'
                            : 'pin-button'}"
                        aria-label={isPinned ? "Unpin message" : "Pin message"}
                        title={isPinned ? "Unpin message" : "Pin message"}
                        on:click={() => onTogglePin(message)}
                    >
                        {#if isPinned}
                            <Minus size={16} />
                        {:else}
                            <Plus size={16} />
                        {/if}
                    </button>
                {/if}
            {/if}
        </div>
    </div>
    <div class="card-content">
        {message.content}
    </div>
</div>

<style>
    .message-card {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        min-height: min-content;
    }

    .card-header {
        background-color: #f9fafb;
        padding: 0.75rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #e5e7eb;
        flex-shrink: 0;
    }

    .metadata {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        flex-wrap: wrap;
        flex: 1;
    }

    .role-badge,
    .entity-badge,
    .doc-type-badge,
    .branch-badge,
    .timestamp,
    .message-id {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        flex-shrink: 0;
    }

    .role-badge.assistant {
        background-color: #e6ffe6;
        color: #16a34a;
    }

    .role-badge.user {
        background-color: #e6f3ff;
        color: #2563eb;
    }

    .entity-badge {
        background-color: #fef3c7;
        color: #b45309;
    }

    .doc-type-badge {
        background-color: #f3e8ff;
        color: #7e22ce;
    }

    .branch-badge {
        background-color: #e0e7ff;
        color: #4338ca;
    }

    .timestamp {
        color: #666;
        font-size: 0.875rem;
        background-color: #f3f4f6;
    }

    .message-id {
        font-family: monospace;
        font-size: 0.875rem;
        color: #666;
        background-color: #f3f4f6;
    }

    .message-actions {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        flex-shrink: 0;
    }

    .action-button {
        background: none;
        border: none;
        cursor: pointer;
        padding: 0.25rem;
        border-radius: 0.25rem;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
    }

    .edit-button {
        color: #666;
    }

    .edit-button:hover {
        background-color: rgba(0, 0, 0, 0.1);
        color: #000;
    }

    .pin-button {
        color: #22c55e;
    }

    .pin-button:hover {
        background-color: rgba(34, 197, 94, 0.1);
        color: #16a34a;
    }

    .unpin-button {
        color: #ef4444;
    }

    .unpin-button:hover {
        background-color: rgba(239, 68, 68, 0.1);
        color: #dc2626;
    }

    .card-content {
        padding: 1rem;
        white-space: pre-wrap;
        word-break: break-word;
        line-height: 1.5;
        flex: 1;
        min-height: 0;
        overflow-y: auto;
    }

    .jump-button {
        color: #2196f3;
    }

    .jump-button:hover {
        background-color: rgba(33, 150, 243, 0.1);
        color: #1976d2;
    }

    .delete-button {
        color: #ef4444;
    }

    .delete-button:hover {
        background-color: rgba(239, 68, 68, 0.1);
        color: #dc2626;
    }
</style>
