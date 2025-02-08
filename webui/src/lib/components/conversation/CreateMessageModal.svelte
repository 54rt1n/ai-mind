<!-- src/lib/components/CreateMessageModal.svelte -->
<script lang="ts">
    import { fade } from "svelte/transition";
    import { X } from "lucide-svelte";
    import { createEventDispatcher } from "svelte";
    import type { ConversationMessage, PipelineType } from "../../types";

    const dispatch = createEventDispatcher<{
        close: void;
        save: { message: Partial<ConversationMessage> };
    }>();

    export let isOpen = false;
    export let conversationId: string;

    const documentTypes: Array<{ value: PipelineType; label: string }> = [
        "adventure",
        "analysis",
        "chore",
        "daydream",
        "inventory",
        "journal",
        "librarian",
        "poetry",
        "ponder",
        "summary",
    ].map((type) => ({
        value: type as PipelineType,
        label: type.charAt(0).toUpperCase() + type.slice(1),
    }));

    let message: Partial<ConversationMessage> = {
        conversation_id: conversationId,
        document_type: "conversation",
        role: "user",
        content: "",
        timestamp: Math.floor(Date.now() / 1000),
        branch: 0,
        sequence_no: 0,
        speaker_id: "",
        listener_id: "",
    };

    $: if (isOpen) {
        message.conversation_id = conversationId;
        message.timestamp = Math.floor(Date.now() / 1000);
    }

    function handleKeydown(event: KeyboardEvent) {
        if (!isOpen) return;

        if (event.key === "Escape") {
            dispatch("close");
        }

        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
            handleSave();
        }
    }

    function handleSave() {
        if (!message.content?.trim()) return;
        dispatch("save", { message });
    }
</script>

<svelte:window on:keydown={handleKeydown} />

{#if isOpen}
    <div class="modal-backdrop" transition:fade role="presentation">
        <div class="modal-content" role="dialog" aria-modal="true">
            <header class="modal-header">
                <h2>Create New Message</h2>
                <div class="header-actions">
                    <div class="keyboard-hints">
                        <span>Ctrl + Enter to save</span>
                        <span>•</span>
                        <span>Esc to cancel</span>
                    </div>
                    <button
                        class="close-button"
                        on:click={() => dispatch("close")}
                    >
                        <X size={20} />
                    </button>
                </div>
            </header>

            <div class="modal-body">
                <div class="form-group">
                    <label for="doc_id">Message ID</label>
                    <input
                        type="text"
                        id="doc_id"
                        bind:value={message.doc_id}
                    />
                </div>

                <div class="form-group">
                    <label for="role">Role</label>
                    <select id="role" bind:value={message.role}>
                        <option value="user">User</option>
                        <option value="assistant">Assistant</option>
                        <option value="system">System</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="document_type">Document Type</label>
                    <select
                        id="document_type"
                        bind:value={message.document_type}
                    >
                        {#each documentTypes as { value, label }}
                            <option {value}>{label}</option>
                        {/each}
                    </select>
                </div>

                <div class="form-group">
                    <label for="user_id">User ID</label>
                    <input
                        type="text"
                        id="user_id"
                        bind:value={message.user_id}
                    />
                </div>

                <div class="form-group">
                    <label for="persona_id">Persona ID</label>
                    <input
                        type="text"
                        id="persona_id"
                        bind:value={message.persona_id}
                    />
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label for="branch">Branch</label>
                        <input
                            type="number"
                            id="branch"
                            bind:value={message.branch}
                            min="0"
                        />
                    </div>

                    <div class="form-group">
                        <label for="sequence_no">Sequence Number</label>
                        <input
                            type="number"
                            id="sequence_no"
                            bind:value={message.sequence_no}
                            min="0"
                        />
                    </div>
                </div>

                <div class="form-group">
                    <label for="content">Content</label>
                    <textarea
                        id="content"
                        bind:value={message.content}
                        class="content-textarea"
                        placeholder="Enter message content..."
                        rows="6"
                    />
                </div>
            </div>

            <footer class="modal-footer">
                <button
                    class="cancel-button"
                    on:click={() => dispatch("close")}
                >
                    Cancel
                </button>
                <button class="save-button" on:click={handleSave}>
                    Create Message
                </button>
            </footer>
        </div>
    </div>
{/if}

<style>
    .modal-backdrop {
        position: fixed;
        inset: 0;
        background-color: rgb(0 0 0 / 0.6);
        backdrop-filter: blur(4px);
        display: grid;
        place-items: center;
        z-index: 50;
        padding: 1.5rem;
    }

    .modal-content {
        background-color: white;
        border-radius: 1rem;
        width: 95vw;
        max-width: 40rem;
        max-height: 90vh;
        display: flex;
        flex-direction: column;
    }

    .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #e5e7eb;
    }

    .header-actions {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }

    .keyboard-hints {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.875rem;
        color: #6b7280;
    }

    .close-button {
        padding: 0.5rem;
        border: none;
        background: none;
        cursor: pointer;
    }

    .modal-body {
        flex: 1;
        padding: 1.5rem;
        overflow-y: auto;
    }

    .form-group {
        margin-bottom: 1rem;
    }

    .form-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }

    label {
        display: block;
        font-size: 0.875rem;
        font-weight: 500;
        color: #374151;
        margin-bottom: 0.5rem;
    }

    input,
    select,
    textarea {
        width: 100%;
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        font-size: 0.875rem;
    }

    input:focus,
    select:focus,
    textarea:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgb(147 197 253 / 0.25);
    }

    .content-textarea {
        resize: vertical;
        min-height: 100px;
    }

    .modal-footer {
        padding: 1rem 1.5rem;
        border-top: 1px solid #e5e7eb;
        display: flex;
        justify-content: flex-end;
        gap: 0.75rem;
    }

    button {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 500;
        cursor: pointer;
    }

    .cancel-button {
        background-color: white;
        border: 1px solid #e5e7eb;
    }

    .save-button {
        background-color: #3b82f6;
        border: none;
        color: white;
    }

    .save-button:hover {
        background-color: #2563eb;
    }
</style>
