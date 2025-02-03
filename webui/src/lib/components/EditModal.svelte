<!-- src/lib/components/EditModal.svelte -->
<script lang="ts">
    import { fade } from 'svelte/transition';
    import { X } from 'lucide-svelte';
    import { createEventDispatcher } from 'svelte';

    const dispatch = createEventDispatcher();

    export let isOpen = false;
    export let initialContent = '';

    let content = '';

    // Simple assignment when modal opens
    $: if (isOpen) content = initialContent;

    function handleKeydown(event: KeyboardEvent) {
        if (!isOpen) return;

        if (event.key === 'Escape') {
            dispatch('close');
        }

        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            dispatch('save', { content });
        }
    }
</script>

<svelte:window on:keydown={handleKeydown} />

{#if isOpen}
    <div class="modal-backdrop" transition:fade role="presentation">
        <div class="modal-content" role="dialog" aria-modal="true">
            <header class="modal-header">
                <h2>Edit Message</h2>
                <div class="header-actions">
                    <div class="keyboard-hints">
                        <span>Ctrl + Enter to save</span>
                        <span>•</span>
                        <span>Esc to cancel</span>
                    </div>
                    <button class="close-button" on:click={() => dispatch('close')}>
                        <X size={20} />
                    </button>
                </div>
            </header>

            <div class="modal-body">
                <textarea
                    bind:value={content}
                    class="content-textarea"
                    placeholder="Edit your message..."
                />
            </div>

            <footer class="modal-footer">
                <button class="cancel-button" on:click={() => dispatch('close')}>
                    Cancel
                </button>
                <button class="save-button" on:click={() => dispatch('save', { content })}>
                    Save Changes
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
        max-width: 75rem;
        height: 90vh;
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
        min-height: 0;
    }

    .content-textarea {
        width: 100%;
        height: 100%;
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        font-family: inherit;
        font-size: 1rem;
        line-height: 1.5;
        resize: none;
    }

    .content-textarea:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgb(147 197 253 / 0.25);
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