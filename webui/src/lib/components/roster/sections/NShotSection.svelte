<!-- src/lib/components/roster/sections/NShotSection.svelte -->
<script lang="ts">
    import EditSection from './EditSection.svelte';
    import { Plus, Trash2 } from 'lucide-svelte';
    import type { Persona, PersonaNestedSection } from '$lib/types';
    import { rosterStore } from '$lib/store/rosterStore';

    export let formData: Partial<Persona>;

    $: nshot = (formData.nshot || {}) as PersonaNestedSection;
</script>

<EditSection title="N-Shot Data">
    <div class="section-header">
        <h4>N-Shot Data</h4>
        <button type="button" class="add-button" on:click|preventDefault={rosterStore.addPifEntry}>
            <Plus size={16} />
            Add Entry
        </button>
    </div>
    
    {#if Object.entries(nshot).length === 0}
        <p class="empty-state">No N-Shot data yet.</p>
    {/if}
    {#each Object.entries(nshot) as [key, value]}
        {#if key !== 'core_traits'}
            <div class="nshot-item">
                <div class="nshot-header">
                    <input
                        type="text"
                        placeholder="Scenario Name"
                        value={key}
                        on:change={(e) => rosterStore.updateNshotKey(key, e.currentTarget.value)}
                    />
                    <button 
                        type="button"
                        class="remove-button"
                        on:click|preventDefault={() => rosterStore.removeNshotEntry(key)}
                        title="Remove entry"
                    >
                        <Trash2 size={16} />
                    </button>
                </div>
                <div class="nshot-content">
                    <div class="input-group">
                        <label for={`human-${key}`}>Human:</label>
                        <textarea
                            id={`human-${key}`}
                            rows="3"
                            placeholder="Enter the human's message..."
                            bind:value={nshot[key].human}
                        ></textarea>
                    </div>
                    <div class="input-group">
                        <label for={`assistant-${key}`}>Assistant:</label>
                        <textarea
                            id={`assistant-${key}`}
                            rows="3"
                            placeholder="Enter the assistant's response..."
                            bind:value={nshot[key].assistant}
                        ></textarea>
                    </div>
                </div>
            </div>
        {/if}
    {/each}
</EditSection>

<style>
    .section-header {
        margin: 1.5rem 0 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .section-header h4 {
        margin: 0;
        font-size: 0.875rem;
        font-weight: 500;
        color: #4b5563;
    }

    .empty-state {
        color: #6b7280;
        font-size: 0.875rem;
        font-style: italic;
    }

    .nshot-item {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .nshot-header {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    .nshot-content {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .input-group {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .input-group label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #4b5563;
    }

    input {
        width: 100%;
        max-width: 250px;
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        transition: all 0.2s;
    }

    textarea {
        width: 100%;
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        resize: vertical;
        min-height: 80px;
        transition: all 0.2s;
        line-height: 1.4;
    }

    input:disabled {
        background-color: #f3f4f6;
        color: #6b7280;
    }

    input:focus,
    textarea:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgb(59 130 246 / 0.1);
    }

    .add-button {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        color: #059669;
        background: #d1fae5;
        border: none;
        border-radius: 0.375rem;
        cursor: pointer;
        transition: all 0.2s;
    }

    .add-button:hover {
        background: #a7f3d0;
    }

    .remove-button {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.25rem;
        color: #ef4444;
        background: none;
        border: none;
        border-radius: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
    }

    .remove-button:hover {
        background: #fee2e2;
    }
</style> 