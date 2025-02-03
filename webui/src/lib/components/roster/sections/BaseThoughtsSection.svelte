<!-- src/lib/components/roster/sections/BaseThoughtsSection.svelte -->
<script lang="ts">
    import EditSection from './EditSection.svelte';
    import { Plus, Trash2 } from 'lucide-svelte';
    import type { Persona } from '$lib/types';
    import { rosterStore } from '$lib/store/rosterStore';

    export let formData: Partial<Persona>;

    $: thoughts = formData.base_thoughts || [];
</script>

<EditSection title="Base Thoughts">
    <div class="section-header">
        <button type="button" class="add-button" on:click={rosterStore.addBaseThought}>
            <Plus size={16} />
            Add Thought
        </button>
    </div>

    {#if thoughts.length === 0}
        <p>No base thoughts yet.</p>
    {/if}
    {#each thoughts as thought, index}
        <div class="thought-item">
            <textarea
                rows="2"
                placeholder="Enter base thought"
                bind:value={thoughts[index]}
            ></textarea>
            <button 
                type="button"
                class="remove-button"
                on:click={() => rosterStore.removeBaseThought(index)}
                disabled={thoughts.length === 1}
            >
                <Trash2 size={16} />
            </button>
        </div>
    {/each}
</EditSection>

<style>
    .section-header {
        margin-bottom: 1rem;
    }

    .thought-item {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }

    textarea {
        flex: 1;
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        resize: vertical;
        min-height: 60px;
        transition: border-color 0.2s;
    }

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

    .remove-button:disabled {
        color: #9ca3af;
        cursor: not-allowed;
    }

    .remove-button:disabled:hover {
        background: none;
    }
</style> 