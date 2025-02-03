<!-- src/lib/components/roster/sections/AttributesSection.svelte -->
<script lang="ts">
    import EditSection from './EditSection.svelte';
    import { Plus, Trash2 } from 'lucide-svelte';
    import type { Persona, PersonaSection } from '$lib/types';
    import { rosterStore } from '$lib/store/rosterStore';

    export let formData: Partial<Persona>;

    $: attributes = formData.attributes ?? {};
</script>

<EditSection title="Attributes">
    <div class="section-header">
        <h4>Attributes</h4>
        <button type="button" class="add-button" on:click={rosterStore.addAttribute}>
            <Plus size={16} />
            Add Attribute
        </button>
    </div>
    
    {#each Object.entries(attributes) as [key, value]}
        <div class="dictionary-item">
            <input
                type="text"
                placeholder="Key"
                value={key}
                on:change={(e) => rosterStore.updateAttributeKey(key, e.currentTarget.value)}
            />
            <input
                type="text"
                placeholder="Value"
                bind:value={attributes[key]}
            />
            <button 
                type="button"
                class="remove-button"
                on:click={() => rosterStore.removeAttribute(key)}
            >
                <Trash2 size={16} />
            </button>
        </div>
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

    .dictionary-item {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }

    input {
        flex: 1;
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        transition: border-color 0.2s;
    }

    input:disabled {
        background-color: #f3f4f6;
        color: #6b7280;
    }

    input:focus {
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