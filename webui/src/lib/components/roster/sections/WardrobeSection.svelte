<!-- src/lib/components/roster/sections/WardrobeSection.svelte -->
<script lang="ts">
    import EditSection from './EditSection.svelte';
    import { Plus, Trash2, GripVertical } from 'lucide-svelte';
    import type { Persona } from '$lib/types';
    import { rosterStore } from '$lib/store/rosterStore';

    export let formData: Partial<Persona>;

    $: wardrobe = formData.wardrobe || {};
</script>

<EditSection title="Wardrobe">
    <div class="section-header">
        <div class="current-outfit">
            <label for="current_outfit">Current Outfit</label>
            <select
                id="current_outfit"
                bind:value={formData.current_outfit}
            >
                {#each Object.keys(wardrobe) as outfitName}
                    <option value={outfitName}>{outfitName}</option>
                {/each}
            </select>
        </div>
        <button type="button" class="add-button" on:click={() => rosterStore.addOutfit()}>
            <Plus size={16} />
            Add Outfit
        </button>
    </div>

    {#if Object.entries(wardrobe).length === 0}
        <p>No outfits yet.</p>
    {/if}
    <div class="outfits-grid">
        {#each Object.entries(wardrobe) as [outfitName, outfit]}
            <div class="outfit-section">
                <div class="outfit-header">
                    <div class="outfit-title">
                        <GripVertical size={16} class="grip" />
                        {#if outfitName === 'default'}
                            <span class="outfit-name">default</span>
                        {:else}
                            <input
                                type="text"
                                class="outfit-name"
                                value={outfitName}
                                on:change={(e) => rosterStore.updateOutfitName(outfitName, e.currentTarget.value)}
                                placeholder="Outfit name"
                            />
                        {/if}
                    </div>
                    {#if outfitName !== 'default'}
                        <button 
                            type="button"
                            class="remove-button"
                            on:click={() => rosterStore.removeOutfit(outfitName)}
                        >
                            <Trash2 size={16} />
                        </button>
                    {/if}
                </div>
                <div class="outfit-details">
                    {#if Object.entries(outfit).length === 0}
                        <p>No pieces yet.</p>
                    {/if}
                    <div class="pieces-list">
                        {#each Object.entries(outfit) as [piece, description]}
                            <div class="outfit-piece">
                                <div class="piece-header">
                                    <input
                                        type="text"
                                        class="piece-name"
                                        value={piece}
                                        on:change={(e) => rosterStore.updateWardrobePieceName(outfitName, piece, e.currentTarget.value)}
                                        placeholder="Piece name (e.g. Top, Bottom)"
                                    />
                                    {#if Object.keys(outfit).length > 1}
                                        <button 
                                            type="button"
                                            class="remove-piece-button"
                                            on:click={() => rosterStore.removeWardrobePiece(outfitName, piece)}
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    {/if}
                                </div>
                                <textarea
                                    rows="3"
                                    placeholder="Describe this piece in detail"
                                    bind:value={wardrobe[outfitName][piece]}
                                ></textarea>
                            </div>
                        {/each}
                    </div>
                    <button 
                        type="button" 
                        class="add-piece-button" 
                        on:click={() => rosterStore.addWardrobePiece(outfitName)}
                    >
                        <Plus size={16} />
                        Add Piece
                    </button>
                </div>
            </div>
        {/each}
    </div>
</EditSection>

<style>
    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 2rem;
        gap: 2rem;
    }

    .current-outfit {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        flex: 1;
    }

    .current-outfit label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #374151;
    }

    .current-outfit select {
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        width: 100%;
        max-width: 300px;
        background-color: white;
    }

    .outfits-grid {
        display: grid;
        gap: 1.5rem;
    }

    .outfit-section {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    }

    .outfit-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e5e7eb;
    }

    .outfit-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .grip {
        color: #9ca3af;
        cursor: grab;
    }

    .grip:active {
        cursor: grabbing;
    }

    .outfit-name {
        font-weight: 600;
        color: #111827;
        border: none;
        background: none;
        padding: 0.5rem;
        font-size: 1rem;
        border: 1px solid transparent;
        border-radius: 0.375rem;
        min-width: 200px;
    }

    input.outfit-name {
        transition: all 0.2s;
    }

    input.outfit-name:hover {
        border-color: #e5e7eb;
    }

    input.outfit-name:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgb(59 130 246 / 0.1);
    }

    span.outfit-name {
        color: #6b7280;
        font-style: italic;
    }

    .outfit-details {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .pieces-list {
        display: grid;
        gap: 1rem;
    }

    .outfit-piece {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        padding: 1rem;
    }

    .piece-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .piece-name {
        flex: 1;
        font-weight: 500;
        color: #111827;
        border: 1px solid transparent;
        background: none;
        padding: 0.375rem 0.5rem;
        font-size: 0.875rem;
        border-radius: 0.375rem;
        transition: all 0.2s;
    }

    .piece-name:hover {
        border-color: #e5e7eb;
    }

    .piece-name:focus {
        outline: none;
        border-color: #3b82f6;
        background: white;
        box-shadow: 0 0 0 3px rgb(59 130 246 / 0.1);
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
        background: white;
    }

    textarea:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgb(59 130 246 / 0.1);
    }

    .add-button {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        color: #059669;
        background: #ecfdf5;
        border: none;
        border-radius: 0.375rem;
        cursor: pointer;
        transition: all 0.2s;
        white-space: nowrap;
    }

    .add-button:hover {
        background: #d1fae5;
    }

    .remove-button, .remove-piece-button {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem;
        color: #ef4444;
        background: none;
        border: none;
        border-radius: 0.375rem;
        cursor: pointer;
        transition: all 0.2s;
    }

    .remove-piece-button {
        padding: 0.375rem;
    }

    .remove-button:hover, .remove-piece-button:hover {
        background: #fee2e2;
    }

    .add-piece-button {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        color: #6b7280;
        background: white;
        border: 1px dashed #d1d5db;
        border-radius: 0.375rem;
        cursor: pointer;
        transition: all 0.2s;
        width: 100%;
        justify-content: center;
    }

    .add-piece-button:hover {
        border-color: #3b82f6;
        color: #3b82f6;
        background: #f8fafc;
    }
</style> 