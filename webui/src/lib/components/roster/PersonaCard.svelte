<!-- lib/components/roster/PersonaCard.svelte -->
<script lang="ts">
    import { Edit2, Trash2, ChevronDown, ChevronUp } from 'lucide-svelte';
    import type { Persona } from '$lib/types';
    
    export let persona: Persona;
    export let onEdit: (persona: Persona) => void;
    export let onDelete: (personaId: string) => void;
    
    let expanded = false;

    function toggleExpanded() {
        expanded = !expanded;
    }

    function handleDelete() {
        if (confirm(`Are you sure you want to delete ${persona.name}?`)) {
            onDelete(persona.persona_id);
        }
    }
</script>

<div class="persona-card">
    <div class="card-header">
        <div class="persona-info">
            <h3>{persona.name}</h3>
            <span class="persona-id">{persona.persona_id}</span>
        </div>
        <div class="card-actions">
            <button 
                class="action-button edit"
                on:click={() => onEdit(persona)}
                title="Edit persona"
            >
                <Edit2 size={16} />
            </button>
            <button 
                class="action-button delete"
                on:click={handleDelete}
                title="Delete persona"
            >
                <Trash2 size={16} />
            </button>
            <button 
                class="action-button expand"
                on:click={toggleExpanded}
                title={expanded ? "Show less" : "Show more"}
            >
                {#if expanded}
                    <ChevronUp size={16} />
                {:else}
                    <ChevronDown size={16} />
                {/if}
            </button>
        </div>
    </div>

    <div class="card-content">
        <div class="basic-info">
            <p class="full-name">{persona.full_name}</p>
            <p class="strategy">Strategy: {persona.chat_strategy}</p>
            <p class="location">Location: {persona.default_location}</p>
        </div>

        {#if expanded}
            <div class="expanded-content">
                <div class="section">
                    <h4>Attributes</h4>
                    <ul>
                        {#each Object.entries(persona.attributes) as [key, value]}
                            <li><strong>{key}:</strong> {value}</li>
                        {/each}
                    </ul>
                </div>

                <div class="section">
                    <h4>Features</h4>
                    <ul>
                        {#each Object.entries(persona.features) as [key, value]}
                            <li><strong>{key}:</strong> {value}</li>
                        {/each}
                    </ul>
                </div>

                <div class="section">
                    <h4>Wardrobe</h4>
                    <p class="current-outfit">Current: {persona.current_outfit}</p>
                    <ul>
                        {#each Object.entries(persona.wardrobe) as [name, outfit]}
                            <li>
                                <strong>{name}:</strong> {outfit.style} - {outfit.outfit}
                            </li>
                        {/each}
                    </ul>
                </div>
            </div>
        {/if}
    </div>
</div>

<style>
    .persona-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        overflow: hidden;
        transition: all 0.2s;
    }

    .persona-card:hover {
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background: #f9fafb;
        border-bottom: 1px solid #e5e7eb;
    }

    .persona-info h3 {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
    }

    .persona-id {
        font-size: 0.875rem;
        color: #6b7280;
    }

    .card-actions {
        display: flex;
        gap: 0.5rem;
    }

    .action-button {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem;
        border: none;
        background: none;
        border-radius: 0.375rem;
        cursor: pointer;
        transition: all 0.2s;
        width: 32px;
        height: 32px;
    }

    .action-button.edit {
        color: #2196f3;
    }

    .action-button.edit:hover {
        background: rgba(33, 150, 243, 0.1);
    }

    .action-button.delete {
        color: #ef4444;
    }

    .action-button.delete:hover {
        background: rgba(239, 68, 68, 0.1);
    }

    .action-button.expand {
        color: #6b7280;
    }

    .action-button.expand:hover {
        background: rgba(107, 114, 128, 0.1);
    }

    .card-content {
        padding: 1rem;
    }

    .basic-info {
        margin-bottom: 1rem;
    }

    .full-name {
        font-size: 1rem;
        color: #374151;
        margin: 0 0 0.5rem 0;
    }

    .strategy, .location {
        font-size: 0.875rem;
        color: #6b7280;
        margin: 0.25rem 0;
    }

    .expanded-content {
        border-top: 1px solid #e5e7eb;
        padding-top: 1rem;
        margin-top: 1rem;
    }

    .section {
        margin-bottom: 1rem;
    }

    .section h4 {
        font-size: 0.875rem;
        font-weight: 600;
        color: #374151;
        margin: 0 0 0.5rem 0;
    }

    .section ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .section li {
        font-size: 0.875rem;
        color: #6b7280;
        margin: 0.25rem 0;
    }

    .current-outfit {
        font-size: 0.875rem;
        color: #374151;
        margin: 0 0 0.5rem 0;
    }
</style>