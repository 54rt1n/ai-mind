<!-- src/lib/components/roster/PersonaTile.svelte -->
<script lang="ts">
    import type { Persona } from '$lib/types';
    import { Plus } from 'lucide-svelte';
    import { configStore } from '$lib/store/configStore';
    export let persona: Persona;

    $: isActive = $configStore.persona_id === persona.persona_id;

    function formatCount(count: number): string {
        return count.toString().padStart(2, '0');
    }

    function handleActivate() {
        if (isActive) {
            // If already active, clear it
            configStore.updateField("persona", "");
            configStore.updateField("persona_id", "");
        } else {
            // Set as active persona
            configStore.updateField("persona", persona.name);
            configStore.updateField("persona_id", persona.persona_id);
        }
    }
</script>

<div 
    class="persona-tile"
    role="button"
    tabindex="0"
>
    <div class="header">
        <div class="title">
            <h3>{persona.name}</h3>
            <span class="version">v{persona.persona_version}</span>
        </div>
        <div class="meta">
            <button 
                class="action-button {isActive ? 'active' : 'activate'}"
                on:click|stopPropagation={handleActivate}
                title={isActive ? "Deactivate persona" : "Activate persona"}
            >
                <span
                    class="icon-rotate"
                    style:transform={`rotate(${isActive ? 45 : 0}deg)`}
                >
                    <Plus size={16} />
                </span>
            </button>
            <span class="strategy">{persona.chat_strategy}</span>
        </div>
    </div>

    <div class="content">
        <div class="details">
            <p class="full-name">{persona.full_name}</p>
            <p class="location">{persona.default_location}</p>
            <p class="outfit">
                <span class="label">Current Outfit</span>
                {persona.current_outfit}
            </p>
        </div>

        <div class="stats">
            <div class="stat">
                <span class="stat-value">{formatCount(Object.keys(persona.attributes).length)}</span>
                <span class="stat-label">Attributes</span>
            </div>
            <div class="stat">
                <span class="stat-value">{formatCount(Object.keys(persona.features).length)}</span>
                <span class="stat-label">Features</span>
            </div>
            <div class="stat">
                <span class="stat-value">{formatCount(Object.keys(persona.wardrobe).length)}</span>
                <span class="stat-label">Wardrobe</span>
            </div>
            <div class="stat">
                <span class="stat-value">{formatCount(persona.wakeup.length)}</span>
                <span class="stat-label">Wakeup</span>
            </div>
            <div class="stat">
                <span class="stat-value">{formatCount(persona.base_thoughts.length)}</span>
                <span class="stat-label">Thoughts</span>
            </div>
            <div class="stat">
                <span class="stat-value">{formatCount(Object.keys(persona.pif).length)}</span>
                <span class="stat-label">PIF</span>
            </div>
            <div class="stat">
                <span class="stat-value">{formatCount(Object.keys(persona.nshot).length)}</span>
                <span class="stat-label">N-Shot</span>
            </div>
        </div>
    </div>

    {#if persona.birthday}
        <div class="footer">
            <span class="birthday">Born {persona.birthday}</span>
        </div>
    {/if}
</div>

<style>
    .persona-tile {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        padding: 1.25rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        flex-direction: column;
        gap: 1.25rem;
    }

    .persona-tile:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.1);
        border-color: #94a3b8;
    }

    .persona-tile:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(148, 163, 184, 0.3);
    }

    .header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
    }

    .title {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    h3 {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
    }

    .version {
        font-size: 0.75rem;
        font-weight: 500;
        color: #64748b;
        background: #f1f5f9;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
    }

    .meta {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .strategy {
        font-size: 0.75rem;
        color: #64748b;
        background: #f1f5f9;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        display: flex;
        align-items: center;
    }

    .content {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        align-items: start;
    }

    .details {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .full-name, .location, .outfit {
        margin: 0;
        font-size: 0.875rem;
        color: #475569;
    }

    .outfit .label {
        display: block;
        font-size: 0.75rem;
        color: #64748b;
        margin-bottom: 0.25rem;
    }

    .stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
    }

    .stat {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.25rem;
        padding: 0.5rem;
        border-radius: 0.5rem;
        background: #f8fafc;
        transition: background-color 0.2s;
    }

    .stat:hover {
        background: #f1f5f9;
    }

    .stat-value {
        font-size: 1.125rem;
        font-weight: 600;
        color: #334155;
        font-variant-numeric: tabular-nums;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-align: center;
    }

    .footer {
        padding-top: 0.75rem;
        border-top: 1px solid #e2e8f0;
    }

    .birthday {
        font-size: 0.75rem;
        color: #64748b;
    }

    @media (max-width: 768px) {
        .content {
            grid-template-columns: 1fr;
            gap: 1rem;
        }

        .details {
            padding-bottom: 1rem;
            border-bottom: 1px solid #e2e8f0;
        }

        .stats {
            grid-template-columns: repeat(4, 1fr);
        }
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

    .action-button.activate {
        color: #4caf50;
        background-color: rgba(76, 175, 80, 0.1);
    }

    .action-button.active {
        color: #f44336;
        background-color: rgba(244, 67, 54, 0.1);
    }

    .icon-rotate {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.2s ease;
    }
</style> 