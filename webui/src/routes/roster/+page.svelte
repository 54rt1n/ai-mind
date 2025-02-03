<!-- src/routes/roster/+page.svelte -->
<script lang="ts">
    import { onMount } from 'svelte';
    import { goto } from '$app/navigation';
    import { Users } from 'lucide-svelte';
    import { rosterStore } from '$lib/store/rosterStore';
    import PersonaTile from '$lib/components/roster/PersonaTile.svelte';

    let loading = true;
    let error: string | null = null;
    $: personas = $rosterStore.personas;

    onMount(async () => {
        try {
            await rosterStore.fetchPersonas();
        } catch (e) {
            error = "Failed to load roster";
        } finally {
            loading = false;
        }
    });

    function navigateToPersona(personaId: string) {
        goto(`/roster/${personaId}`);
    }
</script>

<svelte:head>
    <title>Persona Roster | MindAI</title>
</svelte:head>

<div class="page">
    <header class="page-header">
        <h1>Persona Roster</h1>
        <div class="header-meta">
            <div class="persona-count">
                <Users size={20} />
                <span>{personas.length} Personas</span>
            </div>
        </div>
    </header>

    {#if loading}
        <div class="loading-state">
            <div class="loading-spinner" />
            <p>Loading roster...</p>
        </div>
    {:else if error}
        <div class="error-state">
            <p>{error}</p>
            <button on:click={() => window.location.reload()}>Retry</button>
        </div>
    {:else if personas.length === 0}
        <div class="empty-state">
            <Users size={48} />
            <h2>No Personas Found</h2>
            <p>No personas are currently available in the roster.</p>
        </div>
    {:else}
        <div class="personas-grid">
            {#each personas as persona (persona.persona_id)}
                <div on:click={() => navigateToPersona(persona.persona_id)} on:keydown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        navigateToPersona(persona.persona_id);
                    }
                }}>
                    <PersonaTile {persona} />
                </div>
            {/each}
        </div>
    {/if}
</div>

<style>
    .page {
        max-width: 1600px;
        margin: 0 auto;
        padding: 2rem;
    }

    .page-header {
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    h1 {
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
        color: #1e293b;
    }

    .header-meta {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .persona-count {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: #f1f5f9;
        border-radius: 2rem;
        color: #64748b;
        font-weight: 500;
    }

    .personas-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
        gap: 1.5rem;
    }

    .loading-state,
    .error-state,
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        margin: 2rem 0;
    }

    .loading-spinner {
        width: 2.5rem;
        height: 2.5rem;
        border: 3px solid #f1f5f9;
        border-top-color: #94a3b8;
        border-radius: 50%;
        margin: 0 auto 1rem;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .error-state {
        color: #dc2626;
    }

    .error-state button {
        margin-top: 1rem;
        padding: 0.5rem 1rem;
        background: #dc2626;
        color: white;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
        font-weight: 500;
        transition: background-color 0.2s;
    }

    .error-state button:hover {
        background: #b91c1c;
    }

    .empty-state h2 {
        margin: 1rem 0 0.5rem;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
    }

    .empty-state p {
        margin: 0;
        color: #64748b;
    }

    @media (max-width: 768px) {
        .page {
            padding: 1rem;
        }

        .personas-grid {
            grid-template-columns: 1fr;
        }

        .page-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 1rem;
        }
    }
</style>