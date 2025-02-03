<!-- src/routes/roster/[id]/+page.svelte -->
<script lang="ts">
    import { page } from '$app/stores';
    import { onMount } from 'svelte';
    import { ArrowLeft, Edit } from 'lucide-svelte';
    import type { Persona } from '$lib/types';
    import PersonaModal from '$lib/components/roster/PersonaModal.svelte';
    import { rosterStore } from '$lib/store/rosterStore';
    let loading = true;
    let error: string | null = null;
    let isModalOpen = false;
    let activeSection: 'basic' | 'attributes' | 'features' | 'wardrobe' | 'thoughts' | 'wakeup' | 'pif' | 'nshot' | null = null;

    onMount(async () => {
        try {
            const response = await rosterStore.getPersona($page.params.id);
            rosterStore.setActivePersona(response);
        } catch (e) {
            error = "Failed to load persona";
        } finally {
            loading = false;
        }
    });

    $: persona = $rosterStore.activePersona;

    function openEditModal(section: typeof activeSection = null) {
        activeSection = section;
        isModalOpen = true;
    }

    function handleModalClose() {
        isModalOpen = false;
        activeSection = null;
    }

    async function handleModalCancel() {
        isModalOpen = false;
        activeSection = null;
        rosterStore.cancelUpdates();
    }

    async function handleModalSave(event: CustomEvent<Partial<Persona>>) {
        try {
            loading = true;
            await rosterStore.updatePersona($page.params.id, event.detail);
            isModalOpen = false;
            activeSection = null;
        } catch (e) {
            error = "Failed to update persona";
        } finally {
            loading = false;
        }
    }
</script>

<svelte:head>
    <title>Persona Roster: {persona?.name || 'Loading...'} | MindAI</title>
</svelte:head>

<div class="page">
    <header class="page-header">
        <a href="/roster" class="back-link">
            <ArrowLeft />
            <span>Back to Roster</span>
        </a>
        <h1>{persona?.name || 'Loading...'}</h1>
    </header>

    {#if loading}
        <div class="loading">Loading persona data...</div>
    {:else if error}
        <div class="error">
            <p>{error}</p>
            <button on:click={() => window.location.reload()}>Retry</button>
        </div>
    {:else if persona}
        <div class="section-grid">
            <!-- Basic Info Section -->
            <section class="info-section">
                <div class="section-header">
                    <h2>Basic Information</h2>
                    <button class="edit-button" on:click={() => openEditModal('basic')}>
                        <Edit size={16} />
                        Edit
                    </button>
                </div>
                <div class="section-content">
                    <div class="info-grid">
                        <div class="info-item">
                            <label>Name</label>
                            <span>{persona.name}</span>
                        </div>
                        <div class="info-item">
                            <label>Full Name</label>
                            <span>{persona.full_name}</span>
                        </div>
                        <div class="info-item">
                            <label>Version</label>
                            <span>{persona.persona_version}</span>
                        </div>
                        <div class="info-item">
                            <label>Chat Strategy</label>
                            <span>{persona.chat_strategy}</span>
                        </div>
                        <div class="info-item">
                            <label>Default Location</label>
                            <span>{persona.default_location}</span>
                        </div>
                        {#if persona.birthday}
                            <div class="info-item">
                                <label>Birthday</label>
                                <span>{persona.birthday}</span>
                            </div>
                        {/if}
                    </div>
                </div>
            </section>

            <!-- Attributes Section -->
            <section class="info-section">
                <div class="section-header">
                    <h2>Attributes</h2>
                    <button class="edit-button" on:click={() => openEditModal('attributes')}>
                        <Edit size={16} />
                        Edit
                    </button>
                </div>
                <div class="section-content">
                    <div class="info-grid">
                        {#each Object.entries(persona.attributes ?? {}) as [key, value]}
                            <div class="info-item">
                                <label>{key}</label>
                                <span>{value}</span>
                            </div>
                        {/each}
                    </div>
                </div>
            </section>

            <!-- Features Section -->
            <section class="info-section">
                <div class="section-header">
                    <h2>Features</h2>
                    <button class="edit-button" on:click={() => openEditModal('features')}>
                        <Edit size={16} />
                        Edit
                    </button>
                </div>
                <div class="section-content">
                    {#each Object.entries(persona.features ?? {}) as [key, value]}
                        <div class="feature-item">
                            <h3>{key}</h3>
                            <p>{value}</p>
                        </div>
                    {/each}
                </div>
            </section>

            <!-- Wardrobe Section -->
            <section class="info-section">
                <div class="section-header">
                    <h2>Wardrobe</h2>
                    <button class="edit-button" on:click={() => openEditModal('wardrobe')}>
                        <Edit size={16} />
                        Edit
                    </button>
                </div>
                <div class="section-content">
                    <div class="current-outfit">
                        Current Outfit: <strong>{persona.current_outfit}</strong>
                    </div>
                    {#each Object.entries(persona.wardrobe ?? {}) as [outfitName, outfit]}
                        <div class="outfit-section">
                            <h3>{outfitName}</h3>
                            <div class="outfit-details">
                                {#each Object.entries(outfit ?? {}) as [piece, description]}
                                    <div class="outfit-piece">
                                        <label>{piece}</label>
                                        <p>{description}</p>
                                    </div>
                                {/each}
                            </div>
                        </div>
                    {/each}
                </div>
            </section>

            <!-- Base Thoughts Section -->
            <section class="info-section">
                <div class="section-header">
                    <h2>Base Thoughts</h2>
                    <button class="edit-button" on:click={() => openEditModal('thoughts')}>
                        <Edit size={16} />
                        Edit
                    </button>
                </div>
                <div class="section-content">
                    {#each persona.base_thoughts ?? [] as thought}
                        <div class="thought-item">
                            <p>{thought}</p>
                        </div>
                    {/each}
                </div>
            </section>

            <!-- Wakeup Messages Section -->
            <section class="info-section">
                <div class="section-header">
                    <h2>Wakeup Messages</h2>
                    <button class="edit-button" on:click={() => openEditModal('wakeup')}>
                        <Edit size={16} />
                        Edit
                    </button>
                </div>
                <div class="section-content">
                    {#each persona.wakeup ?? [] as message}
                        <div class="message-item">
                            <p>{message}</p>
                        </div>
                    {/each}
                </div>
            </section>

            <!-- PIF Data Section -->
            <section class="info-section">
                <div class="section-header">
                    <h2>PIF Data</h2>
                    <button class="edit-button" on:click={() => openEditModal('pif')}>
                        <Edit size={16} />
                        Edit
                    </button>
                </div>
                <div class="section-content">
                    {#each Object.entries(persona.pif ?? {}) as [key, value]}
                        <div class="pif-item">
                            <h3>{key}</h3>
                            <p>{value}</p>
                        </div>
                    {/each}
                </div>
            </section>

            <!-- N-Shot Data Section -->
            <section class="info-section">
                <div class="section-header">
                    <h2>N-Shot Data</h2>
                    <button class="edit-button" on:click={() => openEditModal('nshot')}>
                        <Edit size={16} />
                        Edit
                    </button>
                </div>
                <div class="section-content">
                    {#each Object.entries(persona.nshot ?? {}) as [key, value]}
                        <div class="nshot-item">
                            <h3>{key}</h3>
                            <div class="nshot-example">
                                <h4>Human</h4>
                                <p>{value.human}</p>
                                <h4>Assistant</h4>
                                <p>{value.assistant}</p>
                            </div>
                        </div>
                    {/each}
                </div>
            </section>
        </div>
    {/if}
</div>

<PersonaModal
    bind:isOpen={isModalOpen}
    persona={persona ?? null}
    initialSection={activeSection}
    on:close={handleModalClose}
    on:cancel={handleModalCancel}
    on:save={handleModalSave}
/>

<style>
    .page {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }

    .page-header {
        margin-bottom: 2rem;
    }

    .back-link {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #666;
        text-decoration: none;
        margin-bottom: 1rem;
    }

    .back-link:hover {
        color: #333;
    }

    h1 {
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
    }

    .section-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        gap: 1.5rem;
    }

    .info-section {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        overflow: hidden;
    }

    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background: #f9fafb;
        border-bottom: 1px solid #e5e7eb;
    }

    .section-header h2 {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 600;
    }

    .edit-button {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.375rem;
        background: #4CAF50;
        color: white;
        cursor: pointer;
        font-size: 0.875rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }

    .edit-button:hover {
        background: #43A047;
    }

    .section-content {
        padding: 1rem;
    }

    .info-grid {
        display: grid;
        gap: 1rem;
    }

    .info-item {
        display: grid;
        gap: 0.25rem;
    }

    .info-item label {
        font-size: 0.875rem;
        color: #666;
        font-weight: 500;
    }

    .info-item span {
        color: #111;
    }

    .feature-item {
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e5e7eb;
    }

    .feature-item:last-child {
        margin-bottom: 0;
        padding-bottom: 0;
        border-bottom: none;
    }

    .feature-item h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        font-weight: 600;
    }

    .feature-item p {
        margin: 0;
        white-space: pre-wrap;
    }

    .current-outfit {
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: #f3f4f6;
        border-radius: 0.375rem;
        font-size: 0.875rem;
    }

    .outfit-section {
        margin-bottom: 1.5rem;
    }

    .outfit-section h3 {
        margin: 0 0 0.75rem 0;
        font-size: 1rem;
        font-weight: 600;
        color: #4CAF50;
    }

    .outfit-details {
        display: grid;
        gap: 0.75rem;
    }

    .outfit-piece {
        background: #f9fafb;
        padding: 0.75rem;
        border-radius: 0.375rem;
    }

    .outfit-piece label {
        display: block;
        font-size: 0.875rem;
        font-weight: 500;
        color: #666;
        margin-bottom: 0.25rem;
    }

    .outfit-piece p {
        margin: 0;
        white-space: pre-wrap;
    }

    .thought-item,
    .message-item {
        background: #f9fafb;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin-bottom: 0.75rem;
    }

    .thought-item:last-child,
    .message-item:last-child {
        margin-bottom: 0;
    }

    .thought-item p,
    .message-item p {
        margin: 0;
        white-space: pre-wrap;
    }

    .nshot-item,
    .pif-item {
        margin-bottom: 1.5rem;
    }

    .nshot-item:last-child,
    .pif-item:last-child {
        margin-bottom: 0;
    }

    .nshot-item h3,
    .pif-item h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        font-weight: 600;
    }

    .nshot-item p,
    .pif-item p {
        margin: 0;
        white-space: pre-wrap;
    }

    .loading,
    .error {
        text-align: center;
        padding: 2rem;
        background: white;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
    }

    .error {
        color: #dc2626;
    }

    .error button {
        margin-top: 1rem;
        padding: 0.5rem 1rem;
        background: #dc2626;
        color: white;
        border: none;
        border-radius: 0.375rem;
        cursor: pointer;
    }

    .error button:hover {
        background: #b91c1c;
    }

    @media (max-width: 768px) {
        .page {
            padding: 1rem;
        }

        .section-grid {
            grid-template-columns: 1fr;
        }
    }
</style>