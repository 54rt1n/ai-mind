<!-- src/lib/components/roster/PersonaModal.svelte -->
<script lang="ts">
    import { createEventDispatcher } from 'svelte';
    import { fade } from 'svelte/transition';
    import { X } from 'lucide-svelte';
    import type { Persona } from '$lib/types';
    import BasicInfoSection from './sections/BasicInfoSection.svelte';
    import AttributesSection from './sections/AttributesSection.svelte';
    import FeaturesSection from './sections/FeaturesSection.svelte';
    import WardrobeSection from './sections/WardrobeSection.svelte';
    import BaseThoughtsSection from './sections/BaseThoughtsSection.svelte';
    import WakeupMessagesSection from './sections/WakeupMessagesSection.svelte';
    import PifDataSection from './sections/PifDataSection.svelte';
    import NShotSection from './sections/NShotSection.svelte';
    import { rosterStore } from '$lib/store/rosterStore';

    const dispatch = createEventDispatcher<{
        cancel: void;
        save: Partial<Persona>;
    }>();

    export let isOpen = false;
    export let persona: Partial<Persona> | null = null;
    export let initialSection: 'basic' | 'attributes' | 'features' | 'wardrobe' | 'thoughts' | 'wakeup' | 'pif' | 'nshot' | null = null;

    // Form state
    let activeSection = initialSection;

    const sections = [
        { id: 'basic', label: 'Basic Info' },
        { id: 'attributes', label: 'Attributes' },
        { id: 'features', label: 'Features' },
        { id: 'wardrobe', label: 'Wardrobe' },
        { id: 'thoughts', label: 'Thoughts' },
        { id: 'wakeup', label: 'Wakeup' },
        { id: 'pif', label: 'PIF' },
        { id: 'nshot', label: 'N-Shot' }
    ] as const;


    // Reset form when modal opens/closes or persona changes
    $: activePersona = $rosterStore.activePersona ?? {};
    $: if (isOpen) {
        activeSection = initialSection;
    }

    function handleCancel() {
        dispatch('cancel');
    }

    function handleSubmit() {
        const formData = activePersona;
        if (!formData) {
            console.error('No form data to save');
            return;
        }
        dispatch('save', formData);
    }

    function handleKeydown(event: KeyboardEvent) {
        if (event.key === 'Escape') {
            handleCancel();
        }
    }

    function setActiveSection(section: typeof activeSection) {
        activeSection = section;
    }

    async function handleSectionCancel() {
        if (!persona?.persona_id) return;
        try {
            const response = await rosterStore.getPersona(persona.persona_id);
            rosterStore.setActivePersona(response);
        } catch (error) {
            console.error('Failed to reload persona:', error);
        }
    }
</script>

<svelte:window on:keydown={handleKeydown} />

{#if isOpen}
    <div class="modal-backdrop" transition:fade>
        <div class="modal-content" role="dialog" aria-modal="true">
            <header class="modal-header">
                <h2>{persona ? 'Edit' : 'Create'} Persona</h2>
                <div class="section-tabs">
                    {#each sections as { id, label }}
                        <button 
                            type="button"
                            class="tab-button" 
                            class:active={activeSection === id}
                            on:click={() => setActiveSection(id)}
                        >
                            {label}
                        </button>
                    {/each}
                </div>
                <button type="button" class="close-button" on:click={handleCancel}>
                    <X size={20} />
                </button>
            </header>

            <form class="modal-body" on:submit|preventDefault={handleSubmit}>
                <div class="form-grid">
                    {#if !activeSection || activeSection === 'basic'}
                        <BasicInfoSection bind:formData={activePersona} on:cancel={handleSectionCancel} />
                    {/if}
                    {#if !activeSection || activeSection === 'attributes'}
                        <AttributesSection bind:formData={activePersona} on:cancel={handleSectionCancel} />
                    {/if}
                    {#if !activeSection || activeSection === 'features'}
                        <FeaturesSection bind:formData={activePersona} on:cancel={handleSectionCancel} />
                    {/if}
                    {#if !activeSection || activeSection === 'wardrobe'}
                        <WardrobeSection bind:formData={activePersona} on:cancel={handleSectionCancel} />
                    {/if}
                    {#if !activeSection || activeSection === 'thoughts'}
                        <BaseThoughtsSection bind:formData={activePersona} on:cancel={handleSectionCancel} />
                    {/if}
                    {#if !activeSection || activeSection === 'wakeup'}
                        <WakeupMessagesSection bind:formData={activePersona} on:cancel={handleSectionCancel} />
                    {/if}
                    {#if !activeSection || activeSection === 'pif'}
                        <PifDataSection bind:formData={activePersona} on:cancel={handleSectionCancel} />
                    {/if}
                    {#if !activeSection || activeSection === 'nshot'}
                        <NShotSection bind:formData={activePersona} on:cancel={handleSectionCancel} />
                    {/if}
                </div>

                <footer class="modal-footer">
                    <button type="button" class="cancel-button" on:click={handleCancel}>
                        Cancel
                    </button>
                    <button type="submit" class="save-button">
                        {persona ? 'Update' : 'Create'} Persona
                    </button>
                </footer>
            </form>
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
        max-width: 900px;
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
        gap: 1rem;
    }

    .modal-header h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
        white-space: nowrap;
    }

    .section-tabs {
        display: flex;
        gap: 0.5rem;
        overflow-x: auto;
        padding: 0.25rem;
        -ms-overflow-style: none;
        scrollbar-width: none;
    }

    .section-tabs::-webkit-scrollbar {
        display: none;
    }

    .tab-button {
        padding: 0.5rem 1rem;
        border: none;
        background: none;
        color: #6b7280;
        font-size: 0.875rem;
        font-weight: 500;
        cursor: pointer;
        white-space: nowrap;
        border-radius: 0.375rem;
        transition: all 0.2s;
    }

    .tab-button:hover {
        background: #f3f4f6;
        color: #111827;
    }

    .tab-button.active {
        background: #4CAF50;
        color: white;
    }

    .close-button {
        padding: 0.5rem;
        border: none;
        background: none;
        cursor: pointer;
        color: #6b7280;
        transition: color 0.2s;
        flex-shrink: 0;
    }

    .close-button:hover {
        color: #111827;
    }

    .modal-body {
        flex: 1;
        overflow-y: auto;
        padding: 1.5rem;
    }

    .form-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
    }

    .modal-footer {
        padding: 1rem 1.5rem;
        border-top: 1px solid #e5e7eb;
        display: flex;
        justify-content: flex-end;
        gap: 0.75rem;
    }

    .cancel-button,
    .save-button {
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
    }

    .cancel-button {
        color: #374151;
        background: white;
        border: 1px solid #e5e7eb;
    }

    .cancel-button:hover {
        background: #f9fafb;
    }

    .save-button {
        color: white;
        background: #4CAF50;
        border: none;
    }

    .save-button:hover {
        background: #43A047;
    }
</style>