<!-- src/lib/components/PipelineTriggers.svelte -->
<script lang="ts">
    import { goto } from '$app/navigation';
    import { pipelineStore } from '$lib/store/pipelineStore';
    import { taskStore } from '$lib/store/taskStore';
    import { Brain, FileText, Compass, CheckSquare, Cloud, Book, Library, Feather, MessageCircleHeart, Heart } from 'lucide-svelte';
    import { type PipelineType } from '$lib/types';
    
    export let latestPersonaId: string | null = null;
    export let conversationId: string;

    let showDropdown = false;
    
    const iconMap : Record<string, ConstructorOfATypedSvelteComponent>= {
        'brain': Brain,
        'file-text': FileText,
        'compass': Compass,
        'check-square': CheckSquare,
        'cloud': Cloud,
        'book': Book,
        'library': Library,
        'feather': Feather,
        'thought-bubble': MessageCircleHeart,
        'heart': Heart,
    };

    function getPipelineIcon(iconType: string) {
        return iconMap[iconType];
    }

    async function handlePipelineSelect(pipelineType: PipelineType) {
        showDropdown = false;

        try {
            if (!latestPersonaId) {
                alert('No persona ID found in conversation');
                await goto('/chat-matrix');
                return;
            }

            // Update pipeline type and form data
            pipelineStore.setPipelineType(pipelineType);
            pipelineStore.updateFormData({
                user_id: latestPersonaId,
                persona_id: latestPersonaId,
                conversation_id: conversationId,
                mood: $pipelineStore.formData.mood || 'Delighted',
                top_n: $pipelineStore.formData.top_n,
                guidance: $pipelineStore.formData.guidance,
                query_text: $pipelineStore.formData.query_text,
            });

            await taskStore.submitTask(pipelineType, $pipelineStore.formData);
            await goto('/pipeline-tasks');
        } catch (e) {
            alert(`Failed to start ${pipelineType} pipeline`);
        }
    }
</script>

<div class="pipeline-triggers">
    <button 
        class="trigger-button"
        on:click={() => showDropdown = !showDropdown}
        aria-expanded={showDropdown}
        aria-haspopup="true"
    >
        Start Pipeline
    </button>

    {#if showDropdown}
        <div 
            class="pipeline-dropdown"
            role="menu"
        >
            {#each pipelineStore.getPipelineInfo() as pipeline}
                <button
                    class="pipeline-option"
                    on:click={() => handlePipelineSelect(pipeline.type)}
                    role="menuitem"
                >
                    <svelte:component this={iconMap[pipeline.icon]} size={16} />
                    <div class="pipeline-info">
                        <span class="pipeline-name">{pipeline.name}</span>
                        <span class="pipeline-description">{pipeline.description}</span>
                    </div>
                </button>
            {/each}
        </div>
    {/if}
</div>

<style>
    .pipeline-triggers {
        position: relative;
        display: inline-block;
    }

    .trigger-button {
        background-color: #2196f3;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.375rem;
        cursor: pointer;
        font-size: 0.875rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: background-color 0.2s;
    }

    .trigger-button:hover {
        background-color: #1976d2;
    }

    .pipeline-dropdown {
        position: absolute;
        top: 100%;
        left: 0;
        z-index: 50;
        min-width: 280px;
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        border: 1px solid #e5e7eb;
        margin-top: 0.5rem;
        padding: 0.5rem;
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .pipeline-option {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        border: none;
        background: none;
        width: 100%;
        text-align: left;
        cursor: pointer;
        border-radius: 0.375rem;
        transition: background-color 0.2s;
    }

    .pipeline-option:hover {
        background-color: #f3f4f6;
    }

    .pipeline-info {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .pipeline-name {
        font-weight: 500;
        color: #111827;
    }

    .pipeline-description {
        font-size: 0.75rem;
        color: #6b7280;
    }
</style>