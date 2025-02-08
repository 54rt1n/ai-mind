<!-- src/lib/components/toolbox/DragDropContext.svelte -->
<script lang="ts">
    import { toolboxStore, type ActionCard } from '$lib/store/toolboxStore';
    import { onMount } from 'svelte';

    let draggedCard: ActionCard | null = null;
    let dragOverCard: ActionCard | null = null;
    let cards: ActionCard[] = [];

    $: cards = $toolboxStore.cards;

    function handleDragStart(event: DragEvent, card: ActionCard) {
        draggedCard = card;
        if (event.dataTransfer) {
            event.dataTransfer.effectAllowed = 'move';
            event.dataTransfer.setData('text/plain', card.id);
        }
    }

    function handleDragOver(event: DragEvent, card: ActionCard) {
        event.preventDefault();
        if (draggedCard && draggedCard.id !== card.id) {
            dragOverCard = card;
        }
    }

    function handleDragEnd() {
        if (draggedCard && dragOverCard) {
            const newPosition = cards.findIndex(card => card.id === dragOverCard?.id);
            toolboxStore.updateCardPosition(draggedCard.id, newPosition);
        }
        draggedCard = null;
        dragOverCard = null;
    }

    function handleDragLeave() {
        dragOverCard = null;
    }
</script>

<div class="drag-drop-context">
    {#each cards as card (card.id)}
        <div
            class="draggable-item"
            class:dragging={draggedCard?.id === card.id}
            class:drag-over={dragOverCard?.id === card.id}
            draggable="true"
            on:dragstart={(e) => handleDragStart(e, card)}
            on:dragover={(e) => handleDragOver(e, card)}
            on:dragleave={handleDragLeave}
            on:dragend={handleDragEnd}
        >
            <slot {card} />
        </div>
    {/each}
</div>

<style>
    .drag-drop-context {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .draggable-item {
        cursor: move;
    }

    .draggable-item.dragging {
        opacity: 0.5;
    }

    .draggable-item.drag-over {
        position: relative;
    }

    .draggable-item.drag-over::before {
        content: '';
        position: absolute;
        top: -0.5rem;
        left: 0;
        right: 0;
        height: 2px;
        background: #4CAF50;
        border-radius: 1px;
    }
</style> 