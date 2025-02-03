<!-- src/lib/components/DocumentTypeFilter.svelte -->
<script lang="ts">
    import type { ChatMessage } from '../types';
    import { createEventDispatcher } from 'svelte';
    
    const dispatch = createEventDispatcher<{
        filterChange: string | null
    }>();
    
    export let messages: ChatMessage[] = [];
    export let activeDocumentType: string | null = null;
    
    // Calculate document type counts
    $: documentTypeCounts = messages.reduce((acc, message) => {
        acc[message.document_type] = (acc[message.document_type] || 0) + 1;
        return acc;
    }, {} as Record<string, number>);
    
    // Sort document types alphabetically
    $: documentTypes = Object.keys(documentTypeCounts).sort();
    
    function handleFilterClick(documentType: string | null) {
        activeDocumentType = activeDocumentType === documentType ? null : documentType;
        dispatch('filterChange', activeDocumentType);
    }
    
    function getBadgeColor(documentType: string): string {
        const colors: Record<string, string> = {
            // Blues
            conversation: '#e6f3ff',
            analysis: '#dbeafe',
            
            // Greens
            chore: '#dcfce7',
            codex: '#d1fae5',
            
            // Purples
            pondering: '#f3e8ff',
            reflection: '#ede9fe',
            journal: '#fef3c7',
            highlight: '#fff7ed',
            'battle-score': '#ffe4e6',
            arena: '#fce7f3',
            daydream: '#e0f2fe',
            poem: '#dbeafe',
            summary: '#e0e7ff',
            adventure: '#fff1f2',
            brainstorm: '#fef2f2',
            'ner-task': '#ccfbf1',
            'silver-band': '#cffafe',
            step: '#f5f5f4',
            report: '#f3f4f6',
            default: '#f3f4f6'
        };
        return colors[documentType.toLowerCase()] || colors.default;
    }
    
    function getTextColor(documentType: string): string {
        const colors: Record<string, string> = {
            conversation: '#1d4ed8',
            analysis: '#2563eb',
            chore: '#16a34a',
            codex: '#059669',
            pondering: '#7e22ce',
            reflection: '#6d28d9',
            journal: '#b45309',
            highlight: '#c2410c',
            'battle-score': '#be123c',
            arena: '#be185d',
            daydream: '#0284c7',
            poem: '#2563eb',
            summary: '#4338ca',
            adventure: '#be123c',
            brainstorm: '#dc2626',
            'ner-task': '#0f766e',
            'silver-band': '#0e7490',
            step: '#44403c',
            report: '#4b5563',
            default: '#666666'
        };
        return colors[documentType.toLowerCase()] || colors.default;
    }

    function getDisplayName(documentType: string): string {
        return documentType
            .split('-')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
</script>

<div class="filter-container">
    <button
        class="filter-badge"
        class:active={activeDocumentType === null}
        style="background-color: #f3f4f6; color: #666666"
        on:click={() => handleFilterClick(null)}
    >
        All: {messages.length}
    </button>
    
    {#each documentTypes as documentType}
        <button
            class="filter-badge"
            class:active={activeDocumentType === documentType}
            style="background-color: {getBadgeColor(documentType)}; color: {getTextColor(documentType)}"
            on:click={() => handleFilterClick(documentType)}
        >
            {getDisplayName(documentType)}: {documentTypeCounts[documentType]}
        </button>
    {/each}
</div>

<style>
    .filter-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding: 0.75rem;
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .filter-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.375rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        border: none;
        cursor: pointer;
        transition: all 0.2s ease;
        line-height: 1;
    }
    
    .filter-badge:hover {
        filter: brightness(0.95);
        transform: translateY(-1px);
    }
    
    .filter-badge.active {
        box-shadow: 0 0 0 2px currentColor;
        transform: translateY(-1px);
    }
    
    .filter-badge:active {
        transform: translateY(0px);
    }
</style>