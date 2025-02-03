<!-- src/lib/components/BranchFilter.svelte -->
<script lang="ts">
    import type { ChatMessage } from '../types';
    import { createEventDispatcher } from 'svelte';
    
    const dispatch = createEventDispatcher<{
        filterChange: number | null
    }>();
    
    export let messages: ChatMessage[] = [];
    export let activeBranch: number | null = null;
    
    // Calculate branch counts
    $: branchCounts = messages.reduce((acc, message) => {
        if (typeof message.branch === 'number') {
            acc[message.branch] = (acc[message.branch] || 0) + 1;
        }
        return acc;
    }, {} as Record<number, number>);
    
    // Sort branches numerically
    $: branches = Object.keys(branchCounts)
        .map(Number)
        .sort((a, b) => a - b);
    
    function handleFilterClick(branch: number | null) {
        activeBranch = activeBranch === branch ? null : branch;
        dispatch('filterChange', activeBranch);
    }
    
    function getBadgeColor(branch: number): string {
        const colors = [
            '#e6f3ff', // Light Blue
            '#dcfce7', // Light Green
            '#f3e8ff', // Light Purple
            '#fef3c7', // Light Yellow
            '#ffe4e6', // Light Red
            '#e0f2fe', // Light Sky Blue
            '#fff1f2', // Light Rose
            '#ccfbf1', // Light Teal
            '#f5f5f4', // Light Gray
        ];
        return colors[branch % colors.length];
    }
    
    function getTextColor(branch: number): string {
        const colors = [
            '#1d4ed8', // Blue
            '#16a34a', // Green
            '#7e22ce', // Purple
            '#b45309', // Yellow
            '#be123c', // Red
            '#0284c7', // Sky Blue
            '#be123c', // Rose
            '#0f766e', // Teal
            '#44403c', // Gray
        ];
        return colors[branch % colors.length];
    }
</script>

<div class="filter-container">
    <button
        class="filter-badge"
        class:active={activeBranch === null}
        style="background-color: #f3f4f6; color: #666666"
        on:click={() => handleFilterClick(null)}
    >
        All Branches: {messages.length}
    </button>
    
    {#each branches as branch}
        <button
            class="filter-badge"
            class:active={activeBranch === branch}
            style="background-color: {getBadgeColor(branch)}; color: {getTextColor(branch)}"
            on:click={() => handleFilterClick(branch)}
        >
            Branch {branch}: {branchCounts[branch]}
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