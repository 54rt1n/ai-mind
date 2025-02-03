<!-- src/lib/components/EmotionFilter.svelte -->
<script lang="ts">
    import type { ChatMessage } from '../types';
    import { createEventDispatcher } from 'svelte';
    
    const dispatch = createEventDispatcher<{
        filterChange: string | null
    }>();
    
    export let messages: ChatMessage[] = [];
    export let activeEmotion: string | null = null;
    
    // Calculate emotion counts across all emotion fields
    $: emotionCounts = messages.reduce((acc, message) => {
        // Gather all emotions from a-d fields
        const emotions = [
        message.emotion_a?.split(' ') || [],
        message.emotion_b?.split(' ') || [],
        message.emotion_c?.split(' ') || [],
        message.emotion_d?.split(' ') || []
    ].flat().filter(Boolean); // Flatten array and remove empty values
    
        // Count each emotion
        emotions.forEach(emotion => {
            if (emotion) {
                acc[emotion] = (acc[emotion] || 0) + 1;
            }
        });
        
        return acc;
    }, {} as Record<string, number>);
    
    // Sort emotions alphabetically
    $: emotions = Object.keys(emotionCounts).sort();
    
    function handleFilterClick(emotion: string | null) {
        activeEmotion = activeEmotion === emotion ? null : emotion;
        dispatch('filterChange', activeEmotion);
    }
    
    function getBadgeColor(emotion: string): string {
        // Emotional color mapping - you can expand this
        const colors: Record<string, string> = {
            // Positive emotions - warm colors
            'happy': '#fff7ed',
            'joyful': '#ffedd5',
            'excited': '#fee2e2',
            'content': '#fef3c7',
            
            // Negative emotions - cool colors
            'sad': '#eff6ff',
            'angry': '#fef2f2',
            'frustrated': '#fff1f2',
            'anxious': '#f3e8ff',
            
            // Neutral emotions - gray tones
            'neutral': '#f3f4f6',
            'calm': '#f0fdf4',
            'thoughtful': '#f0f9ff',
            
            // Default color for unmapped emotions
            'default': '#f3f4f6'
        };
        return colors[emotion.toLowerCase()] || colors.default;
    }
    
    function getTextColor(emotion: string): string {
        // Matching text colors for the backgrounds
        const colors: Record<string, string> = {
            'happy': '#9a3412',
            'joyful': '#c2410c',
            'excited': '#dc2626',
            'content': '#b45309',
            
            'sad': '#1d4ed8',
            'angry': '#dc2626',
            'frustrated': '#be123c',
            'anxious': '#7e22ce',
            
            'neutral': '#4b5563',
            'calm': '#166534',
            'thoughtful': '#0369a1',
            
            'default': '#4b5563'
        };
        return colors[emotion.toLowerCase()] || colors.default;
    }
</script>

<div class="filter-container">
    <button
        class="filter-badge"
        class:active={activeEmotion === null}
        style="background-color: #f3f4f6; color: #666666"
        on:click={() => handleFilterClick(null)}
    >
        All Emotions ({messages.length})
    </button>
    
    {#each emotions as emotion}
        <button
            class="filter-badge"
            class:active={activeEmotion === emotion}
            style="background-color: {getBadgeColor(emotion)}; color: {getTextColor(emotion)}"
            on:click={() => handleFilterClick(emotion)}
        >
            {emotion}: {emotionCounts[emotion]}
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