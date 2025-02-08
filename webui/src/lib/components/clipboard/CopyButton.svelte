<!-- src/lib/components/CopyButton.svelte -->
<script lang="ts">
    import { clipboardStore } from '$lib/store/clipboardStore';
    import { Copy } from 'lucide-svelte';

    export let content: string;

    function handleCopy() {
        clipboardStore.setContent(content);
    }
</script>

<button 
    class="copy-button" 
    on:click|stopPropagation={handleCopy}
    title="Copy to workspace"
>
    <Copy size={16} />
</button>

<style>
    .copy-button {
        opacity: 0;
        background: none;
        border: none;
        padding: 0.5rem;
        border-radius: 0.375rem;
        cursor: pointer;
        color: #6b7280;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .copy-button:hover {
        background-color: rgba(0, 0, 0, 0.05);
        color: #374151;
    }

    :global(.message:hover) .copy-button {
        opacity: 1;
    }
</style>