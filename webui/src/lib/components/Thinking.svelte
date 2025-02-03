
<script lang="ts">
    import { chatStore } from "$lib/store/chatStore";

</script>

{#if $chatStore.loading}
    <div class="loading-indicator">Thinking...</div>
{/if}

<style>
    .loading-indicator {
        font-weight: bold;
        display: flex;
        align-items: center;
        height: 38px;
    }

    @keyframes ellipsis {
        0% { content: '.'; }
        33% { content: '..'; }
        66% { content: '...'; }
    }

    .loading-indicator::after {
        content: '.';
        display: inline-block;
        animation: ellipsis 1.5s infinite;
    }
</style>