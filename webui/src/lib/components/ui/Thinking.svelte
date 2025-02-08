<script lang="ts">
    export let loading = false;
</script>

{#if loading}
    <div class="loading-container">
        <div class="loading-indicator">Thinking...</div>
    </div>
{/if}

<style>
    .loading-container {
        display: flex;
        align-items: center;
    }

    .loading-indicator {
        font-weight: bold;
        display: flex;
        align-items: center;
        height: 38px;
    }

    @keyframes ellipsis {
        0% {
            content: ".";
        }
        33% {
            content: "..";
        }
        66% {
            content: "...";
        }
    }

    .loading-indicator::after {
        content: ".";
        display: inline-block;
        animation: ellipsis 1.5s infinite;
    }
</style>
