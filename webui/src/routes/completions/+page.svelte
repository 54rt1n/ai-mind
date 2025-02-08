<!-- src/routes/completions/+page.svelte -->
<script lang="ts">
    import AdvancedSettingsPanel from '$lib/components/model/AdvancedSettingsPanel.svelte';
    import CompletionPanel from '$lib/components/completion/CompletionPanel.svelte';
    import ClipboardPanel from '$lib/components/clipboard/ClipboardPanel.svelte';
</script>

<svelte:head>
    <title>Text Completions | MindAI</title>
</svelte:head>

<main class="completions-page">
    <h1>Text Completions</h1>
    <div class="panels-container">
        <ClipboardPanel />
        <CompletionPanel />
    </div>
</main>

<style>
    .completions-page {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }

    h1 {
        margin-bottom: 2rem;
        font-size: 2rem;
        font-weight: 600;
        color: #1F2937;
    }

    .panels-container {
        display: flex;
        flex-direction: column;
        gap: 2rem;
    }

    @media (max-width: 768px) {
        .completions-page {
            padding: 1rem;
        }
    }
</style>