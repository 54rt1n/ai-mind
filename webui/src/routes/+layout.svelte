<!-- src/routes/+layout.svelte -->
<script lang="ts">
    import { onMount } from 'svelte';
    import Header from '$lib/components/Header.svelte';
    import { modelStore } from '$lib/store/modelStore';
    import '../app.css';

    onMount(() => {
        modelStore.initialize();
    });
</script>

<Header />

<main class="main-container">
    <slot />
</main>

<style>
    .main-container {
        max-width: 1200px;
        margin: 20px auto;
        padding: 0 20px;
    }
</style>