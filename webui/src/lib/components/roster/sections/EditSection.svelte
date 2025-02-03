<!-- src/lib/components/roster/sections/EditSection.svelte -->
<script lang="ts">
    export let title: string;
</script>

<div class="form-section">
    <div class="section-header">
        <h3>{title}</h3>
    </div>
    <slot />
</div>

<style>
    .form-section {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1.5rem;
    }

    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }

    h3 {
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
        color: #374151;
    }

    .cancel-button {
        padding: 0.5rem 1rem;
        border: 1px solid #e5e7eb;
        background: white;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        font-weight: 500;
        color: #374151;
        cursor: pointer;
        transition: all 0.2s;
    }

    .cancel-button:hover {
        background: #f3f4f6;
    }
</style> 