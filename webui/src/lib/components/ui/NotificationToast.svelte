<script lang="ts">
    import { fly } from 'svelte/transition';
    import { Loader, ClipboardCheck, AlertCircle, Info, AlertTriangle, Star } from 'lucide-svelte';
    import type { NotificationType } from '$lib/types';
    import "$lib/../styles/notification.css";

    export let notificationType: NotificationType = 'info';
    export let message: string;
    export let loading: boolean = false;
    export let duration: number = 2000;
    export let show: boolean = false;

    $: if (show && !loading && duration > 0) {
        setTimeout(() => {
            show = false;
        }, duration);
    }
</script>

{#if show}
    <div class="notification {notificationType}" transition:fly={{ y: 20, duration: 300 }}>
        {#if loading}
            <Loader size={16} class="spin" />
        {:else if notificationType === 'success'}
            <Star size={16} />
        {:else if notificationType === 'error'}
            <AlertCircle size={16} />
        {:else if notificationType === 'info'}
            <Info size={16} />
        {:else if notificationType === 'copy'}
            <ClipboardCheck size={16} />
        {:else if notificationType === 'warning'}
            <AlertTriangle size={16} />
        {/if}
        {message}
    </div>
{/if}

<style>
    .notification {
        position: fixed;
        bottom: 1rem;
        right: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        border-radius: 0.375rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        z-index: 1000;
    }

    .notification.success {
        background-color: #10b981;
        color: white;
    }

    .notification.loading {
        background-color: #6366f1;
        color: white;
    }

    .notification.info {
        background-color: #3b82f6;
        color: white;
    }

    .notification.warning {
        background-color: #f59e0b;
        color: white;
    }

    .notification.copy {
        background-color: #10b981;
        color: white;
    }

    .notification.error {
        background-color: #ef4444;
        color: white;
    }

    .spin {
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
</style>
