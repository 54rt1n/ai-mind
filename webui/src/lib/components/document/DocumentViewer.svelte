<!-- src/lib/components/DocumentViewer.svelte -->
<script lang="ts">
    import { configStore } from '$lib/store/configStore';
    import { clipboardStore } from '$lib/store/clipboardStore';
    import { slide } from 'svelte/transition';
    import { api } from '$lib/api';
    import { WrapText, AlignJustify, Copy, FolderMinus, X as XIco } from 'lucide-svelte';
    import "$lib/../styles/notification.css";
    import NotificationToast from '../ui/NotificationToast.svelte';
    import type { NotificationParams } from '$lib/types';

    export let show = false;
    export let onClose: () => void;

    let wrapLines = true;  // Default to wrapped
    let notificationProps : NotificationParams = {
        show: false,
        notificationType: 'info',
        message: '',
        loading: false
    };

    function toggleWrap() {
        wrapLines = !wrapLines;
    }
    let content = '';
    let loading = false;
    let error: string | null = null;
    let viewerElement: HTMLDivElement;

    $: if ($configStore.selectedDocument && show) {
        loadDocument($configStore.selectedDocument.name);
    }

    async function loadDocument(documentName: string) {
        loading = true;
        error = null;
        try {
            content = await api.getDocumentContents(documentName);
        } catch (e) {
            error = "Failed to load document contents";
            console.error(e);
        } finally {
            loading = false;
        }
    }

    // Format line numbers to have consistent width
    function formatLineNumber(num: number, total: number): string {
        const width = total.toString().length;
        return num.toString().padStart(width, ' ');
    }

    function copyToClipStore() {
        clipboardStore.setContent(content);
        notificationProps = {
            show: true,
            notificationType: 'copy',
            message: 'Copied to local clip workspace!',
            loading: false
        }
    }

    function documentForget() {
        configStore.documentSelect(null);
        onClose();
    }

    $: lines = content.split('\n');
    $: maxLines = lines.length;
</script>


{#if show && $configStore.selectedDocument}
    <div 
        class="viewer-overlay"
        transition:slide={{ duration: 300 }}
        bind:this={viewerElement}
    >
        <div class="viewer-header">
            <div class="viewer-title">
                <span class="filename">{$configStore.selectedDocument.name}</span>
                <span class="line-count">({lines.length} lines)</span>
            </div>
            <div class="viewer-controls">
                <button
                    class="control-button forget"
                    on:click={documentForget}
                    title="Deselect Document"
                >
                    <FolderMinus size={16} />
                </button>
                <button 
                    class="control-button copy"
                    on:click={copyToClipStore}
                    title="Copy to Clipboard"
                >
                   <Copy size={16} />
                </button>
                <button 
                    class="control-button" 
                    class:active={wrapLines}
                    on:click={toggleWrap}
                    title={wrapLines ? "Disable line wrap" : "Enable line wrap"}
                >
                    {#if wrapLines}
                        <WrapText size={16} />
                    {:else}
                        <AlignJustify size={16} />
                    {/if}
                </button>
                <button class="close-button" on:click={onClose}>
                    <XIco size={16} />
                </button>
            </div>
        </div>

        <div class="viewer-content">
            {#if loading}
                <div class="loading">Loading document contents...</div>
            {:else if error}
                <div class="error">{error}</div>
            {:else}
                <pre class="code-container" class:wrap={wrapLines}><code>{#each lines as line, i}
<span class="line">
<span class="line-number">{formatLineNumber(i + 1, maxLines)}</span>
<span class="line-content">{line}</span>
</span>{/each}</code></pre>
            {/if}
        </div>
    </div>
{/if}

<NotificationToast {...notificationProps} />

<style>
    .viewer-overlay {
        position: fixed;
        top: 0;
        left: 0;
        bottom: 0;
        width: 50%;
        max-width: 800px;
        background: #f8f9fa;
        box-shadow: 4px 0 15px rgba(0, 0, 0, 0.1);
        z-index: 100;
        display: flex;
        flex-direction: column;
    }

    .viewer-header {
        background: #ffffff;
        padding: 1rem;
        border-bottom: 1px solid #e5e7eb;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-shrink: 0;
    }

    .viewer-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .filename {
        font-weight: 600;
        color: #1f2937;
    }

    .line-count {
        color: #6b7280;
        font-size: 0.875rem;
    }

    .close-button {
        background: none;
        border: none;
        color: #6b7280;
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 0.375rem;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .close-button:hover {
        background-color: #f3f4f6;
        color: #1f2937;
    }

    .viewer-content {
        flex: 1;
        overflow-y: auto;
        padding: 1rem;
    }

    .code-container {
        margin: 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.875rem;
        line-height: 1.5;
        tab-size: 4;
    }

    .line {
        display: flex;
        white-space: pre;
    }

    .line:hover {
        background-color: #e5e7eb;
    }

    .line-number {
        color: #6b7280;
        padding-right: 1rem;
        user-select: none;
        text-align: right;
        min-width: 3rem;
    }

    .line-content {
        flex: 1;
        padding-left: 0.5rem;
    }

    .loading {
        padding: 1rem;
        color: #6b7280;
        text-align: center;
    }

    .error {
        padding: 1rem;
        color: #dc2626;
        background: #fee2e2;
        border-radius: 0.375rem;
    }

    .viewer-controls {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .control-button {
        background: none;
        border: none;
        color: #6b7280;
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 0.375rem;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .control-button:hover {
        background-color: #f3f4f6;
        color: #1f2937;
    }

    .control-button.active {
        color: #2196F3;
        background-color: rgba(33, 150, 243, 0.1);
    }

    .control-button.copy {
        color: #10b981;
    }

    .control-button.copy:hover {
        color: #059669;
    }

    .control-button.forget {
        color: #dc2626;
    }

    .control-button.forget:hover {
        color: #b91c1c;
    }

    .code-container {
        margin: 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.875rem;
        line-height: 1.5;
        tab-size: 4;
    }

    .code-container.wrap .line {
        white-space: pre-wrap;
        word-wrap: break-word;
    }

    .code-container:not(.wrap) .line {
        white-space: pre;
    }

    .line {
        display: flex;
        min-width: min-content;
    }

    .line:hover {
        background-color: #e5e7eb;
    }

    .line-number {
        color: #6b7280;
        padding-right: 1rem;
        user-select: none;
        text-align: right;
        min-width: 3rem;
        flex-shrink: 0;
        position: sticky;
        left: 0;
        background: inherit;
    }

    .line-content {
        flex: 1;
        padding-left: 0.5rem;
    }

    /* Add horizontal scrolling for unwrapped lines */
    .viewer-content {
        flex: 1;
        overflow: auto;
        padding: 1rem;
    }

    /* Make line numbers sticky when scrolling horizontally */
    .viewer-content:not(.wrap) {
        overflow-x: auto;
    }

    @media (max-width: 768px) {
        .viewer-overlay {
            width: 100%;
            max-width: none;
        }

        .line-number {
            min-width: 2.5rem;
            padding-right: 0.5rem;
        }
    }
</style>