<!-- src/lib/components/clipboard/ClipboardPanel.svelte -->
<script lang="ts">
    import { clipboardStore } from "$lib/store/clipboardStore";
    import { configStore } from "$lib/store/configStore";
    import { Copy, CircleX } from "lucide-svelte";
    import "$lib/../styles/meta-panels.css";
    import type { NotificationParams } from "$lib/types";
    import NotificationToast from "$lib/components/ui/NotificationToast.svelte";

    let extractedContent = "";

    let notificationProps: NotificationParams = {
        show: false,
        notificationType: "info",
        message: "",
        loading: false,
    };

    function handlePatternChange(event: Event) {
        const input = event.target as HTMLInputElement;
        clipboardStore.setPattern(input.value);
        extractedContent = clipboardStore.extract($clipboardStore).trim();
    }

    function handleClipboardChange(event: Event) {
        const input = event.target as HTMLInputElement;
        clipboardStore.setContent(input.value);
        extractedContent = clipboardStore.extract($clipboardStore).trim();
    }

    // Also update extracted content when we get new clipboard content
    $: if ($clipboardStore.content) {
        extractedContent = clipboardStore.extract($clipboardStore).trim();
    }

    async function copyToClipboard(content: string) {
        try {
            try {
                await navigator.clipboard.writeText(content);
            } catch (e) {
                const textarea = document.createElement("textarea");
                textarea.value = content;
                textarea.style.position = "fixed";
                document.body.appendChild(textarea);
                textarea.select();

                try {
                    document.execCommand("copy");
                    document.body.removeChild(textarea);
                } catch (execError) {
                    document.body.removeChild(textarea);
                    throw execError;
                }
            }

            notificationProps = {
                show: true,
                notificationType: "copy",
                message: "Copied to clipboard!",
                loading: false,
            };
        } catch (error) {
            notificationProps = {
                show: true,
                notificationType: "error",
                message: "Copy failed",
                loading: false,
            };
            console.error("Copy failed:", error);
        }
    }

    function clearClipboard() {
        clipboardStore.clear();
        extractedContent = "";
    }
</script>

<div class="meta-panel">
    <button
        class="meta-panel-header collapsible"
        on:click={() =>
            configStore.updateField(
                "showClipboard",
                !$configStore.showClipboard,
            )}
    >
        <span>Clipboard</span>
        <span class="toggle-icon"
            >{$configStore.showClipboard ? "▼" : "▶"}</span
        >
    </button>

    {#if $configStore.showClipboard}
        <div class="meta-panel-content">
            <div class="clipboard-section">
                <div class="clipboard-controls">
                    <input
                        type="text"
                        placeholder="Enter extraction pattern (regex)"
                        value={$clipboardStore.pattern}
                        on:input={handlePatternChange}
                        class="pattern-input"
                    />
                    <div class="button-group">
                        <button
                            class="control-button copy"
                            on:click={() => copyToClipboard(extractedContent)}
                            disabled={!$clipboardStore.content}
                            title="Copy to clipboard"
                        >
                            <Copy size={16} />
                        </button>
                        <button
                            class="control-button clear"
                            on:click={clearClipboard}
                            disabled={!$clipboardStore.content}
                            title="Clear clipboard"
                        >
                            <CircleX size={16} />
                        </button>
                    </div>
                </div>

                <div class="content-preview">
                    <textarea
                        bind:value={$clipboardStore.content}
                        placeholder="Content will appear here..."
                        rows="5"
                        on:input={handleClipboardChange}
                    />
                </div>

                {#if $clipboardStore.pattern}
                    <div class="extracted-preview">
                        <h4>Extracted Content:</h4>
                        <pre>{extractedContent}</pre>
                    </div>
                {/if}
            </div>
        </div>
    {/if}
</div>

<NotificationToast {...notificationProps} />

<style>
    .clipboard-section {
        width: 100%;
        min-width: 0;
    }

    .clipboard-controls {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    .pattern-input {
        flex: 1;
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        font-family: monospace;
    }

    .button-group {
        display: flex;
        gap: 0.25rem;
        height: 32px;
    }

    .control-button {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 32px;
        width: 32px;
        padding: 0;
        border: none;
        border-radius: 0.375rem;
        cursor: pointer;
        transition: all 0.2s;
        color: white;
        flex-shrink: 0;
    }

    .control-button.copy {
        background-color: #10b981;
    }

    .control-button.copy:hover:not(:disabled) {
        background-color: #059669;
    }

    .control-button.clear {
        background-color: #ef4444;
    }

    .control-button.clear:hover:not(:disabled) {
        background-color: #dc2626;
    }

    .control-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .content-preview {
        margin-bottom: 1rem;
    }

    .content-preview textarea {
        width: 100%;
        padding: 0.75rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        font-family: monospace;
        resize: vertical;
        background-color: #f9fafb;
    }

    .extracted-preview {
        padding: 0.75rem;
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        max-height: 250px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }

    .extracted-preview h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.875rem;
        color: #4b5563;
    }

    .extracted-preview pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: monospace;
        font-size: 0.875rem;
        color: #1f2937;
    }
</style>
