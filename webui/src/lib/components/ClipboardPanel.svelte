<!-- src/lib/components/ClipboardPanel.svelte -->
<script lang="ts">
    import { clipboardStore } from "$lib/store/clipboardStore";
    import { configStore } from "$lib/store/configStore";
    import { workspaceStore, WorkspaceCategory } from "$lib/store/workspaceStore";
    import {
        Copy,
        Trash2,
        ArrowRightCircle,
        ArrowUpCircle, CircleX
    } from "lucide-svelte";
    import "$lib/../styles/meta-panels.css";
    import type { NotificationParams } from "$lib/types";
    import NotificationToast from "$lib/components/NotificationToast.svelte";

    let extractedContent = "";
    let uploadFileName = "";
    $: defaultFilename = `${$workspaceStore.name}.${new Date().toISOString().split("T")[0]}.txt`;

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

    function handleWorkspaceChange(event: Event) {
        const input = event.target as HTMLTextAreaElement;
        workspaceStore.setContent(input.value);
    }

    function handleIncludeInMessagesChange(event: Event) {
        const checkbox = event.target as HTMLInputElement;
        clipboardStore.setIncludeInMessages(checkbox.checked);
    }

    function handleDeleteWorkspace() {
        if (
            confirm(
                `Are you sure you want to delete the ${$workspaceStore.name} workspace content?`,
            )
        ) {
            workspaceStore.deleteCurrent();
        }
    }

    // Also update extracted content when we get new clipboard content
    $: if ($clipboardStore.content) {
        extractedContent = clipboardStore.extract($clipboardStore).trim();
    }

    async function handleSaveRemoteCopy() {
        notificationProps = {
            show: true,
            notificationType: "loading",
            message: "Saving...",
            loading: true,
        };

        try {
            const fileName = uploadFileName || defaultFilename;
            await workspaceStore.saveRemoteCopy(fileName);
            notificationProps = {
                show: true,
                notificationType: "success",
                message: `File upload complete! ${fileName} saved.`,
                loading: false,
                duration: 5000,
            };
        } catch (error) {
            if (error instanceof Error) {
                notificationProps = {
                    show: true,
                    notificationType: "error",
                    message: "Save failed: " + error.message,
                    loading: false,
                    duration: 5000,
                };
            } else {
                notificationProps = {
                    show: true,
                    notificationType: "error",
                    message: "Save failed: Unknown error",
                    loading: false,
                    duration: 5000,
                };
            }
        }
    }

    async function copyToClipboard(content: string) {
        try {
            try {
                await navigator.clipboard.writeText(content);
            } catch (e) {
                const textarea = document.createElement("textarea");
                textarea.value = extractedContent;
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

    function appendToWorkspace() {
        if (extractedContent) {
            const currentContent = $workspaceStore.content;
            workspaceStore.setContent(currentContent + "\n" + extractedContent);
        }
    }

    $: workspaceSelected = $workspaceStore.name !== WorkspaceCategory.None;
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
        <span>Workspace & Clipboard</span>
        <span class="toggle-icon"
            >{$configStore.showClipboard ? "▼" : "▶"}</span
        >
    </button>

    {#if $configStore.showClipboard}
        <div class="meta-panel-content">
            <div class="grid-container">
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
                                on:click={() =>
                                    copyToClipboard(extractedContent)}
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
                            <button
                                class="control-button append"
                                on:click={appendToWorkspace}
                                disabled={!extractedContent}
                                title="Append to workspace"
                            >
                                <ArrowRightCircle size={16} />
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

                <div class="workspace-section">
                    <div class="workspace-tabs">
                        {#each workspaceStore
                            .getCategories()
                            .filter((cat) => cat !== WorkspaceCategory.None) as category}
                            <button
                                class="tab-button"
                                class:active={$workspaceStore.name === category}
                                data-category={category}
                                on:click={() =>
                                    workspaceStore.changeCategory(category)}
                            >
                                {category}
                            </button>
                        {/each}
                    </div>
                    <textarea
                        class="workspace-textarea"
                        bind:value={$workspaceStore.content}
                        on:input={handleWorkspaceChange}
                        placeholder="Use this workspace for notes, drafts, or any text you want to keep handy..."
                    />
                    <div>Word Count: {$workspaceStore.wordCount}</div>
                </div>
            </div>
            <div class="workspace-include">
                <div class="workspace-actions">
                    <label class="include-label">
                        <input
                            type="checkbox"
                            checked={$clipboardStore.includeInMessages}
                            on:change={handleIncludeInMessagesChange}
                        />
                        Include workspace in messages
                    </label>
                </div>
                <div class="workspace-actions">
                    <input
                        type="text"
                        class="filename-input"
                        bind:value={uploadFileName}
                        placeholder={defaultFilename}
                    />
                    <div class="button-group">
                        <button
                            class="workspace-button upload"
                            on:click={handleSaveRemoteCopy}
                            disabled={!workspaceSelected}
                            title="Save workspace content"
                        >
                            <ArrowUpCircle size={16} />
                        </button>
                        <button
                            class="workspace-button copy"
                            on:click={() =>
                                copyToClipboard($workspaceStore.content)}
                            disabled={!workspaceSelected}
                            title="Copy workspace content"
                        >
                            <Copy size={16} />
                        </button>
                        <button
                            class="workspace-button delete"
                            on:click={handleDeleteWorkspace}
                            disabled={!workspaceSelected}
                            title="Delete workspace content"
                        >
                            <Trash2 size={16} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    {/if}
</div>

<NotificationToast {...notificationProps} />

<style>
    .grid-container {
        display: grid;
        /* Make each section take up equal space */
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }

    .clipboard-section,
    .workspace-section {
        /* Ensure each section stays within its column */
        width: 100%;
        min-width: 0; /* Prevent content from forcing expansion */
    }

    .workspace-textarea {
        flex: 1;
        min-height: 400px;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
            monospace;
        font-size: 0.875rem;
        resize: none;
        background-color: #f9fafb;
    }

    .workspace-textarea:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        background-color: white;
    }

    @media (max-width: 768px) {
        .grid-container {
            grid-template-columns: 1fr;
        }
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

    .control-button,
    .workspace-button {
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

    /* Clipboard control colors */
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

    .control-button.append {
        background-color: #6366f1;
    }
    .control-button.append:hover:not(:disabled) {
        background-color: #4f46e5;
    }

    /* Workspace action colors */
    .workspace-button.copy {
        background-color: #10b981;
    }
    .workspace-button.copy:hover {
        background-color: #059669;
    }

    .workspace-button.delete {
        background-color: #ef4444;
    }
    .workspace-button.delete:hover {
        background-color: #dc2626;
    }

    .workspace-button.upload {
        background-color: #f97316;
    }
    .workspace-button.upload:hover {
        background-color: #ea580c;
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

    .workspace-include {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }

    .include-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #666;
        font-size: 0.875rem;
    }

    .workspace-tabs {
        display: flex;
        width: 100%;
        margin-bottom: -1px;
    }

    .tab-button {
        flex: 1;
        padding: 0.25rem;
        border: 1px solid #e5e7eb;
        border-bottom: none;
        border-radius: 0.5rem 0.5rem 0 0;
        font-size: 0.875rem;
        cursor: pointer;
        transition: all 0.2s;
        opacity: 0.7;
    }

    .tab-button.active {
        opacity: 1;
        border-bottom: 3px solid white;
    }

    .tab-button[data-category="Schedule"] {
        background-color: #2563eb;
        color: white;
    }
    .tab-button[data-category="Plan"] {
        background-color: #059669;
        color: white;
    }
    .tab-button[data-category="Tasks"] {
        background-color: #dc2626;
        color: white;
    }
    .tab-button[data-category="Notes"] {
        background-color: #d97706;
        color: white;
    }
    .tab-button[data-category="Health"] {
        background-color: #a855f7;
        color: white;
    }
    .tab-button[data-category="Andi"] {
        background-color: #0284c7;
        color: white;
    }
    .tab-button[data-category="Fun"] {
        background-color: #db2777;
        color: white;
    }

    .workspace-include {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }

    .workspace-actions {
        padding-top: 10px;
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
        height: 32px;
    }

    .control-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .filename-input {
        flex: 1;
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        font-family: monospace;
        height: 32px;
    }

    .filename-input:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
</style>
