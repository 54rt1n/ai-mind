<!-- src/lib/components/clipboard/WorkspacePanel.svelte -->
<script lang="ts">
    import { configStore } from "$lib/store/configStore";
    import { workspaceStore } from "$lib/store/workspaceStore";
    import { get } from "svelte/store";
    import { Copy, Trash2, ArrowUpCircle, Check, Hammer } from "lucide-svelte";
    import "$lib/../styles/meta-panels.css";
    import type { NotificationParams, CompletionMessage } from "$lib/types";
    import { WorkspaceCategory } from "$lib/types";
    import NotificationToast from "$lib/components/ui/NotificationToast.svelte";
    import ModelSelect from "../model/ModelSelect.svelte";
    import { clipboardStore } from "$lib/store/clipboardStore";

    let uploadFileName = "";
    $: defaultFilename = `${$workspaceStore.name}.${new Date().toISOString().split("T")[0]}.txt`;
    let lastConversationHistory: CompletionMessage[] = [];

    let notificationProps: NotificationParams = {
        show: false,
        notificationType: "info",
        message: "",
        loading: false,
    };

    function handleWorkspaceChange(event: Event) {
        const input = event.target as HTMLTextAreaElement;
        workspaceStore.setContent(input.value);
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

    $: workspaceSelected = $workspaceStore.name !== WorkspaceCategory.None;

    export function setConversationHistory(
        conversationHistory: CompletionMessage[],
    ) {
        lastConversationHistory = conversationHistory;
    }

    export async function clearAndGenerate(
        conversationHistory: CompletionMessage[],
    ): Promise<boolean> {
        lastConversationHistory = conversationHistory;
        if (conversationHistory.length === 0) {
            console.log("clearAndGenerate failure", conversationHistory);
            throw new Error("No conversation history");
        }
        const config = get(configStore);
        // We need to make a copy of the config to avoid mutating the store
        const configCopy = {
            ...config,
            maxTokens: 4096,
            temperature: 0.7,
            topP: undefined,
            topK: undefined,
            frequencyPenalty: undefined,
            presencePenalty: undefined,
            repetitionPenalty: undefined,
            minP: undefined,
        };
        return await workspaceStore.generateWorkspaceUpdate(
            configCopy,
            conversationHistory,
            "You are to update the workspace content based on the conversation. Output only the new content.",
        );
    }

    export async function validateAndAcceptWorkspace(): Promise<boolean> {
        if ($workspaceStore.contentStream.trim().length > 0) {
            commitStreamUpdate();
            return true;
        }
        return false;
    }

    function commitStreamUpdate() {
        workspaceStore.commitStreamUpdate();
    }

    async function handleWorkspaceUpdate(autoValidate: boolean = false) {
        if (lastConversationHistory.length === 0) {
            alert(
                "No conversation history available. Please use the Work button in the main chat first.",
            );
            return;
        }
        console.log("Running workspace update...");

        if (!(await clearAndGenerate(lastConversationHistory))) {
            console.error("Failed to generate workspace update.");
            if (!(await clearAndGenerate(lastConversationHistory))) {
                alert("Failed to generate workspace update.");
                console.error("Failed to generate workspace update.");
                return;
            }
        }
        if (autoValidate && !(await validateAndAcceptWorkspace())) {
            console.error("Failed to validate workspace update.");
            return;
        }
    }

    function handleIncludeInMessagesChange(event: Event) {
        const checkbox = event.target as HTMLInputElement;
        configStore.updateField("includeWorkspaceInMessages", checkbox.checked);
    }
</script>

<div class="meta-panel">
    <button
        class="meta-panel-header collapsible"
        on:click={() =>
            configStore.updateField(
                "showWorkspace",
                !$configStore.showWorkspace,
            )}
    >
        <span>Workspace</span>
        <span class="toggle-icon"
            >{$configStore.showWorkspace ? "▼" : "▶"}</span
        >
    </button>

    {#if $configStore.showWorkspace}
        <div class="meta-panel-content">
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
                <div class="workspace-content">
                    <textarea
                        class="workspace-textarea"
                        bind:value={$workspaceStore.content}
                        on:input={handleWorkspaceChange}
                        placeholder="Use this workspace for notes, drafts, or any text you want to keep handy..."
                    />
                    <textarea
                        class="workspace-textarea"
                        bind:value={$workspaceStore.contentStream}
                        placeholder="Generated content will appear here..."
                        disabled={true}
                    />
                    {#if $workspaceStore.contentStream}
                        <div class="workspace-stream-controls">
                            <button
                                class="control-button accept"
                                on:click={commitStreamUpdate}
                                title="Accept changes"
                            >
                                <Check size={16} />
                            </button>
                        </div>
                    {/if}
                </div>
                <div>Word Count: {$workspaceStore.wordCount}</div>
            </div>
            <ModelSelect
                bind:value={$configStore.workspaceModel}
                category="workspace"
            />
            <div class="workspace-include">
                <div class="workspace-actions">
                    <label class="include-label">
                        <input
                            type="checkbox"
                            checked={$configStore.includeWorkspaceInMessages}
                            on:change={handleIncludeInMessagesChange}
                        />
                        Include workspace in messages
                    </label>
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
                        <button
                            class="workspace-button work"
                            on:click={() => handleWorkspaceUpdate(false)}
                            disabled={!workspaceSelected ||
                                lastConversationHistory.length === 0}
                            title="Update workspace from conversation"
                        >
                            <Hammer size={16} />
                        </button>
                        {#if lastConversationHistory.length > 0}
                            <span
                                class="conversation-count"
                                title="Available conversation turns"
                            >
                                ({lastConversationHistory.length})
                            </span>
                        {/if}
                    </div>
                </div>
            </div>
        </div>
    {/if}
</div>

<NotificationToast {...notificationProps} />

<style>
    .workspace-section {
        width: 100%;
        min-width: 0;
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

    .workspace-include {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
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

    .workspace-content {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .workspace-stream-controls {
        display: flex;
        justify-content: flex-end;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }

    .control-button.accept {
        background-color: #10b981;
    }

    .control-button.accept:hover:not(:disabled) {
        background-color: #059669;
    }

    .workspace-button.work {
        background-color: #8b5cf6;
        color: white;
    }

    .workspace-button.work:hover:not(:disabled) {
        background-color: #7c3aed;
    }

    .conversation-count {
        color: #6b7280;
        font-size: 0.875rem;
        padding: 0 0.5rem;
        display: flex;
        align-items: center;
    }

    .include-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #666;
        font-size: 0.875rem;
        margin-right: 1rem;
    }
</style>
