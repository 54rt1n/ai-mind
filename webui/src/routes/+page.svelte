<script lang="ts">
    import { onMount } from "svelte";
    import { configStore } from "$lib/store/configStore";
    import { Edit2 } from "lucide-svelte";
    import "../styles/chat.css";
    import "../styles/meta-panels.css";
    import EditModal from "$lib/components/conversation/EditModal.svelte";
    import MessageFormatPanel from "$lib/components/chat/MessageFormatPanel.svelte";
    import AdvancedSettingsPanel from "$lib/components/model/AdvancedSettingsPanel.svelte";
    import ConversationSettingsPanel from "$lib/components/chat/ConversationSettingsPanel.svelte";
    import ThoughtPanel from "$lib/components/thought/ThoughtPanel.svelte";
    import { thoughtStore } from "$lib/store/thoughtStore";
    import { chatStore, canRetry } from "$lib/store/chatStore";
    import { clipboardStore } from "$lib/store/clipboardStore";
    import { workspaceStore } from "$lib/store/workspaceStore";
    import ClipboardPanel from "$lib/components/clipboard/ClipboardPanel.svelte";
    import CopyButton from "$lib/components/clipboard/CopyButton.svelte";
    import Thinking from "$lib/components/ui/Thinking.svelte";

    let isModalOpen = false;
    let editingContent = "";
    let editingIndex: number | null = null;
    let messageContainer: HTMLDivElement;
    let userInput: string = "";
    let formElement: HTMLFormElement;
    let previousMessageCount = 0;

    let thoughtPanelComponent: ThoughtPanel;
    let clipboardPanelComponent: ClipboardPanel;

    // Model validation
    $: canChat = !!$configStore.chatModel;
    $: canThink = !!$configStore.thoughtModel && !!$configStore.chatModel;
    $: canWork = !!$configStore.workspaceModel;

    $: modelWarning = !canChat
        ? "No chat model selected"
        : $configStore.autoThink && !canThink
          ? "Both chat and thought models must be selected for thinking"
          : "";

    // Button tooltips
    $: sendTooltip = !canChat ? "No chat model selected" : "";
    $: thinkTooltip = !canThink
        ? "Chat and thought models must be selected"
        : "";
    $: workTooltip = !canWork ? "No workspace model selected" : "";

    function prepareMessage(inputMessage: string) {
        if ($configStore.emotionalStateEnabled) {
            const emotionalState =
                configStore.formatters.getEmotionalStateHeader($configStore);
            inputMessage = emotionalState + "\n\n" + inputMessage;
        }

        if ($configStore.sentimentMeterEnabled) {
            const sentiment =
                configStore.formatters.getSentimentMeter($configStore);
            inputMessage = inputMessage + "\n\n" + sentiment;
        }

        // Add footer if selected
        if ($configStore.selectedFooter) {
            inputMessage = inputMessage + "\n\n" + $configStore.selectedFooter;
        }
        return inputMessage;
    }

    function scrollToBottom() {
        if (messageContainer && $configStore.autoScroll) {
            messageContainer.scrollTop = messageContainer.scrollHeight;
        }
    }

    // Subscribe to chat store to detect new messages
    $: {
        if ($chatStore.conversationHistory.length > previousMessageCount) {
            scrollToBottom();
        }
        previousMessageCount = $chatStore.conversationHistory.length;
    }

    // Handle new messages during streaming
    $: if ($chatStore.loading) {
        scrollToBottom();
    }

    onMount(() => {
        chatStore.loadConversationHistory();
        previousMessageCount = $chatStore.conversationHistory.length;
        scrollToBottom();

        if ("serviceWorker" in navigator) {
            navigator.serviceWorker
                .register("/service-worker.js")
                .then((registration) => {
                    // console.log("Service Worker registered successfully:", registration.scope);
                })
                .catch((error) => {
                    // console.log("Service Worker registration failed:", error);
                });
        }
    });

    async function _SendMessage(
        fullMessage: string,
        skipAppend: boolean = false,
    ): Promise<boolean> {
        const workspaceContent = $clipboardStore.includeInMessages
            ? $workspaceStore.content
            : undefined;
        await chatStore.sendMessage(fullMessage, $configStore, {
            workspaceContent,
            skipAppend,
        });
        return true;
    }

    function handleSendMessage(event: Event) {
        event.preventDefault();
        if (userInput.trim()) {
            let fullMessage = prepareMessage(userInput);
            userInput = "";
            if ($configStore.autoThink) {
                runThink();
            } else {
                _SendMessage(fullMessage);
            }
        }
    }

    function retrySendMessage(event: Event) {
        // This is simple, we just invoke the retryy function
        const workspaceContent = $clipboardStore.includeInMessages
            ? $workspaceStore.content
            : undefined;
        chatStore.retry($configStore, workspaceContent);
    }

    function handleKeydown(event: KeyboardEvent) {
        if (event.key === "Enter" && event.shiftKey) {
            event.preventDefault();
            if ($configStore.autoThink) {
                runThink();
            } else {
                formElement.requestSubmit();
            }
        }
    }

    function startEditing(index: number, content: string) {
        editingIndex = index;
        editingContent = content;
        isModalOpen = true;
    }

    function handleSave({ detail }: CustomEvent<{ content: string }>) {
        if (editingIndex !== null) {
            chatStore.updateMessage(editingIndex, detail.content);
            isModalOpen = false;
            editingIndex = null;
        }
    }

    function appendMessage(): string {
        const fullMessage = prepareMessage(userInput);
        chatStore.appendMessage("user", fullMessage);
        userInput = "";
        return fullMessage;
    }

    function handleClose() {
        isModalOpen = false;
        editingIndex = null;
    }

    function formatTimestamp(timestamp: number): string {
        return new Date(timestamp * 1000).toLocaleString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
            month: "short",
            day: "numeric",
            year: "numeric",
        });
    }

    async function thinkPanel(fullMessage: string, autoAccept: boolean = true) {
        console.log("Advancing think...");
        if (
            !(await thoughtPanelComponent.advanceThought(
                $chatStore.conversationHistory,
            ))
        ) {
            console.error("Failed to advance thought.");
            if (
                !(await thoughtPanelComponent.advanceThought(
                    $chatStore.conversationHistory,
                ))
            ) {
                console.error("Failed to advance thought.");
                return;
            }
        }
        if (
            autoAccept &&
            !(await thoughtPanelComponent.validateAndAcceptThought())
        ) {
            console.error("Failed to validate thought.");
            return;
        }
        $configStore.thoughtContent = $thoughtStore.thoughtContent;
        console.log("Generating response");
        await _SendMessage(fullMessage, true);
    }

    async function runThink() {
        userInput = userInput.trim();
        if (userInput.length === 0) {
            alert("Please enter a message.");
            return;
        }
        const fullMessage = appendMessage();
        await thinkPanel(fullMessage);
    }

    async function reThink() {
        const lastTurn =
            $chatStore.conversationHistory[
                $chatStore.conversationHistory.length - 1
            ];
        if (lastTurn.role !== "user") {
            alert("Please use the 'Think' button to generate a new thought.");
            return;
        }
        const fullMessage = lastTurn.content;
        await thinkPanel(fullMessage);
    }

    async function runWork() {
        console.log("Running workspace update...");
        if (
            !(await clipboardPanelComponent.clearAndGenerate(
                $chatStore.conversationHistory,
            ))
        ) {
            alert("Failed to generate workspace update.");
            console.error("Failed to generate workspace update.");
            return;
        }
    }
</script>

<svelte:head>
    <title>MindAI Chat</title>
</svelte:head>

<main class="main-chat">
    <h1>MindAI Chat</h1>
    <ClipboardPanel bind:this={clipboardPanelComponent} />
    <ThoughtPanel bind:this={thoughtPanelComponent} />
    <div class="message-container-wrapper">
        <div class="message-container" bind:this={messageContainer}>
            <div class="chat-container">
                {#each $chatStore.conversationHistory as message, index}
                    <div
                        class="message {message.role}"
                        role="article"
                        aria-label="{message.role} message"
                    >
                        <div class="message-header">
                            <div class="message-info">
                                <strong>{message.role}</strong>
                                <span class="timestamp"
                                    >{formatTimestamp(message.timestamp)}</span
                                >
                            </div>
                            <div class="message-actions">
                                <CopyButton content={message.content} />
                                <button
                                    class="edit-button"
                                    aria-label="Edit message"
                                    title="Edit message"
                                    on:click={() =>
                                        startEditing(index, message.content)}
                                >
                                    <Edit2 size={16} />
                                </button>
                            </div>
                        </div>
                        <span class="message-content">{message.content}</span>
                    </div>
                {/each}
            </div>
        </div>
        <div class="auto-scroll-toggle">
            <label>
                <input type="checkbox" bind:checked={$configStore.autoScroll} />
                Auto-scroll
            </label>
        </div>
    </div>

    <form on:submit={handleSendMessage} bind:this={formElement}>
        <div class="meta-panel">
            <div class="meta-panel-header">Message Input</div>
            <div class="meta-panel-content">
                {#if $chatStore.loading}
                    <Thinking />
                {/if}
                <div class="meta-input">
                    <div class="input-controls">
                        <textarea
                            bind:value={userInput}
                            rows="3"
                            placeholder="Type your message here..."
                            on:keydown={handleKeydown}
                        />
                        {#if modelWarning}
                            <div class="model-warning">
                                ⚠️ {modelWarning}
                            </div>
                        {/if}
                        <div class="input-options">
                            <label class="auto-option">
                                <input
                                    type="checkbox"
                                    bind:checked={$configStore.autoThink}
                                />
                                Auto-think
                            </label>
                        </div>
                    </div>
                    <div class="button-group">
                        {#if !$configStore.autoThink}
                            <button
                                class="control-button send"
                                type="submit"
                                disabled={$chatStore.loading || !canChat}
                                title={$chatStore.loading
                                    ? "Processing..."
                                    : sendTooltip}
                            >
                                Send
                            </button>
                            <button
                                class="control-button retry"
                                type="button"
                                on:click={retrySendMessage}
                                disabled={$chatStore.loading ||
                                    !$canRetry ||
                                    !canChat}
                                title={$chatStore.loading
                                    ? "Processing..."
                                    : !$canRetry
                                      ? "No message to retry"
                                      : sendTooltip}
                            >
                                Retry
                            </button>
                        {:else}
                            <button
                                class="control-button think"
                                type="button"
                                on:click={runThink}
                                disabled={$chatStore.loading ||
                                    !userInput.trim() ||
                                    !canThink}
                                title={$chatStore.loading
                                    ? "Processing..."
                                    : !userInput.trim()
                                      ? "Please enter a message"
                                      : thinkTooltip}
                            >
                                Think
                            </button>
                            <button
                                class="control-button rethink"
                                type="button"
                                on:click={reThink}
                                disabled={$chatStore.loading ||
                                    !$canRetry ||
                                    !canThink}
                                title={$chatStore.loading
                                    ? "Processing..."
                                    : !$canRetry
                                      ? "No message to rethink"
                                      : thinkTooltip}
                            >
                                Rethink
                            </button>
                        {/if}
                        <button
                            class="control-button append"
                            type="button"
                            on:click={appendMessage}
                            disabled={$chatStore.loading || !userInput.trim()}
                            title={$chatStore.loading
                                ? "Processing..."
                                : !userInput.trim()
                                  ? "Please enter a message"
                                  : ""}
                        >
                            Append
                        </button>
                        <button
                            class="control-button swap"
                            type="button"
                            on:click={() => chatStore.swapRoles()}
                            disabled={$chatStore.loading || !$canRetry}
                            title={$chatStore.loading
                                ? "Processing..."
                                : !$canRetry
                                  ? "No messages to swap"
                                  : ""}
                        >
                            Swap
                        </button>
                        <button
                            class="control-button pop"
                            type="button"
                            on:click={() => chatStore.popMessage()}
                            disabled={$chatStore.loading || !$canRetry}
                            title={$chatStore.loading
                                ? "Processing..."
                                : !$canRetry
                                  ? "No messages to remove"
                                  : ""}
                        >
                            Pop
                        </button>
                        <button
                            class="control-button work"
                            type="button"
                            on:click={runWork}
                            disabled={$chatStore.loading || !canWork}
                            title={$chatStore.loading
                                ? "Processing..."
                                : workTooltip}
                        >
                            Work
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </form>

    <MessageFormatPanel />
    <AdvancedSettingsPanel />
    <ConversationSettingsPanel />
    <EditModal
        isOpen={isModalOpen}
        initialContent={editingContent}
        on:close={handleClose}
        on:save={handleSave}
    />
</main>

<style>
    .message-container-wrapper {
        position: relative;
        margin-bottom: 20px;
    }

    .message-container {
        border: 1px solid #ccc;
        height: 400px;
        overflow-y: auto;
        padding: 10px;
    }

    .auto-scroll-toggle {
        position: absolute;
        bottom: 10px;
        right: 10px;
        background: rgba(255, 255, 255, 0.9);
        padding: 4px 8px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        z-index: 10;
        display: flex;
        align-items: center;
        gap: 4px;
        font-size: 12px;
        border: 1px solid #e5e7eb;
    }

    .auto-scroll-toggle label {
        display: flex;
        align-items: center;
        gap: 4px;
        cursor: pointer;
        user-select: none;
        color: #666;
    }

    .auto-scroll-toggle input[type="checkbox"] {
        width: 14px;
        height: 14px;
        cursor: pointer;
        margin: 0;
    }

    .footer-select {
        width: 100%;
        padding: 8px;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-size: 14px;
        background-color: white;
        height: 40px;
        cursor: pointer;
    }

    .message {
        margin-bottom: 10px;
        padding: 10px;
        border-radius: 5px;
        position: relative;
    }

    .message:focus {
        outline: 2px solid #4caf50;
    }

    .message-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
    }

    .edit-button {
        background: none;
        border: none;
        cursor: pointer;
        padding: 5px;
        border-radius: 3px;
        display: flex;
        align-items: center;
    }

    .edit-button:hover {
        background-color: rgba(0, 0, 0, 0.1);
    }

    .user {
        background-color: #e6f3ff;
    }

    .assistant {
        background-color: #e6ffe6;
    }

    .message-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex: 1;
    }

    .timestamp {
        font-size: 0.85rem;
        color: #666;
        font-weight: normal;
    }

    .message-actions {
        display: flex;
        gap: 0.25rem;
    }

    .message-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
    }

    .edit-button {
        opacity: 0;
        background: none;
        border: none;
        cursor: pointer;
        padding: 5px;
        border-radius: 3px;
        color: #666;
        transition:
            opacity 0.2s,
            background-color 0.2s;
    }

    .message:hover .edit-button {
        opacity: 1;
    }

    .edit-button:hover {
        background-color: rgba(0, 0, 0, 0.1);
        color: #333;
    }

    .button-group {
        display: flex;
        gap: 0.5rem;
        justify-content: flex-end;
        margin-top: 0.5rem;
    }

    .control-button {
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.375rem;
        cursor: pointer;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.2s;
        opacity: 0.95;
        position: relative;
    }

    .control-button:disabled {
        cursor: not-allowed;
        opacity: 0.5;
        filter: grayscale(40%);
    }

    /* Improve tooltip appearance */
    .control-button[title]:disabled:hover::before {
        content: attr(title);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        padding: 0.5rem;
        background-color: #333;
        color: white;
        font-size: 0.75rem;
        border-radius: 0.25rem;
        white-space: nowrap;
        z-index: 10;
        pointer-events: none;
        margin-bottom: 0.5rem;
    }

    .control-button[title]:disabled:hover::after {
        content: "";
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        border-width: 0.25rem;
        border-style: solid;
        border-color: #333 transparent transparent transparent;
        margin-bottom: 0.25rem;
    }

    .control-button:not(:disabled):hover {
        opacity: 1;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .control-button.send,
    .control-button.think {
        background-color: #22c55e;
        color: white;
    }

    .control-button.send:not(:disabled):hover,
    .control-button.think:not(:disabled):hover {
        background-color: #16a34a;
    }

    .control-button.retry,
    .control-button.rethink {
        background-color: #15803d;
        color: white;
    }

    .control-button.retry:not(:disabled):hover,
    .control-button.rethink:not(:disabled):hover {
        background-color: #166534;
    }

    .control-button.append {
        background-color: #0ea5e9;
        color: white;
    }

    .control-button.append:not(:disabled):hover {
        background-color: #0284c7;
    }

    .control-button.swap {
        background-color: #46b5cb;
        color: white;
    }

    .control-button.swap:not(:disabled):hover {
        background-color: #3aa3b9;
    }

    .control-button.pop {
        background-color: #ef4444;
        color: white;
    }

    .control-button.pop:not(:disabled):hover {
        background-color: #dc2626;
    }

    .control-button.work {
        background-color: #8b5cf6;
        color: white;
    }

    .control-button.work:not(:disabled):hover {
        background-color: #7c3aed;
    }

    .input-controls {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .input-options {
        display: flex;
        gap: 1rem;
        align-items: center;
    }

    .auto-option {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.875rem;
        color: #666;
        cursor: pointer;
    }

    .auto-option input[type="checkbox"] {
        width: 14px;
        height: 14px;
        cursor: pointer;
        margin: 0;
    }

    .model-warning {
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 0.375rem;
        color: #856404;
        font-size: 0.875rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
