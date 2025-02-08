<!-- src/lib/components/ConversationSettingsPanel.svelte -->
<script lang="ts">
    import { chatStore, canGoBack, canRetry } from "$lib/store/chatStore";
    import { configStore } from "$lib/store/configStore";
    import "$lib/../styles/meta-panels.css";
</script>

<div class="meta-panel">
    <button
        class="meta-panel-header collapsible"
        on:click={() =>
            configStore.updateField("showControls", !$configStore.showControls)}
    >
        <span>Conversation Settings</span>
        <span class="toggle-icon">{$configStore.showControls ? "▼" : "▶"}</span
        >
    </button>
    {#if $configStore.showControls}
        <div class="meta-panel-content">
            <div class="meta-input">
                <input
                    type="text"
                    bind:value={$chatStore.conversationId}
                    placeholder="Conversation ID"
                    on:input={(e) =>
                        chatStore.setConversationId(e.currentTarget.value)}
                />
                <div class="button-container">
                    <button
                        class="control-button"
                        type="button"
                        on:click={chatStore.clearChat}
                        disabled={$chatStore.loading}
                    >
                        Clear Chat
                    </button>
                    <button
                        class="control-button"
                        type="button"
                        on:click={chatStore.saveConversation}
                        disabled={$chatStore.loading}
                    >
                        Save Conversation
                    </button>
                    <button
                        class="control-button"
                        type="button"
                        on:click={chatStore.goBack}
                        disabled={$chatStore.loading || !$canGoBack}
                    >
                        Back
                    </button>
                    <button
                        class="control-button"
                        type="button"
                        on:click={configStore.reset}
                        disabled={$chatStore.loading}
                    >
                        Reset Chat
                    </button>
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
    .button-container {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }

    .control-button {
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        background-color: #4caf50;
        color: white;
        cursor: pointer;
        font-size: 0.9rem;
    }

    .control-button:hover {
        background-color: #45a049;
    }

    .control-button:disabled {
        background-color: #cccccc;
        cursor: not-allowed;
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
</style>
