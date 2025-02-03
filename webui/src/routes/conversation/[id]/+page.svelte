<!-- src/routes/conversation/[id]/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { page } from "$app/stores";
    import { goto } from "$app/navigation";
    import { api } from "$lib/api";
    import { configStore } from "$lib/store/configStore";
    import { type ChatMessage } from "$lib/types";
    import { taskStore } from "$lib/store/taskStore";
    import { pipelineStore } from "$lib/store/pipelineStore";
    import EditModal from "$lib/components/EditModal.svelte";
    import ChatMessageCard from "$lib/components/ChatMessageCard.svelte";
    import DocumentTypeFilter from "$lib/components/DocumentTypeFilter.svelte";
    import BranchFilter from "$lib/components/BranchFilter.svelte";
    import TextFilter from '$lib/components/TextFilter.svelte';
    import CreateMessageModal from '$lib/components/CreateMessageModal.svelte';
    import PipelineTriggers from '$lib/components/PipelineTriggers.svelte';
    import EmotionFilter from '$lib/components/EmotionFilter.svelte';

    let conversation: ChatMessage[] = [];
    let loading = true;
    let error: string | null = null;
    let showCreateModal = false;
    let isModalOpen = false;
    let editingMessageId: string | null = null;
    let editingContent = "";
    let latestPersonaId: string | null = null;

    let activeTextFilter = '';
    let activeBranch: number | null = null;
    let activeDocumentType: string | null = null;
    let activeEmotion: string | null = null;

    // Update the filtering chain to include emotions
    $: textFilteredConversation = conversation.filter((message) => {
        return !activeTextFilter || 
            message.content.toLowerCase().includes(activeTextFilter.toLowerCase());
    });

    // Branch filter uses text filtered results
    $: branchFilteredConversation = textFilteredConversation.filter((message) => {
        return activeBranch === null || message.branch === activeBranch;
    });

    // Document type filter uses branch filtered results
    $: documentFilteredConversation = branchFilteredConversation.filter((message) => {
        return activeDocumentType === null || message.document_type === activeDocumentType;
    });

    // Emotion filter uses document type filtered results
    $: filteredConversation = documentFilteredConversation.filter((message) => {
        if (activeEmotion === null) return true;
        return [
            message.emotion_a?.split(' ') || [],
        message.emotion_b?.split(' ') || [],
        message.emotion_c?.split(' ') || [],
        message.emotion_d?.split(' ') || []
        ].flat().some(emotion => emotion === activeEmotion);
    });
    $: conversationId = $page.params.id;
    $: pinnedMessages = $configStore?.pinnedMessages || [];

    function formatConversationId(id: string): string {
        let formatted = id.replace(/^(conv_|conversation_)/, "");
        const parts = formatted.split(/[_-]/);

        if (parts[0]?.length >= 13) {
            const date = new Date(parseInt(parts[0]));
            parts[0] = date.toLocaleDateString();
        }

        return parts
            .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
            .join(" ");
    }

    // Find the latest user_id and persona_id from conversation messages
    $: {
        if (conversation.length > 0) {
            // Find latest message with persona_id
            const personaMessage = [...conversation]
                .reverse()
                .find((msg) => msg.persona_id);
            if (personaMessage) {
                latestPersonaId = personaMessage.persona_id;
            }
        }
    }

    onMount(async () => {
        await loadConversation();
    });

    function isMessagePinned(docId: string): boolean {
        return pinnedMessages.some((msg) => msg.doc_id === docId);
    }

    function handleTogglePin(message: ChatMessage) {
        const currentPinned = $configStore?.pinnedMessages || [];
        if (isMessagePinned(message.doc_id)) {
            configStore.updateField(
                "pinnedMessages",
                currentPinned.filter((msg) => msg.doc_id !== message.doc_id),
            );
        } else {
            configStore.updateField("pinnedMessages", [
                ...currentPinned,
                message,
            ]);
        }
    }

    function handleEdit(docId: string, content: string) {
        editingMessageId = docId;
        editingContent = content;
        isModalOpen = true;
    }

    async function loadConversation() {
        try {
            const response = await api.getConversation(conversationId);
            conversation = response.data;
            loading = false;
        } catch (e) {
            error = "Failed to load conversation data";
            loading = false;
        }
    }

    async function deleteConversation() {
        if (confirm("Are you sure you want to delete this conversation?")) {
            try {
                await api.deleteConversation(conversationId);
                await goto("/chat-matrix");
            } catch (e) {
                error = "Failed to delete conversation";
            }
        }
    }

    async function alertRedirect(message: string) {
        alert(message);
        await goto("/chat-matrix");
    }

    async function startAnalysis() {
        try {
            if (!latestPersonaId) {
                return await alertRedirect(
                    "No persona ID found in conversation",
                );
            }

            // Use the pipeline store to update form data with conversation values
            pipelineStore.updateFormData({
                user_id: latestPersonaId,
                persona_id: latestPersonaId,
                conversation_id: conversationId,
                mood: $pipelineStore.formData.mood || "Delighted",
                top_n: $pipelineStore.formData.top_n,
                guidance: $pipelineStore.formData.guidance,
                query_text: $pipelineStore.formData.query_text,
            });

            await taskStore.submitTask("analysis", $pipelineStore.formData);
            await goto("/pipeline-tasks");
        } catch (e) {
            error = "Failed to start analysis";
            alert(error);
        }
    }

    async function startSummary() {
        try {
            if (!latestPersonaId) {
                return await alertRedirect(
                    "No persona ID found in conversation",
                );
            }

            // Use the pipeline store to update form data with conversation values
            pipelineStore.updateFormData({
                user_id: latestPersonaId,
                persona_id: latestPersonaId,
                conversation_id: conversationId,
                mood: $pipelineStore.formData.mood || "Delighted",
                top_n: $pipelineStore.formData.top_n,
                guidance: $pipelineStore.formData.guidance,
                query_text: $pipelineStore.formData.query_text,
            });

            await taskStore.submitTask("summary", $pipelineStore.formData);
            await goto("/pipeline-tasks");
        } catch (e) {
            error = "Failed to start summary";
            alert(error);
        }
    }

    async function handleCreateMessage(event: CustomEvent<{ message: Partial<ChatMessage> }>) {
        try {
            await api.createMessage(event.detail.message);
            await loadConversation();
            showCreateModal = false;
        } catch (error) {
            console.error('Error creating message:', error);
            alert('Failed to create message. Please try again.');
        }
    }

    async function handleSave({ detail }: CustomEvent<{ content: string }>) {
        if (!editingMessageId) return;

        try {
            await api.updateMessage(
                conversationId,
                editingMessageId,
                detail.content,
            );
            await loadConversation();
            isModalOpen = false;
            editingMessageId = null;
        } catch (e) {
            error = "Failed to update message";
        }
    }

    async function handleDeleteMessage(docId: string) {
        if (!confirm("Are you sure you want to delete this document?")) {
            return;
        }

        try {
            await api.deleteMessage(conversationId, docId);
            conversation = conversation.filter((msg) => msg.doc_id !== docId);
            filteredConversation = filteredConversation.filter(
                (msg) => msg.doc_id !== docId,
            );
        } catch (error) {
            console.error("Failed to delete document:", error);
            alert("Failed to delete document. Please try again.");
        }
    }
</script>

<svelte:head>
    <title>Conversation {conversationId} | MindAI</title>
</svelte:head>

<main>
    <div class="page-header">
        <div class="header-content">
            <div class="title-section">
                <h1>Conversation</h1>
                <div class="conversation-id">
                    {formatConversationId(conversationId)}
                </div>
            </div>
            <div class="conversation-info">
                {#if latestPersonaId}
                    <div class="info-item">
                        <span class="info-label">Assistant:</span>
                        <span class="info-value">{latestPersonaId}</span>
                    </div>
                {/if}
            </div>
        </div>
        <div class="action-buttons">
            <div class="analyze-buttons">
                <PipelineTriggers {latestPersonaId} {conversationId} />
                <button class="create-button" on:click={() => showCreateModal = true}>
                    + Create Message
                </button>
            </div>
            <button class="delete-button" on:click={deleteConversation}>
                Delete Conversation
            </button>
        </div>
    </div>

    {#if loading}
        <p>Loading conversation data...</p>
    {:else if error}
        <p class="error" role="alert">{error}</p>
    {:else}
    <TextFilter messages={conversation} bind:activeFilter={activeTextFilter} />
    <BranchFilter messages={textFilteredConversation} bind:activeBranch />
    <DocumentTypeFilter messages={branchFilteredConversation} bind:activeDocumentType />
    <EmotionFilter messages={documentFilteredConversation} bind:activeEmotion />
        <div
            class="conversation-container"
            role="log"
            aria-label="Conversation messages"
        >
            {#each filteredConversation as message}
                <ChatMessageCard
                    {message}
                    showActions={true}
                    onEdit={handleEdit}
                    onTogglePin={handleTogglePin}
                    onDelete={handleDeleteMessage}
                />
            {/each}
        </div>
    {/if}

    <EditModal
        isOpen={isModalOpen}
        initialContent={editingContent}
        on:close={() => {
            isModalOpen = false;
            editingMessageId = null;
        }}
        on:save={handleSave}
    />
    <CreateMessageModal
    isOpen={showCreateModal}
    {conversationId}
    on:close={() => showCreateModal = false}
    on:save={handleCreateMessage}
/>
</main>

<style>
    .page-header {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .header-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .title-section {
        display: flex;
        align-items: baseline;
        gap: 1rem;
    }

    .title-section h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
    }

    .conversation-id {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
            monospace;
        font-size: 1.1rem;
        color: #4b5563;
        padding: 0.25rem 0.75rem;
        background-color: #e5e7eb;
        border-radius: 0.25rem;
    }

    .conversation-info {
        display: flex;
        gap: 2rem;
        margin-left: auto;
        padding-left: 2rem;
    }

    .info-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .info-label {
        font-weight: 600;
        color: #4b5563;
    }

    .info-value {
        padding: 0.25rem 0.75rem;
        background-color: #e5e7eb;
        border-radius: 0.25rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
            monospace;
        font-size: 0.9rem;
        color: #1f2937;
    }

    .action-buttons {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }

    .analyze-buttons {
        display: flex;
        gap: 0.75rem;
    }

    .analyze-button {
        background-color: #2196f3;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: all 0.2s ease;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .analyze-button:hover {
        background-color: #1976d2;
        transform: translateY(-1px);
    }

    .delete-button {
        background-color: #ef4444;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .delete-button:hover {
        background-color: #dc2626;
        transform: translateY(-1px);
    }

    .conversation-container {
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        height: 600px;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
        gap: 1rem;
        background-color: #f9fafb;
    }

    .error {
        color: #ef4444;
        font-weight: bold;
    }

    @media (max-width: 768px) {
        .header-content {
            flex-direction: column;
            align-items: flex-start;
            gap: 1rem;
        }

        .title-section {
            flex-direction: column;
            align-items: flex-start;
        }

        .conversation-info {
            flex-direction: column;
            gap: 0.5rem;
        }

        .action-buttons {
            flex-direction: column;
            gap: 1rem;
        }

        .analyze-buttons {
            width: 100%;
        }

        .analyze-button,
        .delete-button {
            flex: 1;
        }
    }

    .create-button {
        background-color: #10b981;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: all 0.2s ease;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .create-button:hover {
        background-color: #059669;
        transform: translateY(-1px);
    }
</style>
