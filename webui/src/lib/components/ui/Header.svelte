<!-- src/lib/components/Header.svelte -->
<script lang="ts">
    import { page } from "$app/stores";
    import { configStore } from "$lib/store/configStore";
    import PinnedMessagesModal from "../conversation/PinnedMessagesModal.svelte";
    import { FileText } from "lucide-svelte";
    import DocumentViewer from "../document/DocumentViewer.svelte";

    let showPinnedModal = false;
    let showViewer = false;
    let showUserIdInput = false;

    function toggleViewer() {
        if ($configStore.selectedDocument) {
            showViewer = !showViewer;
        }
    }

    function handleCloseViewer() {
        showViewer = false;
    }

    function handleUserIdSubmit() {
        showUserIdInput = false;
    }

    function handleKeyPress(event: KeyboardEvent) {
        if (event.key === "Enter") {
            handleUserIdSubmit();
        }
    }

    // Safely get the pinned messages count
    $: pinnedCount = $configStore?.pinnedMessages?.length || 0;
    $: selectedDocument = $configStore?.selectedDocument;
</script>

<header>
    <nav>
        <div class="left">
            <a href="/" class="logo">AI-MIND</a>
        </div>
        <div class="center">
            <a href="/" class:active={$page.url.pathname === "/"}>Home</a>
            <a
                href="/completions"
                class:active={$page.url.pathname === "/completions"}
                >Completions</a
            >
            <a
                href="/chat-matrix"
                class:active={$page.url.pathname === "/chat-matrix"}
                >Chat Matrix</a
            >
            <a href="/search" class:active={$page.url.pathname === "/search"}
                >Search</a
            >
            <a
                href="/pipeline-tasks"
                class:active={$page.url.pathname === "/pipeline-tasks"}
                >Pipeline Tasks</a
            >
            <a
                href="/documents"
                class:active={$page.url.pathname === "/documents"}>Documents</a
            >
            <a href="/roster" class:active={$page.url.pathname === "/roster"}
                >Roster</a
            >
            <a href="/program" class:active={$page.url.pathname === "/program"}
                >Program</a
            >
        </div>
        <div class="right">
            {#if selectedDocument}
                <button
                    class="active-document"
                    on:click={toggleViewer}
                    title="Click to view document contents"
                >
                    <FileText size={16} />
                    {selectedDocument.name}
                </button>
            {/if}
            {#if $configStore.persona}
                <div class="active-persona">
                    Persona: {$configStore.persona}
                </div>
            {/if}
            <button
                class="pinned-count"
                on:click={() => (showPinnedModal = true)}
            >
                {pinnedCount}
            </button>
            {#if showUserIdInput}
                <div class="user-input-container">
                    <input
                        type="text"
                        bind:value={$configStore.user_id}
                        placeholder="Enter username"
                        class="user-input"
                        on:keypress={handleKeyPress}
                    />
                    <button class="user-input-btn" on:click={handleUserIdSubmit}
                        >✓</button
                    >
                </div>
            {:else if $configStore.user_id}
                <span
                    class="clickable user-display"
                    on:click={() => (showUserIdInput = true)}
                    >Welcome, {$configStore.user_id}</span
                >
            {:else}
                <span
                    class="clickable user-display"
                    on:click={() => (showUserIdInput = true)}
                    >No username set</span
                >
            {/if}
        </div>
    </nav>
</header>

<DocumentViewer show={showViewer} onClose={handleCloseViewer} />
<PinnedMessagesModal
    show={showPinnedModal}
    onClose={() => (showPinnedModal = false)}
/>

<style>
    header {
        background-color: #4caf50;
        padding: 10px 20px;
        color: white;
    }

    nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
    }

    .left,
    .center,
    .right {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .logo {
        font-size: 1.5em;
        font-weight: bold;
        text-decoration: none;
        color: white;
    }

    .center a {
        text-decoration: none;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        transition: background-color 0.3s;
    }

    .center a:hover,
    .center a.active {
        background-color: rgba(255, 255, 255, 0.2);
    }

    .right span {
        font-style: italic;
    }

    .pinned-count {
        background-color: white;
        color: #4caf50;
        border: none;
        border-radius: 0.5rem;
        padding: 0.25rem 0.75rem;
        font-weight: bold;
        cursor: pointer;
    }

    .active-document {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background-color: #2196f3;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.25rem 0.75rem;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.2s;
    }

    .active-document:hover {
        background-color: #5ac83e;
    }

    .active-persona {
        background-color: #388e3c;
        color: white;
        border-radius: 0.5rem;
        padding: 0.25rem 0.75rem;
        font-weight: 500;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
    }

    .clickable {
        cursor: pointer;
    }

    .user-input-container {
        display: flex;
        align-items: stretch;
        gap: 0.25rem;
        height: 32px;
    }

    .user-input {
        background-color: rgba(255, 255, 255, 0.9);
        border: none;
        border-radius: 0.25rem;
        padding: 0.25rem 0.5rem;
        font-size: 0.9rem;
        color: #333;
        width: 150px;
        height: 100%;
    }

    .user-input-btn {
        background-color: rgba(255, 255, 255, 0.9);
        border: none;
        border-radius: 0.25rem;
        width: 32px;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        color: #388e3c;
        font-weight: bold;
        padding: 0;
    }

    .user-input-btn:hover {
        background-color: white;
    }

    .user-display {
        font-size: 0.9rem;
        font-style: normal;
        opacity: 0.9;
    }

    .user-display:hover {
        opacity: 1;
    }
</style>
