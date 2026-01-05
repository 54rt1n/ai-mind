<!-- src/routes/chat-matrix/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { api } from "$lib/api";
    import { configStore } from "$lib/store/configStore";
    import { ArrowUp, ArrowDown, RefreshCw } from "lucide-svelte";
    import { writable } from "svelte/store";

    const sortColumn = writable<string | null>(null);
    const sortDirection = writable<"asc" | "desc">("asc");
    let matrixData: Record<string, Record<string, number>> = {};
    let loading = true;
    let error: string | null = null;
    let rebuildLoading = false;

    onMount(async () => {
        const persona_id = $configStore.persona_id;
        if (!persona_id) {
            error = "No persona selected. Please select a persona from the roster.";
            loading = false;
            return;
        }

        try {
            const response = await api.getChatMatrix(persona_id);
            matrixData = response.data;
            loading = false;
        } catch (e) {
            error = "Failed to load chat matrix data";
            loading = false;
        }
    });

    $: sortedData = Object.entries(matrixData).sort((a, b) => {
        if (!$sortColumn) {
            return a[0].localeCompare(b[0]);
        }

        const valueA =
            $sortColumn === "conversation_id" ? a[0] : a[1][$sortColumn] || 0;
        const valueB =
            $sortColumn === "conversation_id" ? b[0] : b[1][$sortColumn] || 0;

        return $sortDirection === "asc"
            ? valueA < valueB
                ? -1
                : valueA > valueB
                  ? 1
                  : 0
            : valueB < valueA
              ? -1
              : valueB > valueA
                ? 1
                : 0;
    });

    function getColumnSortOrder(column: string): number {
        const orderMap: Record<string, number> = {
            timestamp_max: 1,
            conversation: 2,
            summary: 3,
            analysis: 4,
            highlight: 5,
        };
        return orderMap[column] || 6;
    }

    // Replace the existing columns definition with:
    $: columns = Object.keys(Object.values(matrixData)[0] || {}).sort(
        (a, b) => {
            const orderA = getColumnSortOrder(a);
            const orderB = getColumnSortOrder(b);
            if (orderA === orderB) {
                return a.localeCompare(b); // alphabetical for same priority
            }
            return orderA - orderB;
        },
    );

    function handleSort(column: string) {
        if ($sortColumn !== column) {
            $sortColumn = column;
            $sortDirection = column === "conversation_id" ? "asc" : "desc";
        } else {
            $sortDirection = $sortDirection === "asc" ? "desc" : "asc";
        }
    }

    function getSortIcon(columnName: string) {
        if ($sortColumn !== columnName) return null;
        return $sortDirection === "asc" ? ArrowUp : ArrowDown;
    }

    function formatTimestamp(timestamp: number): string {
        return new Date(timestamp * 1000).toLocaleString("en-US", {
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
        });
    }

    async function handleRebuildIndex() {
        const persona_id = $configStore.persona_id;
        if (!persona_id) {
            error = "No persona selected. Please select a persona from the roster.";
            return;
        }

        if (!confirm(`Rebuild index for ${$configStore.persona}? This may take a few moments.`)) {
            return;
        }

        rebuildLoading = true;
        try {
            const result = await api.rebuildIndex(persona_id, false);
            alert(`Index rebuild started for ${$configStore.persona}`);
        } catch (e) {
            error = "Failed to start index rebuild";
        } finally {
            rebuildLoading = false;
        }
    }
</script>

<svelte:head>
    <title>Chat Matrix | </title>
</svelte:head>

<main>
    <div class="header-row">
        <h1>Chat Matrix</h1>
        <div class="action-bar">
            <button
                class="rebuild-button"
                on:click={handleRebuildIndex}
                disabled={rebuildLoading || !$configStore.persona_id}
                title="Rebuild memory index for selected persona"
            >
                <RefreshCw size={14} />
                <span>{rebuildLoading ? 'Rebuilding...' : 'Rebuild Index'}</span>
            </button>
        </div>
    </div>

    {#if $configStore.persona_id && !loading && !error}
        <div class="persona-info">
            Showing conversations for: <strong>{$configStore.persona}</strong> ({$configStore.persona_id})
        </div>
    {/if}

    {#if loading}
        <p>Loading chat matrix data...</p>
    {:else if error}
        <p class="error">{error}</p>
    {:else}
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th
                            class="sortable-header {$sortColumn ===
                            'conversation_id'
                                ? 'active'
                                : ''}"
                            on:click={() => handleSort("conversation_id")}
                        >
                            <div class="header-content">
                                <span>Conversation ID</span>
                                {#if $sortColumn === "conversation_id"}
                                    <svelte:component
                                        this={getSortIcon("conversation_id")}
                                        size={16}
                                    />
                                {/if}
                            </div>
                        </th>
                        {#each columns as column}
                            <th
                                class="sortable-header {$sortColumn === column
                                    ? 'active'
                                    : ''}"
                                on:click={() => handleSort(column)}
                            >
                                <div class="header-content">
                                    <span>{column}</span>
                                    {#if $sortColumn === column}
                                        {#if $sortDirection === "asc"}
                                            <ArrowUp size={16} />
                                        {:else}
                                            <ArrowDown size={16} />
                                        {/if}
                                    {/if}
                                </div>
                            </th>
                        {/each}
                    </tr>
                </thead>
                <tbody>
                    {#each sortedData as [conversationId, data]}
                        <tr>
                            <td>
                                <a href="/conversation/{conversationId}"
                                    >{conversationId}</a
                                >
                            </td>
                            {#each columns as column}
                                <td>
                                    {#if column === "timestamp_max"}
                                        {formatTimestamp(data[column])}
                                    {:else}
                                        {data[column]?.toFixed(0) || "-"}
                                    {/if}
                                </td>
                            {/each}
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>
    {/if}
</main>

<style>
    .table-container {
        overflow-x: auto;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
    }

    th,
    td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }

    th {
        background-color: #4caf50;
        color: white;
    }

    tr:nth-child(even) {
        background-color: #f2f2f2;
    }

    tr:hover {
        background-color: #ddd;
    }

    .error {
        color: red;
        font-weight: bold;
    }

    a {
        color: #4caf50;
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
    }

    .sortable-header {
        cursor: pointer;
        user-select: none;
    }

    .sortable-header:hover {
        background-color: #45a049;
    }

    .sortable-header.active {
        background-color: #3d8b40;
    }

    .header-content {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .header-content span {
        white-space: nowrap;
    }

    .persona-info {
        background-color: #e8f4f8;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin-bottom: 20px;
        border-radius: 4px;
    }

    .header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }

    .action-bar {
        display: flex;
        gap: 10px;
    }

    .rebuild-button {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        font-size: 14px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.2s;
    }

    .rebuild-button:hover:not(:disabled) {
        background-color: #45a049;
    }

    .rebuild-button:disabled {
        background-color: #cccccc;
        cursor: not-allowed;
    }
</style>
