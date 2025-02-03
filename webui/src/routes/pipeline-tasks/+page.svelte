<!-- src/routes/pipeline-tasks/+page.svelte -->
<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { taskStore } from "$lib/store/taskStore";
    import { pipelineStore } from "$lib/store/pipelineStore";
    import { RefreshCw } from "lucide-svelte";
    import PipelineSettingsPanel from "$lib/components/PipelineSettingsPanel.svelte";

    onMount(() => {
        taskStore.fetchTasks();
    });

    function submitTask() {
        taskStore.submitTask(
            $pipelineStore.pipelineType,
            $pipelineStore.formData,
        );
    }

    function resetSettings() {
        pipelineStore.reset();
    }

    function retryTask(taskId: number) {
        taskStore.retryTask(taskId);
    }

    function removeTask(taskId: number) {
        taskStore.removeTask(taskId);
    }

    function strToNumber(str: string): number {
        return Number(str);
    }

    let autoRefresh = false;
    let refreshInterval: ReturnType<typeof setInterval> | undefined = undefined;
    let countdown = 0;

    function startRefreshTimer() {
        // Only start if not already running
        if (refreshInterval === undefined) {
            countdown = 30; // Initialize countdown
            refreshInterval = setInterval(() => {
                if (countdown > 0) {
                    countdown--;
                } else {
                    taskStore.fetchTasks();
                    countdown = 30; // Reset after refresh
                }
            }, 1000);
        }
    }

    function stopRefreshTimer() {
        if (refreshInterval) {
            clearInterval(refreshInterval);
            refreshInterval = undefined;
            countdown = 0;
        }
    }

    // Toggle handler
    function handleAutoRefreshToggle(enabled: boolean) {
        autoRefresh = enabled;
        if (enabled) {
            startRefreshTimer();
        } else {
            stopRefreshTimer();
        }
    }

    // Cleanup
    onDestroy(stopRefreshTimer);
</script>

<svelte:head>
    <title>Pipeline Tasks | MindAI</title>
</svelte:head>

<main class="main-page">
    <h1>Pipeline Tasks</h1>

    <div class="task-sections">
        <section class="current-tasks">
            <h2>Current Tasks</h2>
            {#if $taskStore.loading}
                <p>Loading tasks...</p>
            {:else if $taskStore.error}
                <p class="error">{$taskStore.error}</p>
            {:else}
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Status</th>
                                <th>Pipeline</th>
                                <th>Conversation</th>
                                <th>Progress</th>
                                <th>Name</th>
                                <th>Timestamp</th>
                                <th>Finished On</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {#each $taskStore.tasks as task}
                                <tr>
                                    <td>{task.id}</td>
                                    <td class="status-cell {task.job_status}"
                                        >{task.job_status}</td
                                    >
                                    <td>{task.data.pipeline_type}</td>
                                    <td>
                                        <a
                                            href="/conversation/{task.data
                                                .config.conversation_id}"
                                        >
                                            {task.data.config.conversation_id}
                                        </a>
                                    </td>
                                    <td>
                                        <div class="progress-bar">
                                            <div
                                                class="progress-fill"
                                                style="width: {task.progress}%"
                                            ></div>
                                            <span class="progress-text"
                                                >{task.progress}%</span
                                            >
                                        </div>
                                    </td>
                                    <td>{task.name}</td>
                                    <td
                                        >{new Date(
                                            task.timestamp,
                                        ).toLocaleString()}</td
                                    >
                                    <td>
                                        {task.finishedOn
                                            ? new Date(
                                                  task.finishedOn,
                                              ).toLocaleString()
                                            : "N/A"}
                                    </td>
                                    <td class="action-buttons">
                                        <button
                                            class="retry-button"
                                            on:click={() =>
                                                retryTask(strToNumber(task.id))}
                                            disabled={!(
                                                task.job_status === "failed" ||
                                                task.job_status === "active"
                                            )}
                                        >
                                            Retry
                                        </button>
                                        <button
                                            class="cancel-button"
                                            on:click={() =>
                                                removeTask(
                                                    strToNumber(task.id),
                                                )}
                                        >
                                            Cancel
                                        </button>
                                    </td>
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                </div>
            {/if}
            <div class="refresh-controls">
                <button
                    class="refresh-button"
                    on:click={() => taskStore.fetchTasks()}
                    disabled={$taskStore.loading}
                    title="Refresh tasks"
                >
                    <RefreshCw size={16} />
                </button>
                <label class="refresh-label">
                    <input
                        type="checkbox"
                        checked={autoRefresh}
                        on:change={(e) =>
                            handleAutoRefreshToggle(e.currentTarget.checked)}
                    />
                    Auto-refresh every 30 seconds
                </label>
                {#if autoRefresh}
                    <span class="refresh-status"
                        >Next refresh in {countdown} seconds</span
                    >
                {/if}
            </div>
        </section>

        <section class="create-task">
            <h2>Create New Task</h2>
            <PipelineSettingsPanel />
            <button
                class="submit-button"
                on:click={submitTask}
                disabled={$taskStore.loading}
            >
                Create Task
            </button>
            <button
                class="submit-button"
                on:click={resetSettings}
                disabled={$taskStore.loading}
            >
                Reset Settings
            </button>
        </section>
    </div>
</main>

<style>
    .main-page {
        padding: 2rem;
    }

    .task-sections {
        display: flex;
        flex-direction: column;
        gap: 2rem;
    }

    .table-container {
        overflow-x: auto;
        background: white;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1rem;
    }

    th,
    td {
        padding: 0.75rem 1rem;
        text-align: left;
        border-bottom: 1px solid #e5e7eb;
    }

    th {
        background-color: #f9fafb;
        font-weight: 600;
        color: #374151;
    }

    tr:hover {
        background-color: #f9fafb;
    }

    .status-cell {
        text-transform: capitalize;
        font-weight: 500;
    }

    .status-cell.completed {
        color: #059669;
    }

    .status-cell.failed {
        color: #dc2626;
    }

    .status-cell.active {
        color: #2563eb;
    }

    .progress-bar {
        width: 100px;
        height: 20px;
        background-color: #e5e7eb;
        border-radius: 10px;
        overflow: hidden;
        position: relative;
    }

    .progress-fill {
        height: 100%;
        background-color: #4ade80;
        transition: width 0.3s ease;
    }

    .progress-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 0.75rem;
        font-weight: 500;
        color: #1f2937;
    }

    .action-buttons {
        display: flex;
        gap: 0.5rem;
    }

    button {
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.375rem;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.2s;
    }

    button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .retry-button {
        background-color: #3b82f6;
        color: white;
    }

    .retry-button:hover:not(:disabled) {
        background-color: #2563eb;
    }

    .cancel-button {
        background-color: #ef4444;
        color: white;
    }

    .cancel-button:hover {
        background-color: #dc2626;
    }

    .submit-button {
        background-color: #10b981;
        color: white;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
    }

    .submit-button:hover:not(:disabled) {
        background-color: #059669;
    }

    .error {
        color: #dc2626;
        font-weight: 500;
    }

    .refresh-controls {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background-color: #f9fafb;
        border-top: 1px solid #e5e7eb;
    }

    .refresh-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #4b5563;
        font-size: 0.875rem;
        cursor: pointer;
    }

    .refresh-label input[type="checkbox"] {
        width: 1rem;
        height: 1rem;
        cursor: pointer;
    }

    .refresh-status {
        font-size: 0.875rem;
        color: #6b7280;
        font-style: italic;
    }

    @media (max-width: 768px) {
        main {
            padding: 1rem;
        }

        .action-buttons {
            flex-direction: column;
        }
    }
    .refresh-button {
        background: none;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 0.5rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
        color: #666;
    }

    .refresh-button:hover:not(:disabled) {
        background: #f5f5f5;
        border-color: #ccc;
        color: #333;
    }

    .refresh-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .loading {
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
</style>
