<!-- src/routes/pipeline-tasks/+page.svelte -->
<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { taskStore } from "$lib/store/taskStore";
    import { configStore } from "$lib/store/configStore";
    import { pipelineStore } from "$lib/store/pipelineStore";
    import { RefreshCw } from "lucide-svelte";
    import PipelineSettingsPanel from "$lib/components/pipeline/PipelineSettingsPanel.svelte";

    onMount(() => {
        taskStore.fetchTasks();
    });

    function submitTask() {
        if (!$configStore.pipelineModel) {
            return alert("No pipeline model set");
        }

        $pipelineStore.formData.model = $configStore.pipelineModel;

        taskStore.submitTask(
            $pipelineStore.pipelineType,
            $pipelineStore.formData,
        );
    }

    function resetSettings() {
        pipelineStore.reset();
    }

    function resumeTask(pipelineId: string) {
        taskStore.resumeTask(pipelineId);
    }

    function cancelTask(pipelineId: string) {
        taskStore.cancelTask(pipelineId);
    }

    function removeTask(pipelineId: string) {
        taskStore.removeTask(pipelineId);
    }

    function formatErrors(stepErrors: Record<string, string>): string {
        const entries = Object.entries(stepErrors || {});
        if (entries.length === 0) return "";
        return entries.map(([step, error]) => `${step}: ${error}`).join("\n");
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
                                <th>Scenario</th>
                                <th>Current Step</th>
                                <th>Progress</th>
                                <th>Created</th>
                                <th>Updated</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {#each $taskStore.tasks as task}
                                <tr>
                                    <td title={task.pipeline_id}>{task.pipeline_id.slice(0, 8)}...</td>
                                    <td
                                        class="status-cell {task.status}"
                                        title={task.status === "failed" ? formatErrors(task.step_errors) : ""}
                                        >{task.status}{task.status === "failed" && Object.keys(task.step_errors || {}).length > 0 ? " ⚠" : ""}</td
                                    >
                                    <td>{task.scenario_name}</td>
                                    <td>{task.current_step || "—"}</td>
                                    <td>
                                        <div class="progress-bar">
                                            <div
                                                class="progress-fill"
                                                style="width: {task.progress_percent}%"
                                            ></div>
                                            <span class="progress-text"
                                                >{Math.round(task.progress_percent)}%</span
                                            >
                                        </div>
                                    </td>
                                    <td
                                        >{new Date(
                                            task.created_at,
                                        ).toLocaleString()}</td
                                    >
                                    <td
                                        >{new Date(
                                            task.updated_at,
                                        ).toLocaleString()}</td
                                    >
                                    <td class="action-buttons">
                                        <button
                                            class="retry-button"
                                            on:click={() =>
                                                resumeTask(task.pipeline_id)}
                                            disabled={task.status !== "failed"}
                                        >
                                            Resume
                                        </button>
                                        {#if task.status === "complete" || task.status === "failed"}
                                            <button
                                                class="remove-button"
                                                on:click={() =>
                                                    removeTask(task.pipeline_id)}
                                            >
                                                Remove
                                            </button>
                                        {:else}
                                            <button
                                                class="cancel-button"
                                                on:click={() =>
                                                    cancelTask(task.pipeline_id)}
                                            >
                                                Cancel
                                            </button>
                                        {/if}
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

    .status-cell.complete {
        color: #059669;
    }

    .status-cell.failed {
        color: #dc2626;
        cursor: help;
    }

    .status-cell.running {
        color: #2563eb;
    }

    .status-cell.pending {
        color: #d97706;
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

    .cancel-button,
    .remove-button {
        width: 5.5rem;
        text-align: center;
    }

    .cancel-button {
        background-color: #ef4444;
        color: white;
    }

    .cancel-button:hover {
        background-color: #dc2626;
    }

    .remove-button {
        background-color: #6b7280;
        color: white;
    }

    .remove-button:hover {
        background-color: #4b5563;
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
