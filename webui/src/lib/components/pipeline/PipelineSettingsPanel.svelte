<!-- src/lib/components/PipelineSettingsPanel.svelte -->
<script lang="ts">
    import { pipelineStore } from "$lib/store/pipelineStore";
    import { configStore } from "$lib/store/configStore";
    import type { PipelineType } from "$lib/types";
    import "$lib/../styles/meta-panels.css";
    import { modelStore } from "$lib/store/modelStore";
    import ModelSelect from "$lib/components/model/ModelSelect.svelte";

    const pipelineTypes: PipelineType[] = [
        "analyst",
        "coder",
        "dreamer",
        "journaler",
        "philosopher",
        "reporter",
        "summarizer",
    ];
</script>

<div class="meta-panel">
    <button
        class="meta-panel-header collapsible"
        on:click={pipelineStore.toggleExpanded}
    >
        <span>Pipeline Configuration</span>
        <span class="toggle-icon">{$pipelineStore.isExpanded ? "▼" : "▶"}</span
        >
    </button>

    {#if $pipelineStore.isExpanded}
        <div class="meta-panel-content">
            <div class="meta-input">
                <label>
                    Pipeline Type:
                    <select
                        bind:value={$pipelineStore.pipelineType}
                        on:change={(e) =>
                            pipelineStore.setPipelineType(
                                e.currentTarget.value,
                            )}
                    >
                        {#each pipelineTypes as type}
                            <option value={type}>{type}</option>
                        {/each}
                    </select>
                </label>
            </div>
            <div class="meta-input">
                <div class="model-select-container">
                    {#if Array.isArray($modelStore.models)}
                        <ModelSelect
                            bind:value={$configStore.pipelineModel}
                            category="analysis"
                        />
                        <span>Model: {$configStore.pipelineModel}</span>
                    {:else}
                        <span>No Models Found</span>
                    {/if}
                </div>
                {#if $modelStore.error}
                    <span class="error-message">{$modelStore.error}</span>
                {/if}
            </div>
            <div class="meta-input">
                <label>
                    User ID:
                    <input
                        type="text"
                        value={$pipelineStore.formData.user_id || ""}
                        on:input={(e) =>
                            pipelineStore.updateFormData({
                                user_id: e.currentTarget.value,
                            })}
                    />
                </label>
            </div>

            <div class="meta-input">
                <label>
                    Persona ID:
                    <input
                        type="text"
                        value={$pipelineStore.formData.persona_id || ""}
                        on:input={(e) =>
                            pipelineStore.updateFormData({
                                persona_id: e.currentTarget.value,
                            })}
                    />
                </label>
            </div>

            <div class="meta-input">
                <label>
                    Conversation ID:
                    <input
                        type="text"
                        value={$pipelineStore.formData.conversation_id || ""}
                        on:input={(e) =>
                            pipelineStore.updateFormData({
                                conversation_id: e.currentTarget.value,
                            })}
                    />
                </label>
            </div>

            <div class="meta-input">
                <label>
                    Mood:
                    <input
                        type="text"
                        value={$pipelineStore.formData.mood || ""}
                        on:input={(e) =>
                            pipelineStore.updateFormData({
                                mood: e.currentTarget.value,
                            })}
                    />
                </label>
            </div>

            <div class="meta-input">
                <label>
                    Guidance:
                    <textarea
                        value={$pipelineStore.formData.guidance || ""}
                        on:input={(e) =>
                            pipelineStore.updateFormData({
                                guidance: e.currentTarget.value,
                            })}
                    ></textarea>
                </label>
            </div>

            <div class="meta-input">
                <label>
                    Top N:
                    <input
                        type="number"
                        value={$pipelineStore.formData.top_n || ""}
                        on:input={(e) =>
                            pipelineStore.updateFormData({
                                top_n:
                                    parseInt(e.currentTarget.value) ||
                                    undefined,
                            })}
                    />
                </label>
            </div>

            {#if $pipelineStore.pipelineType !== "analyst"}
                <div class="meta-input">
                    <label>
                        Query:
                        <textarea
                            value={$pipelineStore.formData.query_text || ""}
                            on:input={(e) =>
                                pipelineStore.updateFormData({
                                    query_text: e.currentTarget.value,
                                })}
                        ></textarea>
                    </label>
                </div>
            {/if}
        </div>
    {/if}
</div>

<style>
    input[type="text"],
    input[type="number"],
    select,
    textarea {
        width: 100%;
        padding: 0.5rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 0.9rem;
    }

    textarea {
        min-height: 100px;
        resize: vertical;
    }

    .model-select-container {
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }

    @keyframes spin {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }

    .error-message {
        color: #dc2626;
        font-size: 0.875rem;
        margin-top: 0.25rem;
        display: block;
    }
</style>
