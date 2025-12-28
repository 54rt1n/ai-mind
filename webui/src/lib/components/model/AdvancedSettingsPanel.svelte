<!-- src/lib/components/AdvancedSettingsPanel.svelte -->
<script lang="ts">
    import { configStore } from "$lib/store/configStore";
    import "$lib/../styles/meta-panels.css";
    import ModelSelect from "./ModelSelect.svelte";

    export let hidePersona = false;

    const tokenOptions = [
        { value: 256, label: "Quarter CTX (256)" },
        { value: 384, label: "Mid CTX (384)" },
        { value: 512, label: "Half CTX (512)" },
        { value: 768, label: "Large CTX (768)" },
        { value: 1024, label: "Full CTX (1024)" },
        { value: 1536, label: "Extra Large CTX (1536)" },
        { value: 2048, label: "Huge CTX (2048)" },
        { value: 3072, label: "Massive CTX (3072)" },
        { value: 4096, label: "Maximum CTX (4096)" },
    ];
</script>

<div class="meta-panel">
    <button
        class="meta-panel-header collapsible"
        on:click={() =>
            configStore.updateField("showAdvanced", !$configStore.showAdvanced)}
    >
        <span>Advanced Settings</span>
        <span class="toggle-icon">{$configStore.showAdvanced ? "▼" : "▶"}</span
        >
    </button>
    {#if $configStore.showAdvanced}
        <div class="meta-panel-content">
            {#if !hidePersona}
                <div class="meta-input">
                    <label for="system">System Message:</label>
                    <textarea
                        id="system"
                        bind:value={$configStore.systemMessage}
                        rows="2"
                        placeholder="Optional system message..."
                    />
                </div>
                <div class="meta-input">
                    <label for="location">Location:</label>
                    <input
                        type="text"
                        id="location"
                        bind:value={$configStore.location}
                        placeholder="Optional location..."
                    />
                </div>
                <div class="meta-input">
                    <label for="mood">Mood:</label>
                    <input
                        type="text"
                        id="mood"
                        bind:value={$configStore.mood}
                        placeholder="Agent Mood..."
                    />
                </div>
            {/if}
            <ModelSelect
                bind:value={$configStore.chatModel}
                category="conversation"
            />
            <div class="meta-input">
                <label for="max-tokens">Max Tokens:</label>
                <select id="max-tokens" bind:value={$configStore.maxTokens}>
                    {#each tokenOptions as option}
                        <option value={option.value}>{option.label}</option>
                    {/each}
                </select>
            </div>
            <div class="meta-input">
                <label for="temperature"
                    >Temperature ({$configStore.temperature}):</label
                >
                <input
                    type="range"
                    id="temperature"
                    bind:value={$configStore.temperature}
                    min="0"
                    max="2.5"
                    step="0.01"
                />
            </div>
            <div class="meta-input">
                <label for="frequencyPenalty"
                    >Frequency Penalty ({$configStore.frequencyPenalty}):</label
                >
                <input
                    type="range"
                    id="frequencyPenalty"
                    bind:value={$configStore.frequencyPenalty}
                    min="0"
                    max="1.5"
                    step="0.01"
                />
            </div>
            <div class="meta-input">
                <label for="presencePenalty"
                    >Presence Penalty ({$configStore.presencePenalty}):</label
                >
                <input
                    type="range"
                    id="presencePenalty"
                    bind:value={$configStore.presencePenalty}
                    min="0"
                    max="2.5"
                    step="0.01"
                />
            </div>
            <div class="meta-input">
                <label for="repetitionPenalty"
                    >Repetition Penalty ({$configStore.repetitionPenalty}):</label
                >
                <input
                    type="range"
                    id="repetitionPenalty"
                    bind:value={$configStore.repetitionPenalty}
                    min="0"
                    max="2.5"
                    step="0.01"
                />
            </div>
            <div class="meta-input">
                <label for="minP">Min P ({$configStore.minP}):</label>
                <input
                    type="range"
                    id="minP"
                    bind:value={$configStore.minP}
                    min="0"
                    max="1"
                    step="0.01"
                />
            </div>
            <div class="meta-input">
                <label for="topP">Top P ({$configStore.topP}):</label>
                <input
                    type="range"
                    id="topP"
                    bind:value={$configStore.topP}
                    min="0"
                    max="1"
                    step="0.01"
                />
            </div>
            <div class="meta-input">
                <label for="topK">Top K ({$configStore.topK}):</label>
                <input
                    type="range"
                    id="topK"
                    bind:value={$configStore.topK}
                    min="-1"
                    max="100"
                    step="1"
                />
            </div>
        </div>
    {/if}
</div>

<style>
    input[type="range"] {
        width: 100%;
        height: 8px;
        border-radius: 4px;
        background: #ddd;
        outline: none;
        padding: 0;
        margin: 8px 0;
    }

    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #4caf50;
        cursor: pointer;
        transition: background 0.15s ease-in-out;
    }

    input[type="range"]::-webkit-slider-thumb:hover {
        background: #45a049;
    }
</style>
