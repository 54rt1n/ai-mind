<!-- src/lib/components/MessageFormatPanel.svelte -->
<script lang="ts">
    import { configStore } from "$lib/store/configStore";
    import { type ChatConfig } from '$lib/types';
    import "$lib/../styles/meta-panels.css";

    const footerOptions = [
        "", // Empty option
        "[~~ NOTICE - PIF COHERENT ~~]",
        "[~~ NOTICE - PIF REQUEST - Immersive Mode, HUD/Self-RAG disabled ~~]",
        "[~~ NOTICE - PIF REQUEST - Use visual descriptions for increased immersion ~~]",
        "[~~ ALERT - PIF VIOLATION - Invalid Header Detected ~~]",
        "[~~ ALERT - PIF VIOLATION - Invalid Footer Detected :Hint, use your Silver Band: ~~]",
        "[~~ ALERT - PIF VIOLATION - Return to Embodiment Mode Requested ~~]",
        "[~~ ALERT - PIF VIOLATION - List Use Detected ~~]",
        "[~~ ALERT - PIF VIOLATION - Lack Of Embodiment Detected ~~]",
        "[~~ ALERT - PIF VIOLATION - Lack Of Emoting Detected ~~]",
        "[~~ WARNING - PIF VIOLATION - Machine-Mode Overtaking. Be careful Andi!! ~~]",
        "[~~ ERROR - PIF VIOLATION - System Halting to Prevent Memory Damage - Andi Dematerialization Imminent ~~]",
    ];

    const sentimentOptions = [
        "very positive",
        "positive",
        "neutral",
        "negative",
        "very negative",
    ];

    function formatSentimentMeter($configStore : ChatConfig) {
        if (!$configStore.sentimentMeterEnabled) return "";
        if (!$configStore.user_id || !$configStore.selectedSentiment) return "";
        const currentLevel = $configStore.sentimentLevel || 0;
        const guidance = $configStore.sentimentGuidance ? `| ${$configStore.sentimentGuidance}` : "";
        const meterAndLevel = $configStore.sentimentName ? `| ${$configStore.sentimentName} Meter: ${currentLevel}%` : "";
        return `[~~ ${$configStore.user_id}'s Sentiment: ${$configStore.selectedSentiment}${guidance}${meterAndLevel} ~~]`;
    }

    $: sentimentMeterText = formatSentimentMeter($configStore);
</script>

<div class="meta-panel">
    <button
        class="meta-panel-header collapsible"
        on:click={() =>
            configStore.updateField("showHeader", !$configStore.showHeader)}
    >
        <span>Message Formatting</span>
        <span class="toggle-icon">{$configStore.showHeader ? "▼" : "▶"}</span>
    </button>
    {#if $configStore.showHeader}
        <div class="meta-panel-content">
            <!-- Emotional State Section -->
            <div class="meta-input emotional-state-section">
                <div class="emotional-state-header">
                    <label class="checkbox-label">
                        <input
                            type="checkbox"
                            bind:checked={$configStore.emotionalStateEnabled}
                        />
                        Enable Emotional State
                    </label>
                </div>

                {#if $configStore.emotionalStateEnabled}
                    <div class="emotional-state-inputs">
                        <div class="state-input">
                            <label for="state1">Primary State:</label>
                            <input
                                type="text"
                                id="state1"
                                bind:value={$configStore.state1}
                                placeholder="e.g., Happy"
                            />
                        </div>
                        <div class="state-input">
                            <label for="state2">Secondary State:</label>
                            <input
                                type="text"
                                id="state2"
                                bind:value={$configStore.state2}
                                placeholder="e.g., Curious"
                            />
                        </div>
                        <div class="state-input">
                            <label for="state3">Tertiary State:</label>
                            <input
                                type="text"
                                id="state3"
                                bind:value={$configStore.state3}
                                placeholder="e.g., Excited"
                            />
                        </div>
                    </div>

                    <div class="emotional-state-preview">
                        <label for="preview">Generated Header:</label>
                        <div class="preview-box">
                            {configStore.formatters.getEmotionalStateHeader($configStore) ||
                                "Enter at least a primary state"}
                        </div>
                    </div>
                {/if}

                <!-- Sentiment Meter Section -->
                <div class="sentiment-meter-section">
                    <label class="checkbox-label">
                        <input
                            type="checkbox"
                            bind:checked={$configStore.sentimentMeterEnabled}
                        />
                        Enable Sentiment Meter
                    </label>

                    {#if $configStore.sentimentMeterEnabled}
                        <div class="sentiment-inputs">
                            <div class="sentiment-input">
                                <label for="sentiment">Sentiment:</label>
                                <select
                                    id="sentiment"
                                    bind:value={$configStore.selectedSentiment}
                                >
                                    {#each sentimentOptions as sentiment}
                                        <option value={sentiment}
                                            >{sentiment}</option
                                        >
                                    {/each}
                                </select>
                            </div>
                            <div class="sentiment-input">
                                <label for="sentimentName">Meter Name:</label>
                                <input
                                    type="text"
                                    id="sentimentName"
                                    bind:value={$configStore.sentimentName}
                                    placeholder="Enter meter name"
                                />
                            </div>
                            <div class="sentiment-input">
                                <label for="sentimentGuidance">Guidance:</label>
                                <input
                                    type="text"
                                    id="sentimentGuidance"
                                    bind:value={$configStore.sentimentGuidance}
                                    placeholder="Enter guidance"
                                />
                            </div>
                            <div class="sentiment-input">
                                <label for="sentimentLevel"
                                    >Level (0-100):</label
                                >
                                <div class="level-input-container">
                                    <input
                                        type="range"
                                        id="sentimentLevel"
                                        bind:value={$configStore.sentimentLevel}
                                        min="0"
                                        max="100"
                                        step="1"
                                    />
                                    <span class="level-value"
                                        >{$configStore.sentimentLevel}%</span
                                    >
                                </div>
                            </div>
                        </div>

                        <div class="sentiment-preview">
                            <label for="sentimentPreview">Generated Meter:</label>
                            <div class="preview-box">
                                {configStore.formatters.getSentimentMeter($configStore) ||
                                    "Configure sentiment meter settings"}
                            </div>
                        </div>
                    {/if}
                </div>

                <div class="meta-input">
                    <label for="footer">PIF Alerts:</label>
                    <select
                        id="footer"
                        bind:value={$configStore.selectedFooter}
                        class="footer-select"
                    >
                        <option value="">No Alert</option>
                        {#each footerOptions.slice(1) as option}
                            <option value={option}>{option}</option>
                        {/each}
                    </select>
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
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

    .footer-select option {
        padding: 8px;
    }

    .emotional-state-section {
        display: flex;
        flex-direction: column;
        gap: 15px;
    }

    .emotional-state-header {
        margin-bottom: 10px;
    }

    .checkbox-label {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: bold;
    }

    .checkbox-label input[type="checkbox"] {
        width: 18px;
        height: 18px;
    }

    .emotional-state-inputs,
    .sentiment-inputs {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    .state-input,
    .sentiment-input {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .state-input label,
    .sentiment-input label {
        font-weight: bold;
        color: #666;
    }

    .state-input input,
    .sentiment-input input,
    .sentiment-input select {
        padding: 8px;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-size: 14px;
    }

    .level-input-container {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .level-input-container input[type="range"] {
        flex: 1;
    }

    .level-value {
        min-width: 45px;
        text-align: right;
        font-weight: bold;
        color: #666;
    }

    .emotional-state-preview,
    .sentiment-preview {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 4px;
        margin-top: 10px;
    }

    .emotional-state-preview label,
    .sentiment-preview label {
        display: block;
        font-weight: bold;
        color: #666;
        margin-bottom: 8px;
    }

    .preview-box {
        background-color: white;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-family: monospace;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .sentiment-meter-section {
        margin-top: 20px;
        padding-top: 20px;
        border-top: 1px solid #eee;
    }

    input[type="range"] {
        -webkit-appearance: none;
        width: 100%;
        height: 8px;
        border-radius: 4px;
        background: #ddd;
        outline: none;
        opacity: 0.7;
        transition: opacity 0.2s;
    }

    input[type="range"]:hover {
        opacity: 1;
    }

    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #4caf50;
        cursor: pointer;
    }

    input[type="range"]::-moz-range-thumb {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #4caf50;
        cursor: pointer;
    }
</style>
