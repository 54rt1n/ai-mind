<!-- src/routes/program/+page.svelte -->
<script lang="ts">
    import { Plus, Play, Trash2, RefreshCw } from "lucide-svelte";
    import ActionCard from "$lib/components/toolbox/ActionCard.svelte";
    import DragDropContext from "$lib/components/toolbox/DragDropContext.svelte";
    import {
        toolboxStore,
        type ActionCard as ActionCardType,
    } from "$lib/store/toolboxStore";
    //import "../../styles/chat.css";
    import "../../styles/meta-panels.css";
    import ClipboardPanel from "$lib/components/clipboard/ClipboardPanel.svelte";
    import { configStore } from "$lib/store/configStore";
    import ThoughtPanel from "$lib/components/thought/ThoughtPanel.svelte";
    import { thoughtStore } from "$lib/store/thoughtStore";
    import NotificationToast from "$lib/components/ui/NotificationToast.svelte";

    let showTypeSelector = false;
    let typeSelectorPosition = { x: 0, y: 0 };
    let thoughtPanelComponent: ThoughtPanel;
    let clipboardPanelComponent: ClipboardPanel;
    let showToast = false;
    let toastMessage = "";
    let toastType: "info" | "error" | "success" | "warning" = "info";

    async function executeCard(card: ActionCardType) {
        try {
            // Check if any card is already running
            if ($toolboxStore.cards.some((c) => c.status === "running")) {
                toastMessage =
                    "Please wait for the current execution to complete before running another action.";
                toastType = "warning";
                showToast = true;
                return;
            }

            // Set card to running state at the start
            toolboxStore.updateCardStatus(card.id, "running");

            let conversationHistory = toolboxStore.getConversationHistory(
                card.id,
            );

            // Skip if thought turns is 0
            if (card.thoughtTurns === 0) {
                await toolboxStore.executeCardCascade(card.id);
                return;
            }

            // First generate initial thoughts
            if (
                !(await thoughtPanelComponent.clearAndGenerate(
                    conversationHistory,
                ))
            ) {
                console.error("Failed to generate initial thought.");
                if (
                    !(await thoughtPanelComponent.clearAndGenerate(
                        conversationHistory,
                    ))
                ) {
                    throw new Error(
                        "Failed to generate initial thought after retry.",
                    );
                }
            }

            if (!(await thoughtPanelComponent.validateAndAcceptThought())) {
                throw new Error("Failed to validate initial thought.");
            }

            // For turns > 1, advance thought the specified number of times
            for (let turn = 1; turn < card.thoughtTurns; turn++) {
                console.log(`Advancing thought turn ${turn + 1}...`);

                if (
                    !(await thoughtPanelComponent.advanceThought(
                        conversationHistory,
                    ))
                ) {
                    console.error(
                        `Failed to advance thought turn ${turn + 1}.`,
                    );
                    if (
                        !(await thoughtPanelComponent.advanceThought(
                            conversationHistory,
                        ))
                    ) {
                        throw new Error(
                            `Failed to advance thought turn ${turn + 1} after retry.`,
                        );
                    }
                }

                if (!(await thoughtPanelComponent.validateAndAcceptThought())) {
                    throw new Error(
                        `Failed to validate thought turn ${turn + 1}.`,
                    );
                }
            }

            // Update the card's thoughts with the latest content
            toolboxStore.updateCardConfig(card.id, {
                thoughts: $thoughtStore.thoughtContent,
            });

            // Now execute the card cascade
            await toolboxStore.executeCardCascade(card.id);

            // Set success state
            toolboxStore.updateCardStatus(card.id, "completed");
        } catch (error) {
            console.error("Card execution failed:", error);
            toolboxStore.updateCardStatus(
                card.id,
                "error",
                error instanceof Error
                    ? error.message
                    : "An unknown error occurred",
            );
        }
    }

    async function handleExecuteCard(
        event: CustomEvent<{ card: ActionCardType }>,
    ) {
        const { card } = event.detail;
        console.log("handleExecuteCard", card);
        await executeCard(card);
    }

    function handleAddCard(event: MouseEvent) {
        const button = event.currentTarget as HTMLButtonElement;
        const rect = button.getBoundingClientRect();
        typeSelectorPosition = {
            x: rect.left,
            y: rect.bottom + window.scrollY,
        };
        showTypeSelector = true;
    }

    function handleTypeSelect(type: "simple" | "tool") {
        const user_id = $configStore.user_id;
        const persona_id = $configStore.persona_id;

        if (user_id === null || persona_id === null) {
            toastMessage = "Please select a user and persona first.";
            toastType = "error";
            showToast = true;
            return;
        }

        let system_message = "";
        if (type === "tool") {
            system_message =
                "Follow the tool instructions carefully and return the results in a JSON object.";
        } else {
            system_message = "";
        }
        toolboxStore.addCard(type, null, user_id, persona_id, system_message);
        showTypeSelector = false;
    }

    function handleClickOutside(event: MouseEvent) {
        const target = event.target as HTMLElement;
        if (
            !target.closest(".type-selector") &&
            !target.closest(".add-card-button")
        ) {
            showTypeSelector = false;
        }
    }
</script>

<svelte:window on:click={handleClickOutside} />

<NotificationToast
    bind:show={showToast}
    message={toastMessage}
    notificationType={toastType}
/>

<div class="program-container">
    <div class="toolbar">
        <div class="left">
            <button class="add-card-button" on:click={handleAddCard}>
                <Plus size={16} />
                Add Card
            </button>
            {#if showTypeSelector}
                <div
                    class="type-selector"
                    style="left: {typeSelectorPosition.x}px; top: {typeSelectorPosition.y}px;"
                >
                    <button on:click={() => handleTypeSelect("simple")}>
                        Simple Action
                    </button>
                    <button on:click={() => handleTypeSelect("tool")}>
                        Tool Action
                    </button>
                </div>
            {/if}
        </div>
        <div class="right">
            <button
                class="reset-button"
                on:click={() => toolboxStore.resetAllCards()}
                title="Reset all cards to idle state"
            >
                <RefreshCw size={16} />
                Reset States
            </button>
            <button
                class="clear-button"
                on:click={() => toolboxStore.clearAll()}
            >
                <Trash2 size={16} />
                Clear All
            </button>
            <button
                class="execute-button"
                on:click={async () => {
                    for (const card of $toolboxStore.cards) {
                        await executeCard(card);
                    }
                }}
            >
                <Play size={16} />
                Execute Pipeline
            </button>
        </div>
    </div>

    <ClipboardPanel bind:this={clipboardPanelComponent} />
    <ThoughtPanel bind:this={thoughtPanelComponent} />

    <div class="cards-container">
        {#if $toolboxStore.loading}
            <div class="loading">Executing pipeline...</div>
        {/if}

        {#if $toolboxStore.error}
            <div class="error-message">{$toolboxStore.error}</div>
        {/if}

        <DragDropContext let:card>
            <ActionCard {card} on:executeCard={handleExecuteCard} />
        </DragDropContext>

        {#if $toolboxStore.cards.length === 0}
            <div class="empty-state">
                <h2>No Action Cards</h2>
                <p>Click "Add Card" to start building your pipeline.</p>
            </div>
        {/if}
    </div>
</div>

<style>
    .program-container {
        padding: 2rem;
    }

    .toolbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }

    .left,
    .right {
        display: flex;
        gap: 1rem;
        align-items: center;
    }

    .add-card-button,
    .clear-button,
    .execute-button {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        font-weight: 500;
        cursor: pointer;
        border: none;
    }

    .add-card-button {
        background: #4caf50;
        color: white;
    }

    .clear-button {
        background: #f3f4f6;
        color: #6b7280;
    }

    .clear-button:hover {
        background: #fee2e2;
        color: #dc2626;
    }

    .execute-button {
        background: #2563eb;
        color: white;
    }

    .type-selector {
        position: absolute;
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        box-shadow:
            0 4px 6px -1px rgba(0, 0, 0, 0.1),
            0 2px 4px -1px rgba(0, 0, 0, 0.06);
        overflow: hidden;
        z-index: 50;
    }

    .type-selector button {
        display: block;
        width: 100%;
        padding: 0.75rem 1rem;
        text-align: left;
        border: none;
        background: none;
        cursor: pointer;
    }

    .type-selector button:hover {
        background: #f3f4f6;
    }

    .type-selector button + button {
        border-top: 1px solid #e5e7eb;
    }

    .cards-container {
        position: relative;
    }

    .loading {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.125rem;
        font-weight: 500;
        color: #4b5563;
    }

    .error-message {
        padding: 1rem;
        margin-bottom: 1rem;
        background: #fee2e2;
        color: #dc2626;
        border-radius: 0.375rem;
        font-weight: 500;
    }

    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        background: #f9fafb;
        border: 2px dashed #e5e7eb;
        border-radius: 0.5rem;
    }

    .empty-state h2 {
        margin: 0 0 0.5rem 0;
        font-size: 1.25rem;
        font-weight: 600;
        color: #374151;
    }

    .empty-state p {
        margin: 0;
        color: #6b7280;
    }

    .reset-button {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        font-weight: 500;
        cursor: pointer;
        border: none;
        background: #f59e0b;
        color: white;
    }

    .reset-button:hover {
        background: #d97706;
    }
</style>
