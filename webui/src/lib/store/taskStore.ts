// lib/store/taskStore.ts
import { writable } from 'svelte/store';
import type { PipelineType, BasePipelineSchema } from '$lib';
import { api } from '$lib';

interface DreamerPipeline {
    pipeline_id: string;
    scenario_name: string;
    status: string;  // 'pending' | 'running' | 'complete' | 'failed'
    current_step: string | null;
    completed_steps: string[];
    failed_steps: string[];
    step_errors: Record<string, string>;
    progress_percent: number;
    created_at: string;
    updated_at: string;
}

interface TaskStore {
    tasks: DreamerPipeline[];
    loading: boolean;
    error: string | null;
}

function createTaskStore() {
    const { subscribe, set, update } = writable<TaskStore>({
        tasks: [],
        loading: false,
        error: null,
    });

    let taskStore = {
        subscribe,
        fetchTasks: async () => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const data = await api.getDreamerPipelines();
                update(store => ({ ...store, tasks: data.pipelines || [], loading: false }));
            } catch (error) {
                update(store => ({ ...store, error: 'Failed to fetch tasks', loading: false }));
            }
        },
        submitTask: async (pipelineType: PipelineType, formData: BasePipelineSchema) => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const result = await api.createDreamerPipeline(
                    pipelineType,
                    formData.conversation_id || '',
                    formData.persona_id || '',
                    formData.model || 'gemini-2.0-flash',
                    {
                        userId: formData.user_id,
                        queryText: formData.query_text,
                        guidance: formData.guidance,
                        mood: formData.mood,
                    },
                );
                if (result.status === 'success') {
                    await taskStore.fetchTasks();
                } else {
                    throw new Error(result.message);
                }
            } catch (error) {
                update(store => ({ ...store, error: 'Failed to submit task', loading: false }));
                alert(error);
            }
        },
        resumeTask: async (pipelineId: string) => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const result = await api.resumeDreamerPipeline(pipelineId);
                if (result.status === 'success') {
                    await taskStore.fetchTasks();
                } else {
                    throw new Error(result.message);
                }
            } catch (error) {
                update(store => ({ ...store, error: 'Failed to resume task', loading: false }));
                alert(error);
            }
        },
        cancelTask: async (pipelineId: string) => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const result = await api.cancelDreamerPipeline(pipelineId);
                if (result.status === 'success') {
                    await taskStore.fetchTasks();
                } else {
                    throw new Error(result.message);
                }
            } catch (error) {
                update(store => ({ ...store, error: 'Failed to cancel task', loading: false }));
                alert(error);
            }
        },
        removeTask: async (pipelineId: string) => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const result = await api.deleteDreamerPipeline(pipelineId);
                if (result.status === 'success') {
                    await taskStore.fetchTasks();
                } else {
                    throw new Error(result.message);
                }
            } catch (error) {
                update(store => ({ ...store, error: 'Failed to remove task', loading: false }));
                alert(error);
            }
        }
    };

    return taskStore;
}

export const taskStore = createTaskStore();