// lib/store/taskStore.ts
import { writable } from 'svelte/store';
import type { PipelineType, BasePipelineSchema } from '$lib';
import { api } from '$lib';

interface Task {
    id: string;
    job_status: string;
    progress: number;
    name: string;
    timestamp: number;
    finishedOn: number | null;
    data: {
        pipeline_type: string;
        config: {
            user_id?: string | null;
            persona_id?: string | null;
            conversation_id?: string | null;
            mood?: string | null;
            no_retry?: boolean | null;
            guidance?: string | null;
            top_n?: number | null;
            query_text?: string | null;
        }
    }
}

interface TaskStore {
    tasks: Task[];
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
                const data = await api.getPipelineTasks();
                update(store => ({ ...store, tasks: data.jobs, loading: false }));
            } catch (error) {
                update(store => ({ ...store, error: 'Failed to fetch tasks', loading: false }));
            }
        },
        submitTask: async (pipelineType: PipelineType, formData: BasePipelineSchema) => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const result = await api.createPipelineTask(pipelineType, formData);
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
        retryTask: async (taskId: number) => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const result = await api.retryPipelineTask(taskId);
                if (result.status === 'success') {
                    await taskStore.fetchTasks();
                } else {
                    throw new Error(result.message);
                }
            } catch (error) {
                update(store => ({ ...store, error: 'Failed to retry task', loading: false }));
                alert(error);
            }
        },
        removeTask: async (taskId: number) => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const result = await api.removePipelineTask(taskId);
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