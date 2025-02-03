// lib/store/notificationStore.ts
import { writable } from 'svelte/store';
import type { NotificationParams } from '$lib/types';

const defaultNotification: NotificationParams = {
    show: false,
    notificationType: 'info',
    message: '',
    loading: false,
    duration: 3000
};

function createNotificationStore() {
    const { subscribe, set, update } = writable<NotificationParams>(defaultNotification);

    let timeoutId: ReturnType<typeof setTimeout>;

    return {
        subscribe,
        show: (params: Omit<NotificationParams, 'show'>) => {
            // Clear any existing timeout
            if (timeoutId) clearTimeout(timeoutId);

            // Show the notification
            set({ ...params, show: true });

            // If duration is specified and not loading, auto-hide
            if (params.duration && !params.loading) {
                timeoutId = setTimeout(() => {
                    set(defaultNotification);
                }, params.duration);
            }
        },
        hide: () => {
            if (timeoutId) clearTimeout(timeoutId);
            set(defaultNotification);
        },
        loading: (message: string) => {
            if (timeoutId) clearTimeout(timeoutId);
            set({
                show: true,
                notificationType: 'loading',
                message,
                loading: true
            });
        },
        success: (message: string, duration = 3000) => {
            if (timeoutId) clearTimeout(timeoutId);
            set({
                show: true,
                notificationType: 'success',
                message,
                loading: false,
                duration
            });
        },
        error: (message: string, duration = 3000) => {
            if (timeoutId) clearTimeout(timeoutId);
            set({
                show: true,
                notificationType: 'error',
                message,
                loading: false,
                duration
            });
        }
    };
}

export const notificationStore = createNotificationStore();