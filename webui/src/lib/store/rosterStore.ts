// lib/store/rosterStore.ts
import { writable } from 'svelte/store';
import { api } from '$lib';
import type { Persona, PersonaSection } from '$lib';
import { browser } from '$app/environment';
import { get } from 'svelte/store';

interface RosterStore {
    personas: Persona[];
    activePersona: Partial<Persona> | null;
    updates: Partial<Persona> | null;
    loading: boolean;
    error: string | null;
}

export function getDefaultFormData(): Partial<Persona> {
    return {
        chat_strategy: 'xmlmemory',
        name: '',
        full_name: '',
        attributes: {},
        features: {},
        wakeup: [''],
        base_thoughts: [''],
        pif: {
            core_traits: ''
        },
        default_location: '',
        wardrobe: {},
        current_outfit: 'default',
        persona_version: '0.1a',
        system_header: 'Please follow directions, being precise and methodical, utilizing Chain of Thought, Self-RAG, and Semantic Keywords.',
        include_date: true
    };
}

function sanitizePropertyName(name: string): string {
    // Replace spaces and special characters with underscores
    return name.replace(/[^a-zA-Z0-9_]/g, '_');
}

function createRosterStore() {
    const STORAGE_KEY = 'personaRoster';
    
    // Initialize from localStorage if in browser environment
    const initialState: RosterStore = browser
        ? JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null') || {
            personas: [],
            activePersona: null,
            updates: null,
            loading: false,
            error: null
        }
        : {
            personas: [],
            activePersona: null,
            updates: null,
            loading: false,
            error: null
        };

    const { subscribe, set, update } = writable<RosterStore>(initialState);

    // Save to localStorage when store updates (browser only)
    if (browser) {
        subscribe(state => {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        });
    }

    return {
        subscribe,
        fetchPersonas: async () => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const response = await api.getRoster();
                update(store => ({
                    ...store,
                    personas: response.personas,
                    loading: false
                }));
            } catch (error) {
                update(store => ({
                    ...store,
                    error: 'Failed to fetch personas',
                    loading: false
                }));
            }
        },
        initialize: async () => {
            const state = get({ subscribe });
            if (state.personas.length === 0) {
                await rosterStore.fetchPersonas();
            }
        },
        getPersona: async (personaId: string) => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const response = await api.getPersona(personaId);
                update(store => {
                    // Find the persona in the roster
                    const persona = store.personas.find(p => p.persona_id === personaId);
                    // If the persona is found, update the active persona, and the entry in the roster
                    if (persona) {
                        const updatedPersonas = store.personas.map(p => p.persona_id === personaId ? response : p);
                        const activeCopy = { ...response };
                        return { ...store, updates: {}, activePersona: activeCopy, personas: updatedPersonas };
                    } else {
                        // If the persona is not found, add it to the roster
                        const updatedPersonas = [...store.personas, response];
                        const activeCopy = { ...response };
                        return { ...store, updates: {}, activePersona: activeCopy, personas: updatedPersonas };
                    }
                });
                return response;
            } catch (error) {
                update(store => ({ ...store, error: 'Failed to fetch persona', loading: false }));
                throw error;
            }
        },
        createPersona: async (persona: Omit<Persona, 'persona_id'>) => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const response = await api.createPersona(persona);
                await rosterStore.fetchPersonas();
                return response;
            } catch (error) {
                update(store => ({
                    ...store,
                    error: 'Failed to create persona',
                    loading: false
                }));
                throw error;
            }
        },
        updatePersona: async (personaId: string, updates: Partial<Persona>) => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                const response = await api.updatePersona(personaId, updates);
                await rosterStore.fetchPersonas();
                return response;
            } catch (error) {
                update(store => ({
                    ...store,
                    error: 'Failed to update persona',
                    loading: false
                }));
                throw error;
            }
        },
        deletePersona: async (personaId: string) => {
            update(store => ({ ...store, loading: true, error: null }));
            try {
                await api.deletePersona(personaId);
                await rosterStore.fetchPersonas();
            } catch (error) {
                update(store => ({
                    ...store,
                    error: 'Failed to delete persona',
                    loading: false
                }));
                throw error;
            }
        },
        setActivePersona: (persona: Partial<Persona> | null) => {
            update(store => ({
                ...store,
                activePersona: persona ? JSON.parse(JSON.stringify(persona)) : null,
                updates: {}
            }));
        },
        updateActivePersona: (updates: Partial<Persona>) => {
            update(store => ({
                ...store,
                activePersona: {
                    ...store.activePersona,
                    ...updates
                }
            }));
        },
        addAttribute: () => {
            update(store => {
                if (!store.activePersona) return store;

                const currentAttributes = store.activePersona.attributes ?? { age: '', background: '' };
                const newKey = `attribute_${Object.keys(currentAttributes).length + 1}`;
                console.log('addAttribute', currentAttributes, newKey);
                return {
                    ...store,
                    activePersona: {
                        ...store.activePersona,
                        attributes: {
                            ...currentAttributes,
                            [newKey]: ''
                        } satisfies PersonaSection
                    }
                };
            });
        },
        addOutfit: () => {
            update(store => {
                if (!store.activePersona) return store;

                const currentWardrobe = store.activePersona.wardrobe ?? {};
                // Sanitize the new outfit name
                const outfitName = sanitizePropertyName(`outfit_${Object.keys(currentWardrobe).length + 1}`);
                return {
                    ...store,
                    activePersona: {
                        ...store.activePersona,
                        wardrobe: {
                            ...currentWardrobe,
                            [outfitName]: {
                                piece_1: ''
                            }
                        }
                    }
                };
            });
        },
        removeOutfit: (outfitName: string) => {
            update(store => {
                if (!store.activePersona?.wardrobe || outfitName === 'default') return store;

                const currentWardrobe = { ...store.activePersona.wardrobe };
                delete currentWardrobe[outfitName];
                return {
                    ...store,
                    activePersona: {
                        ...store.activePersona,
                        wardrobe: currentWardrobe,
                        current_outfit: store.activePersona.current_outfit === outfitName ? 'default' : store.activePersona.current_outfit
                    }
                };
            });
        },
        updateOutfitName: (oldName: string, newName: string) => {
            update(store => {
                if (!store.activePersona?.wardrobe || oldName === 'default') return store;
                if (newName === 'default' || store.activePersona.wardrobe[newName]) return store;

                const currentWardrobe = { ...store.activePersona.wardrobe };
                const outfit = currentWardrobe[oldName];
                delete currentWardrobe[oldName];
                currentWardrobe[newName] = outfit;
                return {
                    ...store,
                    activePersona: {
                        ...store.activePersona,
                        wardrobe: currentWardrobe,
                        current_outfit: store.activePersona.current_outfit === oldName ? newName : store.activePersona.current_outfit
                    }
                };
            });
        },
        addWardrobePiece: (outfitName: string) => {
            update(store => {
                if (!store.activePersona?.wardrobe?.[outfitName]) return store;

                const currentOutfit = store.activePersona.wardrobe[outfitName];
                // Sanitize the new piece name
                const newPiece = sanitizePropertyName(`piece_${Object.keys(currentOutfit).length + 1}`);
                return {
                    ...store,
                    activePersona: {
                        ...store.activePersona,
                        wardrobe: {
                            ...store.activePersona.wardrobe,
                            [outfitName]: {
                                ...currentOutfit,
                                [newPiece]: ''
                            }
                        }
                    }
                };
            });
        },
        updateWardrobePieceName: (outfitName: string, oldPiece: string, newPiece: string) => {
            update(store => {
                if (!store.activePersona?.wardrobe?.[outfitName]) return store;

                const currentOutfit = { ...store.activePersona.wardrobe[outfitName] };
                const description = currentOutfit[oldPiece];
                delete currentOutfit[oldPiece];
                // Sanitize the new piece name to ensure it's a valid JSON property name
                const sanitizedPieceName = sanitizePropertyName(newPiece);
                currentOutfit[sanitizedPieceName] = description;
                return {
                    ...store,
                    activePersona: {
                        ...store.activePersona,
                        wardrobe: {
                            ...store.activePersona.wardrobe,
                            [outfitName]: currentOutfit
                        }
                    }
                };
            });
        },
        removeWardrobePiece: (outfitName: string, piece: string) => {
            update(store => {
                if (!store.activePersona?.wardrobe?.[outfitName]) return store;
                if (Object.keys(store.activePersona.wardrobe[outfitName]).length <= 1) return store;

                const currentOutfit = { ...store.activePersona.wardrobe[outfitName] };
                delete currentOutfit[piece];
                return {
                    ...store,
                    activePersona: {
                        ...store.activePersona,
                        wardrobe: {
                            ...store.activePersona.wardrobe,
                            [outfitName]: currentOutfit
                        }
                    }
                };
            });
        },
        removeAttribute: (key: string) => {
            update(store => {
                if (!store.activePersona?.attributes) {
                    return store;
                }

                const { [key]: _, ...rest } = store.activePersona.attributes;
                return {
                    ...store,
                    activePersona: {
                        ...store.activePersona,
                        attributes: {
                            ...rest,
                        } satisfies PersonaSection
                    }
                };
            });
        },
        updateAttributeKey: (oldKey: string, newKey: string) => {
            update(store => {
                if (!store.activePersona?.attributes) {
                    return store;
                }

                const { [oldKey]: value, ...rest } = store.activePersona.attributes;
                return {
                    ...store,
                    activePersona: {
                        ...store.activePersona,
                        attributes: {
                            ...rest,
                            [newKey]: value
                        } satisfies PersonaSection
                    }
                };
            });
        },
        addBaseThought: () => {
            update(store => {
                if (!store.activePersona) return store;
                return {
                    ...store,
                    activePersona: { ...store.activePersona, base_thoughts: [...(store.activePersona.base_thoughts || []), ''] }
                };
            });
        },
        removeBaseThought: (index: number) => {
            update(store => {
                if (!store.activePersona) return store;
                const thoughts = [...(store.activePersona.base_thoughts || [])];
                thoughts.splice(index, 1);
                return { ...store, activePersona: { ...store.activePersona, base_thoughts: thoughts } };
            });
        },
        addFeature: () => {
            update(store => {
                if (!store.activePersona) return store;
                return { ...store, activePersona: { ...store.activePersona, features: { ...(store.activePersona.features || {}), 'feature_1': '' } } }
            });
        },
        removeFeature: (key: string) => {
            update(store => {
                if (!store.activePersona) return store;
                const { [key]: _, ...rest } = store.activePersona.features || {};
                return { ...store, activePersona: { ...store.activePersona, features: rest } };
            });
        },
        updateFeatureKey: (oldKey: string, newKey: string) => {
            update(store => {
                if (!store.activePersona) return store;
                const { [oldKey]: value, ...rest } = store.activePersona.features || {};
                return { ...store, activePersona: { ...store.activePersona, features: { ...rest, [newKey]: value } } };
            });
        },
        addPifEntry: () => {
            update(store => {
                if (!store.activePersona) return store;
                return { ...store, activePersona: { ...store.activePersona, pif: { ...(store.activePersona.pif || {}), 'pif_1': '' } } };
            });
        },
        removePifEntry: (key: string) => {
            update(store => {
                if (!store.activePersona) return store;
                const { [key]: _, ...rest } = store.activePersona.pif || {};
                return { ...store, activePersona: { ...store.activePersona, pif: rest } };
            });
        },
        updatePifKey: (oldKey: string, newKey: string) => {
            update(store => {
                if (!store.activePersona) return store;
                const { [oldKey]: value, ...rest } = store.activePersona.pif || {};
                return { ...store, activePersona: { ...store.activePersona, pif: { ...rest, [newKey]: value } } };
            });
        },
        addWakeupMessage: () => {
            update(store => {
                if (!store.activePersona) return store;
                return { ...store, activePersona: { ...store.activePersona, wakeup: [...(store.activePersona.wakeup || []), ''] } };
            });
        },
        removeWakeupMessage: (index: number) => {
            update(store => {
                if (!store.activePersona) return store;
                const messages = [...(store.activePersona.wakeup || [])];
                messages.splice(index, 1);
                return { ...store, activePersona: { ...store.activePersona, wakeup: messages } };
            });
        },
        addNshotEntry: () => {
            update(store => {
                if (!store.activePersona) return store;
                return { ...store, activePersona: { ...store.activePersona, nshot: { ...(store.activePersona.nshot || {})} } }
            });
        },
        removeNshotEntry: (key: string) => {
            update(store => {
                if (!store.activePersona) return store;
                const { [key]: _, ...rest } = store.activePersona.nshot || {};
                return { ...store, activePersona: { ...store.activePersona, nshot: rest } };
            });
        },
        updateNshotKey: (oldKey: string, newKey: string) => {
            update(store => {
                if (!store.activePersona) return store;
                const { [oldKey]: value, ...rest } = store.activePersona.nshot || {};
                return { ...store, activePersona: { ...store.activePersona, nshot: { ...rest, [newKey]: value } } };
            });
        },
        cancelUpdates: () => {
            update(store => {
                const originalPersona = store.personas.find(p => p.persona_id === store.activePersona?.persona_id);
                const activePersona = originalPersona ? JSON.parse(JSON.stringify(originalPersona)) : null;
                console.log('cancelUpdates', activePersona);
                return {
                    ...store,
                    updates: {},
                    activePersona: activePersona
                };
            });
        }
    };
}

export const rosterStore = createRosterStore();