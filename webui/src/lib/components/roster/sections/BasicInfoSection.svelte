<!-- src/lib/components/roster/sections/BasicInfoSection.svelte -->
<script lang="ts">
    import EditSection from './EditSection.svelte';
    import type { Persona } from '$lib/types';

    export let formData: Partial<Persona>;
</script>

<EditSection title="Basic Information">
    <div class="form-group">
        <label for="name">Name</label>
        <input
            type="text"
            id="name"
            bind:value={formData.name}
            required
        />
    </div>

    <div class="form-group">
        <label for="full_name">Full Name</label>
        <input
            type="text"
            id="full_name"
            bind:value={formData.full_name}
            required
        />
    </div>

    <div class="form-group">
        <label for="chat_strategy">Chat Strategy</label>
        <select
            id="chat_strategy"
            bind:value={formData.chat_strategy}
            required
        >
            <option value="xmlmemory">XML Memory</option>
            <option value="basic">Basic</option>
        </select>
    </div>

    <div class="form-group">
        <label for="default_location">Default Location</label>
        <input
            type="text"
            id="default_location"
            bind:value={formData.default_location}
            required
        />
    </div>

    <div class="form-group">
        <label for="persona_version">Persona Version</label>
        <input
            type="text"
            id="persona_version"
            bind:value={formData.persona_version}
            placeholder="0.1a"
        />
    </div>

    <div class="form-group">
        <label for="system_header">System Header</label>
        <textarea
            id="system_header"
            bind:value={formData.system_header}
            rows="3"
            placeholder="Enter system header..."
        ></textarea>
    </div>

    <div class="form-group">
        <label class="checkbox-label">
            <input
                type="checkbox"
                bind:checked={formData.include_date}
            />
            Include Date in Thoughts
        </label>
    </div>
</EditSection>

<style>
    .form-group {
        margin-bottom: 1rem;
    }

    .form-group:last-child {
        margin-bottom: 0;
    }

    label {
        display: block;
        font-size: 0.875rem;
        font-weight: 500;
        color: #374151;
        margin-bottom: 0.5rem;
    }

    .checkbox-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        cursor: pointer;
    }

    input[type="text"],
    input[type="number"],
    select,
    textarea {
        width: 100%;
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        transition: border-color 0.2s;
    }

    input[type="text"]:focus,
    input[type="number"]:focus,
    select:focus,
    textarea:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgb(59 130 246 / 0.1);
    }

    textarea {
        resize: vertical;
        min-height: 60px;
    }
</style> 