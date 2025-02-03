<!-- src/routes/documents/+page.svelte -->
<script lang="ts">
    import { onMount } from 'svelte';
    import { documentStore } from '$lib/store/documentStore';
    import { configStore } from '$lib/store/configStore';
    import { Plus, Download, Trash2, Upload, File as FileIcon} from 'lucide-svelte';
    
    let fileInput: HTMLInputElement;
    let dragActive = false;

    onMount(() => {
        documentStore.fetchDocuments();
    });

    function handleFileSelect(event: Event) {
        const input = event.target as HTMLInputElement;
        if (input.files?.length) {
            handleUpload(input.files[0]);
        }
    }

    async function handleUpload(file: File) {
        try {
            await documentStore.uploadDocument(file);
            if (fileInput) {
                fileInput.value = ''; // Reset file input
            }
        } catch (error) {
            console.error('Upload failed:', error);
        }
    }

    function handleDragOver(event: DragEvent) {
        event.preventDefault();
        dragActive = true;
    }

    function handleDragLeave() {
        dragActive = false;
    }

    function handleDrop(event: DragEvent) {
        event.preventDefault();
        dragActive = false;
        
        const files = event.dataTransfer?.files;
        if (files?.length) {
            handleUpload(files[0]);
        }
    }

    async function handleDelete(name: string) {
        if (confirm(`Are you sure you want to delete ${name}?`)) {
            try {
                await documentStore.deleteDocument(name);
            } catch (error) {
                console.error('Delete failed:', error);
            }
        }
    }

    function formatSize(bytes: number): string {
        const units = ['B', 'KB', 'MB', 'GB'];
        let size = bytes;
        let unitIndex = 0;
        
        while (size >= 1024 && unitIndex < units.length - 1) {
            size /= 1024;
            unitIndex++;
        }
        
        return `${size.toFixed(1)} ${units[unitIndex]}`;
    }

    function formatDate(timestamp: number): string {
        return new Date(timestamp).toLocaleString();
    }
</script>

<svelte:head>
    <title>Documents | MindAI</title>
</svelte:head>

<div class="documents-page">
    <header class="page-header">
        <h1>Documents</h1>
        <button class="upload-button" on:click={() => fileInput.click()}>
            <Upload size={20} />
            Upload Document
        </button>
        <input
            type="file"
            bind:this={fileInput}
            on:change={handleFileSelect}
            style="display: none"
        />
    </header>

    <div
        class="upload-zone"
        class:dragActive
        on:dragover={handleDragOver}
        on:dragleave={handleDragLeave}
        on:drop={handleDrop}
    >
        <Upload size={24} />
        <p>Drag and drop a file here, or click the upload button above</p>
    </div>

    {#if $documentStore.loading}
        <div class="loading">Loading documents...</div>
    {:else if $documentStore.error}
        <div class="error">{$documentStore.error}</div>
    {:else if $documentStore.documents.length === 0}
        <div class="empty-state">
            <FileIcon size={48} />
            <p>No documents found</p>
        </div>
    {:else}
        <div class="documents-grid">
            {#each $documentStore.documents as document}
                <div class="document-card">
                    <div class="document-info">
                        <FileIcon size={24} />
                        <div class="document-details">
                            <h3>{document.name}</h3>
                            <p class="document-meta">
                                {formatSize(document.size)} • {formatDate(document.modified_time)}
                            </p>
                        </div>
                    </div>
                    <div class="document-actions">
                        <button
                            class="action-button select-button"
                            class:active={$configStore.selectedDocument?.name === document.name}
                            on:click={() => configStore.documentSelect(document)}
                            title={$configStore.selectedDocument?.name === document.name ? 'Deselect' : 'Select'}
                        >
                            <span 
                                class="icon-rotate"
                                style:transform={`rotate(${$configStore.selectedDocument?.name === document.name ? 45 : 0}deg)`}
                            >
                                <Plus size={20} />
                            </span>
                        </button>
                        <button
                            class="action-button download-button"
                            on:click={() => documentStore.downloadDocument(document.name)}
                            title="Download"
                        >
                            <Download size={20} />
                        </button>
                        <button
                            class="action-button delete-button"
                            on:click={() => handleDelete(document.name)}
                            title="Delete"
                        >
                            <Trash2 size={20} />
                        </button>
                    </div>
                </div>
            {/each}
        </div>
    {/if}
</div>

<style>
    .documents-page {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    .page-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }

    .page-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }

    .upload-button {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: #4CAF50;
        color: white;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
        font-weight: 500;
        transition: background-color 0.2s;
    }

    .upload-button:hover {
        background: #43A047;
    }

    .upload-zone {
        border: 2px dashed #e5e7eb;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
        transition: all 0.2s;
    }

    .upload-zone.dragActive {
        border-color: #4CAF50;
        background: #f0fdf4;
    }

    .documents-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1rem;
    }

    .document-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: box-shadow 0.2s;
    }

    .document-card:hover {
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .document-info {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .document-details h3 {
        margin: 0;
        font-size: 1rem;
        font-weight: 500;
    }

    .document-meta {
        margin: 0.25rem 0 0;
        font-size: 0.875rem;
        color: #6b7280;
    }

    .document-actions {
        display: flex;
        gap: 0.5rem;
    }

    .action-button {
        background: none;
        border: none;
        padding: 0.5rem;
        border-radius: 0.375rem;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .select-button {
        color: #4CAF50;
    }

    .select-button:hover {
        background: rgba(76, 175, 80, 0.1);
    }

    .select-button.active {
        color: #f44336;
    }

    .download-button {
        color: #2196f3;
    }

    .download-button:hover {
        background: rgba(33, 150, 243, 0.1);
    }

    .delete-button {
        color: #f44336;
    }

    .delete-button:hover {
        background: rgba(244, 67, 54, 0.1);
    }

    .icon-rotate {
        display: inline-flex;
        transition: transform 0.2s ease;
    }

    .loading {
        text-align: center;
        padding: 3rem;
        color: #6b7280;
    }

    .error {
        background: #fee2e2;
        border: 1px solid #ef4444;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #dc2626;
        margin: 1rem 0;
    }

    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #6b7280;
        background: #f9fafb;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
    }

    .empty-state p {
        margin: 0.5rem 0 0;
    }

    @media (max-width: 768px) {
        .documents-page {
            padding: 1rem;
        }

        .page-header {
            flex-direction: column;
            gap: 1rem;
            align-items: stretch;
        }

        .upload-button {
            width: 100%;
            justify-content: center;
        }

        .documents-grid {
            grid-template-columns: 1fr;
        }
    }
</style>