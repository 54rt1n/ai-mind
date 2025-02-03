# MindAI API Documentation

## Authentication

All API endpoints require authentication using a bearer token. Include the token in the Authorization header:

```http
Authorization: Bearer your_api_key
```

## Base URL

```
http://localhost:8000
```

## API Endpoints

### Chat Completion

#### POST /v1/chat/completions

Generate a chat completion response.

**Request Body:**
```json
{
  "user_id": "string",
  "persona_id": "string",
  "messages": [
    {
      "timestamp": 0,
      "role": "user",
      "content": "string"
    }
  ],
  "model": "string",
  "temperature": 0.7,
  "max_tokens": null,
  "stream": true,
  "location": null,
  "system_message": null,
  "pinned_messages": null,
  "active_document": null,
  "workspace_content": null,
  "thought_content": null,
  "disable_guidance": false,
  "disable_pif": false
}
```

**Response:**
```json
{
  "id": "string",
  "object": "chat.completion",
  "created": 0,
  "model": "string",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "string"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

#### GET /v1/chat/models

List available chat models.

**Response:**
```json
{
  "models": [
    {
      "name": "string",
      "provider": "string",
      "size": "string",
      "categories": ["string"]
    }
  ]
}
```

### Conversation Management

#### GET /api/conversation

List all conversations.

**Response:**
```json
{
  "status": "success",
  "message": "string",
  "data": [
    {
      "conversation_id": "string",
      "document_type": "string",
      "timestamp_max": 0
    }
  ]
}
```

#### POST /api/conversation

Save a new conversation.

**Request Body:**
```json
{
  "conversation_id": "string",
  "messages": [
    {
      "role": "string",
      "content": "string",
      "timestamp": 0
    }
  ]
}
```

#### GET /api/conversation/{conversation_id}

Get a specific conversation.

**Response:**
```json
{
  "status": "success",
  "message": "string",
  "data": [
    {
      "doc_id": "string",
      "document_type": "string",
      "content": "string",
      "timestamp": 0
    }
  ]
}
```

#### POST /api/conversation/{conversation_id}/remove

Delete a conversation.

### Memory Management

#### GET /api/memory/search

Search through memory documents.

**Query Parameters:**
- `query`: Search query string
- `top_n`: Number of results to return (default: 5)
- `document_type`: Filter by document type

**Response:**
```json
{
  "status": "success",
  "results": [
    {
      "doc_id": "string",
      "document_type": "string",
      "content": "string",
      "score": 0.0
    }
  ]
}
```

#### PUT /api/memory/{conversation_id}/{document_id}

Update a document.

**Request Body:**
```json
{
  "data": {
    "key": "value"
  }
}
```

#### GET /api/memory/{document_id}

Get a specific document.

#### POST /api/memory/{conversation_id}/{document_id}/remove

Delete a document.

### Pipeline Management

#### POST /api/pipeline/task

Create a new pipeline task.

**Request Body:**
```json
{
  "pipeline_type": "string",
  "config": {
    "user_id": "string",
    "persona_id": "string",
    "conversation_id": "string",
    "mood": "string",
    "no_retry": true,
    "guidance": "string",
    "top_n": 0,
    "query_text": "string"
  }
}
```

#### GET /api/pipeline/task

List all pipeline tasks.

#### POST /api/pipeline/task/{task_id}/retry

Retry a failed pipeline task.

#### POST /api/pipeline/task/{task_id}/remove

Remove a pipeline task.

### Document Management

#### GET /api/document/list

List all available documents.

**Response:**
```json
{
  "status": "success",
  "message": "string",
  "documents": [
    {
      "name": "string",
      "modified_time": 0.0,
      "size": 0
    }
  ]
}
```

#### POST /api/document/upload

Upload a new document.

**Request Body:**
- Form data with file

#### GET /api/document/{document_name}

Download a specific document.

#### POST /api/document/{document_name}/remove

Delete a document.

#### GET /api/document/{document_name}/contents

Get the contents of a specific document for browser display.

### Persona Management

#### GET /api/roster

List all available personas.

**Response:**
```json
{
  "personas": [
    {
      "persona_id": "string",
      "persona_version": "string",
      "chat_strategy": "string",
      "name": "string",
      "full_name": "string",
      "attributes": {},
      "features": {},
      "default_location": "string"
    }
  ]
}
```

#### POST /api/roster

Create a new persona.

**Request Body:**
```json
{
  "persona_id": "string",
  "chat_strategy": "string",
  "name": "string",
  "full_name": "string",
  "attributes": {},
  "features": {},
  "default_location": "string"
}
```

#### GET /api/roster/{persona_id}

Get a specific persona.

#### PUT /api/roster/{persona_id}

Update an existing persona.

#### DELETE /api/roster/{persona_id}

Delete a persona.

### Administrative Endpoints

#### POST /api/admin/rebuild_index

Rebuild the search index from JSONL files.

### Report Generation

#### GET /api/report/conversation_matrix

Get conversation analysis matrix.

#### GET /api/report/symbolic_keywords

Get symbolic keywords analysis.

**Query Parameters:**
- `document_type`: Filter by document type

## Error Handling

The API uses standard HTTP status codes:

- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 500: Internal Server Error

Error responses include a detail message:

```json
{
  "detail": "Error message description"
}
```

## Rate Limiting

Rate limiting details are provided in response headers:
- `X-RateLimit-Limit`: Requests per hour limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time until limit resets

## Streaming

For endpoints that support streaming (like chat completions), the response is sent as a series of server-sent events (SSE). Each event contains a JSON payload with the following format:

```json
{
  "id": "string",
  "object": "chat.completion.chunk",
  "created": 0,
  "model": "string",
  "choices": [
    {
      "delta": {
        "content": "string"
      },
      "index": 0,
      "finish_reason": null
    }
  ]
}
```

The final event will have `[DONE]` as its data.