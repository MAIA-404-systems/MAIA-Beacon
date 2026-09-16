# MAIA Beacon - Spécification des API REST

Ce document décrit en détail l'ensemble des endpoints HTTP exposés par **MAIA Beacon** sur le port `11343` (par défaut).

---

## Table des Matières

1. [Endpoints d'Administration Worker (`/api/*`)](#1-endpoints-dadministration-worker-api)
   - [GET /api/status](#get-apistatus)
   - [GET /api/models](#get-apimodels)
   - [POST /api/select](#post-apiselect)
   - [POST /api/stop](#post-apistop)
2. [Endpoints Compatibles OpenAI (`/v1/*`)](#2-endpoints-compatibles-openai-v1)
   - [GET /v1/models](#get-v1models)
   - [POST /v1/chat/completions](#post-v1chatcompletions)
   - [POST /v1/completions](#post-v1completions)

---

## 1. Endpoints d'Administration Worker (`/api/*`)

### `GET /api/status`

Retourne l'état de santé du nœud worker, l'état du moteur actif, la télémétrie GPU (VRAM) et NPU.

#### Réponse Example (`200 OK`) :
```json
{
  "type": "beacon",
  "status": "running",
  "active_engine": "llama",
  "active_model": "Ministral-3-3B-Instruct-2512-Q4_K_M",
  "available_models": [
    "Ministral-3-3B-Instruct-2512-Q4_K_M",
    "Qwen2.5-1.5B-Instruct-int4-ov"
  ],
  "target_device": "GPU",
  "active_mmproj": null,
  "active_context": 16384,
  "active_thinking": false,
  "config": null,
  "error_message": null,
  "npu": {
    "device_name": "Intel(R) AI Boost",
    "device": "NPU"
  },
  "gpu_vram": {
    "total_mib": 8192,
    "used_mib": 3200,
    "free_mib": 4992,
    "max_percent": 90.0,
    "max_limit_mib": 7372.8
  },
  "pid": 12844
}
```

---

### `GET /api/models`

Retourne la liste consolidée de tous les modèles disponibles (GGUF et OpenVINO IR).

#### Réponse Example (`200 OK`) :
```json
{
  "models": [
    "Ministral-3-3B-Instruct-2512-Q4_K_M",
    "Qwen2.5-1.5B-Instruct-int4-ov",
    "mistral-7b-instruct-v0.1.Q4_K_S"
  ]
}
```

---

### `POST /api/select`

Bascule ou pré-chauffe un modèle spécifique sur son moteur respectif (GPU ou NPU).

#### Corps de la Requête (`application/json`) :
```json
{
  "model": "Qwen2.5-1.5B-Instruct-int4-ov",
  "context_size": 16384,
  "thinking": false,
  "thinking_effort": "medium",
  "mmproj": null
}
```

#### Réponse Example (`200 OK`) :
```json
{
  "message": "Model selection started",
  "status": "starting",
  "active_model": "Qwen2.5-1.5B-Instruct-int4-ov"
}
```

---

### `POST /api/stop`

Arrête le moteur d'inférence actif (`llama-server` ou NPU) et libère immédiatement 100% de la VRAM et de la RAM.

#### Réponse Example (`200 OK`) :
```json
{
  "message": "Model stopped successfully"
}
```

---

## 2. Endpoints Compatibles OpenAI (`/v1/*`)

### `GET /v1/models`

Format conforme à la spécification OpenAI pour le listing des modèles.

#### Réponse Example (`200 OK`) :
```json
{
  "object": "list",
  "data": [
    {
      "id": "Ministral-3-3B-Instruct-2512-Q4_K_M",
      "object": "model",
      "created": 1726000000,
      "owned_by": "maia-beacon"
    },
    {
      "id": "Qwen2.5-1.5B-Instruct-int4-ov",
      "object": "model",
      "created": 1726000000,
      "owned_by": "maia-beacon"
    }
  ]
}
```

---

### `POST /v1/chat/completions`

Exécute une génération de réponse de chat. Routage automatique vers GPU ou NPU selon le modèle demandé.

#### Corps de la Requête (`application/json`) :
```json
{
  "model": "Qwen2.5-1.5B-Instruct-int4-ov",
  "messages": [
    {"role": "system", "content": "Tu es un assistant IA."},
    {"role": "user", "content": "Bonjour !"}
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 500
}
```

#### Réponse Streaming SSE (`stream: true`) :
```http
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"id":"chatcmpl-npu-1726000100","object":"chat.completion.chunk","created":1726000100,"model":"Qwen2.5-1.5B-Instruct-int4-ov","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-npu-1726000100","object":"chat.completion.chunk","created":1726000100,"model":"Qwen2.5-1.5B-Instruct-int4-ov","choices":[{"index":0,"delta":{"content":"Bonjour"},"finish_reason":null}]}

data: {"id":"chatcmpl-npu-1726000100","object":"chat.completion.chunk","created":1726000100,"model":"Qwen2.5-1.5B-Instruct-int4-ov","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

### `POST /v1/completions`

Exécute une complétion de texte brute.

#### Corps de la Requête (`application/json`) :
```json
{
  "model": "Ministral-3-3B-Instruct-2512-Q4_K_M",
  "prompt": "La théorie de la relativité est",
  "max_tokens": 100
}
```
