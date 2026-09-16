# MAIA Beacon - Diagrammes UML & Architecture

Ce document rassemble les diagrammes UML décrivant l'architecture modulaire, la structure des classes, les séquences d'inférence et la machine à états de **MAIA Beacon**.

---

## 1. Diagramme de Composants (Component Diagram)

Ce diagramme illustre le découplage entre la couche d'API HTTP (`api.routes`), l'orchestrateur central (`engines.manager`), les fournisseurs d'exécution matériels (`engines.llama_engine` et `engines.npu_engine`) et le module d'optimisation matérielle (`optimizer`).

![Diagramme de Composants](images/component_diagram.svg)

<details>
<summary><b>Voir le code Mermaid</b></summary>

```mermaid
flowchart TD
    Client["Client MAIA_API"]

    subgraph MAIA_Beacon ["MAIA Beacon Worker Node"]
        subgraph APILayer ["Couche API"]
            Routes["api/routes.py"]
            App["app.py / main.py"]
        end

        subgraph ConfigLayer ["Configuration"]
            Config["config.py"]
        end

        subgraph CoreLayer ["Orchestrateur Central"]
            Manager["engines/manager.py"]
        end

        subgraph EngineLayer ["Moteurs d Inference"]
            LlamaEngine["engines/llama_engine.py"]
            NPUEngine["engines/npu_engine.py"]
        end

        subgraph HardwareOpt ["Telemétrie et Optimisation"]
            Optimizer["optimizer.py"]
        end
    end

    subgraph Hardware ["Ressources Materielles"]
        GPU["GPU NVIDIA / Vulkan"]
        LlamaProcess["llama-server.exe"]
        NPU["Intel AI Boost NPU"]
    end

    Client -->|HTTP / REST / SSE| Routes
    App --> Routes
    Routes --> Config
    Routes --> Manager
    Manager --> Config
    Manager -->|Dispatch GPU| LlamaEngine
    Manager -->|Dispatch NPU| NPUEngine
    Manager --> Optimizer
    LlamaEngine --> Optimizer
    LlamaEngine -->|Sous-processus| LlamaProcess
    LlamaProcess -->|VRAM / CUDA / Vulkan| GPU
    NPUEngine -->|OpenVINO LLMPipeline| NPU
```
</details>

---

## 2. Diagramme de Classes (Class Diagram)

Ce diagramme présente la structure orientée objet de l'orchestrateur `EngineManager` et ses interactions avec les sous-modules.

```mermaid
classDiagram
    class Config {
        +BEACON_HOST: str
        +BEACON_PORT: int
        +MODELS_DIR: Path
        +LLAMA_SERVER_EXE: Path
        +TARGET_DEVICE: str
        +IDLE_TIMEOUT_SECONDS: int
    }

    class EngineManager {
        +state: dict
        +state_lock: Lock
        +startup_lock: Lock
        +get_available_models() list
        +detect_engine(model_name: str) str
        +stop_all() void
        +sleep_active_engine() void
        +idle_watchdog() void
        +load_model_task(model_name: str) bool
        +get_status() dict
        +list_v1_models() dict
        +handle_chat_completions(request) Response
        +handle_completions(request) Response
    }

    class LlamaEngine {
        +get_available_gguf_models(models_dir) list
        +is_gguf_model(model_name, models_dir) bool
        +kill_all_llama_servers() void
        +start_llama_server_task() bool
        +proxy_to_llama_server() Response
    }

    class NPUEngine {
        +npu_state: dict
        +get_available_openvino_models(models_dir) list
        +is_openvino_model(model_name, models_dir) bool
        +unload_npu_model() void
        +load_npu_model() bool
        +generate_npu() Response
    }

    class Optimizer {
        +get_gpu_vram() tuple
        +get_max_vram_ratio() float
        +get_npu_info() dict
        +read_gguf_metadata(model_path) dict
    }

    EngineManager --> Config : utilise
    EngineManager --> LlamaEngine : orchestre
    EngineManager --> NPUEngine : orchestre
    EngineManager --> Optimizer : interroge
    LlamaEngine --> Optimizer : calcule VRAM
```

---

## 3. Diagramme de Séquence : Inférence Chat `/v1/chat/completions`

Ce diagramme montre le flux de traitement d'une requête de chat OpenAI, incluant la détection du matériel, le chargement dynamique et le streaming SSE.

![Diagramme de Séquence](images/sequence_diagram.svg)

<details>
<summary><b>Voir le code Mermaid</b></summary>

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant Routes as "api/routes.py"
    participant Manager as "engines/manager.py"
    participant Llama as "engines/llama_engine.py"
    participant NPU as "engines/npu_engine.py"
    participant LlamaProcess as "llama-server.exe"
    participant NPUHW as "Intel NPU"

    Client->>Routes: POST /v1/chat/completions (model, messages)
    Routes->>Manager: handle_chat_completions(request)
    Manager->>Manager: detect_engine(model)

    alt Moteur GPU (GGUF / Turboquant)
        Manager->>Llama: proxy_to_llama_server()
        opt Modèle non chargé ou différent
            Llama->>Llama: kill_all_llama_servers()
            Llama->>LlamaProcess: Lancement sous-processus llama-server
            LlamaProcess-->>Llama: Healthcheck HTTP 200 OK
        end
        Llama->>LlamaProcess: POST /v1/chat/completions (Proxy HTTP)
        LlamaProcess-->>Client: Streaming SSE (data: chunk)
    else Moteur NPU (OpenVINO IR)
        Manager->>NPU: load_npu_model(model)
        NPU->>Llama: kill_all_llama_servers()
        NPU->>NPUHW: og.LLMPipeline(model_dir, NPU)
        NPUHW-->>NPU: Modèle prêt en NPU
        Manager->>NPU: generate_npu(request, json_data)
        NPU->>NPUHW: pipe.generate(prompt, streamer)
        NPUHW-->>Client: Streaming SSE (data: chunk)
    end
```
</details>

---

## 4. Diagramme d'États (State Machine Diagram)

Ce diagramme décrit les transitions d'états de l'orchestrateur `EngineManager.state["status"]`.

![Diagramme d'États](images/state_diagram.svg)

<details>
<summary><b>Voir le code Mermaid</b></summary>

```mermaid
stateDiagram-v2
    [*] --> idle

    idle --> starting : Demande de chargement
    starting --> running : Modèle prêt
    starting --> error : Échec de chargement

    running --> running : Nouvelles requêtes
    running --> sleeping : Inactivité expirée
    running --> idle : Arret demande
    running --> starting : Changement de modèle

    sleeping --> starting : Nouvelle requête d inférence
    sleeping --> idle : Arret demande

    error --> idle : Reinitialisation
```
</details>
