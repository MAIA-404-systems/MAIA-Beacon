# MAIA Beacon - Architecture Technique

## 1. Vue d'Ensemble

**MAIA Beacon** est un nœud d'exécution d'inférence d'IA local haute performance et autonome. Il agit comme un worker pour le réseau **MAIA_API** ou comme un serveur autonome compatible OpenAI.

Il adopte une séparation stricte à 3 niveaux :
1. **Couche API (`api/`)** : Exposition des routes REST et OpenAI (FastAPI pure).
2. **Couche Moteurs (`engines/`)** : Runtime d'inférence (Llama / Turboquant sur GPU, OpenVINO GenAI sur NPU, et parsing GGUF).
3. **Couche Matériel (`hardware/`)** : Détection des cartes GPU/VRAM, inspection de l'accélérateur Intel AI Boost NPU, et télémétrie matérielle.

---

## 2. Découpage Modulaire

```
MAIA-Beacon/
├── config.py                 # Configuration centralisée & Variables d'environnement
├── main.py                   # Point d'entrée principal CLI & Uvicorn runner
├── app.py                    # Point d'entrée secondaire / Alias ASGI
├── api/                      # Couche API HTTP (FastAPI)
│   ├── __init__.py
│   └── routes.py             # Endpoints REST & OpenAI (Pure routing)
├── hardware/                 # Couche Matériel (Hardware Abstraction Layer)
│   ├── __init__.py
│   ├── gpu.py                # Télémétrie VRAM GPU (Vulkan, nvidia-smi, rocm-smi, WMI)
│   ├── npu.py                # Inspection Intel AI Boost NPU (OpenVINO Core)
│   └── manager.py            # HardwareManager Singleton
├── engines/                  # Moteurs d'Inférence & Runtimes
│   ├── __init__.py
│   ├── manager.py            # Orchestrateur central (EngineManager Singleton)
│   ├── gguf_parser.py        # Analyseur binaire des modèles GGUF & Calcul NGL
│   ├── llama_engine.py       # Provider GPU / CPU via llama-server.exe
│   └── npu_engine.py         # Provider Intel NPU via OpenVINO LLMPipeline
├── test/                     # Tests de validation & E2E
│   ├── test_beacon_npu_api.py
│   ├── test_npu.py
│   └── test_run_llm_npu.py
└── docs/                     # Documentation technique & UML
    ├── ARCHITECTURE.md
    ├── UML_DIAGRAMS.md
    └── API_REFERENCE.md
```

---

## 3. Rôles et Responsabilités des Composants

### 3.1 `config.py`
Centralise l'accès aux variables d'environnement (`.env`) avec typage strict :
* `BEACON_HOST` / `BEACON_PORT` : Adresse et port d'écoute (par défaut `0.0.0.0:11343`).
* `MODELS_DIR` : Dossier racine des modèles.
* `LLAMA_SERVER_EXE` : Chemin vers le binaire `llama-server.exe`.
* `TARGET_DEVICE` : Cible matérielle par défaut (`GPU` ou `NPU`).
* `IDLE_TIMEOUT_SECONDS` : Temps d'inactivité avant mise en veille (par défaut 300s).

### 3.2 `api/routes.py`
Couche REST construite sur FastAPI. Elle ne contient aucune logique métier ou matérielle et délègue toutes les opérations au singleton `manager` :
* Administre le cycle de vie ASGI (`lifespan`) avec lancement du watchdog d'inactivité.
* Expose les routes d'administration du worker (`/api/status`, `/api/models`, `/api/select`, `/api/stop`).
* Expose les routes standard OpenAI (`/v1/models`, `/v1/chat/completions`, `/v1/completions`).

### 3.3 `hardware/` (Couche Matériel)
* `hardware/gpu.py` : Mesure la VRAM totale, utilisée et libre (NVIDIA, AMD, Intel, Vulkan, Windows WMI).
* `hardware/npu.py` : Interroge l'accélérateur Intel AI Boost NPU via OpenVINO Core.
* `hardware/manager.py` (`HardwareManager`) : Offre une façade unifiée (`get_telemetry()`) interrogée par `EngineManager` pour exposer la télémétrie système.

### 3.4 `engines/manager.py` (`EngineManager`)
Orchestrateur central garantissant l'intégrité de l'état du serveur et la gestion des ressources :
* **Thread Safety** : Utilise `state_lock` et `startup_lock` pour protéger le dictionnaire d'état `self.state`.
* **Détection du Moteur** : `detect_engine(model_name)` analyse si le modèle demandé est un dossier OpenVINO IR (`npu`) ou un fichier GGUF (`llama`).
* **Commutation Exclusive** : Garantit qu'un seul moteur/modèle occupe la mémoire à un instant $T$.
* **Watchdog d'inactivité** : Met le modèle en état `sleeping` en cas d'inactivité prolongée.

### 3.5 `engines/gguf_parser.py`
Analyseur binaire rapide lisant l'architecture des modèles GGUF, le nombre de couches, la dimension des têtes d'attention et calculant le nombre optimal de couches GPU à décharger (`--n-gpu-layers`).

### 3.6 `engines/llama_engine.py` & `engines/npu_engine.py`
Providers d'exécution d'inférence pour `llama-server.exe` (GGUF) et `OpenVINO GenAI` (NPU).

---

## 4. Gestion de la Mémoire et Auto-Sleep

```
[Requête Inférence] ──> [Status: running] ──> (Inactivité > 300s) ──> [Watchdog] ──> [Status: sleeping]
                                                                                            │
                                                                                    Extinction llama-server
                                                                                    Unload OpenVINO NPU
                                                                                    RAM / VRAM 100% Libérée
```

* **Extinction Manuelle** : `POST /api/stop` arrête immédiatement tout moteur actif.
* **Auto-Sleep** : Après `IDLE_TIMEOUT_SECONDS` sans requête, le watchdog libère la VRAM et la RAM.
