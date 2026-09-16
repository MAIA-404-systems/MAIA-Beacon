# MAIA Beacon - GPU & NPU Worker Node

**MAIA Beacon** est un agent autonome et léger conçu pour s'exécuter sur les machines disposant de GPU (NVIDIA RTX, Apple Silicon) ou de NPU (Intel AI Boost via OpenVINO GenAI) afin d'héberger, gérer et exécuter dynamiquement des modèles d'IA locaux pour le réseau **MAIA_API**.

---

## Fonctionnalités Clés

* **Support Multi-Accélérateur (GPU & Intel NPU)** : Routage dynamique des requêtes d'inférence vers les sous-processus `llama-server.exe` (GGUF / Turboquant) ou les pipelines OpenVINO GenAI (NPU).
* **Gestion Autonome du GPU et de la VRAM** : Optimise automatiquement les couches GPU (`ngl`) et la quantification KV Cache grâce au module `optimizer.py`.
* **Chargement et Commutation Dynamique à la Demande** : Démarre, arrête ou bascule d'un modèle à un autre à la volée via des requêtes REST (`/api/select`, `/api/stop`).
* **Détection Automatique Multimodale (Vision)** : Identifie et injecte automatiquement le projecteur `mmproj` correspondant aux modèles Vision.
* **Mise en Veille Automatique (Auto-Sleep Watchdog)** : Libère automatiquement 100% de la VRAM GPU et RAM NPU après une période d'inactivité configurable.
* **Compatibilité OpenAI** : Expose des endpoints REST (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) prêts à l'emploi.

---

## Installation et Démarrage

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
Copiez le fichier `.env.example` en `.env` et ajustez les chemins selon votre environnement :
```ini
BEACON_PORT=11343
LLAMA_SERVER_EXE=C:/chemin/vers/turboquant/llama-server.exe
MODELS_DIR=C:/chemin/vers/turboquant/models
TARGET_DEVICE=AUTO
IDLE_TIMEOUT_SECONDS=300
```

### 3. Lancement
```bash
python main.py
```
Ou avec uvicorn directement :
```bash
uvicorn api.routes:app --host 0.0.0.0 --port 11343
```
Le serveur démarrera et écoutera par défaut sur le port `11343`.

---

## Documentation

* [Architecture Technique](docs/ARCHITECTURE.md) : Vue d'ensemble du découplage modulaire et de l'orchestration GPU / NPU.
* [Diagrammes UML & Séquences](docs/UML_DIAGRAMS.md) : Diagrammes de composants, classes, séquences et machine à états (Mermaid).
* [Spécification des API REST](docs/API_REFERENCE.md) : Référence complète des endpoints `/api/*` et `/v1/*`.

---

## Remerciements et Mentions

* **llama.cpp** : Le moteur d'inférence.
* **[Fork turboquant de llama.cpp de TheTom](https://github.com/TheTom/llama-cpp-turboquant)** : Version modifiée pour inclure des optimisations.
* **[TurboQuant de Google](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)** : Technologie intégrée.
* **Tutoriel YouTube de Codacus** : [Vidéo de référence](https://youtu.be/8F_5pdcD3HY).
* Une partie du code de ce projet a été générée à l'aide de **Google Gemini** via le logiciel **Antigravity**.

---

## Licence

Ce projet est sous licence **Apache License 2.0**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
