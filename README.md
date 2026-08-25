# MAIA Beacon — GPU Worker Node

**MAIA Beacon** est un agent autonome et léger conçu pour s'exécuter sur les machines disposant de GPU (NVIDIA RTX, Apple Silicon, etc.) afin d'héberger, gérer et exécuter les modèles d'IA locaux (`llama-server.exe`) pour le réseau **MAIA_API**.

---

## Fonctionnalités Clés

* **Gestion Autonome du GPU et de la VRAM** : Optimise automatiquement les couches GPU (`ngl`) et la quantification KV Cache grâce au module `optimizer.py`.
* **Chargement et Commutation Dynamique à la Demande** : Démarre, arrête ou bascule d'un modèle GGUF à un autre via des requêtes REST (`/api/select`, `/api/stop`).
* **Détection Automatique Multimodale (Vision)** : Identifie et injecte automatiquement le projecteur `mmproj` correspondant aux modèles Vision.
* **Mise en Veille Automatique (Auto-Sleep)** : Libère automatiquement la VRAM du GPU après une période d'inactivité (par défaut 5 minutes).
* **Intégration Transparente avec MAIA_API** : Scanné et piloté à distance par le Load Balancer MAIA_API via le port `11345`.
* **Compatibilité OpenAI** : Expose des endpoints REST (`/v1/chat/completions` et `/v1/models`) compatibles avec les standards de l'industrie.

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
IDLE_TIMEOUT_SECONDS=300
```

### 3. Lancement
Sous Windows :
```cmd
start_beacon.bat
```
Sous Linux / macOS :
```bash
bash start_beacon.sh
```
Le serveur démarrera et écoutera par défaut sur le port `11343`.
