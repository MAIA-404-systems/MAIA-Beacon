#!/usr/bin/env bash
cd "$(dirname "$0")"

echo "==================================================="
echo "  MAIA Beacon - Démarrage du Worker GPU"
echo "==================================================="

if [ ! -f ".env" ]; then
    echo "[!] Fichier .env non trouvé. Création depuis .env.example..."
    cp ".env.example" ".env"
fi

python app.py
