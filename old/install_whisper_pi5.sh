#!/bin/bash
# =============================================================================
#  Installation Faster-Whisper + modèles sur Raspberry Pi 5 (16 Go / NVMe)
#  Auteur : Jean-François BRUNET - JFBConseils - 2026
# =============================================================================

set -e  # Arrêt immédiat en cas d'erreur

BOLD="\e[1m"
GREEN="\e[32m"
CYAN="\e[36m"
YELLOW="\e[33m"
RED="\e[31m"
RESET="\e[0m"

echo -e "${CYAN}${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Studio Audio – Installation Whisper  (Pi 5 16Go)  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# =============================================================================
# 1. DÉPENDANCES SYSTÈME
# =============================================================================
echo -e "${BOLD}[1/5] Mise à jour du système et dépendances...${RESET}"

sudo apt update -qq
sudo apt install -y \
    ffmpeg \
    portaudio19-dev \
    python3-dev \
    python3-pip \
    libatlas-base-dev \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    gfortran

echo -e "${GREEN}✔ Dépendances système installées.${RESET}"

# =============================================================================
# 2. PAQUETS PYTHON
# =============================================================================
echo -e "\n${BOLD}[2/5] Installation des paquets Python...${RESET}"

pip install --break-system-packages --upgrade pip

pip install --break-system-packages \
    "numpy>=1.26,<2.0" \
    faster-whisper \
    sounddevice \
    soundfile \
    pydub \
    PyAudio \
    PyPDF2 \
    Pillow

echo -e "${GREEN}✔ Paquets Python installés.${RESET}"

# =============================================================================
# 3. DOSSIER DE CACHE MODÈLES SUR LE NVME
# =============================================================================
echo -e "\n${BOLD}[3/5] Préparation du dossier de cache modèles...${RESET}"

MODELS_DIR="$HOME/.cache/huggingface/hub"
mkdir -p "$MODELS_DIR"

echo -e "  📁 Les modèles seront stockés dans : ${CYAN}$MODELS_DIR${RESET}"
df -h "$MODELS_DIR" | tail -1 | awk '{print "  💾 Espace disponible : " $4}'

# =============================================================================
# 4. TÉLÉCHARGEMENT DES MODÈLES
# =============================================================================
echo -e "\n${BOLD}[4/5] Téléchargement des modèles Faster-Whisper...${RESET}"
echo -e "${YELLOW}  → base       (~145 Mo)   – quasi temps réel${RESET}"
echo -e "${YELLOW}  → medium     (~1,5 Go)   – très bonne qualité FR${RESET}"
echo -e "${YELLOW}  → large-v3-turbo (~1,6 Go) – qualité maximale${RESET}"
echo ""

python3 - <<'EOF'
from faster_whisper import WhisperModel
import sys

models = [
    ("base",            "Temps réel / dictée rapide"),
    ("medium",          "Qualité élevée / usage général"),
    ("large-v3-turbo",  "Qualité maximale / fichiers longs"),
]

for name, usage in models:
    print(f"  ⬇  Téléchargement : {name}  ({usage})")
    try:
        WhisperModel(name, device="cpu", compute_type="int8")
        print(f"  ✔  {name} : OK\n")
    except Exception as e:
        print(f"  ✘  {name} : ERREUR → {e}\n", file=sys.stderr)
EOF

echo -e "${GREEN}✔ Modèles téléchargés et mis en cache.${RESET}"

# =============================================================================
# 5. TEST DE VALIDATION
# =============================================================================
echo -e "\n${BOLD}[5/5] Test de validation...${RESET}"

python3 - <<'EOF'
import importlib, sys

ok = True
for pkg in ["faster_whisper", "sounddevice", "soundfile", "pydub", "numpy"]:
    try:
        importlib.import_module(pkg)
        print(f"  ✔  {pkg}")
    except ImportError:
        print(f"  ✘  {pkg} manquant !", file=sys.stderr)
        ok = False

if ok:
    print("\n  ✅ Tous les modules sont opérationnels.")
else:
    print("\n  ⚠  Certains modules sont manquants.", file=sys.stderr)
    sys.exit(1)
EOF

# =============================================================================
# RÉCAPITULATIF
# =============================================================================
echo -e "\n${CYAN}${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║                    INSTALLATION OK                  ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Modèles disponibles :                               ║"
echo "║   • base           → dictée, quasi temps réel        ║"
echo "║   • medium         → usage quotidien, bon FR         ║"
echo "║   • large-v3-turbo → qualité maximale                ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Utilisation dans studio_stt.py :                    ║"
echo "║   from faster_whisper import WhisperModel            ║"
echo "║   model = WhisperModel(\"medium\",                     ║"
echo "║              device=\"cpu\", compute_type=\"int8\")      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"
