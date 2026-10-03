# Studio Audio — TTS · Record · STT

Studio audio complet sur Raspberry Pi 5, réunissant en une seule application Tkinter à onglets : synthèse vocale (**Text-to-Speech**), enregistrement microphone, et transcription automatique (**Speech-to-Text**) via Faster-Whisper.

## Onglets

### 🔊 Text-to-Speak (TTS)
Basé sur **eSpeak NG** et les voix **MBROLA** :
- Détection automatique des voix eSpeak et MBROLA installées ; langues disponibles déduites des voix réellement présentes sur le système.
- Sélection en cascade langue → voix.
- Réglages vitesse (80–250 mots/min), pitch (0–99), volume (0–200 %), pauses entre les mots.
- Import de texte depuis un fichier `.txt` ou extraction depuis un `.pdf` (via **PyPDF2**, avec recollement des lignes coupées par la mise en page, gestion des repères de paragraphe `§`).
- Lecture intégrale, lecture du seul paragraphe courant, arrêt, et export direct en fichier `.wav`.

*(Ce module reprend la logique du projet [[espeak-tts]], intégrée ici dans un studio plus large.)*

### 🎙 Record
Enregistrement microphone autonome :
- Sélection du périphérique d'entrée, avec détection automatique du micro par défaut du système.
- **Test micro** avec vumètre en direct, indépendant de l'enregistrement.
- Format de sortie **WAV** ou **MP3** (bitrate configurable : 128k/192k/256k/320k, conversion via `pydub`/ffmpeg), avec **durée maximale** optionnelle (arrêt automatique).
- Vumètre en temps réel pendant l'enregistrement, chronomètre affiché dans la barre de statut.
- Fermeture propre des flux audio à la sortie de l'onglet ou de l'application.

### 🗣 Speech-to-Text (STT)
Transcription automatique via **Faster-Whisper**, exécuté localement :
- Choix du modèle Whisper : `tiny` (~75 Mo) à `large-v3` (~3,1 Go), avec indication de la taille de chaque modèle et chargement à la demande (calcul en `int8`, 4 threads CPU adaptés au Pi 5).
- Langue de transcription : détection automatique ou choix explicite (français, anglais, allemand, espagnol, italien, portugais, néerlandais, japonais, chinois, arabe).
- Deux sources audio : **enregistrement direct** au micro (avec vumètre, ré-échantillonnage automatique vers 16 kHz quel que soit le taux d'origine) ou **transcription d'un fichier existant** (`.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.aac`).
- Transcription avec filtrage VAD (suppression des silences) et beam search (beam size 5).
- Résultat affiché avec langue détectée et durée audio ; texte exportable en `.txt` ou copiable dans le presse-papiers.

## Configuration persistante (`studio_audio_config.json`)
Les réglages de chaque onglet sont sauvegardés automatiquement à la fermeture de l'application et rechargés au démarrage suivant :
```json
{
  "tts_speed": 105,
  "tts_pitch": 50,
  "tts_volume": 85,
  "tts_pause": true,
  "tts_lang": "Français",
  "tts_voice": "[MBROLA] mb-fr7",
  "stt_model": "medium",
  "stt_lang": "Automatique",
  "rec_format": "WAV",
  "rec_bitrate": "192k"
}
```

## Lancement
```bash
python3 studio_audio.py
```

## Dépendances
Système :
```bash
sudo apt install espeak-ng mbrola mbrola-fr7 ffmpeg
```

Python :
```bash
pip install numpy scipy PyPDF2 pillow sounddevice soundfile onnxruntime pydub faster-whisper --break-system-packages
```

- **`./install_whisper_pi5.sh`** — script d'installation dédié à Faster-Whisper sur Raspberry Pi 5 (dépendance mentionnée en tête du script principal, non fournie ici).
- `pydub` nécessite `ffmpeg` pour la conversion MP3.
- L'onglet STT reste accessible même sans `faster-whisper` installé, mais affiche un message d'indisponibilité plutôt que le sélecteur de modèle.

## Structure du dépôt
```
Studio_Audio/
├── studio_audio.py               # Application principale (Tkinter, 3 onglets)
├── studio_audio_config.json      # Préférences utilisateur (généré à l'exécution)
├── install_whisper_pi5.sh        # Script d'installation de Faster-Whisper (non inclus ici)
└── icons/                        # Logo utilisé par le splash screen et l'interface (microphone_1.png, non inclus ici)
```

> `studio_audio_config.json` reflète les préférences propres à chaque installation (voix TTS, modèle Whisper, format d'enregistrement) ; à exclure du suivi de version ou à ne fournir qu'à titre d'exemple.

## Prérequis matériels
- Raspberry Pi 5 (4 cœurs exploités pour Faster-Whisper)
- Microphone USB compatible (le code prévoit notamment un micro stéréo type K66)
- `python3-tk`

## Auteur
Jean-François BRUNET - JFBConseils - Juillet 2026
