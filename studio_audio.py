#!/usr/bin/env python3
# =============================================================================
#  Studio Audio – TTS · Record · STT
#  TTS : Text To Speech
#  STT : Speech To Text
#  Modules : eSpeak NG + MBROLA · Enregistrement WAV/MP3 · Faster-Whisper STT
#  Dépendances : ./install_whisper_pi5.sh
#
#  Auteur  : Jean‑François BRUNET - JFBConseils - Mars 2026
# =============================================================================

import sys
import os

os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"      # supprime warnings ONNX

import io
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import threading
import tempfile
from math import gcd
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.signal import resample_poly    

import PyPDF2
from PIL import Image, ImageTk

import sounddevice as sd
import soundfile as sf

import onnxruntime as ort
ort.set_default_logger_severity(3)

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except ImportError:
    PYDUB_OK = False

_stderr_backup = sys.stderr
sys.stderr = io.StringIO()
try:
    from faster_whisper import WhisperModel
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False
sys.stderr = _stderr_backup

# =============================================================================
#  CHEMINS  (relatifs au script → portabilité totale)
# =============================================================================
BASE_DIR    = Path(__file__).parent
DEFAULT_DIR = str(BASE_DIR)
LOGO_PATH   = BASE_DIR / "icons" / "microphone_1.png"
CONFIG_PATH = BASE_DIR / "studio_audio_config.json"

# =============================================================================
#  CONSTANTES
# =============================================================================
FG     = "#ffffff"
BG     = "#222222"
BG2    = "#333333"
ACCENT = "#aaddff"

WHISPER_CPU_THREADS = 4          # Pi 5 → 4 cœurs

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
WHISPER_SIZES  = {
    "tiny":           "~75 Mo",
    "base":           "~145 Mo",
    "small":          "~480 Mo",
    "medium":         "~1,5 Go",
    "large-v3-turbo": "~1,6 Go",
    "large-v3":       "~3,1 Go",
}
WHISPER_LANGS = {
    "Automatique": None, "Français": "fr", "Anglais": "en",
    "Allemand":    "de", "Espagnol": "es", "Italien":  "it",
    "Portugais":   "pt", "Néerlandais": "nl", "Japonais": "ja",
    "Chinois":     "zh", "Arabe":     "ar",
}

# =============================================================================
#  CONFIGURATION PERSISTANTE
# =============================================================================
_CONFIG_DEFAULTS = {
    "tts_speed":    105,
    "tts_pitch":    30,
    "tts_volume":   50,
    "tts_pause":    True,
    "tts_lang":     "Français",
    "tts_voice":    "",
    "stt_model":    "medium",
    "stt_lang":     "Automatique",
    "rec_format":   "WAV",
    "rec_bitrate":  "192k",
}

def load_config() -> dict:
    """Charge la config depuis le JSON ; retourne les défauts si absent/corrompu."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Fusionne avec les défauts pour les clés manquantes
        return {**_CONFIG_DEFAULTS, **data}
    except Exception:
        return dict(_CONFIG_DEFAULTS)

def save_config(cfg: dict) -> None:
    """Sauvegarde silencieuse ; ignore les erreurs d'écriture."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass

# =============================================================================
#  UTILITAIRES PARTAGÉS
# =============================================================================
def get_espeak_voices() -> list[str]:
    voices = []
    base = Path("/usr/lib/aarch64-linux-gnu/espeak-ng-data/voices")
    if not base.is_dir():
        return []
    for f in base.rglob("*"):
        if f.is_file() and not f.name.startswith("."):
            voices.append(f.relative_to(base).as_posix())
    return sorted(voices)

def get_mbrola_voices() -> list[str]:
    path = Path("/usr/share/mbrola")
    if not path.is_dir():
        return []
    return sorted(d.name for d in path.iterdir() if d.is_dir())

def extract_pdf_text(path: str) -> str:
    try:
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as exc:
        raise RuntimeError(f"Impossible de lire le PDF :\n{exc}") from exc

    lines   = text.split("\n")
    cleaned = []
    buffer  = ""
    for line in lines:
        s = line.strip()
        if s == "":
            if buffer:
                cleaned.append(buffer.strip())
                buffer = ""
            continue
        if s.startswith("§"):
            if buffer:
                cleaned.append(buffer.strip())
                buffer = ""
            cleaned.append(s)
            continue
        if s.endswith("."):
            buffer += " " + s
            cleaned.append(buffer.strip())
            buffer = ""
        else:
            buffer += " " + s
    if buffer:
        cleaned.append(buffer.strip())
    return "\n\n".join(cleaned)

def list_input_devices() -> list[tuple[int, str]]:
    return [
        (i, d["name"])
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]

def wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "192k") -> None:
    if not PYDUB_OK:
        raise RuntimeError("pydub non installé : pip install pydub")
    AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3", bitrate=bitrate)

def safe_resample(audio: np.ndarray, src_rate: int, dst_rate: int = 16000) -> np.ndarray:
    if src_rate == dst_rate:
        return audio.astype(np.float32)
    g = gcd(dst_rate, src_rate)
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)

# =============================================================================
#  TTS – processus global  (thread-safe)
# =============================================================================
_tts_process: subprocess.Popen | None = None
_tts_lock = threading.Lock()

def speak_async(cmd: list[str]) -> None:
    """Lance espeak-ng et stocke le processus de façon thread-safe."""
    global _tts_process
    proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)
    with _tts_lock:
        _tts_process = proc
    proc.wait()
    with _tts_lock:
        if _tts_process is proc:
            _tts_process = None

def stop_speech() -> None:
    global _tts_process
    with _tts_lock:
        proc = _tts_process
        _tts_process = None
    if proc:
        proc.terminate()

# =============================================================================
#  SPLASH SCREEN
# =============================================================================
class SplashScreen(tk.Toplevel):
    def __init__(self, root: tk.Tk):
        super().__init__(root)
        self.overrideredirect(True)
        self.configure(bg=BG)
        try:
            img = Image.open(LOGO_PATH)
            img.thumbnail((120, 120), Image.LANCZOS)
            self.logo = ImageTk.PhotoImage(img)
        except Exception:
            self.logo = None
        w, h = 320, 260
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        frame = tk.Frame(self, bg=BG)
        frame.pack(expand=True)
        if self.logo:
            tk.Label(frame, image=self.logo, bg=BG).pack(pady=10)
        tk.Label(frame, text="Studio Audio",
                 fg=ACCENT, bg=BG, font=("Arial", 14, "bold")).pack()
        tk.Label(frame, text="TTS · Record · Speech‑to‑Text",
                 fg="#888888", bg=BG, font=("Arial", 10)).pack(pady=4)
        tk.Label(frame, text="Chargement…",
                 fg=FG, bg=BG, font=("Arial", 11)).pack(pady=10)
        self.update()

# =============================================================================
#  ONGLET – TTS (eSpeak NG + MBROLA)
# =============================================================================
class TabTTS(tk.Frame):
    def __init__(self, parent, cfg: dict):
        super().__init__(parent, bg=BG)
        self._cfg = cfg

        self.lang_var   = tk.StringVar()
        self.voice_var  = tk.StringVar()
        self.speed_var  = tk.IntVar(value=cfg["tts_speed"])
        self.pitch_var  = tk.IntVar(value=cfg["tts_pitch"])
        self.volume_var = tk.IntVar(value=cfg["tts_volume"])
        self.pause_var  = tk.BooleanVar(value=cfg["tts_pause"])

        espeak = get_espeak_voices()
        mbrola = get_mbrola_voices()

        def detect_langs(voices: list[str]) -> list[str]:
            langs: dict[str, bool] = {}
            for v in voices:
                s = Path(v).name.split("-")[0]
                if len(s) == 2:
                    langs[s] = True
            return sorted(langs.keys())

        available = detect_langs(espeak)

        ALL_LABELS = {
            "fr": "Français",    "en": "Anglais",     "de": "Allemand",
            "es": "Espagnol",    "it": "Italien",     "pt": "Portugais",
            "nl": "Néerlandais", "fi": "Finnois",     "hu": "Hongrois",
            "pl": "Polonais",    "cs": "Tchèque",     "da": "Danois",
            "et": "Estonien",    "lv": "Letton",      "lt": "Lituanien",
            "mk": "Macédonien",  "no": "Norvégien",   "bg": "Bulgare",
            "hr": "Croate",      "ga": "Gaélique",    "cy": "Gallois",
            "an": "Aragonais",   "bs": "Bosniaque",   "ca": "Catalan",
            "el": "Grec",        "is": "Islandais",
        }
        LANG_LABELS      = {k: v for k, v in ALL_LABELS.items() if k in available}
        self.lang_codes  = list(LANG_LABELS.keys())
        lang_labels_list = list(LANG_LABELS.values())

        self.voices_by_lang: dict[str, list[str]] = {lang: [] for lang in LANG_LABELS}
        for v in espeak:
            s = Path(v).name.split("-")[0]
            if s in self.voices_by_lang:
                self.voices_by_lang[s].append(f"[eSpeak] {v}")
        for v in mbrola:
            s = v[:2]
            if s in self.voices_by_lang:
                self.voices_by_lang[s].append(f"[MBROLA] mb-{v}")

        try:
            img = Image.open(LOGO_PATH)
            img.thumbnail((150, 80), Image.LANCZOS)
            self.logo_image = ImageTk.PhotoImage(img)
        except Exception:
            self.logo_image = None

        # ── Ligne du haut : logo + sélecteurs ──
        top = tk.Frame(self, bg=BG)
        top.pack(pady=10)
        if self.logo_image:
            tk.Label(top, image=self.logo_image, bg=BG).grid(
                row=0, column=0, rowspan=2, padx=10)

        tk.Label(top, text="Langue :", fg=FG, bg=BG).grid(row=0, column=1, padx=10)
        self.lang_menu = ttk.Combobox(top, textvariable=self.lang_var,
                                      values=lang_labels_list, state="readonly")
        self.lang_menu.grid(row=0, column=2)
        self.lang_menu.bind("<<ComboboxSelected>>", self.update_voice_menu)

        tk.Label(top, text="Voix :", fg=FG, bg=BG).grid(row=1, column=1, padx=10)
        self.voice_menu = ttk.Combobox(top, textvariable=self.voice_var, state="readonly")
        self.voice_menu.grid(row=1, column=2)

        # Restauration depuis config
        saved_lang = cfg.get("tts_lang", "Français")
        if saved_lang in lang_labels_list:
            self.lang_var.set(saved_lang)
            self.lang_menu.current(lang_labels_list.index(saved_lang))
        elif "Français" in lang_labels_list:
            self.lang_var.set("Français")
            self.lang_menu.current(lang_labels_list.index("Français"))
        self.update_voice_menu()
        saved_voice = cfg.get("tts_voice", "")
        if saved_voice and saved_voice in self.voice_menu["values"]:
            self.voice_var.set(saved_voice)
        elif "[MBROLA] mb-fr7" in self.voice_menu["values"]:
            self.voice_var.set("[MBROLA] mb-fr7")

        tk.Checkbutton(top, text="Pauses entre les mots",
                       variable=self.pause_var, fg=FG, bg=BG,
                       selectcolor=BG).grid(row=2, column=2, padx=10, pady=6, sticky="w")

        # ── Sliders ──
        sliders = tk.Frame(self, bg=BG)
        sliders.pack(pady=8)
        for col, (label, var, lo, hi) in enumerate([
            ("Vitesse (mots/min)", self.speed_var,  80,  250),
            ("Pitch",              self.pitch_var,   0,   99),
            ("Volume (%)",         self.volume_var,  0,  200),
        ]):
            tk.Label(sliders, text=label, fg=FG, bg=BG).grid(row=0, column=col, padx=20)
            tk.Scale(sliders, from_=lo, to=hi, orient="horizontal",
                     variable=var, bg=BG, fg=FG).grid(row=1, column=col, padx=20)

        # ── Boutons action ──
        actions = tk.Frame(self, bg=BG)
        actions.pack(pady=8)
        tk.Button(actions, text="Lire",            command=self.speak).pack(side="left", padx=8)
        tk.Button(actions, text="Lire Paragraphe", command=self.speak_paragraph).pack(side="left", padx=8)
        tk.Button(actions, text="Stop",            command=stop_speech).pack(side="left", padx=8)
        tk.Button(actions, text="Enregistrer WAV", command=self.save_wav).pack(side="left", padx=8)

        self.text_box = tk.Text(self, height=14, width=80, bg=BG2, fg=FG)
        self.text_box.pack(pady=8, padx=16)
        self.text_box.bind("<KeyRelease>", self.update_char_count)

        self.char_label = tk.Label(self, text="0 caractères", fg=FG, bg=BG)
        self.char_label.pack()

        bottom = tk.Frame(self, bg=BG)
        bottom.pack(pady=8)
        tk.Button(bottom, text="Charger TXT", command=self.load_txt).pack(side="left", padx=8)
        tk.Button(bottom, text="Charger PDF", command=self.load_pdf).pack(side="left", padx=8)
        tk.Button(bottom, text="Effacer",     command=self.clear_text).pack(side="left", padx=8)

    # ── Sérialisation config ──
    def collect_config(self) -> dict:
        return {
            "tts_speed":  self.speed_var.get(),
            "tts_pitch":  self.pitch_var.get(),
            "tts_volume": self.volume_var.get(),
            "tts_pause":  self.pause_var.get(),
            "tts_lang":   self.lang_var.get(),
            "tts_voice":  self.voice_var.get(),
        }

    # ── Voix ──
    def update_voice_menu(self, event=None):
        idx = self.lang_menu.current()
        if idx < 0:
            self.voice_menu["values"] = []
            self.voice_var.set("")
            return
        lang_code = self.lang_codes[idx]
        voices = self.voices_by_lang.get(lang_code, [])
        self.voice_menu["values"] = voices
        self.voice_var.set(voices[0] if voices else "")

    def update_char_count(self, event=None):
        self.char_label.config(
            text=f"{len(self.text_box.get('1.0', tk.END).strip())} caractères")

    def prepare_text(self, text: str) -> str:
        return text.replace(" ", "  ") if self.pause_var.get() else text

    def _build_cmd(self, text: str) -> list[str] | None:
        voice = self.voice_var.get()
        if not voice:
            messagebox.showwarning("Attention", "Aucune voix sélectionnée.")
            return None
        vc = voice.split("] ")[1]
        return ["espeak-ng",
                f"-v{vc}", f"-s{self.speed_var.get()}",
                f"-p{self.pitch_var.get()}", f"-a{self.volume_var.get()}",
                text]

    # ── Lecture ──
    def speak(self):
        text = self.text_box.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Attention", "Aucun texte à lire.")
            return
        cmd = self._build_cmd(self.prepare_text(text))
        if cmd:
            threading.Thread(target=speak_async, args=(cmd,), daemon=True).start()

    def speak_paragraph(self):
        """Lit le paragraphe sous le curseur.
        Stratégie fiable : on cherche dans le widget Text la dernière ligne
        vide avant le curseur et la première ligne vide après, ce qui délimite
        exactement le bloc de texte sans dépendre de la largeur de la fenêtre.
        """
        cursor_idx = self.text_box.index("insert")

        # Cherche le début du paragraphe (ligne vide précédente ou début du doc)
        start_line = int(cursor_idx.split(".")[0])
        while start_line > 1:
            line_content = self.text_box.get(f"{start_line - 1}.0",
                                             f"{start_line - 1}.end").strip()
            if line_content == "":
                break
            start_line -= 1

        # Cherche la fin du paragraphe (ligne vide suivante ou fin du doc)
        end_line = int(cursor_idx.split(".")[0])
        last_line = int(self.text_box.index("end-1c").split(".")[0])
        while end_line < last_line:
            line_content = self.text_box.get(f"{end_line + 1}.0",
                                             f"{end_line + 1}.end").strip()
            if line_content == "":
                break
            end_line += 1

        para = self.text_box.get(f"{start_line}.0", f"{end_line}.end").strip()
        if not para:
            return
        cmd = self._build_cmd(self.prepare_text(para))
        if cmd:
            threading.Thread(target=speak_async, args=(cmd,), daemon=True).start()

    def save_wav(self):
        text = self.text_box.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Attention", "Aucun texte à enregistrer.")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("Fichier WAV", "*.wav")],
            initialfile="lecture.wav")
        if not filename:
            return
        cmd = self._build_cmd(self.prepare_text(text))
        if cmd:
            cmd = cmd[:-1] + ["-w", filename, cmd[-1]]
            subprocess.Popen(cmd, stderr=subprocess.DEVNULL)

    def load_txt(self):
        f = filedialog.askopenfilename(
            initialdir=DEFAULT_DIR, filetypes=[("Text files", "*.txt")])
        if f:
            with open(f, "r", encoding="utf-8") as fh:
                self.text_box.delete("1.0", tk.END)
                self.text_box.insert(tk.END, fh.read())
            self.update_char_count()

    def load_pdf(self):
        f = filedialog.askopenfilename(
            initialdir=DEFAULT_DIR, filetypes=[("PDF files", "*.pdf")])
        if not f:
            return
        try:
            text = extract_pdf_text(f)
        except RuntimeError as exc:
            messagebox.showerror("Erreur PDF", str(exc))
            return
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, text)
        self.update_char_count()

    def clear_text(self):
        self.text_box.delete("1.0", tk.END)
        self.update_char_count()

# =============================================================================
#  ONGLET – RECORD (Micro → WAV / MP3)
# =============================================================================
class TabRecord(tk.Frame):
    def __init__(self, parent, cfg: dict):
        super().__init__(parent, bg=BG)
        self._cfg = cfg

        self.recording   = False
        self.frames: list[np.ndarray] = []
        self.stream      = None
        self.samplerate  = 44100
        self.channels    = 2        # K66 stéréo
        self.dtype       = "int16"
        self.elapsed     = 0
        self.timer_id    = None
        self._stop_event = threading.Event()

        self.device_var  = tk.StringVar()
        self.format_var  = tk.StringVar(value=cfg.get("rec_format", "WAV"))
        self.bitrate_var = tk.StringVar(value=cfg.get("rec_bitrate", "192k"))
        self.duration_var= tk.IntVar(value=0)
        self.vu_var      = tk.DoubleVar(value=0.0)
        self.status_var  = tk.StringVar(value="Prêt.")

        tk.Label(self, text="🎙  Enregistrement Microphone",
                 fg=ACCENT, bg=BG, font=("Arial", 13, "bold")).pack(pady=(14, 6))

        fd = tk.LabelFrame(self, text=" Périphérique d'entrée ",
                           fg=FG, bg=BG, font=("Arial", 9))
        fd.pack(fill="x", padx=20, pady=6)
        self.device_menu = ttk.Combobox(fd, textvariable=self.device_var,
                                        width=52, state="readonly")
        self.device_menu.grid(row=0, column=0, padx=10, pady=6)
        tk.Button(fd, text="↺ Rafraîchir", command=self.refresh_devices,
                  bg="#444444", fg=FG, relief="flat").grid(row=0, column=1, padx=6)
        self.default_label = tk.Label(fd, text="", fg="#88ff88", bg=BG, font=("Arial", 8))
        self.default_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=10)
        self.refresh_devices()

        fo = tk.LabelFrame(self, text=" Paramètres ",
                           fg=FG, bg=BG, font=("Arial", 9))
        fo.pack(fill="x", padx=20, pady=6)
        tk.Label(fo, text="Format :", fg=FG, bg=BG).grid(
            row=0, column=0, padx=10, pady=4, sticky="e")
        fmt = tk.Frame(fo, bg=BG)
        fmt.grid(row=0, column=1, sticky="w")
        for f in ("WAV", "MP3"):
            tk.Radiobutton(fmt, text=f, variable=self.format_var, value=f,
                           fg=FG, bg=BG, selectcolor=BG,
                           command=self._toggle_bitrate).pack(side="left", padx=4)
        tk.Label(fo, text="Bitrate MP3 :", fg=FG, bg=BG).grid(
            row=0, column=2, padx=(20, 4), sticky="e")
        self.bitrate_menu = ttk.Combobox(fo, textvariable=self.bitrate_var,
                                         values=["128k", "192k", "256k", "320k"],
                                         width=7, state="readonly")
        self.bitrate_menu.grid(row=0, column=3, padx=4)
        tk.Label(fo, text="Durée max (s) :", fg=FG, bg=BG).grid(
            row=1, column=0, padx=10, pady=4, sticky="e")
        tk.Spinbox(fo, from_=0, to=3600, textvariable=self.duration_var,
                   width=6, bg=BG2, fg=FG, insertbackground=FG).grid(
            row=1, column=1, sticky="w", padx=4)
        tk.Label(fo, text="(0 = illimité)", fg="#aaaaaa", bg=BG,
                 font=("Arial", 8)).grid(row=1, column=2, sticky="w")
        self._toggle_bitrate()

        fv = tk.LabelFrame(self, text=" Niveau d'entrée ",
                           fg=FG, bg=BG, font=("Arial", 9))
        fv.pack(fill="x", padx=20, pady=6)
        self.vu_bar = ttk.Progressbar(fv, orient="horizontal", length=460,
                                      maximum=100, variable=self.vu_var,
                                      mode="determinate")
        self.vu_bar.pack(padx=10, pady=6, fill="x")
        self.vu_label = tk.Label(fv, text="0 %", fg=FG, bg=BG, font=("Arial", 9))
        self.vu_label.pack()

        fb = tk.Frame(self, bg=BG)
        fb.pack(pady=12)
        self.btn_record = tk.Button(fb, text="⏺  Enregistrer",
                                    command=self.toggle_record,
                                    bg="#cc0000", fg=FG, width=16,
                                    font=("Arial", 10, "bold"), relief="flat")
        self.btn_record.pack(side="left", padx=10)
        self.btn_test = tk.Button(fb, text=" Test micro ",
                                  command=self.toggle_test,
                                  bg="#446644", fg=FG, width=14,
                                  font=("Arial", 10, "bold"), relief="flat")
        self.btn_test.pack(side="left", padx=10)
        self.btn_save = tk.Button(fb, text=" Sauvegarder ",
                                  command=self.save_recording,
                                  bg="#335588", fg=FG, width=16,
                                  font=("Arial", 10, "bold"), relief="flat",
                                  state="disabled")
        self.btn_save.pack(side="left", padx=10)

        tk.Label(self, textvariable=self.status_var,
                 fg="#aaaaaa", bg=BG, font=("Arial", 9)).pack(pady=4)

        self.bind("<Destroy>", self._on_destroy)

    def collect_config(self) -> dict:
        return {
            "rec_format":  self.format_var.get(),
            "rec_bitrate": self.bitrate_var.get(),
        }

    # ── Périphériques ──
    def refresh_devices(self):
        devs = list_input_devices()
        if not devs:
            self.device_menu["values"] = ["(aucun micro détecté)"]
            self.device_var.set("(aucun micro détecté)")
            self.default_label.config(text="")
            return
        self._raw_devices = devs
        labels = [f"[{i}] {n}" for i, n in devs]
        self.device_menu["values"] = labels
        default_idx = self._find_default_device(devs)
        if default_idx is not None:
            self.device_menu.current(default_idx)
            self.default_label.config(text="✔ Périphérique par défaut du système sélectionné")
        else:
            self.device_menu.current(0)
            self.default_label.config(text="")

    def _find_default_device(self, devs: list[tuple[int, str]]) -> int | None:
        try:
            default = sd.default.device
            default_in = default[0] if isinstance(default, (list, tuple)) else default
            if default_in is not None and default_in >= 0:
                for list_pos, (dev_idx, _) in enumerate(devs):
                    if dev_idx == default_in:
                        return list_pos
        except Exception:
            pass
        keywords = ("default", "pulse", "pipewire", "usb")
        for list_pos, (_, name) in enumerate(devs):
            if any(k in name.lower() for k in keywords):
                return list_pos
        return None

    def _selected_device_index(self) -> int | None:
        sel = self.device_var.get()
        if sel.startswith("["):
            try:
                return int(sel.split("]")[0][1:])
            except ValueError:
                pass
        return None

    def _toggle_bitrate(self):
        self.bitrate_menu.config(
            state="readonly" if self.format_var.get() == "MP3" else "disabled")

    # ── Test micro ──
    _testing = False

    def toggle_test(self):
        if not self._testing:
            self._start_test()
        else:
            self._stop_test()

    def _start_test(self):
        idx = self._selected_device_index()
        if idx is None:
            messagebox.showwarning("Attention", "Aucun micro sélectionné.")
            return
        self._testing = True
        self.btn_test.config(text="⏹  Stop test", bg="#884400")
        self.status_var.set("Test micro en cours… parlez dans le micro.")

        def cb(indata, frames, time, status):
            rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
            pct = min(100.0, rms / 32768 * 400)
            self.after(0, lambda p=pct: self._update_vu(p))

        try:
            self._test_stream = sd.InputStream(
                device=idx, samplerate=self.samplerate,
                channels=self.channels, dtype=self.dtype, callback=cb)
            self._test_stream.start()
        except Exception as e:
            self._testing = False
            self.btn_test.config(text=" Test micro ", bg="#446644")
            messagebox.showerror("Erreur", f"Impossible d'ouvrir le micro :\n{e}")

    def _stop_test(self):
        self._testing = False
        self.btn_test.config(text=" Test micro ", bg="#446644")
        if hasattr(self, "_test_stream") and self._test_stream:
            threading.Thread(target=self._close_stream,
                             args=(self._test_stream,), daemon=True).start()
            self._test_stream = None
        self._update_vu(0)
        self.status_var.set("Prêt.")

    # ── Enregistrement ──
    def toggle_record(self):
        if not self.recording:
            self._start()
        else:
            self._request_stop()

    def _start(self):
        if self._testing:
            self._stop_test()
        idx = self._selected_device_index()
        if idx is None:
            messagebox.showwarning("Attention", "Aucun micro sélectionné.")
            return
        self.frames.clear()
        self.recording = True
        self.elapsed   = 0
        self._stop_event.clear()
        self.btn_record.config(text="⏹  Stop", bg="#884400")
        self.btn_save.config(state="disabled")
        self.status_var.set("⏺ Enregistrement en cours…")
        self._tick()
        max_dur = self.duration_var.get()

        def cb(indata, frames, time, status):
            if self.recording:
                self.frames.append(indata.copy())
            rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
            pct = min(100.0, rms / 32768 * 400)
            self.after(0, lambda p=pct: self._update_vu(p))

        try:
            self.stream = sd.InputStream(
                device=idx, samplerate=self.samplerate,
                channels=self.channels, dtype=self.dtype, callback=cb)
            self.stream.start()
        except Exception as e:
            self.recording = False
            self.btn_record.config(text="⏺  Enregistrer", bg="#cc0000")
            messagebox.showerror("Erreur micro",
                f"Impossible d'ouvrir le périphérique :\n{e}\n\n"
                "Vérifiez qu'aucune autre application n'utilise le micro.")
            return
        if max_dur > 0:
            self.after(max_dur * 1000, self._request_stop)

    def _request_stop(self):
        if not self.recording:
            return
        self.recording = False
        self._stop_event.set()
        if self.timer_id:
            self.after_cancel(self.timer_id)
            self.timer_id = None
        self.btn_record.config(text="⏺  Enregistrer", bg="#cc0000")
        self.status_var.set("Arrêt en cours…")
        self._update_vu(0)
        stream_to_close = self.stream
        self.stream = None
        threading.Thread(
            target=self._close_stream_then_update,
            args=(stream_to_close,), daemon=True
        ).start()

    def _close_stream(self, stream):
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _close_stream_then_update(self, stream):
        self._close_stream(stream)
        self.after(0, self._finalize_stop)

    def _finalize_stop(self):
        if self.frames:
            total = len(self.frames) * self.frames[0].shape[0] / self.samplerate
            self.status_var.set(f"Arrêté — {total:.1f} s enregistrées.")
            self.btn_save.config(state="normal")
        else:
            self.status_var.set("Aucune donnée enregistrée.")

    def _update_vu(self, pct: float):
        try:
            self.vu_var.set(pct)
            self.vu_label.config(text=f"{int(pct)} %")
        except tk.TclError:
            pass

    def _tick(self):
        if self.recording:
            self.elapsed += 1
            m, s = divmod(self.elapsed, 60)
            self.status_var.set(f"⏺ Enregistrement en cours… {m:02d}:{s:02d}")
            self.timer_id = self.after(1000, self._tick)

    # ── Sauvegarde ──
    def save_recording(self):
        if not self.frames:
            messagebox.showwarning("Attention", "Aucun enregistrement.")
            return
        fmt = self.format_var.get()
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

        if fmt == "WAV":
            path = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("Fichier WAV", "*.wav")],
                initialdir=DEFAULT_DIR,
                initialfile=f"enreg_{ts}.wav")
            if not path:
                return
            audio = np.concatenate(self.frames).mean(axis=1, keepdims=True).astype(np.int16)
            sf.write(path, audio, self.samplerate, subtype="PCM_16")
            self.status_var.set(f"✔ Sauvegardé : {os.path.basename(path)}")

        else:  # MP3
            path = filedialog.asksaveasfilename(
                defaultextension=".mp3",
                filetypes=[("Fichier MP3", "*.mp3")],
                initialdir=DEFAULT_DIR,
                initialfile=f"enreg_{ts}.mp3")
            if not path:
                return
            tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(tmp_fd)
            try:
                audio = np.concatenate(self.frames).mean(axis=1, keepdims=True).astype(np.int16)
                sf.write(tmp_wav, audio, self.samplerate, subtype="PCM_16")
                wav_to_mp3(tmp_wav, path, self.bitrate_var.get())
                self.status_var.set(f"✔ Sauvegardé : {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Erreur MP3",
                    f"Conversion échouée :\n{e}\n"
                    "Vérifiez que ffmpeg est installé (sudo apt install ffmpeg).")
            finally:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)

    def _on_destroy(self, event):
        if event.widget is not self:
            return
        self.recording = False
        for s in filter(None, [self.stream,
                                getattr(self, "_test_stream", None)]):
            try:
                s.stop()
                s.close()
            except Exception:
                pass

# =============================================================================
#  ONGLET  – STT (Faster-Whisper)
# =============================================================================
class TabSTT(tk.Frame):
    def __init__(self, parent, cfg: dict):
        super().__init__(parent, bg=BG)
        self._cfg = cfg

        self.recording  = False
        self.frames: list[np.ndarray] = []
        self.stream     = None
        self.samplerate = 44100         # K66 ne supporte pas 16000
        self.channels   = 2             # K66 stéréo
        self.dtype      = "float32"
        self.model      = None
        self.model_name = None
        self.elapsed    = 0
        self.timer_id   = None

        self.device_var = tk.StringVar()
        self.model_var  = tk.StringVar(value=cfg.get("stt_model", "medium"))
        self.lang_var   = tk.StringVar(value=cfg.get("stt_lang", "Automatique"))
        self.task_var   = tk.StringVar(value="transcribe")
        self.vu_var     = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Prêt.")

        tk.Label(self, text="🗣  Speech‑to‑Text  (Faster-Whisper)",
                 fg=ACCENT, bg=BG, font=("Arial", 13, "bold")).pack(pady=(14, 6))

        if not WHISPER_OK:
            tk.Label(self,
                     text="⚠  faster-whisper non installé.\n"
                          "pip install faster-whisper --break-system-packages",
                     fg="#ff8888", bg=BG, font=("Arial", 10)).pack(pady=20)
            return

        # ── Micro ──
        fd = tk.LabelFrame(self, text=" Microphone ",
                           fg=FG, bg=BG, font=("Arial", 9))
        fd.pack(fill="x", padx=20, pady=6)
        self.device_menu = ttk.Combobox(fd, textvariable=self.device_var,
                                        width=48, state="readonly")
        self.device_menu.grid(row=0, column=0, padx=10, pady=6)
        tk.Button(fd, text="↺", command=self.refresh_devices,
                  bg="#444444", fg=FG, relief="flat", width=3).grid(row=0, column=1, padx=4)
        self.refresh_devices()

        # ── Modèle ──
        fm = tk.LabelFrame(self, text=" Modèle Whisper ",
                           fg=FG, bg=BG, font=("Arial", 9))
        fm.pack(fill="x", padx=20, pady=6)
        ttk.Combobox(fm, textvariable=self.model_var, values=WHISPER_MODELS,
                     width=16, state="readonly").grid(row=0, column=0, padx=10, pady=6)
        self.btn_load = tk.Button(fm, text="Charger le modèle",
                                  command=self._load_model_thread,
                                  bg="#335522", fg=FG, relief="flat")
        self.btn_load.grid(row=0, column=1, padx=10)
        self.model_label = tk.Label(fm, text="(aucun modèle chargé)",
                                    fg="#aaaaaa", bg=BG, font=("Arial", 9))
        self.model_label.grid(row=0, column=2, padx=10)
        self.size_label = tk.Label(fm, text=WHISPER_SIZES.get(self.model_var.get(), ""),
                                   fg="#888888", bg=BG, font=("Arial", 8))
        self.size_label.grid(row=0, column=3, padx=4)
        self.model_var.trace_add("write", lambda *a: self.size_label.config(
            text=WHISPER_SIZES.get(self.model_var.get(), "")))

        # Progressbar indéterminée (visible uniquement pendant le chargement)
        self._load_progress = ttk.Progressbar(fm, orient="horizontal", length=200,
                                              mode="indeterminate")
        self._load_progress.grid(row=1, column=0, columnspan=4, padx=10, pady=(0, 6))
        self._load_progress.grid_remove()   # cachée par défaut

        # ── Options ──
        fo = tk.LabelFrame(self, text=" Options ",
                           fg=FG, bg=BG, font=("Arial", 9))
        fo.pack(fill="x", padx=20, pady=6)
        tk.Label(fo, text="Langue :", fg=FG, bg=BG).grid(
            row=0, column=0, padx=10, pady=4, sticky="e")
        ttk.Combobox(fo, textvariable=self.lang_var,
                     values=list(WHISPER_LANGS.keys()),
                     width=14, state="readonly").grid(row=0, column=1, padx=4)
        tk.Label(fo, text="Tâche :", fg=FG, bg=BG).grid(
            row=0, column=2, padx=(20, 4), sticky="e")
        tf = tk.Frame(fo, bg=BG)
        tf.grid(row=0, column=3)
        tk.Radiobutton(tf, text="Transcrire", variable=self.task_var,
                       value="transcribe", fg=FG, bg=BG,
                       selectcolor=BG).pack(side="left")
        tk.Radiobutton(tf, text="Traduire (→ EN)", variable=self.task_var,
                       value="translate", fg=FG, bg=BG,
                       selectcolor=BG).pack(side="left", padx=8)

        # ── VU-mètre ──
        fv = tk.LabelFrame(self, text=" Niveau micro ",
                           fg=FG, bg=BG, font=("Arial", 9))
        fv.pack(fill="x", padx=20, pady=6)
        ttk.Progressbar(fv, orient="horizontal", length=460, maximum=100,
                        variable=self.vu_var, mode="determinate").pack(
            padx=10, pady=6, fill="x")

        # ── Boutons ──
        fb = tk.Frame(self, bg=BG)
        fb.pack(pady=8)
        self.btn_record = tk.Button(fb, text="⏺  Enregistrer",
                                    command=self.toggle_record,
                                    bg="#cc0000", fg=FG, width=16,
                                    font=("Arial", 10, "bold"), relief="flat")
        self.btn_record.pack(side="left", padx=8)
        tk.Button(fb, text=" Depuis fichier ",
                  command=self.transcribe_file,
                  bg="#555555", fg=FG, width=16,
                  font=("Arial", 10, "bold"), relief="flat").pack(side="left", padx=8)
        tk.Button(fb, text="🗑  Effacer",
                  command=self.clear_text,
                  bg="#444444", fg=FG, width=12,
                  font=("Arial", 10, "bold"), relief="flat").pack(side="left", padx=8)

        # ── Zone texte ──
        ft = tk.LabelFrame(self, text=" Transcription ",
                           fg=FG, bg=BG, font=("Arial", 9))
        ft.pack(fill="both", expand=True, padx=20, pady=6)
        self.text_box = tk.Text(ft, height=10, width=80,
                                bg=BG2, fg=FG, insertbackground=FG,
                                font=("Courier", 10), wrap="word")
        self.text_box.pack(padx=6, pady=6, fill="both", expand=True)

        fe = tk.Frame(self, bg=BG)
        fe.pack(pady=6)
        tk.Button(fe, text=" Sauvegarder TXT ",
                  command=self.save_txt,
                  bg="#335588", fg=FG, relief="flat",
                  font=("Arial", 9)).pack(side="left", padx=8)
        tk.Button(fe, text=" Copier ",
                  command=self.copy_text,
                  bg="#445544", fg=FG, relief="flat",
                  font=("Arial", 9)).pack(side="left", padx=8)

        tk.Label(self, textvariable=self.status_var,
                 fg="#aaaaaa", bg=BG, font=("Arial", 9)).pack(pady=(2, 10))

    def collect_config(self) -> dict:
        return {
            "stt_model": self.model_var.get(),
            "stt_lang":  self.lang_var.get(),
        }

    # ── Périphériques ──
    def refresh_devices(self):
        devs = list_input_devices()
        if not devs:
            self.device_menu["values"] = ["(aucun micro détecté)"]
            self.device_var.set("(aucun micro détecté)")
            return
        self.device_menu["values"] = [f"[{i}] {n}" for i, n in devs]
        self.device_menu.current(0)

    def _selected_device_index(self) -> int | None:
        sel = self.device_var.get()
        return int(sel.split("]")[0][1:]) if sel.startswith("[") else None

    # ── Chargement modèle (avec progressbar) ──
    def _load_model_thread(self):
        self.btn_load.config(state="disabled", text="Chargement…")
        self._set_status("Chargement du modèle Whisper, veuillez patienter…")
        self._load_progress.grid()          # affiche la progressbar
        self._load_progress.start(12)       # animation toutes les 12 ms
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        name = self.model_var.get()
        try:
            self.model      = WhisperModel(name, device="cpu",
                                           compute_type="int8",
                                           cpu_threads=WHISPER_CPU_THREADS)
            self.model_name = name
            self.after(0, lambda: self.model_label.config(
                text=f"✔ Modèle « {name} » chargé", fg="#88ff88"))
            self._set_status(f"Modèle {name} prêt.")
        except Exception as e:
            self.after(0, lambda: messagebox.showerror(
                "Erreur", f"Impossible de charger le modèle :\n{e}"))
            self._set_status("Erreur de chargement.")
        finally:
            self.after(0, self._hide_progress)
            self.after(0, lambda: self.btn_load.config(
                state="normal", text="Charger le modèle"))

    def _hide_progress(self):
        self._load_progress.stop()
        self._load_progress.grid_remove()

    # ── Enregistrement ──
    def toggle_record(self):
        if not self.recording:
            self._start_recording()
        else:
            self._stop_and_transcribe()

    def _start_recording(self):
        if not self.model:
            messagebox.showwarning("Attention", "Chargez un modèle Whisper.")
            return
        idx = self._selected_device_index()
        if idx is None:
            messagebox.showwarning("Attention", "Aucun micro sélectionné.")
            return
        self.frames, self.recording, self.elapsed = [], True, 0
        self.btn_record.config(text="⏹  Stop & Transcrire", bg="#884400")
        self._tick()

        def cb(indata, frames, time, status):
            self.frames.append(indata.copy())
            pct = min(100.0, float(np.sqrt(np.mean(indata ** 2))) * 300)
            self.after(0, lambda p=pct: self.vu_var.set(p))

        self.stream = sd.InputStream(device=idx, samplerate=self.samplerate,
                                     channels=self.channels, dtype=self.dtype,
                                     callback=cb)
        self.stream.start()

    def _stop_and_transcribe(self):
        self.recording = False
        if self.timer_id:
            self.after_cancel(self.timer_id)
            self.timer_id = None
        self.btn_record.config(text="⏺  Enregistrer", bg="#cc0000")
        self.vu_var.set(0)

        stream_to_close = self.stream
        self.stream     = None
        frames_copy     = list(self.frames)

        def close_then_transcribe():
            if stream_to_close:
                try:
                    stream_to_close.stop()
                except Exception:
                    pass
                try:
                    stream_to_close.close()
                except Exception:
                    pass
            if not frames_copy:
                self.after(0, lambda: self._set_status("Aucune donnée à transcrire."))
                return
            # Rééchantillonnage dynamique (fini le 44100 codé en dur)
            audio = np.concatenate(frames_copy).mean(axis=1).astype(np.float32)
            audio = safe_resample(audio, self.samplerate, 16000)

            # Fichier temporaire sécurisé
            tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(tmp_fd)
            sf.write(tmp_wav, audio, 16000)
            self.after(0, lambda: self._transcribe_audio(tmp_wav, delete_after=True))

        threading.Thread(target=close_then_transcribe, daemon=True).start()

    def _tick(self):
        if self.recording:
            self.elapsed += 1
            m, s = divmod(self.elapsed, 60)
            self._set_status(f"⏺ Enregistrement… {m:02d}:{s:02d}")
            self.timer_id = self.after(1000, self._tick)

    # ── Transcription ──
    def transcribe_file(self):
        if not self.model:
            messagebox.showwarning("Attention", "Chargez un modèle Whisper.")
            return
        path = filedialog.askopenfilename(
            filetypes=[("Fichiers audio", "*.wav *.mp3 *.ogg *.flac *.m4a *.aac"),
                       ("Tous", "*.*")])
        if path:
            self._transcribe_audio(path, delete_after=False)

    def _transcribe_audio(self, path: str, delete_after: bool = False):
        lang_code = WHISPER_LANGS[self.lang_var.get()]
        task      = self.task_var.get()
        self._set_status("Transcription en cours…")
        self.btn_record.config(state="disabled")

        def run():
            try:
                segments, info = self.model.transcribe(
                    path,
                    language=lang_code,
                    task=task,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                segments = list(segments)
                text = " ".join(seg.text.strip() for seg in segments)
                lang_detected = info.language if lang_code is None else lang_code
                self.after(0, lambda t=text, l=lang_detected: self._insert_text(t, l))
                self.after(0, lambda: self._set_status(
                    f"✔ Terminée — {len(text)} caractères"
                    f" | langue : {lang_detected}"
                    f" | durée : {info.duration:.1f}s"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Erreur Whisper", str(e)))
                self._set_status("Erreur de transcription.")
            finally:
                if delete_after and os.path.exists(path):
                    os.remove(path)
                self.after(0, lambda: self.btn_record.config(state="normal"))

        threading.Thread(target=run, daemon=True).start()

    def _insert_text(self, text: str, lang: str | None = None):
        self.text_box.insert(tk.END, f"\n{text}\n")
        self.text_box.see(tk.END)

    def clear_text(self):
        self.text_box.delete("1.0", tk.END)

    def save_txt(self):
        content = self.text_box.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Attention", "Aucun texte à sauvegarder.")
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Fichier texte", "*.txt")],
            initialfile=f"transcription_{ts}.txt")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self._set_status(f"✔ Sauvegardé : {os.path.basename(path)}")

    def copy_text(self):
        self.clipboard_clear()
        self.clipboard_append(self.text_box.get("1.0", tk.END).strip())
        self._set_status("✔ Texte copié dans le presse‑papiers.")

    def _set_status(self, msg: str):
        self.after(0, lambda: self.status_var.set(msg))

# =============================================================================
#  APPLICATION PRINCIPALE
# =============================================================================
class StudioApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Studio Audio – eSpeak NG · Record · STT")
        root.configure(bg=BG)

        # Sauvegarde config à la fermeture
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",     background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background="#333333", foreground=FG,
                                         padding=[14, 6], font=("Arial", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", "#555555")],
                  foreground=[("selected", ACCENT)])

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=0, pady=0)

        cfg = load_config()

        self.tab_tts    = TabTTS(notebook, cfg)
        self.tab_record = TabRecord(notebook, cfg)
        self.tab_stt    = TabSTT(notebook, cfg)

        notebook.add(self.tab_tts,    text="  🔊  Text-to-Speak  ")
        notebook.add(self.tab_record, text="  🎙   Record         ")
        notebook.add(self.tab_stt,    text="  🗣  Speech‑to‑Text ")

    def _on_close(self):
        cfg = load_config()
        cfg.update(self.tab_tts.collect_config())
        cfg.update(self.tab_record.collect_config())
        cfg.update(self.tab_stt.collect_config())
        save_config(cfg)
        self.root.destroy()

# =============================================================================
#  LANCEMENT
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    splash = SplashScreen(root)

    def start_app():
        splash.destroy()
        root.deiconify()
        StudioApp(root)

    root.after(1500, start_app)
    root.mainloop()
