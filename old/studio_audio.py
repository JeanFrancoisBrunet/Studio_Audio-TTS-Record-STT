#!/usr/bin/env python3
# =============================================================================
#  Studio Audio – TTS · Record · STT
#  TTS : Text to Speak
#  STT : Speech to Text
#  Modules : eSpeak NG + MBROLA · Enregistrement WAV/MP3 · Faster-Whisper STT
#  Auteur  : Jean‑François BRUNET - JFBConseils - Mars 2026
#
#  Dépendances : ./install_whisper_pi5.sh
# =============================================================================

import sys
import os
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"      # supprime warnings ONNX
import io
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import threading
import tempfile
import numpy as np
from datetime import datetime

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

# --- Masquage temporaire de stderr pendant l'import de faster-whisper ---
stderr_backup = sys.stderr
sys.stderr = io.StringIO()

try:
    from faster_whisper import WhisperModel
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False

# --- Restauration de stderr --- pour éviter les warnings ONNX
sys.stderr = stderr_backup

# =============================================================================
#  CONSTANTES
# =============================================================================
FG       = "#ffffff"
BG       = "#222222"
BG2      = "#333333"
ACCENT   = "#aaddff"

DEFAULT_DIR  = "/home/jfbrunet/Projects/Studio_Enregistrement"
LOGO_PATH    = "/home/jfbrunet/Projects/Studio_Enregistrement/icons/microphone_1.png"

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
WHISPER_SIZES  = {
    "tiny": "~75 Mo", "base": "~145 Mo", "small": "~480 Mo",
    "medium": "~1,5 Go", "large-v3-turbo": "~1,6 Go", "large-v3": "~3,1 Go"
}
WHISPER_LANGS = {
    "Automatique": None, "Français": "fr", "Anglais": "en",
    "Allemand": "de",    "Espagnol": "es", "Italien": "it",
    "Portugais": "pt",   "Néerlandais": "nl", "Japonais": "ja",
    "Chinois": "zh",     "Arabe": "ar",
}

# =============================================================================
#  UTILITAIRES PARTAGÉS
# =============================================================================
def get_espeak_voices():
    voices = []
    base = "/usr/lib/aarch64-linux-gnu/espeak-ng-data/voices"
    if not os.path.isdir(base):
        return []
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.startswith("."):
                continue
            rel = os.path.relpath(os.path.join(root, f), base)
            voices.append(rel.replace("\\", "/"))
    return sorted(voices)


def get_mbrola_voices():
    voices = []
    path = "/usr/share/mbrola"
    if os.path.isdir(path):
        for f in os.listdir(path):
            if os.path.isdir(os.path.join(path, f)):
                voices.append(f)
    return sorted(voices)


def extract_pdf_text(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
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


def list_input_devices():
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            devices.append((i, d["name"]))
    return devices


def wav_to_mp3(wav_path, mp3_path, bitrate="192k"):
    if not PYDUB_OK:
        raise RuntimeError("pydub non installé : pip install pydub")
    AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3", bitrate=bitrate)


# TTS – processus global
_tts_process = None

def speak_async(cmd):
    global _tts_process
    _tts_process = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)
    _tts_process.wait()
    _tts_process = None

def stop_speech():
    global _tts_process
    if _tts_process:
        _tts_process.terminate()
        _tts_process = None

# =============================================================================
#  SPLASH SCREEN
# =============================================================================
class SplashScreen(tk.Toplevel):
    def __init__(self, root):
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
#  ONGLET 1 – TTS (eSpeak NG + MBROLA)
# =============================================================================
class TabTTS(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)

        self.lang_var   = tk.StringVar()
        self.voice_var  = tk.StringVar()
        self.speed_var  = tk.IntVar(value=105)
        self.pitch_var  = tk.IntVar(value=30)
        self.volume_var = tk.IntVar(value=50)
        self.pause_var  = tk.BooleanVar(value=True)

        espeak = get_espeak_voices()
        mbrola = get_mbrola_voices()

        def detect_langs(voices):
            langs = {}
            for v in voices:
                s = os.path.basename(v).split("-")[0]
                if len(s) == 2:
                    langs[s] = True
            return sorted(langs.keys())

        available = detect_langs(espeak)

        ALL_LABELS = {
            "fr": "Français",   "en": "Anglais",    "de": "Allemand",
            "es": "Espagnol",   "it": "Italien",    "pt": "Portugais",
            "nl": "Néerlandais","fi": "Finnois",    "hu": "Hongrois",
            "pl": "Polonais",   "cs": "Tchèque",    "da": "Danois",
            "et": "Estonien",   "lv": "Letton",     "lt": "Lituanien",
            "mk": "Macédonien", "no": "Norvégien",  "bg": "Bulgare",
            "hr": "Croate",     "ga": "Gaélique",   "cy": "Gallois",
            "an": "Aragonais",  "bs": "Bosniaque",  "ca": "Catalan",
            "el": "Grec",       "is": "Islandais",
        }
        LANG_LABELS = {k: v for k, v in ALL_LABELS.items() if k in available}
        self.lang_codes  = list(LANG_LABELS.keys())
        lang_labels_list = list(LANG_LABELS.values())

        self.voices_by_lang = {lang: [] for lang in LANG_LABELS}
        for v in espeak:
            s = os.path.basename(v).split("-")[0]
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

        if "Français" in lang_labels_list:
            self.lang_var.set("Français")
            self.lang_menu.current(lang_labels_list.index("Français"))
        self.update_voice_menu()
        if "[MBROLA] mb-fr7" in self.voice_menu["values"]:
            self.voice_var.set("[MBROLA] mb-fr7")

        tk.Checkbutton(top, text="Pauses entre les mots",
                       variable=self.pause_var, fg=FG, bg=BG,
                       selectcolor=BG).grid(row=2, column=2, padx=10, pady=6, sticky="w")

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

        actions = tk.Frame(self, bg=BG)
        actions.pack(pady=8)
        tk.Button(actions, text="Lire",           command=self.speak).pack(side="left", padx=8)
        tk.Button(actions, text="Lire Paragraphe",command=self.speak_paragraph).pack(side="left", padx=8)
        tk.Button(actions, text="Stop",           command=stop_speech).pack(side="left", padx=8)
        tk.Button(actions, text="Enregistrer WAV",command=self.save_wav).pack(side="left", padx=8)

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

    def prepare_text(self, text):
        return text.replace(" ", "  ") if self.pause_var.get() else text

    def _build_cmd(self, text):
        voice = self.voice_var.get()
        if not voice:
            messagebox.showwarning("Attention", "Aucune voix sélectionnée.")
            return None
        vc = voice.split("] ")[1]
        return ["espeak-ng",
                f"-v{vc}", f"-s{self.speed_var.get()}",
                f"-p{self.pitch_var.get()}", f"-a{self.volume_var.get()}",
                text]

    def speak(self):
        text = self.text_box.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Attention", "Aucun texte à lire.")
            return
        cmd = self._build_cmd(self.prepare_text(text))
        if cmd:
            threading.Thread(target=speak_async, args=(cmd,), daemon=True).start()

    def speak_paragraph(self):
        text  = self.text_box.get("1.0", tk.END)
        lines = text.split("\n")
        paragraphs, current = [], []
        for line in lines:
            s = line.strip()
            if s == "" or s.startswith("§") or s.endswith("."):
                if s != "":
                    current.append(s)
                if current:
                    paragraphs.append(" ".join(current).strip())
                    current = []
            else:
                current.append(s)
        if current:
            paragraphs.append(" ".join(current).strip())
        cursor_line  = int(self.text_box.index("insert").split(".")[0])
        line_count   = 1
        target_index = 0
        for i, p in enumerate(paragraphs):
            p_lines = p.count(" ") // 12 + 1
            if line_count + p_lines > cursor_line:
                target_index = i
                break
            line_count += p_lines
        if target_index >= len(paragraphs):
            return
        cmd = self._build_cmd(self.prepare_text(paragraphs[target_index]))
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
        if f:
            self.text_box.delete("1.0", tk.END)
            self.text_box.insert(tk.END, extract_pdf_text(f))
            self.update_char_count()

    def clear_text(self):
        self.text_box.delete("1.0", tk.END)
        self.update_char_count()

# =============================================================================
#  ONGLET 2 – RECORD (Micro → WAV / MP3)
# =============================================================================
class TabRecord(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)

        self.recording   = False
        self.frames      = []
        self.stream      = None
        self.samplerate  = 44100
        self.channels    = 2        # K66 stéréo
        self.dtype       = "int16"
        self.elapsed     = 0
        self.timer_id    = None
        self._stop_event = threading.Event()

        self.device_var  = tk.StringVar()
        self.format_var  = tk.StringVar(value="WAV")
        self.bitrate_var = tk.StringVar(value="192k")
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

    def _find_default_device(self, devs):
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

    def _selected_device_index(self):
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

    def _update_vu(self, pct):
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
            tmp = path.replace(".mp3", "_tmp.wav")
            audio = np.concatenate(self.frames).mean(axis=1, keepdims=True).astype(np.int16)
            sf.write(tmp, audio, self.samplerate, subtype="PCM_16")
            try:
                wav_to_mp3(tmp, path, self.bitrate_var.get())
                os.remove(tmp)
                self.status_var.set(f"✔ Sauvegardé : {os.path.basename(path)}")
            except Exception as e:
                if os.path.exists(tmp):
                    os.remove(tmp)
                messagebox.showerror("Erreur MP3",
                    f"Conversion échouée :\n{e}\n"
                    "Vérifiez que ffmpeg est installé (sudo apt install ffmpeg).")

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
#  ONGLET 3 – STT (Faster-Whisper)
# =============================================================================
class TabSTT(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)

        self.recording  = False
        self.frames     = []
        self.stream     = None
        self.samplerate = 44100         # K66 ne supporte pas 16000
        self.channels   = 2             # K66 stéréo
        self.dtype      = "float32"
        self.model      = None
        self.model_name = None
        self.elapsed    = 0
        self.timer_id   = None

        self.device_var = tk.StringVar()
        self.model_var  = tk.StringVar(value="medium")
        self.lang_var   = tk.StringVar(value="Automatique")
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

        fd = tk.LabelFrame(self, text=" Microphone ",
                           fg=FG, bg=BG, font=("Arial", 9))
        fd.pack(fill="x", padx=20, pady=6)
        self.device_menu = ttk.Combobox(fd, textvariable=self.device_var,
                                        width=48, state="readonly")
        self.device_menu.grid(row=0, column=0, padx=10, pady=6)
        tk.Button(fd, text="↺", command=self.refresh_devices,
                  bg="#444444", fg=FG, relief="flat", width=3).grid(row=0, column=1, padx=4)
        self.refresh_devices()

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
        self.size_label = tk.Label(fm, text=WHISPER_SIZES["medium"],
                                   fg="#888888", bg=BG, font=("Arial", 8))
        self.size_label.grid(row=0, column=3, padx=4)
        self.model_var.trace_add("write", lambda *a: self.size_label.config(
            text=WHISPER_SIZES.get(self.model_var.get(), "")))

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

        fv = tk.LabelFrame(self, text=" Niveau micro ",
                           fg=FG, bg=BG, font=("Arial", 9))
        fv.pack(fill="x", padx=20, pady=6)
        ttk.Progressbar(fv, orient="horizontal", length=460, maximum=100,
                        variable=self.vu_var, mode="determinate").pack(
            padx=10, pady=6, fill="x")

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

    def refresh_devices(self):
        devs = list_input_devices()
        if not devs:
            self.device_menu["values"] = ["(aucun micro détecté)"]
            self.device_var.set("(aucun micro détecté)")
            return
        self.device_menu["values"] = [f"[{i}] {n}" for i, n in devs]
        self.device_menu.current(0)

    def _selected_device_index(self):
        sel = self.device_var.get()
        return int(sel.split("]")[0][1:]) if sel.startswith("[") else None

    def _load_model_thread(self):
        self.btn_load.config(state="disabled", text="Chargement…")
        self._set_status("Chargement du modèle Whisper, veuillez patienter…")
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        name = self.model_var.get()
        try:
            self.model      = WhisperModel(name, device="cpu",
                                           compute_type="int8", cpu_threads=4)
            self.model_name = name
            self.after(0, lambda: self.model_label.config(
                text=f"✔ Modèle « {name} » chargé", fg="#88ff88"))
            self._set_status(f"Modèle {name} prêt.")
        except Exception as e:
            self.after(0, lambda: messagebox.showerror(
                "Erreur", f"Impossible de charger le modèle :\n{e}"))
            self._set_status("Erreur de chargement.")
        finally:
            self.after(0, lambda: self.btn_load.config(
                state="normal", text="Charger le modèle"))

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
        self.stream = None
        frames_copy = list(self.frames)

        def close_then_transcribe():
            if stream_to_close:
                try: stream_to_close.stop()
                except: pass
                try: stream_to_close.close()
                except: pass
            if not frames_copy:
                self.after(0, lambda: self._set_status("Aucune donnée à transcrire."))
                return
            audio = np.concatenate(frames_copy).mean(axis=1).astype(np.float32)
            from scipy.signal import resample_poly
            audio = resample_poly(audio, 16000, 44100).astype(np.float32)
            tmp = tempfile.mktemp(suffix=".wav")
            sf.write(tmp, audio, 16000)
            self.after(0, lambda: self._transcribe_audio(tmp, delete_after=True))

        threading.Thread(target=close_then_transcribe, daemon=True).start()

    def _tick(self):
        if self.recording:
            self.elapsed += 1
            m, s = divmod(self.elapsed, 60)
            self._set_status(f"⏺ Enregistrement… {m:02d}:{s:02d}")
            self.timer_id = self.after(1000, self._tick)

    def transcribe_file(self):
        if not self.model:
            messagebox.showwarning("Attention", "Chargez un modèle Whisper.")
            return
        path = filedialog.askopenfilename(
            filetypes=[("Fichiers audio", "*.wav *.mp3 *.ogg *.flac *.m4a *.aac"),
                       ("Tous", "*.*")])
        if path:
            self._transcribe_audio(path, delete_after=False)

    def _transcribe_audio(self, path, delete_after=False):
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

    def _insert_text(self, text, lang=None):
        ts       = datetime.now().strftime("[%H:%M:%S]")
        lang_str = f" | langue : {lang}" if lang else ""
        #self.text_box.insert(tk.END, f"\n{ts}{lang_str}\n{text}\n")
        self.text_box.insert(tk.END, f"\n{text}\n")     # transcription texte uniquement
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

    def _set_status(self, msg):
        self.after(0, lambda: self.status_var.set(msg))


# =============================================================================
#  APPLICATION PRINCIPALE
# =============================================================================
class StudioApp:
    def __init__(self, root):
        self.root = root
        root.title("Studio Audio – eSpeak NG · Record · STT")
        root.configure(bg=BG)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",        background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",    background="#333333", foreground=FG,
                                            padding=[14, 6], font=("Arial", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", "#555555")],
                  foreground=[("selected", ACCENT)])

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=0, pady=0)

        tab_tts    = TabTTS(notebook)
        tab_record = TabRecord(notebook)
        tab_stt    = TabSTT(notebook)

        notebook.add(tab_tts,    text="  🔊  Text-to-Speak  ")
        notebook.add(tab_record, text="  🎙   Record         ")
        notebook.add(tab_stt,    text="  🗣  Speech‑to‑Text ")

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
