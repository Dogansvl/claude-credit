"""Génère les voix off de « C'EST QUI LUI ? » : narrateur (N01…N09b) et entité (E_*).

Sorties : assets/audio/voice/{ID}.wav (48 kHz stéréo PCM 24 bits), manifest.json (format de
docs/ARCHITECTURE.md), REPORT.md, spectrograms/*.png, qc.json.

Backends (ordre automatique) :
  1. edge   : edge-tts fr-FR-HenriNeural (rate −12 %, pitch −8 Hz) — voix neurale Microsoft, en ligne.
  2. piper  : hors ligne, fr_FR-upmc-medium locuteur « pierre » (choisi par --compare, voir REPORT.md).

Usage :
  python tools/make_voices.py [--backend auto|edge|piper] [--only N01,E_CQL] [--takes 4] [--no-asr]
  python tools/make_voices.py --compare          # banc d'essai des voix piper (écrit comparison.json)

Les durées de parole sont ajustées (dans des limites naturelles) aux fenêtres disponibles dans
edit/edl_spec.py ; les dépassements restants sont signalés dans REPORT.md.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from pipeline.audio import voicefx as vf  # noqa: E402

VOICE_DIR = config.VOICE_DIR
CACHE = config.CACHE_DIR / "voice"

# -----------------------------------------------------------------------------
# Textes (edit/script.md). text = affiché (sous-titres) ; tts = prononcé.
# -----------------------------------------------------------------------------
ESCAPE_GAME_TTS = "eskeïpe gueïme"  # graphie retenue (voir REPORT.md, test --spellings)

LINES: list[dict] = [
    dict(id="N01", kind="narrator", text="Le 30 octobre 2026, quatre amis ont pris la route pour un escape game.",
         tts=f"Le trente octobre deux mille vingt-six, quatre amis ont pris la route pour un {ESCAPE_GAME_TTS}."),
    dict(id="N02", kind="narrator", text="Ils se sont filmés toute la soirée.", tts="Ils se sont filmés toute la soirée."),
    dict(id="N03", kind="narrator", text="Voici ce qu'ils ont filmé.", tts="Voici ce qu'ils ont filmé."),
    dict(id="N04", kind="narrator", text="Ils ne l'ont jamais revu.", tts="Ils ne l'ont jamais revu."),
    dict(id="N05", kind="narrator", text="Lui ne les a plus jamais quittés.", tts="Lui ne les a plus jamais quittés."),
    dict(id="N06", kind="narrator", text="Les organisateurs jurent que cette pièce ne fait pas partie du jeu.",
         tts="Les organisateurs jurent que cette pièce ne fait pas partie du jeu."),
    dict(id="N07a", kind="narrator", text="À 22h12, Dogan s'est mis au piano.",
         tts="À vingt-deux heures douze, Dogane s'est mis au piano."),
    dict(id="N07b", kind="narrator", text="Il ne se souvient pas d'avoir joué.", tts="Il ne se souvient pas d'avoir joué."),
    dict(id="N08a", kind="narrator", text="Sur la photo, ils sont quatre.", tts="Sur la photo, ils sont quatre."),
    dict(id="N08b", kind="narrator", text="Regardez derrière eux.", tts="Regardez derrière eux."),
    dict(id="N09a", kind="narrator", text="À 23h, tout était fini.", tts="À vingt-trois heures, tout était fini."),
    dict(id="N09b", kind="narrator", text="Ils riaient.", tts="Ils riaient."),
    dict(id="E_DOGAN", kind="entity", preset="name", text="Dogan…", tts="Dogane."),
    dict(id="E_GABRIEL", kind="entity", preset="name", text="Gabriel…", tts="Gabriel."),
    dict(id="E_HUGO", kind="entity", preset="name", text="Hugo…", tts="Hugo."),
    dict(id="E_ARSENE", kind="entity", preset="name", text="Arsène…", tts="Arsène."),
    dict(id="E_IDENT", kind="entity", preset="ident", text="Identification terminée.", tts="Identification terminée."),
    dict(id="E_CQL", kind="entity", preset="cql", text="C'est qui lui ?", tts="C'est qui lui ?"),
]
BY_ID = {d["id"]: d for d in LINES}

# -----------------------------------------------------------------------------
# Backends
# -----------------------------------------------------------------------------
EDGE_VOICE = "fr-FR-HenriNeural"
EDGE_RATE = {"narrator": -12, "entity": -14}   # %
EDGE_PITCH = {"narrator": "-8Hz", "entity": "-8Hz"}
EDGE_RATE_MAX = 0                              # accélération maximale admise pour tenir une fenêtre

PIPER_VOICES = {  # nom -> (dossier, fichier onnx, locuteur)
    "upmc-pierre": ("vits-piper-fr_FR-upmc-medium", "fr_FR-upmc-medium.onnx", 1),
    "upmc-jessica": ("vits-piper-fr_FR-upmc-medium", "fr_FR-upmc-medium.onnx", 0),
    "tom": ("vits-piper-fr_FR-tom-medium", "fr_FR-tom-medium.onnx", None),
    "gilles": ("vits-piper-fr_FR-gilles-low", "fr_FR-gilles-low.onnx", None),
    "siwis": ("vits-piper-fr_FR-siwis-medium", "fr_FR-siwis-medium.onnx", None),
}
PIPER_NARRATOR = "tom"         # choisi par le banc d'essai (voir REPORT.md)
PIPER_ENTITY = "tom"           # le souffle de tom disparaît au chuchotement ; articulation nette des prénoms
# Rythme : length_scale préféré (1 = naturel ; > 1 = plus lent) et bornes pour tenir les fenêtres
PIPER_LS = {"narrator": 1.10, "name": 1.0, "ident": 1.0, "cql": 1.0}
PIPER_LS_MIN = {"narrator": 0.94, "name": 0.85, "ident": 0.85, "cql": 0.85}
PIPER_NOISE = dict(noise_scale=0.62, noise_w_scale=0.70)

# Post-traitement narrateur selon la voix (voir pipeline/audio/voicefx.NARRATOR_DEFAULTS)
_EQ_BREATHY = [("hp", 65, 2), ("lowshelf", 150, 3.0), ("peak", 320, -2.0, 1.1), ("peak", 2400, 1.0, 0.9),
               ("highshelf", 4200, -6.0), ("lp", 14000, 2)]
NARRATOR_FX = {
    "edge": dict(semitones=-1.0, exciter_db=-28.0),              # Henri : déjà grave (pitch −8 Hz), MP3 24 kHz
    "tom": dict(semitones=-2.5, eq=_EQ_BREATHY,                    # voix soufflée : aigus domptés, de-esser plus ferme
                deess=dict(freq=4500, thr_db=-34, ratio=4.0, max_cut_db=10)),
    "upmc-pierre": dict(semitones=-2.0, exciter_db=-26.0),        # bande étroite (22 kHz) : exciteur discret
    "gilles": dict(semitones=-1.0, exciter_db=-24.0),
    "upmc-jessica": dict(semitones=-1.0, exciter_db=-26.0),
}
# Compression temporelle maximale admise (rubberband, formants préservés) pour tenir une fenêtre
TEMPO_MAX = {"narrator": 1.10, "name": 1.25, "ident": 1.25, "cql": 1.20}


def narrator_fx(backend) -> dict:
    return dict(NARRATOR_FX["edge"] if backend.name == "edge" else NARRATOR_FX.get(backend.voices["narrator"], {}))


class BackendUnavailable(RuntimeError):
    pass


class EdgeBackend:
    name = "edge"

    def __init__(self, timeout: float = 40.0):
        try:
            import edge_tts  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable("edge-tts non installé") from e
        self.timeout = timeout
        self.label = f"edge-tts:{EDGE_VOICE}"

    def synth(self, text: str, kind: str, speed: float = 1.0, seed: int = 0) -> tuple[np.ndarray, int]:
        """speed > 1 accélère par rapport au débit par défaut du rôle."""
        import edge_tts
        base = EDGE_RATE["narrator" if kind == "narrator" else "entity"]
        rate = int(round((1 + base / 100) * speed * 100 - 100))
        rate = min(rate, EDGE_RATE_MAX if kind == "narrator" else 10)
        pitch = EDGE_PITCH["narrator" if kind == "narrator" else "entity"]
        tmp = Path(tempfile.mkdtemp(prefix="edge_"))
        mp3 = tmp / "a.mp3"
        try:
            async def run():
                c = edge_tts.Communicate(text, voice=EDGE_VOICE, rate=f"{rate:+d}%", pitch=pitch)
                await c.save(str(mp3))
            asyncio.run(asyncio.wait_for(run(), self.timeout))
            if not mp3.exists() or mp3.stat().st_size < 1000:
                raise BackendUnavailable("edge-tts : réponse vide")
            wav = tmp / "a.wav"
            subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i",
                            str(mp3), "-ac", "1", "-ar", "48000", "-c:a", "pcm_f32le", str(wav)], check=True)
            x, sr = sf.read(str(wav), dtype="float32")
            return x, sr
        except BackendUnavailable:
            raise
        except Exception as e:  # réseau bloqué, DNS, TLS, timeout…
            raise BackendUnavailable(f"edge-tts indisponible : {type(e).__name__}: {e}") from e
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def describe(self, kind: str) -> dict:
        return dict(voice=EDGE_VOICE, rate=EDGE_RATE["narrator" if kind == "narrator" else "entity"],
                    pitch=EDGE_PITCH["narrator" if kind == "narrator" else "entity"])


class PiperBackend:
    """Piper (VITS) hors ligne. Le bruit interne du modèle est graine explicite (ONNX RandomNormalLike.seed)."""
    name = "piper"

    def __init__(self, narrator_voice: str = PIPER_NARRATOR, entity_voice: str = PIPER_ENTITY):
        try:
            import piper  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable("piper-tts non installé") from e
        self.voices = {"narrator": narrator_voice, "entity": entity_voice}
        for v in set(self.voices.values()):
            if not self.model_path(v).exists():
                raise BackendUnavailable(f"modèle piper introuvable : {self.model_path(v)}")
        self._proto = {}
        self.label = f"piper:{narrator_voice}" + ("" if entity_voice == narrator_voice else f"+{entity_voice}")

    @staticmethod
    def model_path(voice: str) -> Path:
        d, f, _ = PIPER_VOICES[voice]
        return config.MODELS_DIR / d / f

    def _voice(self, voice: str, seed: int):
        import onnxruntime as ort
        from piper import PiperVoice
        from piper.config import PiperConfig
        path = self.model_path(voice)
        cfg = PiperConfig.from_dict(json.loads(Path(str(path) + ".json").read_text(encoding="utf-8")))
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        try:
            import onnx
            if voice not in self._proto:
                self._proto[voice] = onnx.load(str(path))
            m = onnx.ModelProto()
            m.CopyFrom(self._proto[voice])
            for k, node in enumerate(n for n in m.graph.node if n.op_type == "RandomNormalLike"):
                for a in list(node.attribute):
                    if a.name == "seed":
                        node.attribute.remove(a)
                node.attribute.append(onnx.helper.make_attribute("seed", float(seed * 7919 + k)))
            sess = ort.InferenceSession(m.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"])
        except ImportError:  # sans le paquet onnx : rendu non déterministe (signalé)
            print("  [!] paquet 'onnx' absent : synthèse piper non déterministe")
            sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        return PiperVoice(config=cfg, session=sess)

    def synth(self, text: str, kind: str, length_scale: float = 1.0, seed: int = 0,
              voice: str | None = None) -> tuple[np.ndarray, int]:
        from piper import SynthesisConfig
        voice = voice or self.voices["narrator" if kind == "narrator" else "entity"]
        pv = self._voice(voice, seed)
        spk = PIPER_VOICES[voice][2]
        cfg = SynthesisConfig(speaker_id=spk, length_scale=length_scale, **PIPER_NOISE)
        x = np.concatenate([c.audio_float_array for c in pv.synthesize(text, syn_config=cfg)]).astype(np.float32)
        return x, pv.config.sample_rate

    def describe(self, kind: str) -> dict:
        v = self.voices["narrator" if kind == "narrator" else "entity"]
        return dict(voice=v, model=PIPER_VOICES[v][1], speaker=PIPER_VOICES[v][2], **PIPER_NOISE)


def get_backend(name: str):
    """auto : edge si joignable (test réel d'une phrase courte), sinon piper."""
    errors = []
    order = ["edge", "piper"] if name == "auto" else [name]
    for b in order:
        try:
            be = EdgeBackend() if b == "edge" else PiperBackend()
            if b == "edge":
                be.synth("Bonsoir.", "narrator")  # sonde réseau
            return be, errors
        except BackendUnavailable as e:
            errors.append(f"{b}: {e}")
            print(f"  [backend] {b} indisponible -> {e}")
    raise SystemExit("Aucun backend TTS disponible :\n  " + "\n  ".join(errors))


# -----------------------------------------------------------------------------
# ASR de contrôle (sherpa-onnx + whisper small)
# -----------------------------------------------------------------------------
class ASR:
    def __init__(self):
        import sherpa_onnx
        d = config.MODELS_DIR / "sherpa-onnx-whisper-small"
        enc = d / "small-encoder.int8.onnx"
        dec = d / "small-decoder.int8.onnx"
        if not enc.exists():
            enc, dec = d / "small-encoder.onnx", d / "small-decoder.onnx"
        if not enc.exists():
            raise FileNotFoundError(f"modèle whisper introuvable dans {d}")
        self.rec = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(enc), decoder=str(dec), tokens=str(d / "small-tokens.txt"),
            language="fr", task="transcribe", num_threads=2)

        self.cache_path = CACHE / "asr_cache.json"
        try:
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.cache = {}

    def __call__(self, x: np.ndarray, sr: int) -> str:
        y = vf.resample(vf.mono(x), sr, 16000)
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.5
        y = np.concatenate([np.zeros(8000, np.float32), y.astype(np.float32), np.zeros(8000, np.float32)])
        key = hashlib.sha1(np.round(y, 4).astype(np.float32).tobytes()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        s = self.rec.create_stream()
        s.accept_waveform(16000, y)
        self.rec.decode_stream(s)
        text = s.result.text.strip()
        self.cache[key] = text
        return text

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")


_UNITS = "zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize".split()
_TENS = {2: "vingt", 3: "trente", 4: "quarante", 5: "cinquante", 6: "soixante"}


def _num_fr(n: int) -> str:
    if n < 17:
        return _UNITS[n]
    if n < 20:
        return "dix " + _UNITS[n - 10]
    if n < 70:
        t, u = divmod(n, 10)
        return _TENS[t] + ("" if u == 0 else (" et un" if u == 1 else " " + _UNITS[u]))
    if n < 80:
        return "soixante " + ("et onze" if n == 71 else _num_fr(n - 60))
    if n < 100:
        return "quatre vingt" + ("" if n == 80 else " " + _num_fr(n - 80))
    if n < 1000:
        c, r = divmod(n, 100)
        return ("cent" if c == 1 else _UNITS[c] + " cent") + ("" if r == 0 else " " + _num_fr(r))
    m, r = divmod(n, 1000)
    return ("mille" if m == 1 else _num_fr(m) + " mille") + ("" if r == 0 else " " + _num_fr(r))


# Variantes acceptées par le WER : prénom inconnu de Whisper transcrit phonétiquement, graphie TTS.
ALIASES = {"dogane": "dogan", "doggan": "dogan", "dogen": "dogan", "doguen": "dogan", "dugan": "dogan",
           "escapegame": "escape game", "eskeipe": "escape", "gueime": "game", "ils": "il", "arsen": "arsene"}


def _homophone_key(w: str) -> str:
    """Neutralise les terminaisons muettes (riait/riaient, quitté/quittées, sont/son…) : WER « phonétique »."""
    w = re.sub(r"(aient|ait|ais)$", "ai", w)
    w = re.sub(r"(ees|es|ee)$", "e", w)
    if len(w) > 3:
        w = re.sub(r"[sx]$", "", w)
    return w


def norm_words(s: str) -> list[str]:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[’'`\-_]", " ", s)
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s)
    s = re.sub(r"([a-z])(\d)", r"\1 \2", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    out = []
    for w in s.split():
        if w.isdigit():
            out += _num_fr(int(w)).split()
        elif w == "h":
            out.append("heures")
        elif w == "heure":
            out.append("heures")
        else:
            out += ALIASES.get(w, w).split()
    return [_homophone_key(w) for w in out]


def wer(ref: str, hyp: str) -> float:
    r, h = norm_words(ref), norm_words(hyp)
    d = np.arange(len(h) + 1, dtype=float)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[len(h)] / max(1, len(r)))


# -----------------------------------------------------------------------------
# Fenêtres disponibles dans edit/edl_spec.py
# -----------------------------------------------------------------------------
def _num_or_fallback(v, decisions):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        if "fallback" in v and isinstance(v["fallback"], (int, float)):
            return float(v["fallback"])
        return None
    if isinstance(v, str) and v.startswith("decision:"):
        key = v.split(":", 1)[1]
        off = 0.0
        m = re.match(r"^(.*?)([+-]\d+(?:\.\d+)?)$", key)
        if m:
            key, off = m.group(1), float(m.group(2))
        cur = decisions
        for part in key.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        val = _num_or_fallback(cur, decisions)
        return None if val is None else val + off
    return None


def _shot_dur(shot: dict, decisions: dict) -> tuple[float, bool]:
    if isinstance(shot.get("dur"), (int, float)):
        return float(shot["dur"]), True
    a = _num_or_fallback(shot.get("src_in"), decisions)
    b = _num_or_fallback(shot.get("src_out"), decisions)
    sp = shot.get("speed", 1.0)
    sp = sp if isinstance(sp, (int, float)) else 1.0
    if a is not None and b is not None and b > a:
        return (b - a) / sp, False
    return 2.0, False


def voice_windows(spec_path: Path | None = None, decisions_path: Path | None = None) -> dict:
    """Pour chaque ID de voix : durée de parole maximale compatible avec le montage.

    Fenêtre = du début de parole au début de la voix suivante (ou fin du film, ou fin d'une shot
    `silence=True`), moins une marge. Les voix enchaînées (`after:ID:gap`) partagent la fenêtre.
    """
    spec_path = spec_path or config.EDIT_DIR / "edl_spec.py"
    decisions_path = decisions_path or config.EDIT_DIR / "decisions.json"
    try:
        spec_src = importlib.util.spec_from_file_location("_edl_spec_for_voices", spec_path)
        spec = importlib.util.module_from_spec(spec_src)
        spec_src.loader.exec_module(spec)
        decisions = json.loads(Path(decisions_path).read_text(encoding="utf-8"))
    except Exception as e:  # spec absente ou invalide : pas de contrainte
        print(f"  [fenêtres] edl_spec illisible ({e}) : aucune contrainte de durée")
        return {}
    t, events = 0.0, []
    for shot in spec.SHOTS:
        dur, exact = _shot_dur(shot, decisions)
        for v in shot.get("voice", []) or []:
            vid = v["id"]
            if isinstance(vid, str) and vid.startswith("decision:"):
                cur = decisions
                for part in vid.split(":", 1)[1].split("."):
                    cur = cur.get(part) if isinstance(cur, dict) else None
                vid = cur
            if vid not in BY_ID:
                continue
            ev = dict(id=vid, shot=shot["id"], shot_start=t, shot_end=t + dur, silence=bool(shot.get("silence")),
                      exact=exact)
            tv = v.get("t", 0.0)
            if isinstance(tv, (int, float)):
                ev["t"] = t + float(tv)
            elif isinstance(tv, str) and tv.startswith("after:"):
                _, ref, gap = tv.split(":")
                ev.update(after=ref, gap=float(gap))
            else:
                continue
            events.append(ev)
        t += dur
    film_end = t
    heads = [e for e in events if "t" in e]
    groups = []
    for e in heads:
        chain, cur = [e], e
        while True:
            nxt = next((x for x in events if x.get("after") == cur["id"] and x["shot"] == cur["shot"]), None)
            if nxt is None:
                break
            chain.append(nxt)
            cur = nxt
        later = [h["t"] for h in heads if h["t"] > e["t"] + 1e-6]
        end = min(later) if later else film_end
        if any(c["silence"] for c in chain):
            end = min(end, max(c["shot_end"] for c in chain))
        soft = SOFT_END.get(chain[-1]["id"])
        if soft:
            last = chain[-1]
            end = min(end, last["shot_end"] + soft[1] if soft[0] == "end" else last["shot_start"] + soft[1])
        groups.append(dict(ids=[c["id"] for c in chain], gaps=[c.get("gap", 0.0) for c in chain[1:]],
                           start=round(e["t"], 3), window=round(end - e["t"], 3), shot=e["shot"]))
    return dict(groups=groups, film_end=round(film_end, 3))


def margin_for(vid: str) -> float:
    if vid.startswith("E_") and BY_ID[vid].get("preset") == "name":
        return -0.15  # flashs de ~0,6 s : le murmure peut mordre 0,15 s sur le flash suivant (moins compressé)
    if vid.startswith("E_"):
        return 0.08
    return 0.15


# Fins « souples » tirées de l'intention de la spec (relatives à la shot de la dernière voix du groupe) :
# ("end", d) = fin de shot + d ; ("at", t) = t secondes après le début de la shot.
SOFT_END = {
    "N03": ("end", 0.30),    # avant les arrêts sur image des prénoms
    "N05": ("end", -0.30),   # reste dans le noir de S3c (impact calé sur la fin de la voix)
    "N06": ("end", 0.50),
    "N08b": ("at", 4.45),    # avant le braam de S4m (4,55 s)
    "N09b": ("end", 0.40),
}


def group_limits(windows: dict, natural: dict[str, float]) -> dict[str, float]:
    """Durée de parole max par ID, à partir des durées naturelles mesurées.

    Dans un groupe (voix enchaînées par `after:`), toutes les voix sont resserrées du même facteur
    pour que parole + pauses tiennent dans la fenêtre. Une voix utilisée plusieurs fois prend le minimum.
    """
    out: dict[str, float] = {}
    for g in windows.get("groups", []):
        if not all(i in natural for i in g["ids"]):
            continue
        avail = g["window"] - sum(g["gaps"]) - margin_for(g["ids"][-1])
        total = sum(natural[i] for i in g["ids"])
        f = min(1.0, avail / max(total, 1e-3))
        for i in g["ids"]:
            out[i] = round(min(out.get(i, 1e9), natural[i] * f if f < 1.0 else avail - (total - natural[i])), 3)
    return out


# -----------------------------------------------------------------------------
# Mesures prosodiques (optionnelles : parselmouth)
# -----------------------------------------------------------------------------
def prosody(x: np.ndarray, sr: int) -> dict:
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return {}
    snd = parselmouth.Sound(vf.mono(x).astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=320)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if len(f0) < 10:
        return {}
    st = 12 * np.log2(f0 / np.median(f0))
    harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=60)
    hv = harm.values[harm.values > -200]
    pp = call(snd, "To PointProcess (periodic, cc)", 60, 320)
    jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    return dict(f0_med=round(float(np.median(f0)), 1), f0_st_std=round(float(np.std(st)), 2),
                f0_st_range=round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2),
                hnr_db=round(float(np.mean(hv)), 2), jitter_pct=round(100 * float(jit), 2))


def dnsmos(x: np.ndarray, sr: int) -> dict:
    """DNSMOS P.835 (SIG/BAK/OVRL, modèle ONNX du paquet `speechmos`) : estimation de qualité perçue, optionnelle."""
    try:
        import os
        import onnxruntime as ort
        import speechmos
    except ImportError:
        return {}
    path = os.path.join(os.path.dirname(speechmos.__file__), "dnsmos_models", "sig_bak_ovr.onnx")
    if not os.path.exists(path):
        return {}
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    a = vf.resample(vf.mono(x), sr, 16000).astype(np.float32)
    L = int(9.01 * 16000)
    while len(a) < L:
        a = np.concatenate([a, a])
    polys = [np.poly1d([-0.08397278, 1.22083953, 0.0052439]), np.poly1d([-0.13166888, 1.60915514, -0.39604546]),
             np.poly1d([-0.06766283, 1.11546468, 0.04602535])]
    res = []
    for i in range(0, len(a) - L + 1, 16000):
        o = sess.run(None, {sess.get_inputs()[0].name: a[i:i + L][None, :]})[0][0]
        res.append([p(v) for p, v in zip(polys, o[:3])])
    m = np.mean(res, axis=0)
    return dict(dnsmos_sig=round(float(m[0]), 2), dnsmos_bak=round(float(m[1]), 2), dnsmos_ovrl=round(float(m[2]), 2))


def syllables(text: str) -> int:
    return len(re.findall(r"[aeiouyàâéèêëîïôûùœ]+", text.lower().replace("ent ", " ").replace("es ", " ")))


# -----------------------------------------------------------------------------
# Synthèse d'une ligne : prises multiples (piper), ajustement à la fenêtre
# -----------------------------------------------------------------------------
def _hash(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def cached_tts(backend, text, kind, **kw) -> tuple[np.ndarray, int]:
    key = _hash(backend.label, text, kind, kw, PIPER_NOISE if backend.name == "piper" else EDGE_RATE, "v2")
    f = CACHE / "tts" / f"{key}.wav"
    if f.exists():
        x, sr = sf.read(str(f), dtype="float32")
        return x, sr
    x, sr = backend.synth(text, kind, **kw)
    f.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(f), x, sr, subtype="FLOAT")
    return x, sr


def raw_speech_dur(x, sr) -> float:
    a, b = vf.speech_bounds(vf.resample(vf.mono(x), sr, vf.SR), vf.SR)
    return b - a


def _render(line: dict, x: np.ndarray, sr: int, tempo: float, fx: dict, chain_seed: int):
    """Applique la chaîne de traitement du rôle ; renvoie (y 48 kHz stéréo, info)."""
    if line["kind"] == "narrator":
        return vf.narrator_chain(x, sr, tempo=tempo, **fx)
    return vf.entity_chain(x, sr, preset=line["preset"], seed=chain_seed, tempo=tempo)


def _asr_speech(asr: ASR, y: np.ndarray, info: dict) -> str:
    """ASR sur la partie parlée (sans le pré-swell / l'inspiration, que Whisper prend pour des mots)."""
    i0 = max(0, int((info["onset_s"] - 0.05) * vf.SR))
    return asr(y[i0:], vf.SR)


def make_take(backend, line: dict, takes: int, asr: ASR | None, log: list, max_sd: float | None = None) -> dict:
    """Synthétise plusieurs prises, les traite, et garde la meilleure.

    Candidats par graine : débit préféré (compressé au besoin par rubberband, ≤ TEMPO_MAX) et, s'il faut
    tenir une fenêtre plus courte, débit TTS accéléré (piper length_scale / edge rate). Score : WER de
    l'ASR sur la voix TRAITÉE, dépassement de fenêtre, prosodie (narrateur).
    """
    kind = line["kind"]
    role = "narrator" if kind == "narrator" else line["preset"]
    fx = narrator_fx(backend) if kind == "narrator" else {}
    chain_seed = zlib.crc32(line["id"].encode()) % 1000
    seed0 = zlib.crc32(line["id"].encode()) % 100000
    n_takes = takes if backend.name == "piper" else 1
    cands = []
    for k in range(n_takes):
        seed = seed0 + 1000 * k
        variants = []
        if backend.name == "piper":
            ls = PIPER_LS[role]
            x, sr = cached_tts(backend, line["tts"], kind, length_scale=ls, seed=seed)
            variants.append((x, sr, dict(length_scale=ls, seed=seed)))
            sd = raw_speech_dur(x, sr)
            if max_sd and sd > max_sd * 1.01:
                ls2 = round(max(PIPER_LS_MIN[role], ls * max_sd / sd * 0.98), 3)
                if ls2 < ls - 0.01:
                    x2, sr2 = cached_tts(backend, line["tts"], kind, length_scale=ls2, seed=seed)
                    variants.append((x2, sr2, dict(length_scale=ls2, seed=seed)))
        else:
            x, sr = cached_tts(backend, line["tts"], kind, speed=1.0)
            variants.append((x, sr, dict(speed=1.0)))
            sd = raw_speech_dur(x, sr)
            if max_sd and sd > max_sd * 1.01:
                speed = round(min(1.18, sd / max_sd * 1.02), 3)
                x2, sr2 = cached_tts(backend, line["tts"], kind, speed=speed)
                variants.append((x2, sr2, dict(speed=speed)))
        for x, sr, params in variants:
            sd = raw_speech_dur(x, sr)
            tempo = 1.0
            if max_sd and sd > max_sd:
                tempo = round(min(TEMPO_MAX[role], sd / max_sd * 1.01), 3)
            y, info = _render(line, x, sr, tempo, fx, chain_seed)
            if max_sd and info["speech_dur"] > max_sd * 1.01 and tempo < TEMPO_MAX[role]:
                tempo = round(min(TEMPO_MAX[role], tempo * info["speech_dur"] / max_sd * 1.01), 3)
                y, info = _render(line, x, sr, tempo, fx, chain_seed)
            hyp = _asr_speech(asr, y, info) if asr else ""
            w = wer(line["text"], hyp) if asr else 0.0
            pr = prosody(y, vf.SR) if kind == "narrator" else {}
            over = max(0.0, info["speech_dur"] - max_sd - 0.06) if max_sd else 0.0  # tolérance 60 ms
            score = 6.0 * w + 10.0 * over + 0.3 * max(0.0, tempo - 1.0)  # chevaucher une voix > un mot mal compris
            if pr:
                score += 0.15 * abs(pr["f0_st_std"] - 2.8) + 0.01 * max(0.0, 10.0 - pr["hnr_db"])
            if asr and kind == "entity":  # chuchotement : la couche « oreille » doit aussi être comprise
                w_ear = wer(line["text"], asr(info["layers"]["ear"], vf.SR))
                score += 3.0 * w_ear
            p_ = dict(params, **({"tempo": tempo} if tempo > 1.0 else {}))
            cands.append(dict(y=y, info=info, params=p_, raw=(x, sr), asr=hyp, wer=round(w, 3), prosody=pr,
                              score=round(score, 4)))
            log.append(f"{line['id']} prise {k}: {p_} parole={info['speech_dur']:.2f}s wer={w:.2f} « {hyp} » "
                       f"score={score:.3f}")
    best = min(cands, key=lambda c: c["score"])
    best["raw_sd"] = round(raw_speech_dur(*best["raw"]), 3)
    return best


# -----------------------------------------------------------------------------
# Spectrogrammes
# -----------------------------------------------------------------------------
def spectrogram_sheet(items: list[tuple[str, Path, dict]], out: Path, title: str) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import signal
    except ImportError:
        return None
    n = len(items)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(16, 2.3 * rows), squeeze=False)
    for ax in axs.flat:
        ax.set_visible(False)
    for ax, (vid, path, info) in zip(axs.flat, items):
        ax.set_visible(True)
        x, sr = sf.read(str(path), dtype="float32")
        m = vf.mono(x)
        f, t, S = signal.spectrogram(m, fs=sr, nperseg=1024, noverlap=896, mode="magnitude")
        S = 20 * np.log10(S + 1e-9)
        S -= S.max()
        ax.pcolormesh(t, f / 1000, S, vmin=-90, vmax=0, cmap="magma", shading="auto")
        ax.set_ylim(0, 14)
        ax.axvline(info["onset_s"], color="cyan", lw=1)
        ax.axvline(info["onset_s"] + info["speech_dur"], color="cyan", lw=1, ls="--")
        env = vf.rms_env(m, sr, 10)
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(m)) / sr, 20 * np.log10(env), color="lime", lw=0.7)
        ax2.set_ylim(-90, 0)
        ax2.tick_params(labelsize=6)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{vid}  onset={info['onset_s']:.3f}s  parole={info['speech_dur']:.2f}s  durée={info['dur']:.2f}s  "
                     f"{info['lufs']:.1f} LUFS  crête {info['peak_db']:.1f} dBTP", fontsize=8)
        ax.set_ylabel("kHz", fontsize=7)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=60)
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# Banc d'essai des voix (piper) : --compare
# -----------------------------------------------------------------------------
def compare_voices(voices: list[str], seeds: int, out_json: Path) -> dict:
    asr = ASR()
    be = PiperBackend(narrator_voice=voices[0])
    res = {}
    narr = [l for l in LINES if l["kind"] == "narrator"]
    for v in voices:
        if not PiperBackend.model_path(v).exists():
            print(f"  [compare] {v} : modèle absent")
            continue
        rows, concat = [], []
        for line in narr:
            for k in range(seeds):
                seed = zlib.crc32(line["id"].encode()) % 100000 + 1000 * k
                x, sr = cached_tts(be, line["tts"], "narrator", length_scale=1.0, seed=seed, voice=v)
                y, info = vf.narrator_chain(x, sr, **NARRATOR_FX.get(v, {}))  # chaîne réglée pour chaque voix
                if k == 0:
                    concat += [vf.mono(y), np.zeros(int(0.1 * vf.SR), np.float32)]
                hyp = asr(y, vf.SR)
                pr = prosody(x, sr)
                rows.append(dict(id=line["id"], seed=seed, wer=wer(line["text"], hyp), asr=hyp,
                                 rate=syllables(line["tts"]) / max(0.3, info["speech_dur"]), **pr))
        agg = {k: round(float(np.mean([r[k] for r in rows if k in r])), 3)
               for k in ("wer", "rate", "f0_med", "f0_st_std", "hnr_db", "jitter_pct")}
        # régularité : écart-type inter-phrases de la hauteur médiane (en demi-tons)
        f0s = [r["f0_med"] for r in rows if "f0_med" in r]
        agg["f0_med_spread_st"] = round(float(np.std(12 * np.log2(np.array(f0s) / np.median(f0s)))), 3) if f0s else None
        agg["sr_model"] = int(json.loads(Path(str(PiperBackend.model_path(v)) + ".json").read_text())["audio"]["sample_rate"])
        agg.update(dnsmos(np.concatenate(concat), vf.SR))
        res[v] = dict(summary=agg, rows=rows)
        print(f"  [compare] {v}: {agg}")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def spelling_test(variants: list[str], seeds: int) -> list[dict]:
    """Teste plusieurs graphies de « escape game » (ASR aller-retour, voix narrateur piper)."""
    asr = ASR()
    be = PiperBackend()
    out = []
    for v in variants:
        hits, hyps = 0, []
        for k in range(seeds):
            text = f"Quatre amis ont pris la route pour un {v}."
            x, sr = cached_tts(be, text, "narrator", length_scale=PIPER_LS["narrator"], seed=500 + k)
            y, _ = vf.narrator_chain(x, sr, **narrator_fx(be))
            hyp = asr(y, vf.SR)
            ok = "escape game" in " ".join(norm_words(hyp))
            hits += ok
            hyps.append(hyp)
        try:
            from piper import PiperVoice
            ph = "".join("".join(s) for s in PiperVoice.load(PiperBackend.model_path(PIPER_NARRATOR)).phonemize(v))
        except Exception:
            ph = ""
        out.append(dict(spelling=v, phonemes=ph, asr_hits=f"{hits}/{seeds}", asr=hyps))
        print(f"  [graphie] {v!r:22} {ph:22} {hits}/{seeds}  {hyps[0]}")
    return out


# -----------------------------------------------------------------------------
# Rapport
# -----------------------------------------------------------------------------
def write_report(manifest: dict, qc: dict, windows: dict, maxsd: dict, compare: dict | None,
                 spellings: list | None, backend_errors: list[str], out: Path) -> None:
    L = []
    L.append("# Voix off — rapport de génération\n")
    L.append(f"Généré par `python tools/make_voices.py` le {time.strftime('%Y-%m-%d %H:%M')} — backend : "
             f"**{manifest['backend']}**.\n")
    if backend_errors:
        L.append("Backends écartés automatiquement : " + " ; ".join(f"`{e}`" for e in backend_errors) + "\n")
    L.append("## Voix choisie et pourquoi\n")
    L.append(REPORT_RATIONALE.strip() + "\n")
    if compare:
        L.append("### Banc d'essai (12 phrases du narrateur × prises, même chaîne de traitement)\n")
        L.append("| voix | WER ASR ↓ | DNSMOS OVRL ↑ (SIG / BAK) | syll./s | F0 médiane (Hz) | variation F0 (σ, demi-tons) | dispersion F0 entre phrases (σ, dt) | HNR (dB) ↑ | jitter % ↓ | Fe modèle |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for v, d in compare.items():
            s = d["summary"]
            mos = (f"{s['dnsmos_ovrl']:.2f} ({s['dnsmos_sig']:.2f} / {s['dnsmos_bak']:.2f})" if "dnsmos_ovrl" in s else "")
            L.append(f"| {v} | {s['wer']:.3f} | {mos} | {s['rate']:.2f} | {s['f0_med']:.0f} | {s['f0_st_std']:.2f} | "
                     f"{s['f0_med_spread_st']:.2f} | {s['hnr_db']:.1f} | {s['jitter_pct']:.2f} | {s['sr_model']} |")
        L.append("\nWER : taux d'erreur de mots de Whisper small (sherpa-onnx) sur la voix traitée, après normalisation "
                 "(nombres, homophones muets, prénom inconnu de Whisper). DNSMOS : qualité perçue estimée (Microsoft "
                 "P.835) sur les 12 phrases traitées enchaînées.\n")
        L.append("")
    if spellings:
        L.append("### Graphies testées pour « escape game » (ASR aller-retour, voix retenue)\n")
        L.append("| graphie TTS | phonèmes (espeak) | reconnu « escape game » | exemple ASR |")
        L.append("|---|---|---|---|")
        for s in spellings:
            L.append(f"| `{s['spelling']}` | `{s['phonemes']}` | {s['asr_hits']} | {s['asr'][0]} |")
        ph = next((s_["phonemes"] for s_ in spellings if s_["spelling"] == ESCAPE_GAME_TTS), "")
        L.append(f"\nGraphie retenue : `{ESCAPE_GAME_TTS}` → `{ph}` : la prononciation « à l'anglaise avec l'accent "
                 "français » courante en France ; [ɛskɛp ɡɛm] (sans diphtongue) est souvent entendu « escape d'AIM » "
                 "par l'ASR, et la graphie brute fait lire « escape » [ɛskap] à espeak.\n")
    L.append("## Chaînes de traitement\n")
    L.append(CHAIN_DOC.strip() + "\n")
    L.append("## Fichiers\n")
    L.append("| ID | type | texte affiché | texte TTS | onset (s) | parole (s) | durée (s) | fenêtre parole max (s) | LUFS | crête (dBTP) | réglage | ASR (fichier final) | WER |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for vid, it in manifest["items"].items():
        q = qc.get(vid, {})
        lim = maxsd.get(vid)
        flag = "" if lim is None or it["speech_dur"] <= lim + 0.02 else " ⚠"
        L.append(f"| {vid} | {it['kind']} | {it['text']} | {it['tts_text']} | {it['onset_s']:.3f} | {it['speech_dur']:.2f}{flag} | "
                 f"{it['dur']:.2f} | {'' if lim is None else f'{lim:.2f}'} | {q.get('lufs', '')} | {q.get('peak_db', '')} | "
                 f"{q.get('params', '')} | {q.get('asr_final', '')} | {q.get('wer_final', '')} |")
    L.append("")
    ent = [v for v in manifest["items"] if v.startswith("E_")]
    if ent:
        L.append("### Entité : ASR par couche\n")
        L.append("| ID | ASR couche « oreille » (chuchotée, −1,5 à −2,5 dt, formants préservés) | ASR mix final | ASR TTS brut |")
        L.append("|---|---|---|---|")
        for vid in ent:
            q = qc.get(vid, {})
            L.append(f"| {vid} | {q.get('asr_ear', '')} | {q.get('asr_final', '')} | {q.get('asr_raw', '')} |")
        L.append("")
    if windows.get("groups"):
        L.append("## Fenêtres du montage (edit/edl_spec.py)\n")
        L.append("Fenêtre = du début de parole jusqu'à la voix suivante (ou fin de shot `silence`, ou fin du film).\n")
        L.append("| shot | voix | début abs. approx. (s) | fenêtre (s) | parole + pauses livrées (s) | marge (s) | "
                 "au débit préféré (s) | il manquerait (s) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for g in windows["groups"]:
            used = sum(manifest["items"][i]["speech_dur"] for i in g["ids"] if i in manifest["items"]) + sum(g["gaps"])
            pref = sum(qc.get(i, {}).get("natural_speech", 0.0) for i in g["ids"]) + sum(g["gaps"])
            need = pref + margin_for(g["ids"][-1]) - g["window"]
            L.append(f"| {g['shot']} | {' + '.join(g['ids'])} | {g['start']:.2f} | {g['window']:.2f} | {used:.2f} | "
                     f"{g['window'] - used:+.2f} | {pref:.2f} | {'' if need <= 0.02 else f'{need:.2f}'} |")
        L.append("\n« il manquerait » : allongement de la fenêtre (shot ou position de la voix suivante) qui permettrait "
                 "de garder le débit lent préféré sans compression ; relancer alors `tools/make_voices.py`, qui "
                 "s'adapte automatiquement à `edit/edl_spec.py`.")
        L.append(f"\nDurée totale estimée de la spec : {windows['film_end']:.1f} s (approximative : ancres non résolues).\n")
    L.append("## Spectrogrammes\n")
    L.append("`spectrograms/narrator.png`, `spectrograms/entity.png` (traits cyan : début et fin de parole mesurés).\n")
    L.append(REPORT_LIMITS.strip() + "\n")
    out.write_text("\n".join(L), encoding="utf-8")


REPORT_RATIONALE = """
* **Premier choix automatique : edge-tts `fr-FR-HenriNeural`** (rate −12 %, pitch −8 Hz), voix neurale masculine
  grave et posée. Dans ce conteneur le service est bloqué (`speech.platform.bing.com`) ; sur le PC du client,
  `python tools/make_voices.py` l'utilise automatiquement (sonde réseau réelle, repli piper sinon).
* **Narrateur hors ligne : piper `fr_FR-tom-medium`** (44,1 kHz). Banc d'essai ci-dessous (12 phrases × 3 prises,
  chaîne réglée pour chaque voix) : de loin la meilleure articulation (WER ≈ 2,4 × plus bas que pierre), prosodie
  posée et régulière (σ F0 ≈ 2,2 demi-tons dans la phrase, hauteur médiane la plus stable d'une phrase à l'autre),
  pleine bande. Son défaut — une voix très soufflée (HNR ≈ 5 dB, aigus 4–9 kHz très présents) — est dompté par la
  chaîne (étagère −6 dB au-dessus de 4,2 kHz, de-esser ferme) ; en voix off d'horreur ce grain voilé, proche,
  sert le propos.
* `upmc-pierre` a le timbre le plus propre (meilleur DNSMOS, HNR ≈ 10 dB, prosodie plus mobile) mais des défauts
  d'articulation systématiques qu'un francophone entend : [ʁ] mal réalisé (« Ils riaient » compris « Il viait »,
  « derrière eux » → « d'ailleurs peu »), « s'est mis » → « semis », et une bande limitée à ~10 kHz.
  `gilles-low` (16 kHz) : le plus grave mais grain de vocodeur, nasales absentes du modèle, pire WER, hauteur
  irrégulière d'une phrase à l'autre. `jessica` : féminine (référence).
* **Sélection de prises** : le bruit interne de VITS est à **graine explicite** (attribut `seed` des nœuds ONNX
  `RandomNormalLike`, rendu reproductible) ; 6 prises par phrase, chacune traitée puis réécoutée par l'ASR ; la
  meilleure (intelligibilité, prosodie, tenue de la fenêtre du montage) est gardée.
* **Entité : base tom** également — après chuchotement seule l'enveloppe spectrale (l'articulation) compte, et tom
  prononce nettement mieux les prénoms isolés (« Gabriel » reconnu 3/3 contre 0/3 pour pierre).
* Modèles sherpa-onnx (supertonic, kokoro multi-lang, matcha, zipvoice) : **non évalués** — l'accès au dépôt
  GitHub `k2-fsa/sherpa-onnx` (API et releases) est refusé dans cette session (proxy : « GitHub access to this
  repository is not enabled »). D'après leur documentation : kokoro multi-lang n'a qu'une voix française féminine
  (ff_siwis), matcha et zipvoice sont zh/en ; supertonic 2 annonce le français : à tester sur le PC du client.
"""

CHAIN_DOC = """
**Narrateur** (`pipeline/audio/voicefx.narrator_chain`) : TTS → 48 kHz → transposition **avec préservation des
formants** (rubberband R3 `-F`) : −2,5 demi-tons pour tom (−1 pour edge-Henri, déjà abaissé de 8 Hz) → EQ (passe-haut
65 Hz, chaleur +3 dB à 150 Hz, −2 dB à 320 Hz, +1 dB à 2,4 kHz, étagère −6 dB au-dessus de 4,2 kHz pour le souffle
de tom ; exciteur discret au-dessus de 9 kHz pour les voix à bande étroite) → de-esser (bande > 4,5 kHz) →
compression douce (2,4:1, genou 8 dB, 12/140 ms) → réverbe type plate synthétique 1,2 s, pré-délai 14 ms, 12 % wet →
−20 LUFS intégrés, crête vraie ≤ −3 dBTP (limiteur à anticipation suréchantillonné ×4) → tête 60 ms avant la parole,
queue de réverbe ≤ 0,55 s fondue. Débit : piper `length_scale` 1,10 (plus lent que nature) ; si la fenêtre du montage
l'exige, compression temporelle rubberband ≤ +10 % (formants préservés) ou débit TTS accéléré, la meilleure prise
gagne.

**Entité** (`voicefx.entity_chain`) : TTS → 48 kHz → transposition −3,5 à −5,5 dt **formants vers le bas**
(rubberband sans `-F`) → **chuchotement** : dans les trames voisées, enveloppe spectrale (liftrage cepstral 2,2 ms)
× bruit blanc de phase aléatoire, énergie de trame conservée ; trames non voisées (fricatives, explosions) gardées
telles quelles → consonnes nettes → couches désaccordées (0 / +15…+22 / −22…−27 cents, décalages 8–19 ms,
panoramiques ±0,3–0,5) + couche sous-grave (−7 à −9 dt, passe-bas 2,5 kHz) + **couche sèche « collée à
l'oreille »** (−1 à −2 dt, dominante pour E_IDENT et E_CQL, décalée à gauche avec retard interaural 0,35 ms) → EQ de
proximité (+4,5 dB à 150 Hz, souffle +3 dB à 4 kHz, −3 dB à 7,5 kHz) → de-esser, compression → **pré-swell en
réverbe inversée** `reverse(reverb(reverse(x)))` (IR sombre synthétique) qui monte jusqu'au mot (prénoms 0,6 s,
E_IDENT 1,0 s ; E_CQL : aucun, remplacé par une inspiration très proche) → queue sombre courte → −20 LUFS, crête
≤ −3 dBTP. `onset_s` = début réel de la parole (après pré-swell / inspiration).
"""

REPORT_LIMITS = """
## Limites connues
* Les durées de parole sont contraintes par les fenêtres de `edit/edl_spec.py` (voir tableau) : le débit
  « lent » est conservé partout où la fenêtre le permet ; les phrases N01 et N07a+N07b sont au débit naturel.
  Allonger S2b/S2c (≈ +0,4 s) et S4h (≈ +0,5 s) permettrait un débit plus lent.
* Les prénoms de l'entité doivent tenir dans les flashs de 0,55 s (S4i–S4l) : murmures courts ; leurs queues
  et pré-swells débordent volontairement sur les flashs voisins.
* Voix piper = repli hors ligne ; sur le PC du client, relancer `python tools/make_voices.py` produit la version
  edge-tts (même format, même manifeste), puis relancer `edit/build_edl.py`.
"""


def synthesize_all(backend, ids: list[str], windows: dict, n_takes: int, asr, log: list):
    """Durées naturelles -> limites par groupe -> meilleures prises traitées. Renvoie (natural, maxsd, takes)."""
    # 1) durées naturelles (prise de référence au débit préféré) de toutes les voix des groupes concernés
    need = set(ids)
    for g in windows.get("groups", []):
        if need & set(g["ids"]):
            need |= set(g["ids"])
    natural = {}
    for vid in [l["id"] for l in LINES if l["id"] in need]:
        line = BY_ID[vid]
        role = "narrator" if line["kind"] == "narrator" else line["preset"]
        if backend.name == "piper":
            x, sr = cached_tts(backend, line["tts"], line["kind"], length_scale=PIPER_LS[role],
                               seed=zlib.crc32(vid.encode()) % 100000)
        else:
            x, sr = cached_tts(backend, line["tts"], line["kind"], speed=1.0)
        natural[vid] = raw_speech_dur(x, sr)
    # 2) limites par groupe ; 3) prises (traitées, évaluées) qui tiennent la fenêtre
    maxsd = group_limits(windows, natural)
    takes = {}
    for vid in [l["id"] for l in LINES if l["id"] in ids]:
        takes[vid] = make_take(backend, BY_ID[vid], n_takes, asr, log, max_sd=maxsd.get(vid))
        if asr:
            asr.save()
        # allocation séquentielle : la voix suivante d'un groupe enchaîné récupère (ou rend) l'écart réel
        for g in windows.get("groups", []):
            if vid in g["ids"] and g["ids"].index(vid) < len(g["ids"]) - 1:
                k = g["ids"].index(vid)
                nxt = g["ids"][k + 1]
                avail = g["window"] - sum(g["gaps"]) - margin_for(g["ids"][-1])
                used = sum(takes[i]["info"]["speech_dur"] if i in takes else natural.get(i, 0.0) for i in g["ids"][:k + 1])
                later = sum(natural.get(i, 0.0) for i in g["ids"][k + 2:])
                maxsd[nxt] = round(max(0.3, min(natural.get(nxt, 9.0), avail - used - later)), 3)
    return natural, maxsd, takes


def publish_report(out_dir: Path) -> None:
    """(Re)construit spectrogrammes + REPORT.md à partir de manifest.json, qc.json, comparison.json, spellings.json."""
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    qc_doc = json.loads((out_dir / "qc.json").read_text(encoding="utf-8")) if (out_dir / "qc.json").exists() else {}
    q = qc_doc.get("items", {})
    sheets = {"narrator": [], "entity": []}
    for v, it in manifest["items"].items():
        p = out_dir / it["file"]
        d = {**it, **q.get(v, {})}
        if "lufs" not in d or "peak_db" not in d:
            y, _ = sf.read(str(p), dtype="float32")
            d["lufs"], d["peak_db"] = round(vf.lufs(y), 2), round(vf.true_peak_db(y), 2)
        sheets[it["kind"]].append((v, p, d))
    if sheets["narrator"]:
        spectrogram_sheet(sheets["narrator"], out_dir / "spectrograms" / "narrator.png", f"Narrateur — {manifest['backend']}")
    if sheets["entity"]:
        spectrogram_sheet(sheets["entity"], out_dir / "spectrograms" / "entity.png", f"Entité — {manifest['backend']}")
    load = lambda n: json.loads((out_dir / n).read_text(encoding="utf-8")) if (out_dir / n).exists() else None  # noqa: E731
    windows = qc_doc.get("windows") or voice_windows()
    write_report(manifest, q, windows, qc_doc.get("max_speech", {}), load("comparison.json"), load("spellings.json"),
                 qc_doc.get("backend_errors", []), out_dir / "REPORT.md")


# -----------------------------------------------------------------------------
# Programme principal
# -----------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="auto", choices=["auto", "edge", "piper"])
    ap.add_argument("--only", default="", help="IDs séparés par des virgules (fusionnés dans le manifeste existant)")
    ap.add_argument("--takes", type=int, default=6, help="prises par phrase (piper)")
    ap.add_argument("--no-asr", action="store_true", help="sans ASR de contrôle (plus rapide)")
    ap.add_argument("--compare", action="store_true", help="banc d'essai des voix piper (comparison.json)")
    ap.add_argument("--spellings", action="store_true", help="teste les graphies de « escape game » (spellings.json)")
    ap.add_argument("--report", action="store_true", help="reconstruit seulement REPORT.md et les spectrogrammes")
    ap.add_argument("--out", default=str(VOICE_DIR))
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    if args.compare:
        compare_voices(["upmc-pierre", "tom", "gilles", "upmc-jessica"], 3, out_dir / "comparison.json")
        return 0
    if args.report:
        publish_report(out_dir)
        return 0
    if args.spellings:
        res = spelling_test(["escape game", "escape guème", "eskèpe guème", "èscaipe gaime", "esquèpe guéme",
                             "eskeïpe gueïme"], 4)
        (out_dir / "spellings.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    t0 = time.time()
    backend, backend_errors = get_backend(args.backend)
    print(f"Backend : {backend.label}")
    asr = None
    if not args.no_asr:
        try:
            asr = ASR()
        except Exception as e:
            print(f"  [asr] indisponible : {e}")
    windows = voice_windows()
    ids = [s for s in args.only.split(",") if s] or [l["id"] for l in LINES]
    unknown = [i for i in ids if i not in BY_ID]
    if unknown:
        raise SystemExit(f"IDs inconnus : {unknown}")
    log: list[str] = []
    try:
        natural, maxsd, takes = synthesize_all(backend, ids, windows, args.takes, asr, log)
    except BackendUnavailable as e:  # edge coupé en cours de route : tout refaire en piper (voix homogène)
        if backend.name != "edge" or args.backend == "edge":
            raise
        backend_errors.append(f"edge: {e}")
        print(f"  [backend] edge interrompu ({e}) -> piper")
        backend = PiperBackend()
        log.clear()
        natural, maxsd, takes = synthesize_all(backend, ids, windows, args.takes, asr, log)

    staging = CACHE / f"staging_{backend.name}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    items, qc = {}, {}
    for vid in ids:
        line = BY_ID[vid]
        lim = maxsd.get(vid)
        take = takes[vid]
        y, info = take["y"], dict(take["info"])
        layers = info.pop("layers", None)
        wav = staging / f"{vid}.wav"
        vf.write_wav(wav, y)
        items[vid] = dict(file=f"{vid}.wav", text=line["text"], tts_text=line["tts"], onset_s=info["onset_s"],
                          speech_dur=info["speech_dur"], dur=info["dur"], kind=line["kind"])
        xr, srr = take["raw"]
        q = dict(lufs=info["lufs"], peak_db=info["peak_db"], params=take["params"], prosody=take["prosody"],
                 max_speech=lim, natural_speech=round(natural.get(vid, 0.0), 3))
        if asr:
            yy, _ = sf.read(str(wav), dtype="float32")  # relu depuis le WAV 24 bits écrit
            q["asr_raw"] = asr(xr, srr)
            q["asr_final"] = _asr_speech(asr, yy, info)
            q["wer_final"] = round(wer(line["text"], q["asr_final"]), 3)
            if line["kind"] == "entity":
                q["asr_final_full"] = asr(yy, vf.SR)
            if layers is not None:
                q["asr_ear"] = asr(layers["ear"], vf.SR)
                q["wer_ear"] = round(wer(line["text"], q["asr_ear"]), 3)
                sf.write(str(CACHE / f"{vid}_ear.wav"), layers["ear"], vf.SR)
        qc[vid] = q
        print(f"{vid:9} onset={info['onset_s']:.3f} parole={info['speech_dur']:.2f}s (max {lim}) durée={info['dur']:.2f}s "
              f"{info['lufs']:.1f} LUFS {info['peak_db']:.1f} dBTP | ASR: {q.get('asr_final', '-')}"
              + (f" | oreille: {q.get('asr_ear')}" if 'asr_ear' in q else ""))

    # fusion avec le manifeste existant (--only) puis publication atomique dans assets/audio/voice
    man_path = out_dir / "manifest.json"
    manifest = {"backend": backend.label, "items": {}}
    if args.only and man_path.exists():
        old = json.loads(man_path.read_text(encoding="utf-8"))
        if old.get("backend") != backend.label:
            print(f"  [!] backend différent du manifeste existant ({old.get('backend')}) : fusion quand même")
        manifest["items"] = old.get("items", {})
    for vid in ids:
        shutil.copy2(staging / f"{vid}.wav", out_dir / f"{vid}.wav")
        manifest["items"][vid] = items[vid]
    manifest["items"] = {l["id"]: manifest["items"][l["id"]] for l in LINES if l["id"] in manifest["items"]}
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    qc_path = out_dir / "qc.json"
    old_doc = json.loads(qc_path.read_text(encoding="utf-8")) if (args.only and qc_path.exists()) else {}
    all_items = {**old_doc.get("items", {}), **qc}
    qc_doc = dict(backend=backend.label, backend_errors=backend_errors,
                  backend_settings={k: backend.describe(k) for k in ("narrator", "entity")},
                  windows=windows, max_speech={**old_doc.get("max_speech", {}), **maxsd}, items=all_items,
                  takes_log=old_doc.get("takes_log", []) + log if args.only else log)
    qc_path.write_text(json.dumps(qc_doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    if asr:
        asr.save()
    publish_report(out_dir)
    print(f"OK : {len(ids)} voix -> {out_dir} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
