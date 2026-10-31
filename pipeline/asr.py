"""Transcription française mot à mot (horodatage des mots), plusieurs moteurs.

Ordre des moteurs (le premier disponible gagne) :
  1. faster-whisper (large-v3, sinon medium ; `word_timestamps=True`) — téléchargé depuis Hugging Face ;
  2. openai-whisper (large-v3 / medium) ;
  3. sherpa-onnx :
       a. NeMo Parakeet TDT 0.6B v3 (multilingue dont le français, transducteur hors ligne, horodatage des
          jetons → mots) — `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` ;
       b. Zipformer streaming français (horodatage des jetons, moins précis) ;
       c. Whisper small (sherpa-onnx 2.0 ne renvoie AUCUN horodatage pour Whisper : vérifié ; mots placés
          au prorata dans les zones de parole détectées, moteur marqué « approx »).
     Les modèles sherpa-onnx manquants sont téléchargés depuis les releases GitHub de k2-fsa/sherpa-onnx
     (désactivable : ESCAPE_ASR_DOWNLOAD=0).

Post-traitements : 1 s de silence ajoutée de part et d'autre (sinon Parakeet saute parfois la 1re phrase) ;
zones de parole (Silero VAD, sinon énergie) restées sans mot → nouveau décodage isolé ; débuts/fins de mots
autour des pauses recalés sur l'énergie (avance/retard d'émission des transducteurs, durées TDT trop longues).

Sortie (contrat) : {"backend","language":"fr","words":[{"w","start","end","p"}],"segments":[{"start","end","text"}]}

API :
  transcribe_audio(x, sr=16000, language="fr", backend=None) -> dict    (x : mono float32)
  transcribe_file(path, language="fr", backend=None, t0=None, t1=None) -> dict   (tout fichier lisible par ffmpeg)
  available_backends() -> list[str]
CLI : python -m pipeline.asr fichier.(wav|mp4|mov) [--backend sherpa-parakeet]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import threading
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config  # noqa: E402

SR = 16000
FFMPEG = os.environ.get("ESCAPE_FFMPEG", "ffmpeg")
GH_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
PARAKEET = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
ZIPFORMER_FR = "sherpa-onnx-streaming-zipformer-fr-2023-04-14"
WHISPER_SMALL = "sherpa-onnx-whisper-small"
MAX_CHUNK_S = 25.0
NUM_THREADS = int(os.environ.get("ESCAPE_ASR_THREADS", "2"))
# hallucinations classiques de Whisper sur le silence (français)
HALLUCINATIONS = ["sous-titres realises par", "sous-titrage", "amara.org", "merci d'avoir regarde",
                  "abonnez-vous", "sous titres realises", "radio-canada"]

_lock = threading.Lock()
_models: dict = {}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def load_audio(path, t0: float | None = None, t1: float | None = None, sr: int = SR) -> np.ndarray:
    """Décode n'importe quel média en mono float32 à `sr` Hz (ffmpeg)."""
    cmd = [FFMPEG, "-nostdin", "-v", "error"]
    if t0:
        cmd += ["-ss", f"{t0:.3f}"]
    cmd += ["-i", str(path)]
    if t1 is not None:
        cmd += ["-t", f"{max(0.0, t1 - (t0 or 0.0)):.3f}"]
    cmd += ["-vn", "-map", "0:a:0?", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg n'a pas pu lire l'audio de {path}: {r.stderr.decode(errors='replace')[-400:]}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def _rms_db(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    if n == 0:
        return np.zeros(0)
    fr = x[: n * hop].reshape(n, hop)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)


def _chunks(x: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Découpe en morceaux ≤ MAX_CHUNK_S aux points les plus calmes."""
    n, out, a = len(x), [], 0
    hop = int(0.02 * sr)
    while n - a > MAX_CHUNK_S * sr:
        lo, hi = a + int(15 * sr), a + int(MAX_CHUNK_S * sr)
        e = _rms_db(x[lo:hi], hop)
        k = int(np.argmin(np.convolve(e, np.ones(10) / 10, mode="same"))) if len(e) else 0
        cut = lo + k * hop
        out.append((a, cut))
        a = cut
    out.append((a, n))
    return out


# ---------------------------------------------------------------------------
# Modèles sherpa-onnx (téléchargement + chargement paresseux)
# ---------------------------------------------------------------------------
def _model_dir(name: str) -> Path:
    return config.MODELS_DIR / name


def ensure_sherpa_model(name: str) -> Path | None:
    """Renvoie le dossier du modèle ; le télécharge depuis GitHub s'il manque (sauf ESCAPE_ASR_DOWNLOAD=0)."""
    d = _model_dir(name)
    if d.exists() and any(d.glob("*.onnx")):
        return d
    if os.environ.get("ESCAPE_ASR_DOWNLOAD", "1") == "0":
        return None
    url = GH_RELEASE + name + ".tar.bz2"
    tmpdir = config.MODELS_DIR / f"_dl_{name}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    arch = tmpdir / f"{name}.tar.bz2"
    try:
        log(f"[asr] téléchargement de {url} …")
        with urllib.request.urlopen(url, timeout=60) as r, open(arch, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        with tarfile.open(arch, "r:bz2") as tf:
            try:
                tf.extractall(config.MODELS_DIR, filter="data")
            except TypeError:  # Python < 3.12
                for m in tf.getmembers():
                    p = (config.MODELS_DIR / m.name).resolve()
                    if not str(p).startswith(str(config.MODELS_DIR.resolve())):
                        raise RuntimeError(f"chemin suspect dans l'archive : {m.name}")
                tf.extractall(config.MODELS_DIR)
    except Exception as e:
        log(f"[asr] téléchargement impossible ({name}) : {e}")
        return None
    finally:
        arch.unlink(missing_ok=True)
        try:
            tmpdir.rmdir()
        except OSError:
            pass
    return d if d.exists() else None


def _pick(d: Path, pattern: str) -> str:
    """Préfère la version int8 d'un fichier onnx."""
    c = sorted(d.glob(pattern.replace(".onnx", ".int8.onnx"))) or sorted(d.glob(pattern))
    if not c:
        raise FileNotFoundError(f"{pattern} introuvable dans {d}")
    return str(c[0])


def _load(kind: str):
    with _lock:
        if kind in _models:
            return _models[kind]
        import sherpa_onnx
        m = None
        if kind == "sherpa-parakeet":
            d = ensure_sherpa_model(PARAKEET)
            if d:
                m = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=_pick(d, "encoder*.onnx"), decoder=_pick(d, "decoder*.onnx"),
                    joiner=_pick(d, "joiner*.onnx"), tokens=str(d / "tokens.txt"),
                    num_threads=NUM_THREADS, model_type="nemo_transducer", decoding_method="greedy_search")
        elif kind == "sherpa-zipformer-fr":
            d = ensure_sherpa_model(ZIPFORMER_FR)
            if d:
                m = sherpa_onnx.OnlineRecognizer.from_transducer(
                    encoder=_pick(d, "encoder*.onnx"), decoder=_pick(d, "decoder*.onnx"),
                    joiner=_pick(d, "joiner*.onnx"), tokens=str(d / "tokens.txt"),
                    num_threads=NUM_THREADS, decoding_method="greedy_search")
        elif kind == "sherpa-whisper":
            d = ensure_sherpa_model(WHISPER_SMALL)
            if d:
                pre = "small"
                m = sherpa_onnx.OfflineRecognizer.from_whisper(
                    encoder=_pick(d, f"{pre}-encoder.onnx"), decoder=_pick(d, f"{pre}-decoder.onnx"),
                    tokens=str(d / f"{pre}-tokens.txt"), language="fr", task="transcribe",
                    num_threads=NUM_THREADS)
        elif kind == "faster-whisper":
            from faster_whisper import WhisperModel
            name = os.environ.get("ESCAPE_WHISPER_MODEL", "large-v3")
            last = None
            for nm in dict.fromkeys([name, "large-v3", "medium"]):
                try:
                    m = WhisperModel(nm, device="cpu", compute_type="int8", cpu_threads=max(2, NUM_THREADS))
                    break
                except Exception as e:  # modèle non téléchargeable (Hugging Face bloqué...)
                    last = e
            if m is None:
                raise RuntimeError(f"faster-whisper : aucun modèle chargeable ({last})")
        elif kind == "openai-whisper":
            import whisper
            last = None
            for nm in ("large-v3", "medium"):
                try:
                    m = whisper.load_model(nm, device="cpu")
                    break
                except Exception as e:
                    last = e
            if m is None:
                raise RuntimeError(f"openai-whisper : aucun modèle chargeable ({last})")
        if m is None:
            raise RuntimeError(f"moteur {kind} indisponible")
        _models[kind] = m
        return m


BACKENDS = ["faster-whisper", "openai-whisper", "sherpa-parakeet", "sherpa-zipformer-fr", "sherpa-whisper"]


def available_backends() -> list[str]:
    """Moteurs importables (sans charger les modèles)."""
    out = []
    for b in BACKENDS:
        mod = {"faster-whisper": "faster_whisper", "openai-whisper": "whisper"}.get(b, "sherpa_onnx")
        try:
            __import__(mod)
            out.append(b)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Moteurs
# ---------------------------------------------------------------------------
def _w(word: str, s: float, e: float, p: float | None) -> dict:
    return {"w": word, "start": round(float(s), 3), "end": round(float(max(e, s + 0.02)), 3),
            "p": None if p is None else round(float(p), 3)}


def _run_faster_whisper(x: np.ndarray, language: str) -> list[dict]:
    m = _load("faster-whisper")
    segs, _ = m.transcribe(x, language=language, word_timestamps=True, vad_filter=True, beam_size=5,
                           condition_on_previous_text=False)
    words = []
    for s in segs:
        for w in s.words or []:
            words.append(_w(w.word.strip(), w.start, w.end, w.probability))
    return words


def _run_openai_whisper(x: np.ndarray, language: str) -> list[dict]:
    m = _load("openai-whisper")
    res = m.transcribe(x, language=language, word_timestamps=True, condition_on_previous_text=False, fp16=False)
    return [_w(w["word"].strip(), w["start"], w["end"], w.get("probability"))
            for s in res.get("segments", []) for w in s.get("words", [])]


def _tokens_to_words(tokens: list[str], starts: list[float], durs: list[float] | None,
                     logp: list[float] | None, total: float) -> list[dict]:
    """Regroupe des jetons BPE (« ▁mot » ou « ␠mot ») en mots horodatés ; la ponctuation isolée est collée au mot précédent."""
    words: list[dict] = []
    cur = None
    for i, tok in enumerate(tokens):
        t = starts[i] if i < len(starts) else (starts[-1] if starts else 0.0)
        d = durs[i] if durs and i < len(durs) and durs[i] > 0 else None
        lp = logp[i] if logp and i < len(logp) else None
        txt = tok.replace("▁", " ")
        new_word = txt.startswith(" ") or cur is None
        core = txt.strip()
        if not core:
            continue
        if new_word and re.fullmatch(r"[^\w]+", core) and cur is not None:
            new_word = False  # ponctuation isolée (« ? », « ! ») : rattachée au mot précédent
            cur["punct_only"] = True
        if new_word:
            if cur is not None:
                words.append(cur)
            cur = {"w": core, "start": t, "end": t + (d or 0.0), "lp": [] if lp is None else [lp], "has_d": d is not None}
        else:
            cur["w"] += core
            if not cur.pop("punct_only", False) or d is not None:
                cur["end"] = max(cur["end"], t + (d or 0.0))
            if lp is not None:
                cur["lp"].append(lp)
    if cur is not None:
        words.append(cur)
    out = []
    for k, w in enumerate(words):
        nxt = words[k + 1]["start"] if k + 1 < len(words) else total
        end = w["end"]
        if not w["has_d"] or end <= w["start"]:
            # pas de durée : fin = début du mot suivant, plafonnée (≈ 0,09 s par caractère, max 0,8 s)
            end = min(nxt, w["start"] + min(0.8, 0.09 * max(2, len(w["w"]))))
        end = min(end, nxt) if nxt > w["start"] else end
        p = math.exp(float(np.mean(w["lp"]))) if w["lp"] else None
        out.append(_w(w["w"], w["start"], end, p))
    return out


def _run_sherpa_offline(kind: str, x: np.ndarray, pad: float = 1.0) -> list[dict]:
    """Décodage hors ligne ; `pad` s de silence de part et d'autre (sans quoi Parakeet saute parfois
    la première phrase d'un clip qui commence directement sur la parole)."""
    rec = _load(kind)
    z = np.zeros(int(pad * SR), np.float32)
    s = rec.create_stream()
    s.accept_waveform(SR, np.concatenate([z, x, z]))
    rec.decode_stream(s)
    r = s.result
    toks = list(r.tokens)
    ts = list(r.timestamps)
    durs = list(getattr(r, "durations", []) or [])
    lps = list(getattr(r, "ys_log_probs", []) or [])
    if not toks:
        return []
    if not ts:  # whisper : pas d'horodatage → placement approximatif
        return _approx_times(r.text, x)
    words = _tokens_to_words(toks, [t - pad for t in ts], durs or None, lps or None, len(x) / SR)
    for w in words:
        w["start"] = round(min(max(w["start"], 0.0), len(x) / SR), 3)
        w["end"] = round(min(max(w["end"], w["start"] + 0.02), len(x) / SR + 0.02), 3)
    return words


def _run_sherpa_online(kind: str, x: np.ndarray) -> list[dict]:
    rec = _load(kind)
    s = rec.create_stream()
    s.accept_waveform(SR, x)
    s.accept_waveform(SR, np.zeros(int(0.8 * SR), np.float32))  # vidange du contexte droit
    s.input_finished()
    while rec.is_ready(s):
        rec.decode_stream(s)
    r = rec.get_result_all(s) if hasattr(rec, "get_result_all") else None
    toks = list(getattr(r, "tokens", []) or []) if r is not None else []
    ts = list(getattr(r, "timestamps", []) or []) if r is not None else []
    if not toks:
        return []
    words = _tokens_to_words(toks, ts, None, None, len(x) / SR)
    for w in words:  # ce modèle sort en MAJUSCULES
        w["w"] = w["w"].lower()
    return words


def _energy_regions(x: np.ndarray, thr_rel_db: float = -32.0, min_gap: float = 0.18) -> list[tuple[float, float]]:
    """Zones de parole grossières par énergie (300–3400 Hz) : [(t0, t1)] en secondes (repli sans Silero)."""
    from scipy.signal import butter, sosfiltfilt
    if len(x) < SR // 10:
        return []
    y = sosfiltfilt(butter(4, [300, 3400], btype="bandpass", fs=SR, output="sos"), x)
    e = _rms_db(y, int(0.01 * SR))
    if not len(e):
        return []
    thr = max(np.percentile(e, 98) + thr_rel_db, np.percentile(e, 20) + 8.0)
    act = np.concatenate([[False], e > thr, [False]])
    edges = np.flatnonzero(np.diff(act.astype(int)))
    merged: list[list[float]] = []
    for i, j in zip(edges[::2], edges[1::2]):
        a, b = i * 0.01, j * 0.01
        if merged and a - merged[-1][1] < min_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= 0.08]


def _silero_regions(x: np.ndarray) -> list[tuple[float, float]] | None:
    """Zones de parole par Silero VAD (sherpa-onnx) ; None si le modèle est indisponible."""
    mp = config.MODELS_DIR / "silero_vad.onnx"
    if not mp.exists() and os.environ.get("ESCAPE_ASR_DOWNLOAD", "1") != "0":
        try:
            mp.parent.mkdir(parents=True, exist_ok=True)
            tmp = mp.with_suffix(".part")
            urllib.request.urlretrieve(GH_RELEASE + "silero_vad.onnx", tmp)
            os.replace(tmp, mp)
        except Exception:
            return None
    if not mp.exists():
        return None
    try:
        import sherpa_onnx
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(mp)
        cfg.silero_vad.threshold = 0.4
        cfg.silero_vad.min_silence_duration = 0.15
        cfg.silero_vad.min_speech_duration = 0.1
        cfg.silero_vad.window_size = 512
        cfg.sample_rate = SR
        cfg.num_threads = 1
        vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=max(60, len(x) / SR + 5))
        out = []
        y = np.concatenate([x, np.zeros(SR, np.float32)])
        for i in range(0, len(y), 512):
            vad.accept_waveform(y[i:i + 512])
            while not vad.empty():
                out.append((vad.front.start / SR, (vad.front.start + len(vad.front.samples)) / SR))
                vad.pop()
        vad.flush()
        while not vad.empty():
            out.append((vad.front.start / SR, (vad.front.start + len(vad.front.samples)) / SR))
            vad.pop()
        return [(round(a, 3), round(min(b, len(x) / SR), 3)) for a, b in out]
    except Exception:
        return None


def speech_regions(x: np.ndarray) -> list[tuple[float, float]]:
    """Zones de parole (Silero si disponible, sinon énergie)."""
    r = _silero_regions(x)
    return r if r is not None else _energy_regions(x)


def _approx_times(text: str, x: np.ndarray) -> list[dict]:
    """Texte sans horodatage : mots répartis dans les zones de parole au prorata des caractères."""
    toks = re.findall(r"\S+", text or "")
    if not toks:
        return []
    regs = speech_regions(x) or [(0.0, len(x) / SR)]
    total_speech = sum(b - a for a, b in regs)
    weights = np.array([len(t) + 1 for t in toks], float)
    cum = np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * total_speech

    def at(s):  # temps de parole cumulé → temps réel
        acc = 0.0
        for a, b in regs:
            if s <= acc + (b - a):
                return a + (s - acc)
            acc += b - a
        return regs[-1][1]
    return [_w(t, at(cum[i]), at(cum[i + 1]), 0.3) for i, t in enumerate(toks)]


def _refine_times(words: list[dict], x: np.ndarray, max_shift: float = 0.45, pause: float = 0.2) -> None:
    """Recalage fin sur l'énergie (150 Hz–6 kHz, trames de 10 ms) autour des pauses :
    - début d'un mot qui suit une pause : avancé jusqu'à la 1re trame au-dessus du bruit de fond (+4 dB)
      si l'ASR l'a placé dans le silence (retard d'émission / avance des transducteurs), ou reculé jusqu'à
      l'attaque si l'ASR l'a placé en pleine parole ;
    - fin d'un mot suivi d'une pause : ramenée à la dernière trame de parole (les durées TDT incluent le silence),
      plafonnée à 0,15 s + 0,12 s par caractère.
    Les déplacements sont bornés (`max_shift`) : sur un fond bruité, rien ne bouge."""
    from scipy.signal import butter, sosfiltfilt
    if not words or len(x) < SR // 5:
        return
    y = sosfiltfilt(butter(4, [150, 6000], btype="bandpass", fs=SR, output="sos"), x)
    e = _rms_db(y, int(0.01 * SR))
    floor = float(np.percentile(e, 10))
    act = e > floor + 4.0
    n = len(act)

    def idx(t):
        return int(min(max(round(t / 0.01), 0), n - 1))
    for k, w in enumerate(words):
        prev_end = words[k - 1]["end"] if k else -1.0
        nxt_start = words[k + 1]["start"] if k + 1 < len(words) else len(x) / SR + 1
        if w["start"] - prev_end >= pause:
            i0 = idx(w["start"])
            if not act[i0]:  # placé dans le silence : chercher l'attaque vers l'avant
                lim = idx(w["start"] + max_shift)
                j = next((j for j in range(i0, lim + 1) if act[j] and act[min(j + 1, n - 1)]), None)
                if j is not None:
                    w["start"] = round(j * 0.01, 3)
                    w["end"] = round(max(w["end"], w["start"] + 0.06), 3)
                    for m in range(k + 1, len(words)):  # les mots suivants restent dans l'ordre
                        lo_t = words[m - 1]["start"] + 0.05
                        if words[m]["start"] >= lo_t:
                            break
                        words[m]["start"] = round(lo_t, 3)
                        words[m]["end"] = round(max(words[m]["end"], lo_t + 0.04), 3)
                    for m in range(k, len(words) - 1):
                        if words[m]["end"] > words[m + 1]["start"]:
                            words[m]["end"] = round(max(words[m]["start"] + 0.04, words[m + 1]["start"]), 3)
                    nxt_start = words[k + 1]["start"] if k + 1 < len(words) else nxt_start
            else:  # en pleine parole : reculer jusqu'au début de la zone active
                lim = idx(max(w["start"] - max_shift * 0.6, prev_end + 0.03, 0.0))
                j = i0
                while j - 1 >= lim and act[j - 1]:
                    j -= 1
                w["start"] = round(j * 0.01, 3)
        if nxt_start - w["end"] >= pause or k + 1 == len(words):
            cap = w["start"] + 0.15 + 0.12 * len(re.sub(r"\W", "", w["w"]))
            i1 = idx(min(w["end"], cap))
            lo = idx(w["start"] + 0.06)
            j = i1
            while j > lo and not act[j]:
                j -= 1
            w["end"] = round(max(w["start"] + 0.06, min((j + 1) * 0.01 + 0.03, cap)), 3)
        if w["end"] <= w["start"]:
            w["end"] = round(w["start"] + 0.06, 3)


def _fill_missed(words: list[dict], x: np.ndarray, kind: str) -> list[dict]:
    """Zones de parole (VAD) sans aucun mot reconnu : nouveau décodage isolé de la zone, mots insérés."""
    regs = speech_regions(x)
    added = []
    for a, b in regs:
        if b - a < 0.35 or any(w["start"] < b and w["end"] > a for w in words):
            continue
        a0, b0 = max(0.0, a - 0.25), min(len(x) / SR, b + 0.25)
        ws = _run_sherpa_offline(kind, x[int(a0 * SR):int(b0 * SR)])
        for w in ws:
            w["start"] = round(w["start"] + a0, 3)
            w["end"] = round(w["end"] + a0, 3)
            w["refill"] = True
        added += ws
    return sorted(words + added, key=lambda w: w["start"])


def _segments(words: list[dict], gap: float = 0.6) -> list[dict]:
    segs, cur = [], []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > gap or re.search(r"[.?!…]$", cur[-1]["w"])):
            segs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        segs.append(cur)
    return [{"start": s[0]["start"], "end": s[-1]["end"], "text": " ".join(w["w"] for w in s)} for s in segs]


def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.\-' ]+", " ", s)).strip()


def _drop_hallucinations(words: list[dict]) -> list[dict]:
    text = _norm(" ".join(w["w"] for w in words))
    if not any(h in text for h in HALLUCINATIONS):
        return words
    segs = _segments(words)
    keep = []
    for s in segs:
        if not any(h in _norm(s["text"]) for h in HALLUCINATIONS):
            keep += [w for w in words if s["start"] <= w["start"] <= s["end"]]
    return keep


def transcribe_audio(x: np.ndarray, sr: int = SR, language: str = "fr", backend: str | None = None) -> dict:
    """Transcrit un signal mono. `backend` force un moteur ; sinon le premier disponible dans l'ordre."""
    x = np.asarray(x, np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    order = [backend] if backend else (os.environ.get("ESCAPE_ASR_BACKEND") or "").split(",") if os.environ.get(
        "ESCAPE_ASR_BACKEND") else BACKENDS
    errors = []
    for b in order:
        if not b:
            continue
        try:
            words = []
            for a, e in _chunks(x, SR):
                seg = x[a:e]
                if b == "faster-whisper":
                    ws = _run_faster_whisper(seg, language)
                elif b == "openai-whisper":
                    ws = _run_openai_whisper(seg, language)
                elif b in ("sherpa-parakeet", "sherpa-whisper"):
                    ws = _run_sherpa_offline(b, seg)
                elif b == "sherpa-zipformer-fr":
                    ws = _run_sherpa_online(b, seg)
                else:
                    raise ValueError(f"moteur inconnu : {b}")
                off = a / SR
                for w in ws:
                    w["start"] = round(w["start"] + off, 3)
                    w["end"] = round(w["end"] + off, 3)
                words += ws
            if b == "sherpa-parakeet":
                words = _fill_missed(words, x, b)
            words = _drop_hallucinations(words)
            _refine_times(words, x)
            name = b + ("(approx)" if b == "sherpa-whisper" else "")
            return {"backend": name, "language": language, "words": words, "segments": _segments(words),
                    "duration": round(len(x) / SR, 3), "errors": errors}
        except Exception as e:  # moteur indisponible : suivant
            errors.append(f"{b}: {str(e)[:200]}")
            continue
    return {"backend": "none", "language": language, "words": [], "segments": [], "duration": round(len(x) / SR, 3),
            "errors": errors}


def transcribe_file(path, language: str = "fr", backend: str | None = None,
                    t0: float | None = None, t1: float | None = None) -> dict:
    """Transcrit l'audio d'un fichier (wav, mp4, mov...). Horodatage relatif à `t0` (0 par défaut)."""
    x = load_audio(path, t0, t1)
    out = transcribe_audio(x, SR, language, backend)
    out["file"] = str(path)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Transcription mot à mot (français)")
    ap.add_argument("file")
    ap.add_argument("--backend", default=None, choices=BACKENDS)
    ap.add_argument("--json", action="store_true", help="afficher le JSON complet")
    a = ap.parse_args(argv)
    res = transcribe_file(a.file, backend=a.backend)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        print(f"moteur : {res['backend']}  {res.get('errors') or ''}")
        for s in res["segments"]:
            print(f"[{s['start']:7.2f} → {s['end']:7.2f}] {s['text']}")
    return 0 if res["words"] else 1


if __name__ == "__main__":
    sys.exit(main())
