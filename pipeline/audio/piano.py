"""Leitmotiv du vrai piano de IMG_8354 : extraction (onsets + hauteurs) et variantes.

extract_leitmotif(proxy_8354=None, force=False) -> dict
    Son extrait du proxy (repli : rush original) via ffmpeg, détection d'attaques par flux spectral,
    hauteur par YIN sur plusieurs trames de chaque note, sélection de la suite de 4 à 7 notes la plus
    nette. Cache JSON : CACHE_DIR/piano/leitmotif.json (clé = chemin + taille + date du fichier).
piano_variants(leitmotif=None, proxy=None, force=False) -> {"slow"|"reverse"|"detuned"|"note"|"music_box": Path}
    WAV 48 kHz dans CACHE_DIR/piano/.
Accès pour synth.py : motif_notes(source), note_audio(source), variant_audio(name, source)
    source = "auto" (vrai piano si disponible, sinon None), "real" (exception si indisponible), "fallback" (None).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from . import dsp
from .dsp import SR, config

CLIP = "IMG_8354"
VARIANTS = ("slow", "reverse", "detuned", "note", "music_box")
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_VERSION = 3  # incrémenter si l'algorithme change (invalide le cache)


def cache_dir() -> Path:
    d = config.CACHE_DIR / "piano"
    d.mkdir(parents=True, exist_ok=True)
    return d


def source_path(proxy=None) -> Path | None:
    """Proxy de 8354 si présent, sinon le rush original (même temps audio), sinon None."""
    if proxy is not None:
        p = Path(proxy)
        return p if p.exists() else None
    p = config.proxy_path(CLIP)
    if p.exists():
        return p
    try:
        return config.rush_path(CLIP)
    except FileNotFoundError:
        return None


def _src_key(p: Path) -> str:
    st = p.stat()
    return hashlib.sha1(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}|v{_VERSION}".encode()).hexdigest()[:16]


def midi_name(m: float) -> str:
    k = int(round(m))
    return f"{NOTE_NAMES[k % 12]}{k // 12 - 1}"


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------
def _frames(y: np.ndarray, n: int, hop: int) -> np.ndarray:
    y = np.concatenate([np.zeros(n // 2), y, np.zeros(n)])
    k = 1 + (len(y) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    return y[idx]


def detect_onsets(y: np.ndarray, hop: int = 256, nfft: int = 2048, min_gap: float = 0.09):
    """Attaques par flux spectral log (30 Hz–5 kHz) + seuil adaptatif. Renvoie (temps, forces)."""
    win = np.hanning(nfft)
    X = np.abs(np.fft.rfft(_frames(y, nfft, hop) * win, axis=1))
    fr = np.fft.rfftfreq(nfft, 1 / SR)
    band = (fr > 30) & (fr < 5000)
    L = np.log1p(100.0 * X[:, band] / (X[:, band].max() + 1e-12))
    flux = np.concatenate([[0.0], np.maximum(L[1:] - L[:-1], 0).sum(axis=1)])
    flux = flux / (flux.max() + 1e-12)
    frame_t = np.arange(len(flux)) * hop / SR
    thr = median_filter(flux, size=int(0.6 * SR / hop) | 1) + 0.08
    pk, props = signal.find_peaks(flux, height=thr, distance=max(1, int(min_gap * SR / hop)))
    # raffinement : instant de montée la plus raide de l'enveloppe (2 ms) autour du pic
    env = np.sqrt(np.convolve(y ** 2, np.ones(96) / 96, mode="same"))
    times = []
    for p in pk:
        c = int(frame_t[p] * SR)
        a, b = max(0, c - int(0.04 * SR)), min(len(env) - 1, c + int(0.03 * SR))
        if b - a < 4:
            times.append(frame_t[p])
            continue
        d = np.diff(env[a:b])
        times.append((a + int(np.argmax(d))) / SR)
    return np.array(times), flux[pk]


def yin(frame: np.ndarray, fmin: float = 50.0, fmax: float = 2000.0, thresh: float = 0.15):
    """YIN (de Cheveigné & Kawahara 2002) -> (f0 Hz, apériodicité 0..1)."""
    x = np.asarray(frame, dtype=np.float64)
    W = len(x) // 2
    tmax = min(int(SR / fmin), W - 1)
    tmin = max(2, int(SR / fmax))
    nfft = 1 << int(np.ceil(np.log2(len(x) + W)))
    ac = np.fft.irfft(np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(x[:W], nfft)), nfft)[:tmax + 2]
    cs = np.concatenate([[0.0], np.cumsum(x ** 2)])
    tau = np.arange(tmax + 2)
    d = cs[W] + (cs[tau + W] - cs[tau]) - 2 * ac
    d[0] = 0
    cm = np.ones_like(d)
    csum = np.cumsum(d[1:])
    cm[1:] = d[1:] * np.arange(1, len(d)) / np.maximum(csum, 1e-12)
    cand = np.flatnonzero(cm[tmin:tmax] < thresh)
    if len(cand):
        t = cand[0] + tmin
        while t + 1 < tmax and cm[t + 1] < cm[t]:
            t += 1
    else:
        t = int(np.argmin(cm[tmin:tmax])) + tmin
    if 1 <= t < len(cm) - 1:  # interpolation parabolique
        a, b, c = cm[t - 1], cm[t], cm[t + 1]
        den = a - 2 * b + c
        shift = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    else:
        shift = 0.0
    return SR / (t + shift), float(np.clip(cm[t], 0, 1))


def _note_pitch(y: np.ndarray, t0: float, t1: float):
    """Médiane des f0 YIN sur plusieurs trames de la note (après l'attaque)."""
    f0s, aps = [], []
    for off in (0.04, 0.08, 0.13, 0.19, 0.26):
        a = int((t0 + off) * SR)
        if (t0 + off + 0.085) > t1 or a + 4096 > len(y):
            break
        f, ap = yin(y[a:a + 4096] * np.hanning(4096), 50.0, 2000.0)
        f0s.append(f)
        aps.append(ap)
    if not f0s:
        a = int((t0 + 0.02) * SR)
        if a + 4096 <= len(y):
            f, ap = yin(y[a:a + 4096] * np.hanning(4096))
            f0s, aps = [f], [ap]
    if not f0s:
        return None, 1.0
    f0s, aps = np.array(f0s), np.array(aps)
    good = aps < 0.3
    sel = f0s[good] if good.any() else f0s
    return float(np.median(sel)), float(np.median(aps[good] if good.any() else aps))


def _load_mono(path: Path) -> np.ndarray:
    key = _src_key(path)
    cached = cache_dir() / f"src_{key}.wav"
    if cached.exists():
        return dsp.to_mono(dsp.load(cached))
    a = dsp.ffmpeg_decode(path)
    dsp.save(cached, a)
    return dsp.to_mono(a)


def _analyse(y: np.ndarray) -> list[dict]:
    times, strength = detect_onsets(y)
    dur = len(y) / SR
    notes = []
    for i, t0 in enumerate(times):
        t1 = times[i + 1] if i + 1 < len(times) else min(dur, t0 + 2.0)
        f0, ap = _note_pitch(y, t0, t1)
        a, b = int(t0 * SR), int(min(t1, t0 + 0.12) * SR)
        e = dsp.rms_db(y[a:b]) if b > a else -120.0
        if f0 is None or not (40 < f0 < 2100):
            continue
        m = 69 + 12 * np.log2(f0 / 440.0)
        notes.append({"t": round(float(t0), 4), "ioi": round(float(t1 - t0), 4), "f0": round(f0, 2),
                      "midi": int(round(m)), "cents": round(float((m - round(m)) * 100), 1), "name": midi_name(m),
                      "conf": round(1.0 - ap, 3), "energy_db": round(e, 2), "flux": round(float(strength[i]), 3)})
    return notes


def _select_motif(notes: list[dict]) -> list[dict]:
    best, best_score = None, -1e9
    for L in range(7, 3, -1):
        for i in range(len(notes) - L + 1):
            seq = notes[i:i + L]
            if min(n["conf"] for n in seq) < 0.6:
                continue
            iois = [n["ioi"] for n in seq[:-1]]
            if min(iois) < 0.12 or max(iois) > 1.8:
                continue
            leaps = np.abs(np.diff([n["midi"] for n in seq]))
            if leaps.max() > 19:
                continue
            e = np.mean([10 ** (n["energy_db"] / 20) for n in seq])
            score = 20 * np.log10(e + 1e-9) + 6 * np.log2(L) + 10 * np.mean([n["conf"] for n in seq])
            if score > best_score:
                best, best_score = seq, score
        if best is not None and L <= 5:
            break
    if best is None:
        conf = [n for n in notes if n["conf"] >= 0.5]
        best = conf[:7] if len(conf) >= 4 else []
    return best


def extract_leitmotif(proxy_8354=None, force: bool = False) -> dict:
    """Extrait le leitmotiv (4–7 notes) du vrai piano. Cache JSON dans CACHE_DIR/piano."""
    path = source_path(proxy_8354)
    if path is None:
        return {"ok": False, "reason": f"source {CLIP} introuvable", "notes": []}
    key = _src_key(path)
    cj = cache_dir() / "leitmotif.json"
    if cj.exists() and not force:
        try:
            d = json.loads(cj.read_text(encoding="utf-8"))
            if d.get("key") == key:
                return d
        except Exception:
            pass
    y = _load_mono(path)
    notes = _analyse(y)
    motif = _select_motif(notes)
    out = {"ok": len(motif) >= 4, "key": key, "source": str(path), "method": "flux-onsets + YIN",
           "duration": round(len(y) / SR, 3), "all_notes": notes, "notes": []}
    if motif:
        t0 = motif[0]["t"]
        for j, nt in enumerate(motif):
            d = dict(nt)
            d["t_rel"] = round(nt["t"] - t0, 4)
            d["dur"] = round(nt["ioi"] if j < len(motif) - 1 else min(1.5, max(0.5, nt["ioi"])), 4)
            out["notes"].append(d)
        last = motif[-1]
        out["span"] = [round(t0, 4), round(last["t"] + min(2.0, last["ioi"]), 4)]
        out["summary"] = " ".join(n["name"] for n in motif)
    if not out["ok"]:
        out["reason"] = "moins de 4 notes nettes détectées"
    cj.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Variantes
# ---------------------------------------------------------------------------
def _motif_segment(lm: dict, y: np.ndarray) -> np.ndarray:
    a = max(0, int((lm["span"][0] - 0.05) * SR))
    b = min(len(y), int((lm["span"][1] + 0.6) * SR))
    return dsp.fade(dsp.as_stereo(y[a:b]), 0.01, 0.4)


def _best_note(lm: dict, y: np.ndarray) -> np.ndarray:
    cands = sorted(lm["notes"], key=lambda n: (min(n["ioi"], 2.5) * n["conf"]), reverse=True)
    nt = cands[0]
    a = max(0, int((nt["t"] - 0.008) * SR))
    ln = min(max(0.35, nt["ioi"] - 0.03), 3.0)
    b = min(len(y), a + int(ln * SR))
    seg = dsp.fade(dsp.as_stereo(y[a:b]), 0.004, min(0.15, ln * 0.3))
    return dsp.normalize(dsp.highpass(seg, 40, order=2), -1.0)


def _render_variants(lm: dict, y: np.ndarray) -> dict[str, np.ndarray]:
    seg = _motif_segment(lm, y)
    rng = np.random.default_rng(8354)
    out = {}
    out["slow"] = dsp.reverb(dsp.stretch(seg, 2.0), dsp.make_ir(4.0, 0.03, 0.5, 1), wet=0.35)
    out["reverse"] = dsp.reverb(seg, dsp.make_ir(3.0, 0.02, 0.5, 2), wet=0.5)[::-1].copy()
    n = len(seg)
    t = np.arange(n) / SR
    k = max(6, int(n / SR * 0.8) + 6)
    wob = np.interp(t * 0.8, np.arange(k), rng.uniform(-1, 1, k)) * 18 + 6 * np.sin(2 * np.pi * 3.1 * t)
    rate = 2 ** ((-30 + wob) / 1200)
    src = np.concatenate([[0.0], np.cumsum(rate)[:-1]])
    det = dsp.varispeed(seg, src)
    out["detuned"] = dsp.reverb(dsp.lowpass(det, 6000, order=2), dsp.make_ir(2.5, 0.02, 0.55, 3), wet=0.3)
    out["note"] = dsp.reverb(_best_note(lm, y), dsp.make_ir(6.0, 0.03, 0.5, 4), wet=0.55)
    from . import synth
    out["music_box"] = synth.music_box_leitmotif(None, 0, source="real")
    return {k: dsp.normalize(v, -1.0) for k, v in out.items()}


def piano_variants(leitmotif: dict | None = None, proxy=None, force: bool = False) -> dict[str, Path]:
    """Écrit les variantes slow / reverse / detuned / note / music_box dans CACHE_DIR/piano/. {} si pas de piano."""
    lm = leitmotif or extract_leitmotif(proxy)
    if not lm.get("ok"):
        return {}
    d = cache_dir()
    paths = {v: d / f"{v}.wav" for v in VARIANTS}
    stamp = d / "variants.key"
    if not force and stamp.exists() and stamp.read_text().strip() == lm["key"] and all(p.exists() for p in paths.values()):
        return paths
    y = _load_mono(Path(lm["source"]))
    for name, a in _render_variants(lm, y).items():
        dsp.save(paths[name], a)
    stamp.write_text(lm["key"])
    return paths


# ---------------------------------------------------------------------------
# Accès simplifié pour synth.py
# ---------------------------------------------------------------------------
def _lm_for(source: str) -> dict | None:
    if source == "fallback":
        return None
    lm = extract_leitmotif()
    if not lm.get("ok"):
        if source == "real":
            raise RuntimeError(f"leitmotiv réel indisponible : {lm.get('reason')}")
        return None
    return lm


def motif_notes(source: str = "auto") -> list[tuple] | None:
    """[(t_rel, f0_hz, durée, vélocité), ...] du motif réel, ou None."""
    lm = _lm_for(source)
    if lm is None:
        return None
    e = np.array([n["energy_db"] for n in lm["notes"]])
    vel = np.clip((10 ** ((e - e.max()) / 20)) ** 0.5, 0.5, 1.0)
    return [(n["t_rel"], float(n["f0"]), n["dur"], float(v)) for n, v in zip(lm["notes"], vel)]


def note_audio(source: str = "auto") -> np.ndarray | None:
    """Note réelle isolée, SANS réverbe (N, 2), ou None."""
    lm = _lm_for(source)
    if lm is None:
        return None
    return _best_note(lm, _load_mono(Path(lm["source"])))


def variant_audio(name: str, source: str = "auto") -> np.ndarray | None:
    if name not in VARIANTS:
        raise KeyError(name)
    lm = _lm_for(source)
    if lm is None:
        return None
    return dsp.load(piano_variants(lm)[name])


if __name__ == "__main__":
    import sys

    res = extract_leitmotif(sys.argv[1] if len(sys.argv) > 1 else None, force=True)
    print(json.dumps({k: v for k, v in res.items() if k != "all_notes"}, indent=1, ensure_ascii=False))
    print({k: str(v) for k, v in piano_variants(res, force=True).items()})
