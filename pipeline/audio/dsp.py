"""Briques DSP du projet (48 kHz, float32, stéréo (N, 2)).

Toutes les fonctions acceptent du mono (N,) ou du stéréo (N, 2) et renvoient du stéréo float32
(N, 2), sauf mention contraire. Les temps sont en secondes, les gains en dB.

Pitch / stretch : rubberband CLI (meilleure qualité) -> filtre ffmpeg `rubberband` -> repli
vocodeur de phase pur numpy (toujours disponible, ex. Windows sans rubberband).
Forcer un moteur : variable d'environnement ESCAPE_AUDIO_BACKEND = rubberband | ffmpeg | numpy.
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

try:  # config.py est à la racine du dépôt (réexporté pour piano.py / mixer.py)
    import config  # noqa: F401
except ModuleNotFoundError:  # exécution directe d'un module
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config  # noqa: F401,E402

SR = 48000
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Formes, conversions, E/S
# ---------------------------------------------------------------------------
def as_stereo(x) -> np.ndarray:
    """(N,) | (N,1) | (N,C) -> (N, 2) float32."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    if x.shape[1] > 2:
        return np.ascontiguousarray(x[:, :2])
    return x


def to_mono(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x if x.ndim == 1 else x.mean(axis=1)


def n_samples(t: float) -> int:
    return int(round(float(t) * SR))


def silence(dur: float) -> np.ndarray:
    return np.zeros((n_samples(dur), 2), np.float32)


def resample(x, sr_in: int, sr_out: int = SR) -> np.ndarray:
    """Rééchantillonnage polyphase (qualité studio)."""
    x = np.asarray(x, dtype=np.float32)
    if int(sr_in) == int(sr_out):
        return x
    g = np.gcd(int(sr_in), int(sr_out))
    return signal.resample_poly(x, int(sr_out) // g, int(sr_in) // g, axis=0).astype(np.float32)


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def ffmpeg_decode(path, sr: int = SR, start: float | None = None, dur: float | None = None) -> np.ndarray:
    """Décode la piste audio de n'importe quel fichier (mp4, mov, wav...) en (N, 2) float32."""
    cmd = [_ffmpeg(), "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{float(start):.6f}"]
    cmd += ["-i", str(path)]
    if dur is not None:
        cmd += ["-t", f"{float(dur):.6f}"]
    cmd += ["-vn", "-ac", "2", "-ar", str(int(sr)), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg n'a pas pu décoder {path} : {r.stderr.decode(errors='replace')[-800:]}")
    a = np.frombuffer(r.stdout, dtype=np.float32)
    return a.reshape(-1, 2).copy()


def load(path) -> np.ndarray:
    """Charge un fichier audio (wav/flac/ogg, sinon via ffmpeg) -> (N, 2) float32 à 48 kHz."""
    p = Path(path)
    try:
        data, fs = sf.read(str(p), dtype="float32", always_2d=True)
    except Exception:
        return as_stereo(ffmpeg_decode(p))
    return as_stereo(resample(data, fs, SR))


def save(path, x, subtype: str = "FLOAT") -> Path:
    """Écrit un WAV 48 kHz stéréo. subtype : FLOAT (défaut), PCM_24, PCM_16."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    y = as_stereo(x)
    if subtype.upper().startswith("PCM"):
        y = np.clip(y, -1.0, 1.0)
    sf.write(str(p), y, SR, subtype=subtype)
    return p


def db(x):
    """Amplitude linéaire -> dB (élément par élément)."""
    return 20.0 * np.log10(np.maximum(np.abs(x), _EPS))


def lin(d):
    """dB -> amplitude linéaire."""
    return np.power(10.0, np.asarray(d, dtype=np.float64) / 20.0)


def peak_db(x) -> float:
    x = np.asarray(x)
    return float(db(np.max(np.abs(x)))) if x.size else -240.0


def rms_db(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(10 * np.log10(np.mean(x ** 2) + 1e-24)) if x.size else -240.0


def gain(x, db) -> np.ndarray:  # noqa: F811 (le paramètre masque db() localement, volontairement)
    return (as_stereo(x) * np.float32(lin(db))).astype(np.float32)


def normalize(x, peak: float = -1.0) -> np.ndarray:
    """Normalise le crête échantillon à `peak` dBFS."""
    x = as_stereo(x)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < 1e-9:
        return x.copy()
    return (x * np.float32(lin(peak) / m)).astype(np.float32)


def _ramp(n: int, shape: str = "cos") -> np.ndarray:
    p = (np.arange(n, dtype=np.float64) + 0.5) / max(n, 1)
    if shape == "lin":
        return p
    if shape == "exp":
        return p ** 2.5
    if shape == "sine":  # puissance constante
        return np.sin(0.5 * np.pi * p)
    return 0.5 - 0.5 * np.cos(np.pi * p)  # cosinus surélevé : départ à pente nulle (aucun clic)


def fade(x, fi: float = 0.0, fo: float = 0.0, shape: str = "cos") -> np.ndarray:
    """Fondu d'entrée `fi` et de sortie `fo` (secondes)."""
    y = as_stereo(x).copy()
    n = len(y)
    a = min(n, n_samples(fi))
    b = min(n, n_samples(fo))
    if a > 0:
        y[:a] *= _ramp(a, shape)[:, None].astype(np.float32)
    if b > 0:
        y[n - b:] *= _ramp(b, shape)[::-1, None].astype(np.float32)
    return y


def pad_to(x, n: int) -> np.ndarray:
    """Complète par du silence ou tronque à n échantillons."""
    x = as_stereo(x)
    n = int(n)
    if len(x) >= n:
        return x[:n].copy()
    return np.concatenate([x, np.zeros((n - len(x), 2), np.float32)])


def mix_at(dst: np.ndarray, src, t: float, gain_db: float = 0.0) -> np.ndarray:
    """Ajoute `src` dans `dst` (modifié sur place) à l'instant t (s, peut être négatif)."""
    src = as_stereo(src)
    k = n_samples(t)
    a0, s0 = (k, 0) if k >= 0 else (0, -k)
    m = min(len(dst) - a0, len(src) - s0)
    if m > 0:
        g = np.float32(lin(gain_db))
        dst[a0:a0 + m] += src[s0:s0 + m] * g
    return dst


def pan(x, p: float) -> np.ndarray:
    """Panoramique à puissance constante, p ∈ [-1 (gauche), 1 (droite)] (appliqué sur le mono)."""
    m = to_mono(x)
    a = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.stack([m * np.cos(a), m * np.sin(a)], 1).astype(np.float32) * np.float32(np.sqrt(2))


def width(x, w: float) -> np.ndarray:
    """Largeur stéréo M/S (0 = mono, 1 = inchangé, >1 = élargi)."""
    x = as_stereo(x)
    m = (x[:, 0] + x[:, 1]) * 0.5
    s = (x[:, 0] - x[:, 1]) * 0.5 * w
    return np.stack([m + s, m - s], 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Filtres
# ---------------------------------------------------------------------------
def _clip_hz(hz):
    return np.clip(np.asarray(hz, dtype=np.float64), 1.0, 0.49 * SR)


def _sos_apply(x, sos, zero_phase: bool = False) -> np.ndarray:
    y = as_stereo(x).astype(np.float64)
    if zero_phase and len(y) > 3 * 6 * len(sos):
        y = signal.sosfiltfilt(sos, y, axis=0)
    else:
        y = signal.sosfilt(sos, y, axis=0)
    return y.astype(np.float32)


def lowpass(x, hz: float, order: int = 4, zero_phase: bool = False) -> np.ndarray:
    return _sos_apply(x, signal.butter(order, float(_clip_hz(hz)), "lowpass", fs=SR, output="sos"), zero_phase)


def highpass(x, hz: float, order: int = 4, zero_phase: bool = False) -> np.ndarray:
    return _sos_apply(x, signal.butter(order, float(_clip_hz(hz)), "highpass", fs=SR, output="sos"), zero_phase)


def bandpass(x, hz, order: int = 2, q: float | None = None, zero_phase: bool = False) -> np.ndarray:
    """Passe-bande Butterworth. hz = (bas, haut) ou fréquence centrale (avec q, défaut 1.0)."""
    if np.ndim(hz) == 0:
        q = 1.0 if q is None else float(q)
        c = float(hz)
        k = 1.0 / (2.0 * q)
        lo, hi = c * (np.sqrt(1 + k * k) - k), c * (np.sqrt(1 + k * k) + k)
    else:
        lo, hi = float(hz[0]), float(hz[1])
    lo, hi = float(_clip_hz(lo)), float(_clip_hz(hi))
    if hi <= lo * 1.001:
        hi = lo * 1.01
    return _sos_apply(x, signal.butter(order, [lo, hi], "bandpass", fs=SR, output="sos"), zero_phase)


def _biquad_coefs(kind: str, hz, q, gain_db=0.0):
    """Coefficients RBJ (Audio EQ Cookbook), vectorisés. Renvoie (b0,b1,b2,a1,a2) normalisés."""
    f = _clip_hz(hz)
    q = np.maximum(np.asarray(q, dtype=np.float64), 1e-3)
    A = np.power(10.0, np.asarray(gain_db, dtype=np.float64) / 40.0)
    w = 2 * np.pi * f / SR
    cw, sw = np.cos(w), np.sin(w)
    al = sw / (2 * q)
    if kind == "lowpass":
        b0, b1, b2 = (1 - cw) / 2, 1 - cw, (1 - cw) / 2
        a0, a1, a2 = 1 + al, -2 * cw, 1 - al
    elif kind == "highpass":
        b0, b1, b2 = (1 + cw) / 2, -(1 + cw), (1 + cw) / 2
        a0, a1, a2 = 1 + al, -2 * cw, 1 - al
    elif kind == "bandpass":  # gain 0 dB au centre
        b0, b1, b2 = al, 0 * cw, -al
        a0, a1, a2 = 1 + al, -2 * cw, 1 - al
    elif kind == "notch":
        b0, b1, b2 = 1 + 0 * cw, -2 * cw, 1 + 0 * cw
        a0, a1, a2 = 1 + al, -2 * cw, 1 - al
    elif kind == "peaking":
        b0, b1, b2 = 1 + al * A, -2 * cw, 1 - al * A
        a0, a1, a2 = 1 + al / A, -2 * cw, 1 - al / A
    elif kind in ("lowshelf", "highshelf"):
        sq = 2 * np.sqrt(A) * al
        if kind == "lowshelf":
            b0 = A * ((A + 1) - (A - 1) * cw + sq)
            b1 = 2 * A * ((A - 1) - (A + 1) * cw)
            b2 = A * ((A + 1) - (A - 1) * cw - sq)
            a0 = (A + 1) + (A - 1) * cw + sq
            a1 = -2 * ((A - 1) + (A + 1) * cw)
            a2 = (A + 1) + (A - 1) * cw - sq
        else:
            b0 = A * ((A + 1) + (A - 1) * cw + sq)
            b1 = -2 * A * ((A - 1) + (A + 1) * cw)
            b2 = A * ((A + 1) + (A - 1) * cw - sq)
            a0 = (A + 1) - (A - 1) * cw + sq
            a1 = 2 * ((A - 1) - (A + 1) * cw)
            a2 = (A + 1) - (A - 1) * cw - sq
    else:
        raise ValueError(f"biquad inconnu : {kind}")
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def biquad(x, kind: str, hz: float, q: float = 0.707, gain_db: float = 0.0) -> np.ndarray:
    b0, b1, b2, a1, a2 = _biquad_coefs(kind, hz, q, gain_db)
    sos = np.array([[b0, b1, b2, 1.0, a1, a2]], dtype=np.float64)
    return _sos_apply(x, sos)


def peaking(x, hz: float, db: float, q: float = 1.0) -> np.ndarray:  # noqa: F811
    return biquad(x, "peaking", hz, q, db)


def shelf(x, hz: float, db: float, kind: str = "low", q: float = 0.707) -> np.ndarray:  # noqa: F811
    return biquad(x, "lowshelf" if kind.startswith("low") else "highshelf", hz, q, db)


def resonator(x, hz: float, q: float = 10.0) -> np.ndarray:
    """Passe-bande résonant étroit (gain 0 dB au centre) — utile pour la synthèse modale."""
    return biquad(x, "bandpass", hz, q)


def tv_filter(x, kind: str, hz_curve, q=0.707, gain_db=0.0, block: int = 64) -> np.ndarray:
    """Biquad à paramètres variables dans le temps (fréquence `hz_curve` : scalaire ou (N,)).

    Traitement par blocs avec conservation d'état : adapté aux balayages lents/moyens.
    """
    y = as_stereo(x).astype(np.float64).T.copy()  # (2, N)
    n = y.shape[1]
    hz_curve = np.broadcast_to(np.asarray(hz_curve, dtype=np.float64), (n,))
    q_curve = np.broadcast_to(np.asarray(q, dtype=np.float64), (n,))
    centers = np.minimum(np.arange(0, n, block) + block // 2, n - 1)
    b0, b1, b2, a1, a2 = _biquad_coefs(kind, hz_curve[centers], q_curve[centers], gain_db)
    out = np.empty_like(y)
    zi = np.zeros((1, 2, 2))
    for i, s in enumerate(range(0, n, block)):
        sos = np.array([[b0[i], b1[i], b2[i], 1.0, a1[i], a2[i]]])
        out[:, s:s + block], zi = signal.sosfilt(sos, y[:, s:s + block], axis=-1, zi=zi)
    return out.T.astype(np.float32)


def dc_block(x, hz: float = 15.0) -> np.ndarray:
    return highpass(x, hz, order=2)


# ---------------------------------------------------------------------------
# Dynamique
# ---------------------------------------------------------------------------
def _smooth_attack_release(target: np.ndarray, att_coef: float, rel_coef: float) -> np.ndarray:
    """Lissage à une pôle asymétrique (montée = attaque) au taux de contrôle."""
    out = np.empty_like(target)
    s = 0.0
    for i, v in enumerate(target.tolist()):
        c = att_coef if v > s else rel_coef
        s = v + c * (s - v)
        out[i] = s
    return out


def _level_db(x, hop: int, win_s: float = 0.005) -> np.ndarray:
    """Niveau RMS (dB) lié stéréo, échantillonné tous les `hop` échantillons."""
    p = np.max(as_stereo(x).astype(np.float64) ** 2, axis=1)
    w = max(1, int(win_s * SR))
    p = np.maximum(uniform_filter1d(p, w, mode="constant"), 0.0)
    return 10 * np.log10(p[hop // 2::hop] + 1e-20)


def compress(x, thr_db: float = -18.0, ratio: float = 3.0, attack: float = 0.01, release: float = 0.15,
             makeup: float = 0.0, knee: float = 6.0) -> np.ndarray:
    """Compresseur RMS lié stéréo à genou doux."""
    x = as_stereo(x)
    n = len(x)
    if n == 0:
        return x.copy()
    hop = 32
    lv = _level_db(x, hop)
    over = lv - thr_db
    gr = np.where(over <= -knee / 2, 0.0,
                  np.where(over >= knee / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * max(knee, 1e-6))))
    rate = SR / hop
    gr = _smooth_attack_release(gr, np.exp(-1 / (attack * rate)), np.exp(-1 / (release * rate)))
    pos = np.arange(len(gr)) * hop + hop // 2
    g = lin(makeup - np.interp(np.arange(n), pos, gr))
    return (x * g[:, None]).astype(np.float32)


def activity(key, n: int | None = None, thr_db: float = -42.0, attack: float = 0.08, release: float = 0.30,
             hold: float = 0.15, lookahead: float | None = None) -> np.ndarray:
    """Courbe de présence (0..1, longueur n) d'un signal clé (voix) : attaque/relâche/maintien.

    `lookahead` (défaut = attack) : la courbe démarre AVANT la voix, la musique est déjà baissée
    quand la parole commence (mixage hors-ligne).
    """
    key = as_stereo(key)
    n = len(key) if n is None else int(n)
    hop = 48
    lv = _level_db(key, hop, win_s=0.02)
    rate = SR / hop
    raw = (lv > thr_db).astype(np.float64)
    h = max(1, int(hold * rate))
    raw = maximum_filter1d(raw, size=2 * h + 1)
    la = int((attack if lookahead is None else lookahead) * rate)
    if la > 0:
        raw = np.concatenate([raw[la:], np.zeros(la)])
    p = _smooth_attack_release(raw, np.exp(-1 / (max(attack, 1e-3) * rate / 3)),
                               np.exp(-1 / (max(release, 1e-3) * rate / 3)))
    pos = np.arange(len(p)) * hop + hop // 2
    return np.clip(np.interp(np.arange(n), pos, p, left=p[0] if len(p) else 0, right=0), 0, 1)


def duck(x, key, depth_db: float, attack: float = 0.08, release: float = 0.30, thr_db: float = -42.0,
         hold: float = 0.15) -> np.ndarray:
    """Ducking : baisse `x` de depth_db quand `key` est actif (attaque/relâche en secondes)."""
    x = as_stereo(x)
    p = activity(key, len(x), thr_db, attack, release, hold)
    return (x * lin(-depth_db * p)[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Réverbération (convolution)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=48)
def _make_ir_cached(decay_s: float, predelay: float, damping: float, seed: int, early: bool,
                    width_: float, bright: float) -> np.ndarray:
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 7919])
    n_pre = int(predelay * SR)
    n_tail = max(64, int(decay_s * 1.15 * SR))
    t = np.arange(n_tail) / SR
    noise = rng.standard_normal((n_tail, 2))
    edges = [20, 160, 400, 1000, 2500, 6000, 0.45 * SR]
    tail = np.zeros((n_tail, 2))
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        r = i / (len(edges) - 2)
        T = decay_s * (1.0 - damping * 0.8 * r ** 0.8)
        if i == 0:
            T = decay_s * 0.8
        T = max(T, 0.05)
        sos = signal.butter(2, [lo, hi], "bandpass", fs=SR, output="sos")
        band = signal.sosfilt(sos, noise, axis=0)
        level = 1.0 if i < 4 else bright
        tail += band * (level * np.exp(-6.9078 * t / T))[:, None]
    tail *= (1 - np.exp(-t / 0.012))[:, None]  # densification de la diffusion
    m = (tail[:, 0] + tail[:, 1]) * 0.5
    s = (tail[:, 0] - tail[:, 1]) * 0.5 * width_
    tail = np.stack([m + s, m - s], 1)
    ir = np.zeros((n_pre + n_tail, 2))
    ir[n_pre:] = tail
    if early:  # réflexions précoces éparses, différentes à gauche et à droite
        for ch in range(2):
            k = rng.integers(7, 13)
            ts = np.sort(rng.uniform(0.003, predelay + 0.07, k))
            for tt in ts:
                idx = int(tt * SR)
                if idx < len(ir):
                    ir[idx, ch] += rng.choice([-1, 1]) * 2.2 * np.exp(-tt / 0.05) * rng.uniform(0.4, 1.0)
    ir /= np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True)) + 1e-12
    ir = ir.astype(np.float32)
    ir.setflags(write=False)
    return ir


def make_ir(decay_s: float, predelay: float = 0.02, damping: float = 0.5, seed: int = 0, early: bool = True,
            width: float = 1.0, bright: float = 1.0) -> np.ndarray:
    """Réponse impulsionnelle stéréo synthétique (bruit à décroissance multibande, RT60 = decay_s).

    damping ∈ [0,1] : aigus qui s'éteignent plus vite (0 = carrelage brillant, 1 = salle très mate).
    Énergie normalisée à 1 par canal (le signal mouillé a ~ le même RMS que le sec). Lecture seule.
    """
    return _make_ir_cached(round(float(decay_s), 4), round(float(predelay), 4), round(float(damping), 3),
                           int(seed), bool(early), round(float(width), 3), round(float(bright), 3))


def _convolve_stereo(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    x = as_stereo(x).astype(np.float64)
    ir = as_stereo(ir).astype(np.float64)
    return np.stack([signal.oaconvolve(x[:, c], ir[:, c]) for c in range(2)], 1)


def reverb(x, ir, wet: float = 0.3, tail: bool = True) -> np.ndarray:
    """Réverbe à convolution. Sortie = (1-wet)·sec + wet·mouillé ; longueur len(x)+len(ir)-1 si tail."""
    x = as_stereo(x)
    if len(x) == 0:
        return x.copy()
    w = _convolve_stereo(x, ir)
    n = len(w) if tail else len(x)
    y = np.zeros((n, 2))
    y[:len(x)] = x * (1 - wet)
    y += wet * w[:n]
    return y.astype(np.float32)


def reverse_reverb(x, ir, wet: float = 1.0) -> np.ndarray:
    """Réverbe inversée (le son « aspire » sa propre réverbe avant d'arriver).

    Sortie de longueur len(x) + len(ir) - 1 : le signal original commence à l'indice len(ir) - 1
    (la montée de réverbe le PRÉCÈDE). wet=1 : swell + son sec ; wet<1 : swell atténué.
    """
    x = as_stereo(x)
    r = x[::-1]
    w = _convolve_stereo(r, ir)[::-1]
    y = w * wet
    off = len(ir) - 1
    y[off:off + len(x)] += x
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Interpolation / varispeed / vocodeur de phase
# ---------------------------------------------------------------------------
def interp_cubic(y, pos) -> np.ndarray:
    """Lecture à positions fractionnaires (Catmull-Rom). y (M, C) ; pos (N,) -> (N, C). Zéro hors bornes."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    pos = np.asarray(pos, dtype=np.float64)
    m = len(y)
    if m == 0:
        return np.zeros((len(pos), y.shape[1]), np.float32)
    i = np.floor(pos).astype(np.int64)
    f = (pos - i)[:, None].astype(np.float32)

    def g(k):
        return y[np.clip(k, 0, m - 1)]

    ym1, y0, y1, y2 = g(i - 1), g(i), g(i + 1), g(i + 2)
    c1 = 0.5 * (y1 - ym1)
    c2 = ym1 - 2.5 * y0 + 2 * y1 - 0.5 * y2
    c3 = 0.5 * (y2 - ym1) + 1.5 * (y0 - y1)
    out = ((c3 * f + c2) * f + c1) * f + y0
    out[(pos < 0) | (pos > m - 1)] = 0
    return out.astype(np.float32)


def varispeed(x, src_pos) -> np.ndarray:
    """Lecture « bande magnétique » : out[n] = x[src_pos[n]] (hauteur liée à la vitesse).

    Anti-repliement automatique si la vitesse locale dépasse 1.
    """
    x = as_stereo(x)
    src_pos = np.asarray(src_pos, dtype=np.float64)
    if len(src_pos) > 1:
        vmax = float(np.max(np.abs(np.diff(src_pos))))
        if vmax > 1.05:
            x = lowpass(x, 0.45 * SR / vmax, order=8, zero_phase=True)
    return interp_cubic(x, src_pos)


def _pv_stretch_mono(x: np.ndarray, ana_centers: np.ndarray, n_out: int, nfft: int = 2048,
                     hs: int = 512) -> np.ndarray:
    """Vocodeur de phase à verrouillage de phase (Laroche-Dolson).

    Trame de synthèse m (centrée en m*hs) lue à la position d'analyse ana_centers[m] (quelconque :
    accélération, ralenti, gel). Fréquence instantanée estimée sur 2 trames d'analyse distantes de `dh`.
    """
    win = signal.windows.hann(nfft, sym=False)
    half = nfft // 2
    dh = hs // 2
    xp = np.concatenate([np.zeros(nfft + dh), x.astype(np.float64), np.zeros(nfft + dh)])
    off = nfft + dh
    nb = half + 1
    omega = 2 * np.pi * np.arange(nb) / nfft
    out = np.zeros(n_out + nfft + hs)
    wsum = np.zeros_like(out)
    phase = None
    n_frames = len(ana_centers)
    for m in range(n_frames):
        a = int(round(ana_centers[m])) + off
        lo = a - half
        if lo < dh or lo + nfft > len(xp):
            continue
        X1 = np.fft.rfft(xp[lo:lo + nfft] * win)
        X0 = np.fft.rfft(xp[lo - dh:lo - dh + nfft] * win)
        mag = np.abs(X1)
        a1 = np.angle(X1)
        dphi = a1 - np.angle(X0) - omega * dh
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        inst = omega + dphi / dh
        if phase is None:
            phase = a1.copy()
        else:
            pk = np.flatnonzero((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])) + 1
            if len(pk) == 0:
                phase = phase + inst * hs
            else:
                pk_phase = phase[pk] + inst[pk] * hs
                mids = (pk[:-1] + pk[1:]) / 2.0
                owner = np.searchsorted(mids, np.arange(nb))
                phase = pk_phase[owner] + a1 - a1[pk][owner]
        frame = np.fft.irfft(mag * np.exp(1j * phase), nfft) * win
        s = m * hs - half + half  # trame centrée en m*hs, décalée de `half` dans `out`
        out[s:s + nfft] += frame
        wsum[s:s + nfft] += win ** 2
    out = out[half:half + n_out]
    wsum = wsum[half:half + n_out]
    return (out / np.maximum(wsum, 1e-3 * wsum.max() if wsum.max() > 0 else 1.0)).astype(np.float32)


def _pv_warp(x: np.ndarray, src_of_u: np.ndarray, nfft: int = 2048, hs: int = 512) -> np.ndarray:
    n_out = len(src_of_u)
    frames = np.arange(0, n_out + hs, hs)
    centers = np.interp(frames, np.arange(n_out), src_of_u)
    x = as_stereo(x)
    return np.stack([_pv_stretch_mono(x[:, c], centers, n_out, nfft, hs) for c in range(2)], 1)


# ---------------------------------------------------------------------------
# Moteurs pitch / stretch
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _have_rubberband() -> bool:
    return shutil.which("rubberband") is not None


@functools.lru_cache(maxsize=1)
def _have_ffmpeg_rubberband() -> bool:
    try:
        r = subprocess.run([_ffmpeg(), "-hide_banner", "-filters"], capture_output=True, text=True, timeout=30)
        return " rubberband " in r.stdout
    except Exception:
        return False


def backend() -> str:
    """Moteur pitch/stretch utilisé : 'rubberband' | 'ffmpeg' | 'numpy'."""
    forced = os.environ.get("ESCAPE_AUDIO_BACKEND", "").strip().lower()
    if forced in ("rubberband", "ffmpeg", "numpy"):
        return forced
    if _have_rubberband():
        return "rubberband"
    if _have_ffmpeg_rubberband():
        return "ffmpeg"
    return "numpy"


def _run_rubberband(x: np.ndarray, args: list[str], extra_files: dict[str, str] | None = None) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="rb_") as td:
        src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(str(src), as_stereo(x), SR, subtype="FLOAT")
        for name, content in (extra_files or {}).items():
            (Path(td) / name).write_text(content)
        a = [str(Path(td) / f) if f in (extra_files or {}) else f for f in args]
        for engine in (["-3"], []):
            r = subprocess.run(["rubberband", "-q", *engine, *a, str(src), str(dst)], capture_output=True, text=True)
            if r.returncode == 0 and dst.exists():
                y, fs = sf.read(str(dst), dtype="float32", always_2d=True)
                return as_stereo(resample(y, fs, SR))
        raise RuntimeError(f"rubberband a échoué : {r.stderr[-500:]}")


def _run_ffmpeg_filter(x: np.ndarray, af: str) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="ffaf_") as td:
        src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(str(src), as_stereo(x), SR, subtype="FLOAT")
        r = subprocess.run([_ffmpeg(), "-v", "error", "-nostdin", "-y", "-i", str(src), "-af", af,
                            "-ar", str(SR), "-c:a", "pcm_f32le", str(dst)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg -af {af} a échoué : {r.stderr[-500:]}")
        y, fs = sf.read(str(dst), dtype="float32", always_2d=True)
        return as_stereo(resample(y, fs, SR))


def pitch(x, semitones: float, formant: bool = False) -> np.ndarray:
    """Transposition sans changer la durée (même longueur que x)."""
    x = as_stereo(x)
    if abs(semitones) < 1e-6 or len(x) < 64:
        return x.copy()
    be = backend()
    try:
        if be == "rubberband":
            y = _run_rubberband(x, ["-p", f"{semitones:.6f}"] + (["-F"] if formant else []))
            return pad_to(y, len(x))
        if be == "ffmpeg":
            af = f"rubberband=pitch={2 ** (semitones / 12):.8f}:formant={'preserved' if formant else 'shifted'}"
            return pad_to(_run_ffmpeg_filter(x, af), len(x))
    except Exception as e:  # repli silencieux mais signalé
        print(f"[dsp.pitch] {be} indisponible ({e}) -> repli numpy", file=sys.stderr)
    return warp(x, np.arange(len(x), dtype=np.float64), semitones, backend_="numpy")


def stretch(x, ratio: float) -> np.ndarray:
    """Étirement temporel sans changer la hauteur. ratio = durée sortie / durée entrée (2.0 = 2× plus lent)."""
    x = as_stereo(x)
    ratio = float(ratio)
    n_out = int(round(len(x) * ratio))
    if abs(ratio - 1) < 1e-6 or len(x) < 64:
        return pad_to(x, n_out)
    be = backend()
    try:
        if be == "rubberband":
            return pad_to(_run_rubberband(x, ["-t", f"{ratio:.8f}"]), n_out)
        if be == "ffmpeg":
            return pad_to(_run_ffmpeg_filter(x, f"rubberband=tempo={1 / ratio:.8f}"), n_out)
    except Exception as e:
        print(f"[dsp.stretch] {be} indisponible ({e}) -> repli numpy", file=sys.stderr)
    return pad_to(_pv_warp(x, np.arange(n_out) / ratio), n_out)


def warp(x, src_pos, semitones=None, backend_: str | None = None) -> np.ndarray:
    """Déformation temps/hauteur générale, longueur de sortie = len(src_pos).

    out[n] contient le contenu de x à la position source src_pos[n] (échantillons, croissante pour le
    moteur rubberband), transposé de semitones[n] demi-tons (scalaire ou (N,)), SANS couplage
    vitesse/hauteur (ralenti qui garde la hauteur, dérive de hauteur à durée constante...).
    Méthode : étirement (src -> position intermédiaire U) puis lecture varispeed à la vitesse 2^(st/12).
    """
    x = as_stereo(x)
    src_pos = np.asarray(src_pos, dtype=np.float64)
    n = len(src_pos)
    if n == 0:
        return np.zeros((0, 2), np.float32)
    st = np.zeros(n) if semitones is None else np.broadcast_to(np.asarray(semitones, dtype=np.float64), (n,))
    p = np.power(2.0, st / 12.0)
    U = np.concatenate([[0.0], np.cumsum(p)[:-1]])
    n_y = int(np.ceil(U[-1])) + 4
    src_of_u = np.interp(np.interp(np.arange(n_y), U, np.arange(n)), np.arange(n), src_pos)
    if n > 1:  # extrapolation linéaire au-delà de la dernière position (garde la monotonie)
        beyond = np.arange(n_y) > U[-1]
        slope = (src_pos[-1] - src_pos[-2]) / max(U[-1] - U[-2], 1e-9)
        src_of_u[beyond] = src_pos[-1] + (np.arange(n_y)[beyond] - U[-1]) * slope
    ident = np.allclose(src_of_u, np.arange(n_y) + src_of_u[0], atol=0.75)
    if ident:
        y = interp_cubic(x, src_of_u)
    else:
        be = backend_ or backend()
        y = None
        if be == "rubberband" and np.all(np.diff(src_of_u) > 0):
            try:
                y = _warp_rubberband(x, src_of_u)
            except Exception as e:
                print(f"[dsp.warp] rubberband indisponible ({e}) -> repli numpy", file=sys.stderr)
        if y is None:
            y = _pv_warp(x, src_of_u)
    if np.allclose(p, 1.0):
        return pad_to(y, n)
    lp = float(np.max(p))
    if lp > 1.05:  # lecture accélérée : anti-repliement
        y = lowpass(y, 0.45 * SR / lp, order=8, zero_phase=True)
    return interp_cubic(y, U)


def _warp_rubberband(x: np.ndarray, src_of_u: np.ndarray) -> np.ndarray:
    s0 = int(np.floor(src_of_u[0]))
    s1 = int(np.ceil(src_of_u[-1])) + 1
    seg = pad_to(x[max(0, s0):s1], s1 - s0) if s0 >= 0 else np.concatenate(
        [np.zeros((-s0, 2), np.float32), x[:s1]])
    n_y = len(src_of_u)
    step = 1024
    keys, last_s, last_t = [], -1, -1
    for u in list(range(0, n_y, step)) + [n_y - 1]:
        s = int(round(src_of_u[u])) - s0
        if s > last_s and u > last_t:
            keys.append(f"{s} {u}")
            last_s, last_t = s, u
    ratio = n_y / max(len(seg), 1)
    y = _run_rubberband(seg, ["-M", "timemap.txt", "-t", f"{ratio:.8f}"],
                        extra_files={"timemap.txt": "\n".join(keys) + "\n"})
    return pad_to(y, n_y)


# ---------------------------------------------------------------------------
# Effets
# ---------------------------------------------------------------------------
def distort(x, drive: float = 4.0, asym: float = 0.0, oversample: int = 4, mix: float = 1.0) -> np.ndarray:
    """Saturation tanh suréchantillonnée (anti-repliement). asym > 0 : harmoniques paires."""
    x = as_stereo(x)
    if len(x) == 0:
        return x.copy()
    up = signal.resample_poly(x.astype(np.float64), oversample, 1, axis=0)
    y = (np.tanh(drive * (up + asym)) - np.tanh(drive * asym)) / np.tanh(max(drive, 1e-3))
    y = signal.resample_poly(y, 1, oversample, axis=0)[:len(x)]
    y = pad_to(y.astype(np.float32), len(x))
    if asym:
        y = dc_block(y)
    return (mix * y + (1 - mix) * x).astype(np.float32)


def bitcrush(x, bits: float = 6, downsample: int = 4) -> np.ndarray:
    """Quantification + échantillonneur-bloqueur (son numérique dégradé)."""
    x = as_stereo(x)
    q = 2.0 ** (bits - 1)
    y = np.round(x * q) / q
    if downsample > 1:
        idx = (np.arange(len(y)) // downsample) * downsample
        y = y[idx]
    return y.astype(np.float32)


def whisperize(x, seed: int = 0, amount: float = 1.0) -> np.ndarray:
    """Transforme une voix parlée en chuchotement : enveloppe spectrale (cepstre) excitée par du bruit."""
    x = as_stereo(x)
    n = len(x)
    if n < 2048:
        return x.copy()
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 31337])
    nper, nov = 1024, 768
    out = np.zeros((n, 2), np.float32)
    mono = to_mono(x).astype(np.float64)
    _, _, Z = signal.stft(mono, fs=SR, nperseg=nper, noverlap=nov)
    logm = np.log(np.abs(Z) + 1e-9)
    cep = np.fft.irfft(logm, n=nper, axis=0)
    lift = 36
    cep[lift:nper - lift] = 0
    env = np.exp(np.fft.rfft(cep, axis=0).real)
    for c in range(2):
        ph = np.exp(2j * np.pi * rng.random(Z.shape))
        Zw = amount * env * ph + (1 - amount) * Z
        _, y = signal.istft(Zw, fs=SR, nperseg=nper, noverlap=nov)
        out[:, c] = pad_to(y.astype(np.float32), n)[:, 0]
    out = highpass(out, 170, order=2)
    out = shelf(out, 320, -3.0, "low")
    out = peaking(out, 4200, 4.0, 0.8)
    r_in, r_out = rms_db(x), rms_db(out)
    return gain(out, r_in - r_out)


def detune_double(x, cents: float = 15.0, delay_ms: float = 18.0, mix: float = 0.6) -> np.ndarray:
    """Doublage désaccordé : copie +cents retardée à gauche, copie −cents retardée à droite.

    Sortie de longueur len(x) + retard max (≈ 1,37 × delay_ms).
    """
    x = as_stereo(x)
    a = to_mono(pitch(x, cents / 100.0))
    b = to_mono(pitch(x, -cents / 100.0))
    d1 = int(delay_ms * SR / 1000)
    d2 = int(delay_ms * 1.37 * SR / 1000)
    out = np.zeros((len(x) + d2, 2), np.float32)
    out[:len(x)] += x
    out[d1:d1 + len(a), 0] += mix * a
    out[d2:d2 + len(b), 1] += mix * b
    return out


def stutter(x, grain_s: float, repeats: int) -> np.ndarray:
    """Répète le début de x (grain) `repeats` fois, bords adoucis (bégaiement)."""
    g = fade(as_stereo(x)[:n_samples(grain_s)], 0.0005, 0.0015)
    return np.concatenate([g] * max(1, repeats))


# ---------------------------------------------------------------------------
# Mesures : loudness BS.1770, true peak
# ---------------------------------------------------------------------------
# Pondération K à 48 kHz (ITU-R BS.1770-4) : pré-filtre en étagère + passe-haut RLB.
_KW_SOS = np.array([
    [1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585],
    [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621],
])


def lufs(x) -> float:
    """Loudness intégrée BS.1770 (pyloudnorm). Silence -> -70."""
    import pyloudnorm as pyln

    x = as_stereo(x).astype(np.float64)
    if len(x) < int(0.4 * SR) + 1:
        x = np.concatenate([x, np.zeros((int(0.4 * SR) + 1 - len(x), 2))])
    v = pyln.Meter(SR).integrated_loudness(x)
    return float(v) if np.isfinite(v) else -70.0


def loudness_curve(x, win: float = 3.0, hop: float = 0.1):
    """Loudness glissante non fenêtrée (BS.1770) : renvoie (t_fin_fenêtre, LUFS).

    win=3.0 : court terme (S) ; win=0.4 : momentané (M). Fenêtres partielles au début complétées
    par du silence (convention ffmpeg ebur128).
    """
    x = as_stereo(x).astype(np.float64)
    y = signal.sosfilt(_KW_SOS, x, axis=0)
    p = np.sum(y ** 2, axis=1)
    cs = np.concatenate([[0.0], np.cumsum(p)])
    W, H = int(win * SR), max(1, int(hop * SR))
    ends = np.arange(H, len(p) + 1, H)
    if len(ends) == 0:
        ends = np.array([len(p)])
    ms = (cs[ends] - cs[np.maximum(ends - W, 0)]) / W
    return ends / SR, -0.691 + 10 * np.log10(ms + 1e-20)


def short_term(x):
    return loudness_curve(x, 3.0, 0.1)


def momentary(x):
    return loudness_curve(x, 0.4, 0.1)


@functools.lru_cache(maxsize=1)
def _os4_filter() -> np.ndarray:
    """Filtre d'interpolation ×4 (96 coefficients, Kaiser) découpé en 4 phases (4, 24)."""
    h = signal.firwin(96, 1.0 / 4.0, window=("kaiser", 9.0)) * 4.0
    return h.reshape(24, 4).T.copy()


def _os4_abs_max(x) -> np.ndarray:
    """|x| suréchantillonné ×4, max par échantillon d'origine (phases intermédiaires), canaux liés."""
    x = as_stereo(x)
    n = len(x)
    ph = _os4_filter()
    env = np.abs(x).max(axis=1).astype(np.float32)
    d = 12  # retard de groupe ≈ (96-1)/2/4 échantillons
    for k in range(4):
        for c in range(2):
            y = signal.oaconvolve(x[:, c], ph[k].astype(np.float32))[d:d + n]
            np.maximum(env[:len(y)], np.abs(y), out=env[:len(y)])
    return env


def _tp_envelope(x) -> np.ndarray:
    """Crête vraie estimée par échantillon (points suréchantillonnés ×4 autour, canaux liés)."""
    env = _os4_abs_max(x)
    return np.maximum(env, np.concatenate([env[1:], env[-1:]]))  # point inter-échantillon vers n+1


def true_peak_db(x) -> float:
    """Crête vraie (dBTP) par suréchantillonnage ×4 (filtre polyphase 96 coefficients)."""
    x = as_stereo(x)
    if len(x) == 0:
        return -240.0
    return float(db(np.max(_os4_abs_max(x))))


def lufs_fast(x) -> float:
    """Loudness intégrée BS.1770-4 vectorisée (portes absolue −70 et relative −10). ≈ pyloudnorm."""
    x = as_stereo(x).astype(np.float64)
    if len(x) < int(0.4 * SR):
        x = np.concatenate([x, np.zeros((int(0.4 * SR) - len(x) + 1, 2))])
    y = signal.sosfilt(_KW_SOS, x, axis=0)
    p = np.sum(y ** 2, axis=1)
    cs = np.concatenate([[0.0], np.cumsum(p)])
    W, H = int(0.4 * SR), int(0.1 * SR)
    starts = np.arange(0, len(p) - W + 1, H)
    z = (cs[starts + W] - cs[starts]) / W
    lk = -0.691 + 10 * np.log10(z + 1e-20)
    z = z[lk > -70]
    if len(z) == 0:
        return -70.0
    rel = -0.691 + 10 * np.log10(z.mean()) - 10
    z2 = z[-0.691 + 10 * np.log10(z + 1e-20) > rel]
    return float(-0.691 + 10 * np.log10(z2.mean())) if len(z2) else -70.0


def _release_env(h: np.ndarray, tau: float) -> np.ndarray:
    """e[n] = max(h[n], e[n-1]·exp(-1/tau)) vectorisé (astuce log + cummax)."""
    k = np.arange(len(h), dtype=np.float64) / max(tau, 1.0)
    lg = np.log(h + 1e-9) + k
    return np.exp(np.maximum.accumulate(lg) - k)


def limit(x, ceiling_db: float = -1.0, lookahead_ms: float = 5.0, release_ms: float = 80.0) -> np.ndarray:
    """Limiteur true-peak à anticipation (détection suréchantillonnée ×4, stéréo lié).

    Gain calculé sur les échantillons à venir (hors-ligne : pas de retard), attaque lissée sur la
    fenêtre d'anticipation, relâche exponentielle. Itère si un dépassement inter-échantillon subsiste.
    """
    x = as_stereo(x)
    if len(x) == 0:
        return x.copy()
    L = max(1, int(lookahead_ms * SR / 1000))
    target = ceiling_db
    y = x
    env = _tp_envelope(x)
    for _ in range(4):
        gt = np.minimum(1.0, lin(target) / np.maximum(env, 1e-12))
        gr = -20 * np.log10(gt)  # réduction requise (dB, >= 0)
        if gr.max() <= 1e-4:
            y = x.copy()
        else:
            h = _exact_forward_max(gr, L)
            e = _release_env(h, release_ms * SR / 1000)
            s = _backward_mean(e, L + 1)
            y = (x * lin(-s)[:, None]).astype(np.float32)
        over = true_peak_db(y) - ceiling_db
        if over <= 0.005:
            break
        target -= over + 0.01
    return y.astype(np.float32)


def _exact_forward_max(a: np.ndarray, L: int) -> np.ndarray:
    """max(a[n..n+L]) par doublement de fenêtre (O(N log L)), sans dépendre de `origin`."""
    n = len(a)
    res = a.copy()
    span = 1  # res[n] = max(a[n .. n+span-1])
    while span < L + 1:
        step = min(span, L + 1 - span)
        sh = np.concatenate([res[step:], np.zeros(step)])
        res = np.maximum(res, sh)
        span += step
    return res[:n]


def _backward_mean(a: np.ndarray, w: int) -> np.ndarray:
    """mean(a[n-w+1..n]) (moyenne glissante causale, début complété par a[0])."""
    if len(a) == 0:
        return a.copy()
    ap = np.concatenate([np.full(w - 1, a[0]), a])
    cs = np.concatenate([[0.0], np.cumsum(ap)])
    return (cs[w:] - cs[:-w]) / w


_LAST_MASTER_GAIN: dict = {}


def master(x, lufs_target: float = -14.0, tp: float = -1.0, max_iter: int = 8, g0: float | None = None,
           **kw) -> np.ndarray:
    """Master : gain vers `lufs_target` LUFS intégrés + limiteur true-peak à `tp` dBTP (itératif).

    Alias accepté : master(x, lufs=-14, tp=-1). g0 : gain de départ (dB) pour converger plus vite ;
    le gain final est disponible dans dsp.last_master_gain().
    """
    if "lufs" in kw:
        lufs_target = float(kw.pop("lufs"))
    x = as_stereo(x)
    l0 = lufs_fast(x)
    if l0 <= -69.0:
        return x.copy()
    g = (lufs_target - l0) if g0 is None else float(g0)
    y = x
    for _ in range(max_iter):
        y = limit(gain(x, g), tp, **kw)
        err = lufs_target - lufs_fast(y)
        if abs(err) < 0.03:
            break
        g += err * 1.15
    _LAST_MASTER_GAIN["g"] = g
    return y.astype(np.float32)


def last_master_gain() -> float | None:
    return _LAST_MASTER_GAIN.get("g")


def stats(x) -> dict:
    """Mesures rapides : LUFS intégrés, dBTP, crête, RMS, DC."""
    x = as_stereo(x)
    return {"lufs": round(lufs(x), 2), "true_peak_db": round(true_peak_db(x), 2), "peak_db": round(peak_db(x), 2),
            "rms_db": round(rms_db(x), 2), "dc": float(np.abs(x.mean(axis=0)).max()) if len(x) else 0.0,
            "dur": round(len(x) / SR, 3)}
