"""Bibliothèque de sons 100 % synthétisés (libre de droits, publiable TikTok/Instagram).

Chaque entrée de CATALOG : f(dur=None, seed=0, **params) -> np.ndarray float32 (N, 2) à 48 kHz.
- dur=None : durée par défaut sensée ; sinon la sortie fait EXACTEMENT round(dur * 48000) échantillons.
- Sons « montants » (riser, reverse_swell, whoosh_rev) : leur fin coïncide avec dur (pour caler un impact).
- Déterministe : même (nom, dur, seed, params) -> mêmes échantillons.
- Niveaux : crête normalisée vers −1 dBFS (impacts −0,5, screamer −0,3) ; le `gain_db` des cues de
  l'EDL est relatif à cette référence. Les paramètres inconnus sont ignorés.

API : CATALOG, render(name, dur=None, seed=0, **params), names(), music_box(notes, ...).
"""
from __future__ import annotations

import functools
import zlib

import numpy as np
from scipy.interpolate import CubicSpline

from . import dsp
from .dsp import SR

# ---------------------------------------------------------------------------
# Outils de synthèse
# ---------------------------------------------------------------------------


def _n(dur: float) -> int:
    return max(1, int(round(float(dur) * SR)))


def _t(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / SR


def _rng(seed, salt: str) -> np.random.Generator:
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, zlib.crc32(salt.encode("utf-8"))])


def _lfo(n: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Modulateur aléatoire lisse dans ~[-1, 1] (points aléatoires à `rate` Hz, spline cubique)."""
    k = max(5, int(np.ceil(n / SR * rate)) + 5)
    pts = rng.uniform(-1, 1, k)
    xs = np.arange(n) / SR * rate + 1.0
    return np.clip(CubicSpline(np.arange(k), pts)(xs), -1.3, 1.3)


def _sin(f, n: int | None = None, phase0: float = 0.0) -> np.ndarray:
    f = np.broadcast_to(np.asarray(f, dtype=np.float64), (n,) if n is not None else np.shape(f))
    return np.sin(phase0 + 2 * np.pi * np.cumsum(f) / SR)


def _saw(f, n: int | None = None, phase0: float = 0.0) -> np.ndarray:
    """Dent de scie à bande limitée (PolyBLEP)."""
    f = np.broadcast_to(np.asarray(f, dtype=np.float64), (n,) if n is not None else np.shape(f))
    dt = np.clip(f / SR, 1e-7, 0.5)
    ph = (phase0 + np.cumsum(dt)) % 1.0
    y = 2.0 * ph - 1.0
    m1 = ph < dt
    t1 = ph[m1] / dt[m1]
    y[m1] -= t1 + t1 - t1 * t1 - 1.0
    m2 = ph > 1.0 - dt
    t2 = (ph[m2] - 1.0) / dt[m2]
    y[m2] -= t2 * t2 + t2 + t2 + 1.0
    return y


def _square(f, n: int | None = None, phase0: float = 0.0) -> np.ndarray:
    return 0.5 * (_saw(f, n, phase0) - _saw(f, n, phase0 + 0.5))


def _noise(n: int, rng, color: str = "white") -> np.ndarray:
    w = rng.standard_normal(n)
    if color == "white":
        return w
    spec = np.fft.rfft(w)
    fr = np.fft.rfftfreq(n, 1 / SR)
    fr[0] = fr[1] if n > 1 else 1.0
    spec /= np.sqrt(fr) if color == "pink" else fr  # brown : 1/f²
    y = np.fft.irfft(spec, n)
    return y / (np.std(y) + 1e-12)


def _env(t: np.ndarray, attack: float, tau: float) -> np.ndarray:
    return (1 - np.exp(-t / max(attack, 1e-5))) * np.exp(-t / max(tau, 1e-5))


def _norm(x: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(x))) if np.size(x) else 0.0
    return x / m if m > 1e-12 else x


def _st(mono: np.ndarray) -> np.ndarray:
    return dsp.as_stereo(np.asarray(mono, dtype=np.float32))


def _pan(mono, p) -> np.ndarray:
    """Panoramique à puissance constante (p scalaire ou courbe (N,) dans [-1, 1])."""
    m = np.asarray(mono, dtype=np.float64)
    a = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return (np.stack([m * np.cos(a), m * np.sin(a)], 1) * np.sqrt(2)).astype(np.float32)


@functools.lru_cache(maxsize=16)
def _decor_fir(seed: int, length_ms: float = 22.0) -> np.ndarray:
    rng = np.random.default_rng([seed, 4242])
    L = int(length_ms * SR / 1000)
    h = rng.standard_normal(L) * np.exp(-np.arange(L) / (L / 4))
    h -= h.mean()
    return h / np.sqrt(np.sum(h ** 2))


def _widen(mono, amount: float = 0.6, seed: int = 0) -> np.ndarray:
    """Stéréo par M/S : côté = version décorrélée (compatibilité mono parfaite : L+R = 2·M)."""
    m = np.asarray(mono, dtype=np.float64)
    s = np.convolve(m, _decor_fir(int(seed) % 997), mode="full")[:len(m)] * amount
    return np.stack([m + s, m - s], 1).astype(np.float32)


def _finish(x, n: int, peak: float = -1.0, fi: float = 0.0005, fo: float = 0.02, hp: float = 18.0) -> np.ndarray:
    """Longueur exacte, coupe-DC, fondus anti-clic, normalisation de crête."""
    y = dsp.pad_to(dsp.as_stereo(x), n)
    if hp:
        y = dsp.highpass(y, hp, order=2)
    y = dsp.fade(y, fi, min(fo, len(y) / SR * 0.5))
    y = dsp.normalize(y, peak) if peak is not None else y
    return np.nan_to_num(y).astype(np.float32)


def _hall(seed=0, decay=3.0, damping=0.55, predelay=0.025):
    return dsp.make_ir(decay, predelay=predelay, damping=damping, seed=seed)


# ---------------------------------------------------------------------------
# Impacts et basses
# ---------------------------------------------------------------------------


def impact_big(dur=None, seed=0, tail=3.0, **_):
    """Gros impact de bande-annonce : sub 45→28 Hz, thud, transitoire, métal inharmonique, réverbe 3 s."""
    dur = 5.0 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    rng = _rng(seed, "impact_big")
    f = 28 + 17 * np.exp(-t / 0.25)
    sub = _sin(f, n) * _env(t, 0.003, 0.62)
    sub = np.tanh(2.0 * sub) / np.tanh(2.0)  # harmoniques : le sub reste perceptible sur téléphone
    punch = _sin(55 + 75 * np.exp(-t / 0.04), n) * _env(t, 0.001, 0.22)
    thud = dsp.to_mono(dsp.lowpass(_noise(n, rng), 300, order=4))
    thud = _norm(thud) * _env(t, 0.0015, 0.2)
    body = _norm(dsp.to_mono(dsp.bandpass(_noise(n, rng), (150, 700), order=2))) * _env(t, 0.001, 0.12)
    crack = _norm(dsp.to_mono(dsp.bandpass(_noise(n, rng), (700, 4500), order=2))) * _env(t, 0.0004, 0.05)
    trans = _norm(dsp.to_mono(dsp.highpass(_noise(n, rng), 4000, order=4))) * np.exp(-t / 0.0016)
    trans[t > 0.005] = 0
    ratios = np.array([1.0, 1.593, 2.136, 2.296, 2.653, 3.402, 3.938, 4.567, 5.218, 6.07])
    f0 = 148 * (1 + 0.04 * rng.uniform(-1, 1))
    bend = 1 - 0.018 * (1 - np.exp(-t / 1.2))
    metal = np.zeros((n, 2))
    for k, r in enumerate(ratios):
        part = _sin(f0 * r * bend * (1 + 0.002 * rng.uniform(-1, 1)), n, rng.uniform(0, 6.28))
        part *= _env(t, 0.002, 1.7 / (1 + 0.32 * k)) / (1 + 0.45 * k)
        metal += _pan(part, (-1) ** k * rng.uniform(0.25, 0.8))
    metal /= np.max(np.abs(metal)) + 1e-9
    dry = (_st(sub * 0.7 + punch * 0.85) + _widen(thud * 0.9 + body * 0.75, 0.3, seed)
           + _widen(crack * 0.8 + trans * 0.55, 0.5, seed + 1) + metal * 0.32)
    send = _widen(thud * 0.6 + body * 0.6 + crack * 0.8 + punch * 0.3, 0.45, seed + 2) + metal * 0.45
    wet = dsp.reverb(send, _hall(seed, max(0.5, float(tail)), 0.55), wet=1.0, tail=False)
    y = dry + wet * 0.5
    y = dsp.peaking(y, 140, 3.0, 0.9)
    y = dsp.distort(y, 2.0)
    return _finish(y, n, peak=-0.5, fo=min(0.4, dur * 0.2))


def impact_hit(dur=None, seed=0, **_):
    """Impact court et percutant (0,8 s)."""
    dur = 0.8 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    rng = _rng(seed, "impact_hit")
    sub = _sin(42 + 38 * np.exp(-t / 0.05), n) * _env(t, 0.002, 0.16)
    punch = _sin(90 + 110 * np.exp(-t / 0.02), n) * _env(t, 0.0008, 0.06)
    thud = _norm(dsp.to_mono(dsp.lowpass(_noise(n, rng), 420, order=4))) * _env(t, 0.001, 0.07)
    crack = _norm(dsp.to_mono(dsp.bandpass(_noise(n, rng), (900, 5500), order=2))) * _env(t, 0.0003, 0.028)
    trans = _norm(dsp.to_mono(dsp.highpass(_noise(n, rng), 5000, order=4))) * np.exp(-t / 0.0012)
    trans[t > 0.005] = 0
    dry = _st(np.tanh(1.8 * (sub * 0.9 + punch * 0.7))) + _widen(thud * 0.55 + crack * 0.5 + trans * 0.5, 0.45, seed)
    wet = dsp.reverb(_widen(crack * 0.6 + thud * 0.4, 0.5, seed + 3), dsp.make_ir(0.8, 0.008, 0.6, seed), 1.0, tail=False)
    return _finish(dry + 0.28 * wet, n, peak=-0.5, fo=min(0.15, dur * 0.25))


def impact_sub(dur=None, seed=0, **_):
    """Thump sub 40 Hz qui chute (1,5 s)."""
    dur = 1.5 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    rng = _rng(seed, "impact_sub")
    f = 26 + 16 * np.exp(-t / 0.35)
    sub = _sin(f, n) * _env(t, 0.004, 0.42)
    sub = np.tanh(2.4 * sub) / np.tanh(2.4)
    knock = _norm(dsp.to_mono(dsp.lowpass(_noise(n, rng), 180, order=4))) * _env(t, 0.002, 0.05)
    y = _st(sub + 0.35 * knock)
    y += 0.12 * dsp.reverb(y, dsp.make_ir(1.2, 0.01, 0.7, seed), 1.0, tail=False)
    return _finish(y, n, peak=-1.0, fo=min(0.3, dur * 0.25))


def sub_drop(dur=None, seed=0, f_start=60.0, f_end=25.0, **_):
    """Sub-drop : balayage exponentiel 60→25 Hz, 2 s."""
    dur = 2.0 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    p = t / max(dur, 1e-3)
    f = f_start * (f_end / f_start) ** np.clip(p * 1.15, 0, 1)
    amp = (1 - np.exp(-t / 0.006)) * np.exp(-3.2 * p ** 1.6)
    y = _sin(f, n) * amp
    y = np.tanh(2.2 * y) / np.tanh(2.2)
    air = _norm(dsp.to_mono(dsp.lowpass(_noise(n, _rng(seed, "sub_drop")), 140, order=2))) * amp * 0.08
    return _finish(_st(y + air), n, peak=-1.0, fo=min(0.25, dur * 0.2))


# ---------------------------------------------------------------------------
# Braams, risers, swells, whooshes
# ---------------------------------------------------------------------------


def _braam_core(n: int, t: np.ndarray, rng, amp: np.ndarray, cutoff: np.ndarray, seed: int) -> np.ndarray:
    voices = [(55.0, 1.0), (82.41, 0.75), (110.0, 0.6), (164.8, 0.22)]
    out = np.zeros((n, 2))
    for i, (f0, a) in enumerate(voices):
        for j, cents in enumerate((-13, -5, 4, 12)):
            fv = f0 * 2 ** (cents / 1200) * (1 + 0.0015 * _lfo(n, 0.7, rng))
            v = _saw(fv, n, rng.uniform()) * a
            out += _pan(v, (-0.7 + 0.47 * j) * (1 if i % 2 == 0 else -1))
    out *= 1 + 0.16 * np.sin(2 * np.pi * 27 * t)[:, None]  # rugosité cuivrée
    out = dsp.tv_filter(out, "lowpass", cutoff, q=1.25, block=64)
    out = dsp.tv_filter(out, "lowpass", cutoff * 1.6, q=0.7, block=64)
    out = dsp.distort(out * amp[:, None] * 0.45, 3.2)
    breath = _norm(dsp.to_mono(dsp.bandpass(_noise(n, rng), (300, 1800), order=2))) * _env(t, 0.01, 0.12) * 0.25
    sub = _sin(27.5, n) * amp * 0.45 + _sin(55.0, n) * amp * 0.3
    return out + _widen(breath, 0.5, seed) + _st(sub)


def braam(dur=None, seed=0, **_):
    """BRAAAM de bande-annonce : scies graves désaccordées (55 Hz + quinte + octave), filtre qui s'ouvre/se ferme."""
    dur = 3.0 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    rng = _rng(seed, "braam")
    rel = max(0.4, dur * 0.3)
    amp = (1 - np.exp(-t / 0.028)) * (0.72 + 0.28 * np.exp(-t / 0.5)) * np.clip((dur - t) / rel, 0, 1) ** 1.5
    cutoff = 170 + 2400 * (1 - np.exp(-t / 0.09)) * np.exp(-t / 0.75) + 180 * np.exp(-t / 3)
    y = _braam_core(n, t, rng, amp, cutoff, seed)
    y = dsp.reverb(y, _hall(seed, 3.5, 0.6), wet=0.3, tail=False)
    return _finish(y, n, peak=-1.0, fo=0.05)


def braam_long(dur=None, seed=0, **_):
    """Braam de 5 s avec swell (crescendo puis extinction)."""
    dur = 5.0 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    p = t / dur
    rng = _rng(seed, "braam_long")
    amp = (1 - np.exp(-t / 0.08)) * (0.45 + 0.55 * np.sin(np.pi * np.clip(p / 0.9, 0, 1)) ** 1.2)
    amp *= np.clip((dur - t) / (dur * 0.25), 0, 1) ** 1.3
    cutoff = 160 + 1900 * np.sin(np.pi * np.clip(p / 0.85, 0, 1)) ** 2 + 400 * np.exp(-t / 0.15)
    y = _braam_core(n, t, rng, amp, cutoff, seed)
    y = dsp.reverb(y, _hall(seed, 4.0, 0.6), wet=0.32, tail=False)
    return _finish(y, n, peak=-1.0, fo=0.08)


def riser(dur=None, seed=0, **_):
    """Montée : tons de Shepard + cluster de cordes qui monte + bruit, trémolo qui accélère, fin nette à dur."""
    dur = 4.0 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    p = t / max(dur, 1e-3)
    rng = _rng(seed, "riser")
    K, base, rise = 8, 30.0, 1.6
    shep = np.zeros(n)
    for k in range(K):
        x = (k + rise * p) % K
        f = base * 2 ** x
        g = np.exp(-0.5 * ((x - 4.2) / 1.25) ** 2)
        shep += _sin(f, n, rng.uniform(0, 6.28)) * g
    shep = _norm(shep)
    cl = np.zeros((n, 2))
    pm = 2 ** (8 * p ** 1.4 / 12)
    for i, f0 in enumerate((220.0, 233.08, 329.63, 349.23, 440.0, 466.16)):
        v = _saw(f0 * pm * 2 ** (rng.uniform(-8, 8) / 1200), n, rng.uniform())
        cl += _pan(v, -0.8 + 0.32 * i)
    cl = dsp.tv_filter(cl, "lowpass", 700 + 7000 * p ** 1.8, q=0.9, block=128)
    rate = 3.5 + 24 * p ** 1.7
    trem = 1 - 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(rate) / SR))
    noise = dsp.tv_filter(_widen(_noise(n, rng), 0.8, seed), "bandpass", 300 * (28 ** p), q=1.1, block=128)
    y = (_st(shep) * (0.55 * (0.25 + 0.75 * p))[:, None] + cl * (0.32 * trem)[:, None]
         + noise * (0.9 * p ** 1.5)[:, None])
    y *= (0.03 + 0.97 * p ** 2.2)[:, None]
    y = dsp.reverb(y, dsp.make_ir(1.4, 0.015, 0.5, seed), wet=0.22, tail=False)
    return _finish(y, n, peak=-1.0, fi=0.05, fo=0.004)


def reverse_swell(dur=None, seed=0, **_):
    """Swell de réverbe inversée qui culmine et se termine net à dur."""
    dur = 2.0 if dur is None else dur
    n = _n(dur)
    rng = _rng(seed, "reverse_swell")
    m = _n(0.35)
    tt = _t(m)
    hit = np.zeros((m, 2))
    for i, f0 in enumerate((110.0, 164.81, 220.0, 233.08, 329.63)):
        hit += _pan(_saw(f0 * 2 ** (rng.uniform(-10, 10) / 1200), m, rng.uniform()), -0.6 + 0.3 * i)
    hit = dsp.lowpass(hit, 2600) * _env(tt, 0.002, 0.09)[:, None] * 0.4
    hit += _widen(_norm(dsp.to_mono(dsp.highpass(_noise(m, rng), 2500))) * _env(tt, 0.001, 0.1), 0.8, seed) * 0.5
    ir = dsp.make_ir(max(3.0, dur * 1.4), 0.0, 0.35, seed, early=False)
    w = dsp.reverb(hit, ir, wet=1.0, tail=True)[::-1]
    y = dsp.pad_to(w[-n:] if len(w) >= n else np.concatenate([np.zeros((n - len(w), 2), np.float32), w]), n)
    cym = _norm(dsp.to_mono(dsp.bandpass(_noise(n, rng), (2500, 8000)))) * np.exp(-(dur - _t(n)) / max(0.15, dur * 0.3))
    y = dsp.lowpass(dsp.normalize(y, 0), 9000, order=2) + _widen(cym, 0.7, seed + 1) * 0.12
    return _finish(y, n, peak=-1.0, fi=min(0.3, dur * 0.3), fo=0.003)


def _whoosh_core(n, t, rng, env, fc, pan_curve, seed):
    nz = _noise(n, rng)
    a = dsp.to_mono(dsp.tv_filter(nz, "bandpass", fc, q=1.3, block=64))
    b = dsp.to_mono(dsp.tv_filter(_noise(n, rng), "lowpass", np.maximum(fc * 0.35, 120), q=0.8, block=64))
    y = _norm(a) * env + 0.5 * _norm(b) * env
    return _pan(y, pan_curve)


def whoosh(dur=None, seed=0, **_):
    """Souffle de passage (bande de bruit qui balaie, panoramique gauche→droite)."""
    dur = 0.6 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    p = t / dur
    rng = _rng(seed, "whoosh")
    pk = 0.55
    g = np.log(0.5) / np.log(pk)
    env = np.sin(np.pi * np.clip(p, 0, 1) ** g) ** 2
    fc = 280 + 3200 * env ** 1.3
    y = _whoosh_core(n, t, rng, env, fc, -0.8 + 1.6 * p, seed)
    y = dsp.reverb(y, dsp.make_ir(0.7, 0.01, 0.5, seed), wet=0.18, tail=False)
    return _finish(y, n, peak=-1.0, fi=0.005, fo=0.02)


def whoosh_rev(dur=None, seed=0, **_):
    """Souffle inversé qui aspire et se coupe net à dur (entrée de coupe)."""
    dur = 0.5 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    p = t / dur
    rng = _rng(seed, "whoosh_rev")
    env = p ** 3.2
    fc = 350 + 5500 * p ** 2
    y = _whoosh_core(n, t, rng, env, fc, 0.6 - 1.2 * p, seed)
    ir = dsp.make_ir(0.6, 0.0, 0.4, seed, early=False)
    y = y + 0.4 * dsp.reverse_reverb(y, ir, wet=1.0)[-n:]
    return _finish(y, n, peak=-1.0, fi=0.01, fo=0.003)


def freeze_thump(dur=None, seed=0, **_):
    """Arrêt sur image : thump grave + clic d'obturateur mécanique."""
    dur = 0.6 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    rng = _rng(seed, "freeze_thump")
    th = _sin(45 + 45 * np.exp(-t / 0.03), n) * _env(t, 0.0015, 0.1)
    th = np.tanh(2 * th)
    body = _norm(dsp.to_mono(dsp.lowpass(_noise(n, rng), 250))) * _env(t, 0.001, 0.04) * 0.4
    sh = np.zeros(n)
    for t0, f, a in ((0.0, 4300, 1.0), (0.052, 3600, 0.75), (0.058, 1300, 0.5)):
        k = int(t0 * SR)
        m = n - k
        if m <= 0:
            continue
        tt = _t(m)
        burst = _noise(m, rng) * np.exp(-tt / 0.0012)
        ring = dsp.to_mono(dsp.resonator(burst, f, q=9)) * 3
        sh[k:] += a * (_norm(dsp.to_mono(dsp.highpass(burst, 2500))) * 0.6 + ring) * np.exp(-tt / 0.012)
    y = _st(th * 0.9 + body) + _widen(_norm(sh) * 0.32, 0.3, seed)
    y = dsp.reverb(y, dsp.make_ir(0.45, 0.005, 0.6, seed), wet=0.14, tail=False)
    return _finish(y, n, peak=-1.0, fo=min(0.1, dur * 0.3))


# ---------------------------------------------------------------------------
# Glitchs, bande, HUD
# ---------------------------------------------------------------------------


def _glitch_slices(n, rng, seed, n_slices, long=False):
    out = np.zeros((n, 2))
    bounds = np.sort(rng.choice(np.arange(int(0.006 * SR), n - int(0.004 * SR)), n_slices - 1, replace=False))
    bounds = np.concatenate([[0], bounds, [n]])
    buzz_f = rng.uniform(70, 260)
    for i in range(len(bounds) - 1):
        a, b = int(bounds[i]), int(bounds[i + 1])
        m = b - a
        if m < 16:
            continue
        kind = rng.choice(["stutter", "crush", "beep", "buzz", "drop", "sweep"] if long else
                          ["stutter", "crush", "beep", "buzz", "drop"], p=None)
        if kind == "stutter":
            gl = rng.integers(int(0.003 * SR), int(0.012 * SR))
            src = _noise(gl, rng) * 0.5 + _square(rng.uniform(300, 1500), gl) * 0.5
            seg = np.tile(src, m // gl + 1)[:m]
            seg = dsp.to_mono(dsp.bitcrush(seg, rng.integers(3, 6), 1))
        elif kind == "crush":
            seg = dsp.to_mono(dsp.bitcrush(_noise(m, rng) * 0.7 + _sin(rng.uniform(150, 900), m) * 0.4,
                                           rng.uniform(2.5, 5), int(rng.integers(4, 14))))
        elif kind == "beep":
            seg = _sin(rng.uniform(1800, 6500), m) * 0.6 + _square(rng.uniform(900, 2500), m) * 0.2
        elif kind == "buzz":
            seg = _square(buzz_f * rng.choice([1, 2, 0.5]), m) * 0.8 + _saw(buzz_f * 3.01, m) * 0.2
        elif kind == "sweep":
            seg = _sin(np.geomspace(rng.uniform(2000, 6000), rng.uniform(80, 300), m), m) * 0.7
            seg = dsp.to_mono(dsp.bitcrush(seg, 4, 3))
        else:
            seg = np.zeros(m)
        ramp = min(m // 2, int(0.0004 * SR))
        if ramp > 0:
            r = np.linspace(0, 1, ramp)
            seg[:ramp] *= r
            seg[-ramp:] *= r[::-1]
        seg *= rng.uniform(0.45, 1.0)
        out[a:b] = _pan(seg, rng.uniform(-0.8, 0.8))
    return out


def glitch_short(dur=None, seed=0, **_):
    """Bégaiement numérique bitcrushé (0,15 s)."""
    dur = 0.15 if dur is None else dur
    n = _n(dur)
    rng = _rng(seed, "glitch_short")
    y = _glitch_slices(n, rng, seed, int(rng.integers(5, 8)))
    y = dsp.lowpass(y, 14000)
    return _finish(y, n, peak=-1.0, fi=0.0008, fo=0.004, hp=60)


def glitch_long(dur=None, seed=0, **_):
    """Corruption numérique longue (0,8 s) : bégaiements, bourdonnements, balayages, coupures."""
    dur = 0.8 if dur is None else dur
    n = _n(dur)
    rng = _rng(seed, "glitch_long")
    y = _glitch_slices(n, rng, seed, int(rng.integers(18, 28)), long=True)
    t = _t(n)
    hum = _square(50.0, n) * 0.15 * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 7.3 * t)))
    y += _st(hum)
    y = dsp.lowpass(y, 14000)
    return _finish(y, n, peak=-1.0, fi=0.0008, fo=0.01, hp=50)


def vhs_click(dur=None, seed=0, **_):
    """Clic de touche de magnétoscope/cassette + arrêt de bande."""
    dur = 0.5 if dur is None else dur
    n = _n(dur)
    rng = _rng(seed, "vhs_click")
    clk = np.zeros(n)
    for t0, f, a in ((0.0, 2200, 1.0), (0.034, 1500, 0.8), (0.04, 3400, 0.4)):
        k = int(t0 * SR)
        if k >= n:
            continue
        tt = _t(n - k)
        b = _noise(n - k, rng) * np.exp(-tt / 0.0015)
        clk[k:] += a * (dsp.to_mono(dsp.resonator(b, f, 6)) * 3 + dsp.to_mono(dsp.highpass(b, 3000)) * 0.4)
        clk[k:] += a * 0.6 * _sin(150, n - k) * _env(tt, 0.0005, 0.012)
    k0 = min(int(0.03 * SR), max(0, n - 1))
    tt = _t(n - k0)
    slow = np.exp(-tt / 0.09)  # moteur qui s'arrête
    motor = (_saw(60 * slow, n - k0) * 0.3 + _sin(1000 * slow, n - k0) * 0.25) * slow ** 0.5
    motor = dsp.to_mono(dsp.lowpass(motor, 4000))
    hiss = _norm(dsp.to_mono(dsp.bandpass(_noise(n - k0, rng), (2000, 9000)))) * 0.08 * np.exp(-tt / 0.12)
    y = _st(_norm(clk) * 0.9)
    tail = np.zeros(n)
    tail[k0:] = motor + hiss
    y += _widen(tail, 0.4, seed)
    return _finish(y, n, peak=-1.0, fo=0.03)


def tinnitus(dur=None, seed=0, ramp_to_db=None, fade_in=0.25, freq=4000.0, **_):
    """Acouphène : sinus ~4 kHz avec léger battement.

    ramp_to_db : niveau final RELATIF au début (dB) atteint à dur (ex. +13). Le mixer convertit le
    `ramp_to_db` absolu des cues (−17 pour une cue à −30) en relatif.
    """
    dur = 2.0 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    rng = _rng(seed, "tinnitus")
    f = freq + 12 * _lfo(n, 0.4, rng)
    y = _sin(f, n) + 0.2 * _sin(f + 4.3, n, 1.0) + 0.1 * _sin(freq * 0.955, n, 2.0)
    hiss = _norm(dsp.to_mono(dsp.bandpass(_noise(n, rng), (6000, 9500)))) * 0.03
    y = _norm(y) + hiss
    lvl = np.zeros(n) if ramp_to_db is None else float(ramp_to_db) * (t / dur) ** 1.3
    y = y * dsp.lin(lvl)
    out = _st(y * 0.89)  # référence : 0 dB relatif = crête −1 dBFS
    out = dsp.fade(out, min(fade_in, dur * 0.4), 0.006)
    return dsp.pad_to(out, n).astype(np.float32)


def heartbeat(dur=None, seed=0, bpm_start=60.0, bpm_end=None, bpm=None, **_):
    """Battements de cœur lub-dub graves, tempo bpm_start→bpm_end sur la durée."""
    dur = 6.0 if dur is None else dur
    n = _n(dur)
    rng = _rng(seed, "heartbeat")
    b0 = float(bpm if bpm is not None else bpm_start)
    b1 = float(b0 if bpm_end is None else bpm_end)
    t = _t(n)
    rate = (b0 + (b1 - b0) * t / dur) / 60.0
    ph = np.cumsum(rate) / SR
    beats = np.flatnonzero(np.diff(np.floor(ph + 0.97)) > 0) + 1
    beats = np.concatenate([[int(0.02 * SR)], beats]) if len(beats) == 0 or beats[0] > 0.3 * SR else beats
    out = np.zeros(n)
    L = _n(0.45)
    tt = _t(L)

    def thump(f_hi, f_lo, tau, amp):
        s = _sin(f_lo + (f_hi - f_lo) * np.exp(-tt / 0.025), L) * _env(tt, 0.004, tau)
        s += 0.55 * _sin(2.1 * (f_lo + (f_hi - f_lo) * np.exp(-tt / 0.02)), L) * _env(tt, 0.003, tau * 0.6)
        nz = dsp.to_mono(dsp.lowpass(_noise(L, rng), 160)) * _env(tt, 0.002, 0.025)
        knock = dsp.to_mono(dsp.bandpass(_noise(L, rng), (110, 420))) * _env(tt, 0.002, 0.03)
        return amp * (np.tanh(2.2 * s) + 0.4 * _norm(nz) + 0.55 * _norm(knock))

    for b in beats:
        bpm_here = 60.0 * rate[min(b, n - 1)]
        gap = np.clip(0.33 * 60.0 / bpm_here, 0.13, 0.3)
        for off, args in ((0, (68, 44, 0.075, 1.0)), (int(gap * SR), (82, 52, 0.06, 0.68))):
            k = b + off
            if k >= n:
                continue
            seg = thump(*args) * rng.uniform(0.9, 1.0)
            m = min(L, n - k)
            out[k:k + m] += seg[:m]
    y = dsp.lowpass(_st(out), 1000, order=2)
    y = dsp.peaking(y, 180, 4.0, 1.0)
    y = dsp.reverb(y, dsp.make_ir(0.5, 0.004, 0.8, seed), wet=0.1, tail=False)
    return _finish(y, n, peak=-1.0, fo=0.03, hp=22)


def clock_tick(dur=None, seed=0, bpm=60.0, **_):
    """Tic-tac d'horloge ancienne (balancier), alternance tic/tac."""
    dur = 8.0 if dur is None else dur
    n = _n(dur)
    rng = _rng(seed, "clock_tick")
    period = 60.0 / max(float(bpm), 1.0)
    out = np.zeros((n, 2))
    L = _n(0.18)
    tt = _t(L)
    i = 0
    t0 = 0.03
    while t0 < dur:
        hi = i % 2 == 0
        freqs = (2450, 3720, 5300, 980) if hi else (2050, 3150, 4650, 820)
        exc = _noise(L, rng) * np.exp(-tt / 0.0006)
        s = sum(dsp.to_mono(dsp.resonator(exc, f, q=28 if f > 1500 else 8)) * (1.0 if f > 1500 else 2.5)
                for f in freqs)
        s2 = np.zeros(L)
        d = int(0.007 * SR)
        s2[d:] = s[:-d] * 0.45  # double clic de l'échappement
        wood = dsp.to_mono(dsp.resonator(exc, 900 if hi else 760, q=5)) * 4.0
        tick = _norm(s + s2 + wood) * np.exp(-tt / 0.05) * rng.uniform(0.85, 1.0)
        k = int(t0 * SR)
        m = min(L, n - k)
        out[k:k + m] += _pan(tick[:m], 0.12 if hi else -0.12)
        t0 += period
        i += 1
    out = dsp.peaking(out, 600, 4, 1.2)
    out = dsp.reverb(out, dsp.make_ir(1.3, 0.012, 0.55, seed), wet=0.25, tail=False)
    return _finish(out, n, peak=-1.0, fo=0.02, hp=120)


def scan_sweep(dur=None, seed=0, **_):
    """Balayage de scanner HUD (montée tonale hachée + bruit numérique)."""
    dur = 0.9 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    p = t / dur
    rng = _rng(seed, "scan_sweep")
    f = 420 * (8.0 ** p)
    tone = _sin(f, n) + 0.3 * _sin(3 * f, n) + 0.4 * _sin(f * 1.007, n, 1.0)
    gate = 0.55 + 0.45 * np.tanh(6 * np.sin(2 * np.pi * 38 * t))
    tone *= gate * np.sin(np.pi * np.clip(p, 0, 1)) ** 0.6
    nz = dsp.to_mono(dsp.tv_filter(_noise(n, rng), "bandpass", 800 + 6000 * p, q=2.0)) * 0.5 * p
    y = _pan(_norm(tone) * 0.7 + _norm(nz) * 0.3, -0.5 + p)
    y = dsp.reverb(y, dsp.make_ir(0.5, 0.01, 0.5, seed), wet=0.15, tail=False)
    return _finish(y, n, peak=-1.0, fi=0.004, fo=0.04, hp=150)


def hud_lock(dur=None, seed=0, **_):
    """Blip de verrouillage HUD à 2 tons (quinte montante)."""
    dur = 0.32 if dur is None else dur
    n = _n(dur)
    out = np.zeros(n)
    for t0, f, d in ((0.0, 1318.5, 0.055), (0.07, 1975.5, 0.11)):
        k = int(t0 * SR)
        m = min(_n(d + 0.06), n - k)
        if m <= 0:
            continue
        tt = _t(m)
        s = np.sin(2 * np.pi * f * tt + 0.6 * np.sin(2 * np.pi * 2 * f * tt) * np.exp(-tt / 0.01))
        e = (1 - np.exp(-tt / 0.0015)) * np.where(tt < d, 1.0, np.exp(-(tt - d) / 0.012))
        out[k:k + m] += s * e
    y = _widen(out, 0.15, seed)
    y = dsp.reverb(y, dsp.make_ir(0.35, 0.006, 0.5, seed), wet=0.12, tail=False)
    return _finish(y, n, peak=-1.0, fo=0.02, hp=200)


def marker_scribble(dur=None, seed=0, **_):
    """Feutre sur papier photo : friction de bruit passe-bande à modulation rapide + couinement."""
    dur = 0.35 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    rng = _rng(seed, "marker_scribble")
    speed = np.abs(np.sin(2 * np.pi * rng.uniform(5.5, 8) * t + rng.uniform(0, 3))) ** 0.6
    speed *= 0.6 + 0.4 * _lfo(n, 25, rng)
    rough = 0.6 + 0.4 * np.abs(_lfo(n, 320, rng))
    fr = _norm(dsp.to_mono(dsp.bandpass(_noise(n, rng), (1800, 7500), order=2))) * speed * rough
    body = _norm(dsp.to_mono(dsp.bandpass(_noise(n, rng), (380, 1100), order=2))) * speed * 0.3
    sq_f = 2900 + 600 * speed
    squeak = _sin(sq_f, n) * np.clip(speed - 0.55, 0, 1) * 0.35
    env = np.clip(t / 0.012, 0, 1) * np.clip((dur - t) / 0.05, 0, 1)
    y = _pan((fr + body + squeak) * env, rng.uniform(-0.2, 0.2))
    return _finish(y, n, peak=-1.0, fo=0.01, hp=200)


def drip(dur=None, seed=0, **_):
    """Gouttes d'eau dans une pièce carrelée."""
    dur = 3.0 if dur is None else dur
    n = _n(dur)
    rng = _rng(seed, "drip")
    out = np.zeros((n, 2))
    t0 = 0.08
    L = _n(0.12)
    tt = _t(L)
    while t0 < dur - 0.05:
        f0 = rng.uniform(850, 1500)
        f = f0 * (1 + 1.4 * (1 - np.exp(-tt / 0.012)))
        s = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(tt, 0.0006, rng.uniform(0.012, 0.022))
        s += 0.3 * _noise(L, rng) * np.exp(-tt / 0.0008)
        k = int(t0 * SR)
        m = min(L, n - k)
        out[k:k + m] += _pan(s[:m] * rng.uniform(0.6, 1.0), rng.uniform(-0.6, 0.6))
        t0 += rng.uniform(0.45, 1.25)
    out = dsp.reverb(out, dsp.make_ir(1.9, 0.009, 0.15, seed, bright=1.3), wet=0.55, tail=False)
    return _finish(out, n, peak=-1.0, fo=0.05, hp=150)


def creak(dur=None, seed=0, **_):
    """Grincement de bois (frottement collé-glissé à travers des résonances de bois)."""
    dur = 1.4 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    p = t / dur
    rng = _rng(seed, "creak")
    rate = 14 + 150 * np.sin(np.pi * np.clip(p * 1.05, 0, 1)) ** 1.6 + 25 * _lfo(n, 3, rng)
    ph = np.cumsum(np.maximum(rate, 5) * (1 + 0.18 * _lfo(n, 40, rng))) / SR
    idx = np.flatnonzero(np.diff(np.floor(ph)) > 0)
    pulses = np.zeros(n)
    pulses[idx] = rng.uniform(0.5, 1.0, len(idx))
    exc = dsp.to_mono(dsp.lowpass(pulses, 6000, order=2)) + 0.02 * _noise(n, rng)
    y = np.zeros(n)
    for f, q, a in ((190, 7, 1.0), (430, 9, 0.9), (780, 10, 0.7), (1350, 12, 0.5), (2300, 14, 0.3)):
        y += a * dsp.to_mono(dsp.resonator(exc, f * (1 + 0.02 * rng.uniform(-1, 1)), q))
    env = np.sin(np.pi * np.clip(p, 0, 1)) ** 0.7
    y = _pan(_norm(y) * env, rng.uniform(-0.4, 0.4))
    y = dsp.reverb(y, dsp.make_ir(1.1, 0.01, 0.6, seed), wet=0.25, tail=False)
    return _finish(y, n, peak=-1.0, fo=0.05, hp=80)


def car_pass(dur=None, seed=0, speed_ms=21.0, distance=3.5, **_):
    """Voiture qui passe (moteur + pneus), Doppler par ligne à retard variable, pano G→D."""
    dur = 2.5 if dur is None else dur
    n = _n(dur)
    rng = _rng(seed, "car_pass")
    c = 343.0
    pre = _n(0.25)
    m = n + pre
    fe = 38 * (1 + 0.03 * _lfo(m, 2, rng))
    eng = np.zeros(m)
    for k in range(1, 28):
        eng += _sin(fe * k, m, rng.uniform(0, 6.28)) / k ** 0.9 * (1.3 if k % 2 == 0 else 1.0)
    eng = dsp.to_mono(dsp.lowpass(eng, 1800, order=2))
    tyre = dsp.to_mono(dsp.bandpass(_noise(m, rng, "pink"), (250, 3500), order=2))
    src = _norm(eng) * 0.6 + _norm(tyre) * 0.7
    t = _t(n)
    tc = dur * 0.52
    x = speed_ms * (t - tc)
    r = np.sqrt(x ** 2 + distance ** 2)
    emit = (t - r / c) * SR + pre
    y = dsp.interp_cubic(src, emit)[:, 0] * (distance / r)
    cutoff = np.clip(9000 * distance / r, 900, 12000)
    y = dsp.to_mono(dsp.tv_filter(y, "lowpass", cutoff, q=0.7, block=64))
    pos = np.clip(x / np.maximum(r, 1e-3), -1, 1) * 0.9
    out = _pan(y, pos)
    out = dsp.reverb(out, dsp.make_ir(0.9, 0.02, 0.6, seed), wet=0.1, tail=False)
    return _finish(out, n, peak=-1.0, fi=0.05, fo=0.1, hp=30)


def wind(dur=None, seed=0, **_):
    """Vent en rafales (bruit filtré, sifflements, grondement)."""
    dur = 10.0 if dur is None else dur
    n = _n(dur)
    rng = _rng(seed, "wind")
    gust = np.clip(0.55 + 0.45 * _lfo(n, 0.22, rng), 0.05, 1.2) ** 1.8
    out = np.zeros((n, 2))
    for ch in range(2):
        g = gust * (0.85 + 0.15 * _lfo(n, 0.5, rng))
        fc = 380 + 700 * g
        body = dsp.to_mono(dsp.tv_filter(_noise(n, rng, "pink"), "bandpass", fc, q=0.9, block=256))
        whistle = dsp.to_mono(dsp.tv_filter(_noise(n, rng), "bandpass", 900 + 600 * g, q=25, block=256)) * 2.5
        rumble = dsp.to_mono(dsp.lowpass(_noise(n, rng, "brown"), 140, order=2))
        out[:, ch] = _norm(body) * g + _norm(whistle) * 0.18 * g ** 1.5 + _norm(rumble) * 0.25 * g
    return _finish(out, n, peak=-1.0, fi=0.2, fo=0.2, hp=25)


# ---------------------------------------------------------------------------
# Screamer
# ---------------------------------------------------------------------------


def _scream_voice(n, f0, formants, rng, rough=0.5):
    t = _t(n)
    vib = 2 ** ((55 * np.sin(2 * np.pi * 6.3 * t) + 25 * _lfo(n, 30, rng)) / 1200)
    f = f0 * vib
    src = _saw(f, n)
    sub = np.sin(np.pi * np.cumsum(f) / SR)  # sous-harmonique (voix « double », growl)
    src = src * (1 + rough * sub) + 0.35 * _noise(n, rng)
    y = np.zeros(n)
    for F, bw, g in formants:
        y += g * dsp.to_mono(dsp.resonator(src, F, q=F / bw))
    return _norm(y)


def screamer_stack(dur=None, seed=0, scream=None, **_):
    """Stinger massif du screamer (le son le plus fort du film).

    Couches : burst de bruit distordu, cri synthétique pitché vers le bas (2 voix formantiques),
    sub hit, crissement métallique, transitoire. `scream` (optionnel, audio (N,) ou (N,2) 48 kHz) :
    copies à −5 et −12 demi-tons, distordues, superposées.
    """
    dur = 1.2 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    rng = _rng(seed, "screamer_stack")
    hold = np.clip((dur - t) / max(0.35, dur * 0.35), 0, 1) ** 1.4
    burst = dsp.distort(_widen(_noise(n, rng), 0.9, seed), 8.0)
    burst = dsp.bandpass(burst, (250, 7500), order=2) * (_env(t, 0.002, 0.3) * 0.7 + 0.3 * hold)[:, None]
    f0a = 430 * (0.62 + 0.38 * np.exp(-t / 0.5))
    va = _scream_voice(n, f0a, [(850 * 0.85, 120, 1.0), (1250 * 0.85, 160, 0.7), (2800 * 0.85, 260, 0.45),
                                (3600 * 0.85, 320, 0.25)], rng, 0.4)
    vb = _scream_voice(n, f0a * 0.5, [(850 * 0.7, 110, 1.0), (1250 * 0.7, 150, 0.8), (2800 * 0.7, 250, 0.4)],
                       rng, 0.8)
    venv = (1 - np.exp(-t / 0.015)) * hold
    voice = dsp.distort(_pan(va * venv, -0.25) + _pan(vb * venv, 0.25), 6.0)
    sub = _sin(30 + 26 * np.exp(-t / 0.18), n) * _env(t, 0.002, 0.45)
    sub = np.tanh(2.5 * sub)
    bend = 1 - 0.16 * (1 - np.exp(-t / 0.5))
    screech = np.zeros((n, 2))
    for k, f in enumerate((2350, 3110, 3870, 4520, 5980)):
        fm = f * bend * (1 + 0.01 * np.sin(2 * np.pi * rng.uniform(9, 17) * t))
        screech += _pan(_sin(fm, n, rng.uniform(0, 6)) / (1 + 0.3 * k), (-1) ** k * 0.6)
    screech *= (np.sin(2 * np.pi * 37 * t) * 0.5 + 0.5)[:, None] * (_env(t, 0.008, 0.45) * hold)[:, None]
    click = _norm(dsp.to_mono(dsp.highpass(_noise(n, rng), 3000))) * np.exp(-t / 0.002)
    crack = _norm(dsp.to_mono(dsp.bandpass(_noise(n, rng), (600, 4000)))) * _env(t, 0.0004, 0.05)
    y = (burst * 0.55 + voice * 0.8 + _st(sub) * 0.9 + screech * 0.22 / max(np.max(np.abs(screech)), 1e-9)
         + _widen(click * 0.5 + crack * 0.6, 0.6, seed))
    if scream is not None and np.size(scream) > 64:
        sc = dsp.pad_to(dsp.as_stereo(scream), n)
        for st, g in ((-5, 0.7), (-12, 0.8)):
            y += dsp.distort(dsp.pitch(sc, st), 5.0) * g
    y = dsp.distort(y * 0.8, 2.2)
    y = dsp.compress(y, thr_db=-14, ratio=4, attack=0.001, release=0.08, makeup=4)
    y = dsp.width(y, 1.25)
    return _finish(y, n, peak=-0.3, fo=0.08, hp=25)


# ---------------------------------------------------------------------------
# Piano, boîte à musique
# ---------------------------------------------------------------------------
FALLBACK_MOTIF = [  # motif de secours en la mineur (t, midi, durée) — berceuse harmonique mineure
    (0.00, 76, 0.45), (0.45, 81, 0.45), (0.90, 84, 0.45), (1.35, 83, 0.65), (2.00, 80, 0.45), (2.45, 81, 1.5),
]


def _midi_hz(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=np.float64) - 69) / 12)


def _synth_piano_note(f0: float, n: int, rng) -> np.ndarray:
    t = _t(n)
    B = 0.00035
    y = np.zeros(n)
    for k in range(1, 30):
        fk = k * f0 * np.sqrt(1 + B * k * k)
        if fk > 15000:
            break
        amp = (1.0 / k ** 1.05) * (0.55 + 0.45 * abs(np.sin(k * np.pi * 0.118)))
        env = 0.62 * np.exp(-t / (0.32 / (1 + 0.07 * k))) + 0.38 * np.exp(-t / (3.6 / (1 + 0.22 * k)))
        for d in (-0.7, 0.6):
            y += 0.5 * amp * env * _sin(fk * 2 ** (d / 1200), n, rng.uniform(0, 6.28))
    y *= 1 - np.exp(-t / 0.0015)
    ham = _norm(dsp.to_mono(dsp.bandpass(_noise(n, rng), (200, 1600)))) * np.exp(-t / 0.006) * 0.25
    y = dsp.to_mono(dsp.lowpass(_norm(y) + ham, 7500, order=2))
    return y


def _piano_dry_note(source: str, seed: int):
    if source != "fallback":
        try:
            from . import piano
            a = piano.note_audio(source=source)
            if a is not None:
                return dsp.as_stereo(a)
        except Exception as e:
            if source == "real":
                raise
            print(f"[synth] note de piano réelle indisponible ({e}) -> note synthétisée")
    rng = _rng(seed, "piano_note")
    return dsp.fade(_widen(_synth_piano_note(220.0, _n(4.5), rng), 0.25, seed), 0.0, 1.2)


def piano_note(dur=None, seed=0, source="auto", reverb_s=6.0, **_):
    """Une note du vrai piano de 8354 avec une longue réverbe (repli : note de piano synthétisée)."""
    dur = 7.0 if dur is None else dur
    n = _n(dur)
    dry = _piano_dry_note(source, seed)
    y = dsp.reverb(dry, dsp.make_ir(reverb_s, 0.03, 0.5, seed), wet=0.5, tail=True)
    return _finish(y, n, peak=-1.0, fo=min(1.0, dur * 0.2), hp=40)


def _music_box_note(f: float, n: int, rng, vel: float = 1.0) -> np.ndarray:
    t = _t(n)
    tau = 1.1 * (1000.0 / max(f, 100)) ** 0.35
    idx = 1.8 * np.exp(-t / 0.035) + 0.15
    y = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * 2.0 * f * t)) * np.exp(-t / tau)
    y += 0.3 * np.sin(2 * np.pi * f * 1.0009 * t) * np.exp(-t / tau)  # léger battement de lame
    if f * 6.27 < 18000:
        y += 0.22 * np.sin(2 * np.pi * f * 6.27 * t) * np.exp(-t / 0.05)
    if f * 17.55 < 20000:
        y += 0.07 * np.sin(2 * np.pi * f * 17.55 * t) * np.exp(-t / 0.012)
    pluck = _noise(n, rng) * np.exp(-t / 0.0012) * 0.12
    y = (y + dsp.to_mono(dsp.highpass(pluck, 2500))) * (1 - np.exp(-t / 0.0007))
    return y * vel


def music_box(notes, dur=None, seed=0, slowdown=0.6, sag_cents=35.0, detune_cents=9.0, transpose=0,
              loop=True, reverb_s=2.4, mechanism=True):
    """Re-synthèse en boîte à musique. notes = [(t, midi|f0_hz, durée[, vélocité]), ...] (t relatifs).

    Le motif boucle sur `dur`, ralentit (ritardando ×(1+slowdown)) et baisse de `sag_cents` comme un
    ressort qui se détend ; chaque note est légèrement désaccordée.
    """
    rng = _rng(seed, "music_box")
    notes = [tuple(nt) for nt in notes]
    period = max(nt[0] + nt[2] for nt in notes)
    dur = (2 * period + 1.5) if dur is None else dur
    n = _n(dur)
    out = np.zeros((n, 2))
    L = _n(3.0)
    cursor = 0.0
    while cursor < dur - 0.15:
        for nt in notes:
            t_rel, pitch_v = nt[0], nt[1]
            vel = nt[3] if len(nt) > 3 else 1.0
            prog = min(1.0, cursor / max(dur, 1e-3))
            stretch = 1 + slowdown * prog ** 1.6
            tn = cursor + t_rel * stretch
            if tn >= dur - 0.1:
                break
            f = float(_midi_hz(pitch_v + transpose)) if pitch_v < 140 else float(pitch_v) * 2 ** (transpose / 12)
            cents = rng.uniform(-detune_cents, detune_cents) - sag_cents * (tn / dur) ** 2
            s = _music_box_note(f * 2 ** (cents / 1200), L, rng, vel * (1 - 0.25 * prog))
            k = int(tn * SR)
            m = min(L, n - k)
            out[k:k + m] += _pan(s[:m], rng.uniform(-0.25, 0.25))
            tick = _noise(_n(0.01), rng) * np.exp(-_t(_n(0.01)) / 0.0015) * 0.05
            kt = max(0, k - _n(0.004))
            mt = min(len(tick), n - kt)
            out[kt:kt + mt] += dsp.as_stereo(dsp.to_mono(dsp.bandpass(tick, (2500, 6000))))[:mt]
        if not loop:
            break
        cursor += period * (1 + slowdown * min(1.0, cursor / max(dur, 1e-3)) ** 1.6)
    if mechanism:
        t = _t(n)
        whirr_rate = 31 * (1 - 0.35 * t / dur)
        whirr = dsp.to_mono(dsp.bandpass(_noise(n, rng), (1500, 4500))) * (
            0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(whirr_rate) / SR))
        out += _widen(_norm(whirr) * 0.012, 0.5, seed)
    out = dsp.shelf(out, 320, -8, "low")
    out = dsp.peaking(out, 1400, 3.5, 0.9)
    out = dsp.reverb(out, dsp.make_ir(reverb_s, 0.02, 0.45, seed), wet=0.33, tail=False)
    return _finish(out, n, peak=-1.0, fo=0.3, hp=150)


def _motif(source: str):
    """Notes (t, f0_hz, durée, vél.) du vrai piano (piano.py) transposées en registre de boîte à musique."""
    if source != "fallback":
        try:
            from . import piano
            m = piano.motif_notes(source=source)
            if m:
                return m
        except Exception as e:
            if source == "real":
                raise
            print(f"[synth] motif réel indisponible ({e}) -> motif de secours")
    return [(t, float(_midi_hz(mi)), d, 1.0) for t, mi, d in FALLBACK_MOTIF]


def music_box_leitmotif(dur=None, seed=0, source="auto", slowdown=0.6, **_):
    """Boîte à musique jouant le motif extrait du vrai piano (repli : motif en la mineur), qui ralentit."""
    notes = _motif(source)
    fs = np.array([nt[1] for nt in notes])
    med = float(np.median(12 * np.log2(fs / 440.0) + 69))
    shift = 12 * int(np.round((84 - med) / 12))  # registre aigu de boîte à musique (~C6)
    notes = [(t, f * 2 ** (shift / 12), d, v) for t, f, d, v in notes]
    return music_box(notes, dur=dur, seed=seed, slowdown=slowdown)


def _piano_variant(name: str, dur, seed, source):
    try:
        from . import piano
        y = piano.variant_audio(name, source=source)
    except Exception as e:
        if source == "real":
            raise
        print(f"[synth] variante piano '{name}' indisponible ({e}) -> repli boîte à musique")
        y = None
    if y is None:
        y = music_box_leitmotif(None, seed, source="fallback")
    n = _n(dur) if dur is not None else len(y)
    return _finish(y, n, peak=-1.0, fo=min(0.5, n / SR * 0.2), hp=30)


def piano_slow(dur=None, seed=0, source="auto", **_):
    """Motif du vrai piano ralenti ×0,5 (hauteur conservée) + réverbe."""
    return _piano_variant("slow", dur, seed, source)


def piano_reverse(dur=None, seed=0, source="auto", **_):
    """Motif du vrai piano à l'envers + réverbe."""
    return _piano_variant("reverse", dur, seed, source)


def piano_detuned(dur=None, seed=0, source="auto", **_):
    """Motif du vrai piano désaccordé (−30 cents + vibrato instable)."""
    return _piano_variant("detuned", dur, seed, source)


# ---------------------------------------------------------------------------
# Lits : drones, nappes
# ---------------------------------------------------------------------------


def drone_low(dur=None, seed=0, darker=False, **_):
    """Drone sombre évolutif : sub 30–45 Hz + scies désaccordées filtrées + bruit modulé + grondements."""
    dur = 12.0 if dur is None else dur
    n = _n(dur)
    rng = _rng(seed, "drone_low")
    fsub = 37.5 + 7.5 * _lfo(n, 0.05, rng)
    sub = _sin(fsub, n) * (0.8 + 0.2 * _lfo(n, 0.1, rng))
    cluster = np.zeros((n, 2))
    for semi, a in ((0, 1.0), (1, 0.42), (7, 0.5), (6, 0.22), (12, 0.35), (13, 0.15)):
        for j in range(3):
            cents = rng.uniform(-9, 9) + 6 * _lfo(n, 0.07, rng)
            v = _saw(55.0 * 2 ** (semi / 12) * 2 ** (cents / 1200), n, rng.uniform())
            cluster += _pan(v * a * (0.75 + 0.25 * _lfo(n, 0.09, rng)), rng.uniform(-0.8, 0.8))
    lo, span = (150.0, 150.0) if darker else (240.0, 320.0)
    cutoff = lo + span * (0.5 + 0.5 * _lfo(n, 0.045, rng))
    raw = cluster
    cluster = dsp.tv_filter(raw, "lowpass", cutoff, q=1.6, block=256)
    cluster = dsp.lowpass(cluster, 1100 if darker else 1800, order=2)
    cluster /= np.max(np.abs(cluster)) + 1e-9
    moan_fc = (260 if darker else 340) * 2 ** (0.9 * _lfo(n, 0.06, rng))  # formant lent : gémissement
    moan = dsp.tv_filter(raw, "bandpass", moan_fc, q=3.0, block=256)
    moan /= np.max(np.abs(moan)) + 1e-9
    cluster = cluster + moan * 0.45
    nz = _widen(_noise(n, rng, "brown"), 0.7, seed)
    nz = dsp.bandpass(nz, (50, 300 if darker else 450), order=2)
    nz = nz / (np.max(np.abs(nz)) + 1e-9) * ((0.5 + 0.5 * _lfo(n, 0.08, rng)) ** 2)[:, None]
    air = np.zeros((n, 2))
    if not darker:
        air = dsp.bandpass(_widen(_noise(n, rng), 0.9, seed + 1), (1500, 4000), order=2)
        air = air / (np.max(np.abs(air)) + 1e-9) * (0.5 + 0.5 * _lfo(n, 0.12, rng))[:, None] * 0.05
    groan = np.zeros((n, 2))
    k_ev = max(1, int(dur / 7.0))
    for _e in range(k_ev):
        t0 = rng.uniform(0.0, max(0.1, dur - 3.0))
        ln = min(rng.uniform(3.0, 5.5), dur - t0)
        if ln <= 0.5:
            continue
        m = _n(ln)
        tt = _t(m)
        base = rng.uniform(140, 260) * (0.7 if darker else 1.0)
        bend = 2 ** (-1.2 * tt / ln / 12)
        g = np.zeros(m)
        for r, a in ((1.0, 1.0), (2.76, 0.5), (5.4, 0.25), (1.51, 0.35)):
            g += a * _sin(base * r * bend * (1 + 0.003 * np.sin(2 * np.pi * 4.7 * tt)), m, rng.uniform(0, 6.28))
        g *= np.sin(np.pi * tt / ln) ** 2
        k = int(t0 * SR)
        groan[k:k + m] += _pan(g, rng.uniform(-0.7, 0.7))[:len(groan) - k]
    groan /= np.max(np.abs(groan)) + 1e-9
    y = (_st(sub) * (0.24 if darker else 0.18) + cluster * (0.5 if darker else 0.55) + nz * 0.3 + air
         + groan * (0.07 if darker else 0.09))
    y *= (0.85 + 0.15 * _lfo(n, 0.06, rng))[:, None]
    y = dsp.reverb(y, dsp.make_ir(4.5, 0.03, 0.62, seed), wet=0.33, tail=False)
    return _finish(y, n, peak=-1.0, fi=0.02, fo=0.03, hp=22)


def drone_tension(dur=None, seed=0, **_):
    """Cluster aigu dissonant façon cordes (battements lents, micro-intervalles)."""
    dur = 10.0 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    rng = _rng(seed, "drone_tension")
    base = 880.0
    offs = [-12.0, -11.5, -0.5, 0.0, 0.35, 1.0, 1.45, 6.0, 6.5, 11.6]
    out = np.zeros((n, 2))
    total_amp = np.zeros(n)
    for i, o in enumerate(offs):
        f = base * 2 ** (o / 12)
        vib = 2 ** ((rng.uniform(4, 11) * np.sin(2 * np.pi * rng.uniform(4.3, 6.1) * t + rng.uniform(0, 6))
                     + 4 * _lfo(n, 0.3, rng)) / 1200)
        a = np.clip(0.55 + 0.45 * _lfo(n, rng.uniform(0.05, 0.15), rng), 0, 1) ** 1.5
        a *= 0.6 if o < -6 else 1.0
        v = 0.7 * _sin(f * vib, n, rng.uniform(0, 6)) + 0.3 * _saw(f * vib, n, rng.uniform())
        out += _pan(v * a, rng.uniform(-0.9, 0.9))
        total_amp += a
    out = dsp.lowpass(out, 5500, order=2)
    bow = dsp.bandpass(_widen(_noise(n, rng), 0.9, seed), (2500, 7000), order=2)
    bow = bow / (np.max(np.abs(bow)) + 1e-9) * (total_amp / len(offs))[:, None] * 0.12
    y = out / (np.max(np.abs(out)) + 1e-9) + bow
    y = dsp.reverb(y, dsp.make_ir(3.5, 0.03, 0.45, seed), wet=0.45, tail=False)
    return _finish(y, n, peak=-1.0, fi=0.05, fo=0.05, hp=120)


def warm_pad(dur=None, seed=0, **_):
    """Nappe chaude majeure (faux happy end du Burger King)."""
    dur = 8.0 if dur is None else dur
    n = _n(dur)
    t = _t(n)
    rng = _rng(seed, "warm_pad")
    chord = [(146.83, 1.0), (220.0, 0.8), (293.66, 0.7), (369.99, 0.6), (440.0, 0.45), (659.25, 0.18)]
    out = np.zeros((n, 2))
    for i, (f, a) in enumerate(chord):
        for j, c in enumerate((-8, 0, 7)):
            drift = 2 ** ((c + 3 * _lfo(n, 0.3, rng)) / 1200)
            out += _pan(_saw(f * drift, n, rng.uniform()) * a, (-0.7 + 0.7 * j) * (1 if i % 2 else -1) * 0.8)
    cutoff = 1100 + 250 * _lfo(n, 0.15, rng)
    out = dsp.tv_filter(out, "lowpass", cutoff, q=0.8, block=256)
    out = dsp.lowpass(out, 3000, order=2)
    out = out / (np.max(np.abs(out)) + 1e-9) + _st(_sin(73.42, n) * 0.25)
    att = min(1.5, dur * 0.3)
    out *= (np.clip(t / att, 0, 1) ** 1.5)[:, None]
    out = dsp.distort(out * 0.6, 1.4)
    out = dsp.reverb(out, dsp.make_ir(3.0, 0.03, 0.5, seed), wet=0.35, tail=False)
    return _finish(out, n, peak=-1.0, fo=min(1.0, dur * 0.2), hp=30)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
CATALOG = {
    "impact_big": impact_big, "impact_hit": impact_hit, "impact_sub": impact_sub, "sub_drop": sub_drop,
    "braam": braam, "braam_long": braam_long, "riser": riser, "reverse_swell": reverse_swell,
    "whoosh": whoosh, "whoosh_rev": whoosh_rev, "freeze_thump": freeze_thump,
    "glitch_short": glitch_short, "glitch_long": glitch_long, "vhs_click": vhs_click,
    "tinnitus": tinnitus, "heartbeat": heartbeat, "clock_tick": clock_tick, "scan_sweep": scan_sweep,
    "hud_lock": hud_lock, "marker_scribble": marker_scribble, "drip": drip, "creak": creak,
    "car_pass": car_pass, "screamer_stack": screamer_stack, "piano_note": piano_note,
    "drone_low": drone_low, "drone_tension": drone_tension, "music_box_leitmotif": music_box_leitmotif,
    "warm_pad": warm_pad, "wind": wind,
    # extras (variantes du vrai piano, utilisables dans l'EDL)
    "piano_slow": piano_slow, "piano_reverse": piano_reverse, "piano_detuned": piano_detuned,
}

REQUIRED = ["impact_big", "impact_hit", "impact_sub", "sub_drop", "braam", "braam_long", "riser", "reverse_swell",
            "whoosh", "whoosh_rev", "freeze_thump", "glitch_short", "glitch_long", "vhs_click", "tinnitus",
            "heartbeat", "clock_tick", "scan_sweep", "hud_lock", "marker_scribble", "drip", "creak", "car_pass",
            "screamer_stack", "piano_note", "drone_low", "drone_tension", "music_box_leitmotif", "warm_pad", "wind"]


def names() -> list[str]:
    return list(CATALOG)


def render(name: str, dur=None, seed: int = 0, **params) -> np.ndarray:
    """Rend l'entrée `name` du CATALOG -> (N, 2) float32 (N = round(dur*48000) si dur est donné)."""
    if name not in CATALOG:
        raise KeyError(f"SFX inconnu : {name!r} (disponibles : {', '.join(CATALOG)})")
    y = CATALOG[name](dur=None if dur is None else float(dur), seed=int(seed), **params)
    y = dsp.as_stereo(y)
    if dur is not None:
        y = dsp.pad_to(y, _n(dur))
    return np.ascontiguousarray(np.nan_to_num(y), dtype=np.float32)
