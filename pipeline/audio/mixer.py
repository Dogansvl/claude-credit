"""Mixage et master du film : mix(edl, quality) -> RENDER_DIR/mix_{quality}.wav (+ rapport JSON).

Bus :
  src   : son source des shots (`edl.src_audio`, sinon dérivé des shots) extrait des proxies à 48 kHz,
          nivelé par segment (répliques : crête momentanée −15 LUFS ≈ voix off à −20 LUFS intégrés ;
          ambiances : −20 LUFS) AVANT `gain_db`, vitesse (varispeed), src_fx
          (pitch_drift, slow_down, scream_layer), mute_ranges, fondus ≥ 9 ms à chaque coupe.
  vox   : voix off (`edl.voices`, fichiers du manifeste placés à t_file).
  sfx   : cues ponctuelles re-synthétisées à la bonne durée (synth.render, cache disque).
  music : lits musicaux et lits SFX (`edl.music`) rendus à la longueur exacte, automation, fondus,
          ducking de `duck` dB sous voix et répliques (attaque 80 ms / relâche 300 ms).
Extensions (intelligibilité) : son source −6 dB et traînes SFX −5 dB sous les voix off, puis
« ducker d'intelligibilité » automatique : dans la bande parole (300 Hz–4 kHz), le fond (lits, SFX hors
attaque, son source sous les voix off) est abaissé juste ce qu'il faut pour rester INTEL_TARGET_DB sous la
voix (narrateur 10 dB, entité 8 dB, répliques 7 dB) ; sous 160 Hz seulement 30 % de la réduction (le poids
des impacts et des sub-drops est conservé).
Silences (`edl.silence`) : zéro absolu (rampes 5 ms) sur tous les bus, sauf voix allow_in_silence.
Cue `params.pre_silence` (s) : trou franc (zéro absolu, hors voix off) juste AVANT la cue (« suck-out »).
Son source : détection de musique captée (autoradio, sono) signalée ; `src_fx` `{"type": "demusic"}` (ou
ESCAPE_SRC_DEMUSIC=auto) l'atténue (demucs si installé, sinon suppression des composantes tenues).
Master : −14 LUFS intégrés, −1,5 dBTP (marge AAC pour garantir ≤ −1 dBTP après encodage) ; le screamer doit
porter le maximum de loudness court terme ET momentané avec une marge nette (SCREAMER_MARGIN_S/M) : sa section
est densifiée (saturation douce suréchantillonnée), les sections concurrentes abaissées en douceur, re-master.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from . import dsp, synth
from .dsp import SR, config

DIALOG_MIN_GAIN_DB = -4.0      # segments source à gain >= ce seuil = répliques (clé de ducking)
DIALOG_REF_M = -15.0           # répliques : crête de loudness momentané visée (≈ voix off à −20 LUFS)
AMBIENCE_REF_LUFS = -20.0      # ambiances : loudness intégré visé avant gain_db
SRC_LEVEL_RANGE = (-15.0, 15.0)
CUT_FADE_S = 0.009             # fondu minimal à chaque coupe de son source
SILENCE_RAMP_S = 0.005
DUCK_ATTACK, DUCK_RELEASE = 0.08, 0.30
SRC_DUCK_UNDER_VO_DB = 6.0     # son source baissé sous narrateur/entité (extension du contrat)
SFX_TAIL_DUCK_DB = 5.0         # traînes des SFX (> 0,5 s après leur attaque) baissées sous les voix off
DIALOG_KEY_THR_DB = -34.0      # seuil de la clé « répliques » (RMS 20 ms, après nivellement et gain)
SCREAMER_SPAN_S = 1.35         # durée de la section screamer si la shot du screamer est introuvable
SCREAMER_TAIL_S = 0.45         # traîne (noir qui suit) incluse dans la section densifiée
SCREAMER_MARGIN_S = 2.5        # LU : court terme du screamer au-dessus du 2e pic du film (exigé ≥ 1, idéal ≥ 2)
SCREAMER_MARGIN_M = 3.3        # LU : momentané du screamer au-dessus du 2e pic (exigé ≥ 3)
MASTER_LUFS = -14.0
MASTER_TP = -1.5               # −1 dBTP exigé ; 0,5 dB de marge pour le dépassement du codage AAC (mesuré ~+0,3 dB)
MAX_SECTION_CUT_DB = 9.0       # abaissement maximal cumulé d'un passage concurrent (avant master)
SUBSONIC_HZ = 28.0
INTEL_BAND = (300.0, 4000.0)   # bande d'intelligibilité de la parole
INTEL_TARGET_DB = {"narrator": 10.0, "entity": 8.0, "dialogue": 7.0}
INTEL_MAX_DB = 14.0
INTEL_LOW_SHARE = 0.3          # part de la réduction appliquée sous INTEL_XOVER_HZ
INTEL_XOVER_HZ = 160.0
CUE_PROTECT_S = 0.12           # attaque des SFX jamais baissée par le ducker d'intelligibilité
MUSIC_WARN_SCORE = 0.55        # score de musique captée dans le son source au-delà duquel on prévient
BUS_TRIM_DB = {"src": 0.0, "vox": 0.0, "sfx": 0.0, "music": 6.0}  # équilibre des bus (lits normalisés en crête)
PIANO_SOUNDS = ("piano_note", "music_box_leitmotif", "piano_slow", "piano_reverse", "piano_detuned")


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
def _cache_dir(sub: str) -> Path:
    d = config.CACHE_DIR / "audio" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


_CODE_KEY: str | None = None


def _code_key() -> str:
    """Empreinte du code de synthèse (invalide le cache disque si synth/dsp/piano changent)."""
    global _CODE_KEY
    if _CODE_KEY is None:
        h = hashlib.sha1()
        for name in ("dsp.py", "synth.py", "piano.py"):
            h.update((Path(__file__).parent / name).read_bytes())
        _CODE_KEY = h.hexdigest()[:12]
    return _CODE_KEY


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest()
    if isinstance(v, float):
        return round(v, 6)
    return v


def render_sfx(name: str, dur: float | None, seed: int, params: dict | None = None) -> np.ndarray:
    """synth.render avec cache disque (CACHE_DIR/audio/sfx/<hash>.npy)."""
    params = dict(params or {})
    extra = ""
    if name in PIANO_SOUNDS and params.get("source", "auto") != "fallback":
        try:
            from . import piano
            extra = piano.extract_leitmotif().get("key", "")
        except Exception:
            extra = ""
    key = hashlib.sha1(json.dumps([name, None if dur is None else round(float(dur), 6), int(seed),
                                   _jsonable(params), _code_key(), extra]).encode()).hexdigest()[:20]
    p = _cache_dir("sfx") / f"{name}_{key}.npy"
    if p.exists():
        try:
            return np.load(p)
        except Exception:
            pass
    y = synth.render(name, dur=dur, seed=seed, **params)
    np.save(p, y)
    return y


def _auto_seed(*parts) -> int:
    return zlib.crc32("|".join(str(x) for x in parts).encode()) % 10007


_SRC_MEM: dict[str, np.ndarray] = {}


def clip_audio(clip: str) -> np.ndarray | None:
    """Piste audio complète d'un clip (proxy, sinon rush original) en (N, 2) 48 kHz, avec cache."""
    if clip in _SRC_MEM:
        return _SRC_MEM[clip]
    path = config.proxy_path(clip)
    if not path.exists():
        try:
            path = config.rush_path(clip)
        except FileNotFoundError:
            return None
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:12]
    cp = _cache_dir("src") / f"{clip}_{key}.npy"
    if cp.exists():
        a = np.load(cp)
    else:
        try:
            a = dsp.ffmpeg_decode(path)
        except RuntimeError:
            return None
        np.save(cp, a)
    _SRC_MEM[clip] = a
    return a


def segment_level_gain(y: np.ndarray, dialog: bool) -> float:
    """Gain de nivellement d'un segment source (les prises iPhone ont des niveaux très variables).

    Répliques : crête de loudness momentané -> DIALOG_REF_M (un cri et une phrase dite arrivent au même
    niveau de référence ; le `gain_db` de l'EDL règle ensuite leur rapport). Ambiances : loudness
    intégré -> AMBIENCE_REF_LUFS. Segments quasi muets : 0 dB. Borné à SRC_LEVEL_RANGE.
    """
    if len(y) < 64:
        return 0.0
    yy = y if len(y) >= int(0.45 * SR) else dsp.pad_to(y, int(0.45 * SR))
    if dialog:
        _, M = dsp.momentary(yy)
        ref = float(M.max())
        target = DIALOG_REF_M
    else:
        ref = dsp.lufs_fast(yy)
        target = AMBIENCE_REF_LUFS
    if ref < -65:
        return 0.0
    return float(np.clip(target - ref, *SRC_LEVEL_RANGE))


# ---------------------------------------------------------------------------
# Segments de son source
# ---------------------------------------------------------------------------
def _speed_pair(speed) -> tuple[float, float]:
    if isinstance(speed, (list, tuple)):
        return float(speed[0]), float(speed[-1])
    return (float(speed), float(speed)) if speed is not None else (1.0, 1.0)


def derive_src_audio(edl: dict) -> list[dict]:
    """Repli si l'EDL n'a pas de `src_audio` : segments dérivés des shots `clip` non muets."""
    segs = []
    for sh in edl.get("shots", []):
        if sh.get("kind") != "clip" or not sh.get("clip"):
            continue
        au = sh.get("audio") or {}
        if au.get("mute_src"):
            continue
        v0, v1 = _speed_pair(sh.get("speed", 1.0))
        play = sh["dur"] if sh.get("freeze_after") is None else min(sh["dur"], float(sh["freeze_after"]))
        segs.append({
            "clip": sh["clip"], "src_in": float(sh.get("src_in", 0.0)),
            "src_out": float(sh.get("src_in", 0.0)) + play * (v0 + v1) / 2, "t": float(sh["start"]),
            "dur": play, "speed": sh.get("speed", 1.0), "gain_db": float(au.get("src_gain_db", 0.0)),
            "fade_in": float(au.get("fade_in", CUT_FADE_S)), "fade_out": float(au.get("fade_out", CUT_FADE_S)),
            "src_fx": au.get("src_fx", []),
            "mute_ranges": [[sh["start"] + a, sh["start"] + b] for a, b in au.get("mute_ranges", [])],
            "shot": sh.get("id"),
        })
    return segs


def _seg_duration(seg: dict) -> float:
    if seg.get("dur") is not None:
        return float(seg["dur"])
    v0, v1 = _speed_pair(seg.get("speed", 1.0))
    return max(0.0, (float(seg["src_out"]) - float(seg["src_in"])) / max((v0 + v1) / 2, 1e-3))


def render_segment(seg: dict, a: np.ndarray, warnings: list) -> np.ndarray:
    """Rend un segment source (sans gain ni placement) : vitesse, src_fx, longueur exacte."""
    D = _seg_duration(seg)
    n = max(0, int(round(D * SR)))
    if n == 0:
        return np.zeros((0, 2), np.float32)
    tau = np.arange(n) / SR
    v0, v1 = _speed_pair(seg.get("speed", 1.0))
    v = v0 + (v1 - v0) * tau / max(D, 1e-6)
    mult = np.ones(n)
    st = np.zeros(n)
    pitched = False
    layers = []
    for fx in seg.get("src_fx") or []:
        typ = fx.get("type")
        t0 = float(fx.get("t0", 0.0))
        t1 = float(fx.get("t1", D) if fx.get("t1") is not None else D)
        ramp = np.clip((tau - t0) / max(t1 - t0, 1e-3), 0, 1)
        if typ == "slow_down":
            mult *= 1 + (float(fx.get("to", 0.6)) - 1) * ramp
            pitched = True
        elif typ == "pitch_drift":
            st += float(fx.get("semitones", -1.0)) * ramp
            pitched = True
        elif typ == "scream_layer":
            layers.append(fx)
        else:
            warnings.append(f"src_fx inconnu ignoré : {typ} ({seg.get('clip')})")
    rate = v * mult
    src_pos = float(seg["src_in"]) * SR + np.concatenate([[0.0], np.cumsum(rate)[:-1]])
    lo = int(max(0, np.floor(src_pos.min()) - 8))
    hi = int(min(len(a), np.ceil(src_pos.max()) + 8))
    if hi <= lo:
        warnings.append(f"segment hors du clip {seg.get('clip')} (src_in={seg['src_in']})")
        return np.zeros((n, 2), np.float32)
    if src_pos.min() < 0:
        warnings.append(f"segment {seg.get('clip')} commence avant le début du clip "
                        f"({src_pos.min() / SR:.2f} s, complété par du silence)")
    if src_pos.max() > len(a) - 1:
        warnings.append(f"segment {seg.get('clip')} dépasse la fin du clip de "
                        f"{(src_pos.max() - len(a)) / SR:.2f} s (complété par du silence)")
    piece = a[lo:hi]
    local = src_pos - lo
    if abs(v0 - 1) > 1e-6 or abs(v1 - 1) > 1e-6:
        st = st + 12 * np.log2(np.maximum(v, 1e-3))  # vitesse = bande magnétique (la hauteur suit)
    if not pitched:
        if np.allclose(rate, 1.0) and abs(local[0] - round(local[0])) < 1e-6:
            k = int(round(local[0]))
            if k < 0:  # src_in négatif : silence avant le début du clip
                y = dsp.pad_to(np.concatenate([np.zeros((-k, 2), np.float32), piece[:max(0, n + k)]]), n)
            else:
                y = dsp.pad_to(piece[k:k + n], n)
        else:
            y = dsp.varispeed(piece, local)
    else:
        y = dsp.warp(piece, local, st)
    for fx in layers:
        y = scream_layer(y, fx)
    return dsp.pad_to(y, n)


def scream_layer(y: np.ndarray, fx: dict | None = None) -> np.ndarray:
    """Copies −5 et −12 demi-tons distordues SOUS le cri original (monstrueux, plus large)."""
    fx = fx or {}
    out = y.astype(np.float32).copy()
    shifts = fx.get("semitones", [-5, -12])
    gains = fx.get("gains_db", [-3.0, -4.5])
    for i, (s, g) in enumerate(zip(shifts, gains)):
        layer = dsp.distort(dsp.pitch(y, float(s), formant=False), 4.0 + 2 * i)
        layer = dsp.lowpass(layer, 7000 if i == 0 else 4500, order=2)
        d = int((0.004 + 0.007 * i) * SR)
        layer = np.concatenate([np.zeros((d, 2), np.float32), layer])[:len(y)]
        layer = dsp.pan(layer, -0.35 if i == 0 else 0.35)
        out += dsp.gain(layer, g)
    return out


def _apply_mute_ranges(y: np.ndarray, t_start: float, ranges, ramp: float = 0.008) -> np.ndarray:
    if not ranges:
        return y
    g = np.ones(len(y))
    tau = t_start + np.arange(len(y)) / SR
    for a, b in ranges:
        a, b = float(a), float(b)
        g = np.minimum(g, np.clip(np.maximum((a - tau) / ramp, (tau - b) / ramp), 0, 1))
    return (y * g[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Lits musicaux
# ---------------------------------------------------------------------------
def _bed_gain_curve(bed: dict, t0: float, n: int) -> np.ndarray:
    tau = t0 + np.arange(n) / SR
    t1 = t0 + n / SR
    auto = bed.get("automation") or []
    pts = []
    for p in auto:
        tt, dbv = p[0], float(p[1])
        if isinstance(tt, str):  # jetons non résolus par le builder
            tt = t1 if "end" in tt else t0
        pts.append((float(tt), dbv))
    if pts:
        pts.sort()
        level = np.interp(tau, [p[0] for p in pts], [p[1] for p in pts])
    else:
        level = np.full(n, float(bed.get("gain_db", 0.0)))
    g = dsp.lin(level)
    fi = min(n, int(float(bed.get("fade_in") or 0.0) * SR))
    fo = min(n, int(float(bed.get("fade_out") or 0.0) * SR))
    if fi > 0:
        g[:fi] *= dsp._ramp(fi)
    if fo > 0:
        g[n - fo:] *= dsp._ramp(fo)[::-1]
    return g


def _silence_mask(n: int, ranges, ramp: float = SILENCE_RAMP_S) -> np.ndarray:
    g = np.ones(n, np.float32)
    if not ranges:
        return g
    tau = np.arange(n) / SR
    for a, b in ranges:
        a, b = float(a), float(b)
        g = np.minimum(g, np.clip(np.maximum((a - tau) / ramp, (tau - b) / ramp), 0, 1).astype(np.float32))
    return g


# ---------------------------------------------------------------------------
# Intelligibilité : ducker automatique dans la bande parole
# ---------------------------------------------------------------------------
_FR_HOP = 0.01  # pas des courbes de niveau (s)


def band_level_db(x: np.ndarray, band=INTEL_BAND, win: float = 0.1) -> np.ndarray:
    """Niveau (dB) de x dans `band`, fenêtre glissante `win`, échantillonné tous les 10 ms."""
    m = dsp.to_mono(x).astype(np.float64)
    sos = signal.butter(2, band, "bandpass", fs=SR, output="sos")
    p = signal.sosfiltfilt(sos, m) ** 2 if len(m) > 64 else m ** 2
    h = int(_FR_HOP * SR)
    nf = max(1, len(p) // h)
    fr = p[:nf * h].reshape(nf, h).mean(axis=1)
    k = max(1, int(round(win / _FR_HOP)))
    return 10 * np.log10(uniform_filter1d(fr, k, mode="nearest") + 1e-20)


def intel_reduction(fg: np.ndarray, bg: np.ndarray, presence: np.ndarray, target_db: float) -> np.ndarray:
    """Réduction (dB, par trame de 10 ms) à appliquer au fond pour qu'il reste `target_db` sous la voix.

    Niveau de voix = crête syllabique (max sur 250 ms du niveau 100 ms) ; seulement là où la voix est
    présente et à moins de 35 dB de son maximum (pas de pompage sur les traînes de réverbe).
    """
    L_fg = band_level_db(fg)
    L_bg = band_level_db(bg)
    n = min(len(L_fg), len(L_bg))
    L_fg, L_bg = L_fg[:n], L_bg[:n]
    held = maximum_filter1d(L_fg, 25)
    h = int(_FR_HOP * SR)
    pr = presence[: n * h: h][:n] if len(presence) >= n * h else np.zeros(n)
    act = (pr > 0.5) & (held > held.max() - 35)
    return np.where(act, np.clip(L_bg - (held - target_db), 0.0, INTEL_MAX_DB), 0.0)


def _smooth_reduction(r: np.ndarray, n: int, attack: float = 0.04, release: float = 0.3, hold: float = 0.15) -> np.ndarray:
    """Trames 10 ms -> courbe par échantillon : maintien, anticipation (= attaque), attaque/relâche."""
    if not np.any(r > 0):
        return np.zeros(n)
    rate = 1.0 / _FR_HOP
    r = maximum_filter1d(r, max(1, int(hold * rate)) * 2 + 1)
    la = int(attack * rate)
    r = np.concatenate([r[la:], np.zeros(la)])
    r = dsp._smooth_attack_release(r, np.exp(-1 / max(attack * rate, 1e-3)), np.exp(-1 / max(release * rate, 1e-3)))
    pos = (np.arange(len(r)) + 0.5) * _FR_HOP * SR
    return np.interp(np.arange(n), pos, r, left=0.0, right=0.0)


def apply_split_gain(x: np.ndarray, r_db: np.ndarray, low_share: float = INTEL_LOW_SHARE) -> np.ndarray:
    """Baisse x de r_db au-dessus de INTEL_XOVER_HZ et de r_db·low_share en dessous (séparation à phase nulle)."""
    if not np.any(r_db > 0.01):
        return x
    sos = signal.butter(2, INTEL_XOVER_HZ, "lowpass", fs=SR, output="sos")
    lo = signal.sosfiltfilt(sos, x.astype(np.float64), axis=0)
    hi = x - lo
    return (lo * dsp.lin(-r_db * low_share)[:, None] + hi * dsp.lin(-r_db)[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Musique captée dans le son source (vrais rushs : sono du Burger King, accueil, autoradio)
# ---------------------------------------------------------------------------
def music_score(y: np.ndarray) -> float:
    """0..1 : part de l'énergie 150 Hz–4 kHz portée par des composantes TENUES (notes, accords, basse)
    sur ≥ 0,4 s. La parole (harmoniques qui glissent, syllabes) reste basse (~0,2–0,4) ; une musique
    de fond tenue monte au-delà de ~0,55."""
    m = dsp.to_mono(y).astype(np.float64)
    if len(m) < int(1.0 * SR):
        return 0.0
    f, _, Z = signal.stft(m, SR, nperseg=4096, noverlap=4096 - 1024)
    band = (f >= 150) & (f <= 4000)
    A = np.abs(Z[band])
    if A.sum() <= 0:
        return 0.0
    k = max(3, int(0.4 * SR / 1024) | 1)
    from scipy.ndimage import median_filter
    held = median_filter(A, size=(1, k), mode="nearest")
    return float(np.clip(np.minimum(held, A).sum() / (A.sum() + 1e-12), 0.0, 1.0))


def demusic(y: np.ndarray, amount_db: float = 12.0, use_demucs: bool = True) -> tuple[np.ndarray, str]:
    """Atténue la musique captée sous la voix : demucs (si installé : séparation « vocals ») sinon
    suppression des composantes tenues (masque doux, la parole garde ses transitoires). Renvoie (y, méthode)."""
    y = dsp.as_stereo(y)
    if use_demucs:
        try:
            import importlib.util
            import subprocess
            import tempfile
            if importlib.util.find_spec("demucs") is not None:
                with tempfile.TemporaryDirectory(prefix="demucs_") as td:
                    src = Path(td) / "in.wav"
                    dsp.save(src, y)
                    r = subprocess.run([sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
                                        "-o", td, str(src)], capture_output=True, text=True, timeout=900)
                    voc = next(Path(td).rglob("vocals.wav"), None)
                    if r.returncode == 0 and voc is not None:
                        v = dsp.pad_to(dsp.load(voc), len(y))
                        rest = y - v
                        return (v + dsp.gain(rest, -amount_db)).astype(np.float32), "demucs"
        except Exception:
            pass
    out = np.zeros_like(y)
    for c in range(2):
        f, _, Z = signal.stft(y[:, c].astype(np.float64), SR, nperseg=4096, noverlap=4096 - 512)
        A = np.abs(Z) + 1e-12
        from scipy.ndimage import median_filter
        k = max(3, int(0.5 * SR / 512) | 1)
        held = median_filter(A, size=(1, k), mode="nearest")
        frac = np.clip(np.minimum(held, A) / A, 0, 1)            # part tenue de chaque point temps-fréquence
        g = 1.0 - frac * (1.0 - dsp.lin(-amount_db))
        _, z = signal.istft(Z * g, SR, nperseg=4096, noverlap=4096 - 512)
        out[:, c] = dsp.pad_to(z.astype(np.float32), len(y))[:, 0]
    return out, "tenues"


# ---------------------------------------------------------------------------
# Screamer : densité + marges
# ---------------------------------------------------------------------------
def _soft_clip_os(x: np.ndarray, c: float) -> np.ndarray:
    """c·tanh(x/c) suréchantillonné ×4 (saturation douce sans repliement grossier)."""
    up = signal.resample_poly(x.astype(np.float64), 4, 1, axis=0)
    up = c * np.tanh(up / c)
    return dsp.pad_to(signal.resample_poly(up, 1, 4, axis=0).astype(np.float32), len(x))


def _screamer_span(edl: dict, ts: float) -> tuple[float, float]:
    """(fin du cri, fin de la section densifiée) d'après la shot qui contient le screamer."""
    for sh in edl.get("shots", []):
        a = float(sh.get("start", -1))
        if a - 0.02 <= ts < a + float(sh.get("dur", 0)):
            end = a + float(sh["dur"])
            return end, end + SCREAMER_TAIL_S
    return ts + SCREAMER_SPAN_S - SCREAMER_TAIL_S, ts + SCREAMER_SPAN_S


def _densify(x: np.ndarray, ts: float, t_end: float, t_tail: float, master_gain_db: float) -> np.ndarray:
    """Saturation douce de la section screamer, calée sur le plafond du master (après son gain)."""
    a = max(0, int((ts - 0.005) * SR))
    b = min(len(x), int(t_tail * SR))
    if b - a < 64:
        return x
    g = float(dsp.lin(master_gain_db))
    c = float(dsp.lin(MASTER_TP - 0.3))
    seg = x[a:b]
    sat = _soft_clip_os(seg * g, c) / g
    tau = (np.arange(b - a) / SR) + (a / SR)
    w = np.clip((t_tail - tau) / max(t_tail - t_end, 1e-3), 0, 1)        # plein pendant le cri, fondu sur la traîne
    w = np.minimum(w, np.clip((tau - (ts - 0.005)) / 0.003, 0, 1))[:, None].astype(np.float32)
    out = x.copy()
    out[a:b] = seg * (1 - w) + sat * w
    return out


def screamer_check(y: np.ndarray, ts: float, t_tail: float) -> dict:
    """Loudness court terme (3 s) et momentané (0,4 s) du screamer vs le reste du film."""
    tS, S = dsp.short_term(y)
    tM, M = dsp.momentary(y)
    span = t_tail - ts

    def split(t_end, L, win):
        ov = np.clip(np.minimum(t_end, t_tail) - np.maximum(t_end - win, ts), 0, None)
        inside = ov >= min(0.6 * span, 0.5 * win)
        other = ov <= 0.02
        return inside, other
    iS, oS = split(tS, S, 3.0)
    iM, oM = split(tM, M, 0.4)
    s_scr = float(S[iS].max()) if iS.any() else -99.0
    m_scr = float(M[iM].max()) if iM.any() else -99.0
    s_oth = float(S[oS].max()) if oS.any() else -99.0
    m_oth = float(M[oM].max()) if oM.any() else -99.0
    i, j = int(np.argmax(S)), int(np.argmax(M))
    d = {"max_short_term_lufs": round(float(S[i]), 2), "max_short_term_t": round(float(tS[i]), 2),
         "max_short_term_window": [round(float(tS[i] - 3.0), 2), round(float(tS[i]), 2)],
         "screamer_short_term_lufs": round(s_scr, 2), "other_max_short_term_lufs": round(s_oth, 2),
         "other_max_short_term_t": round(float(tS[oS][np.argmax(S[oS])]), 2) if oS.any() else None,
         "max_momentary_lufs": round(float(M[j]), 2), "max_momentary_t": round(float(tM[j]), 2),
         "screamer_momentary_lufs": round(m_scr, 2), "other_max_momentary_lufs": round(m_oth, 2),
         "other_max_momentary_t": round(float(tM[oM][np.argmax(M[oM])]), 2) if oM.any() else None,
         "margin_short_term_lu": round(s_scr - s_oth, 2), "margin_momentary_lu": round(m_scr - m_oth, 2),
         "momentary_max_in_screamer": bool(ts - 0.05 <= tM[j] <= t_tail + 0.4)}
    d["_curves"] = (tS, S, oS, tM, M, oM, s_scr, m_scr)
    return d


def _excess_curve(n: int, d: dict, ts: float, t_tail: float) -> np.ndarray:
    """Réduction supplémentaire (dB, par échantillon) des passages qui dépassent (screamer − marge)."""
    tS, S, oS, tM, M, oM, s_scr, m_scr = d["_curves"]
    add = np.zeros(n)
    for t_end, L, other, win, lim in ((tS, S, oS, 3.0, s_scr - SCREAMER_MARGIN_S),
                                      (tM, M, oM, 0.4, m_scr - SCREAMER_MARGIN_M)):
        ex = np.where(other, L - lim, -1.0)
        for k in np.flatnonzero(ex > 0):
            a, b = max(0, int((t_end[k] - win) * SR)), min(n, int(t_end[k] * SR))
            np.maximum(add[a:b], ex[k] + 0.25, out=add[a:b])
    # jamais sur le screamer (ni sur sa seconde d'approche)
    a, b = max(0, int((ts - 0.05) * SR)), min(n, int((t_tail + 0.05) * SR))
    add[a:b] = 0.0
    if add.any():  # rampes douces (~0,2 s) : pas de « trou » audible
        hop = 480
        fr = add[: len(add) // hop * hop].reshape(-1, hop).max(axis=1)
        fr = uniform_filter1d(maximum_filter1d(fr, 21), 21)
        add = np.interp(np.arange(n), (np.arange(len(fr)) + 0.5) * hop, fr)
        add[a:b] = 0.0
    return add


# ---------------------------------------------------------------------------
# Mix
# ---------------------------------------------------------------------------
def _duration(edl: dict) -> float:
    d = (edl.get("meta") or {}).get("duration")
    if d:
        return float(d)
    return max((float(s["start"]) + float(s["dur"]) for s in edl.get("shots", [])), default=0.0)


def _voice_path(v: dict) -> Path | None:
    f = v.get("file")
    cands = []
    if f:
        p = Path(f)
        cands += [p] if p.is_absolute() else [config.REPO / p, config.VOICE_DIR / p.name]
    if v.get("id"):
        cands.append(config.VOICE_DIR / f"{v['id']}.wav")
    for c in cands:
        if c.exists():
            return c
    return None


def _screamer_time(edl: dict) -> float | None:
    for c in edl.get("cues", []):
        if c.get("sfx") == "screamer_stack":
            return float(c["t"])
    for s in edl.get("src_audio", []) or []:
        if any((fx or {}).get("type") == "scream_layer" for fx in s.get("src_fx") or []):
            return float(s["t"])
    return None


def build_buses(edl: dict, level_sources: bool = True, log=print) -> tuple[dict, dict]:
    """Construit les 4 bus (avant master). Renvoie (buses, info)."""
    dur = _duration(edl)
    N = int(round(dur * SR))
    warnings: list[str] = list()
    buses = {k: np.zeros((N, 2), np.float32) for k in ("src", "vox", "sfx", "music")}
    dialog = np.zeros((N, 2), np.float32)
    info: dict = {"duration": dur, "samples": N, "warnings": warnings, "src_level_gain_db": [], "src_music": []}
    demusic_mode = (__import__("os").environ.get("ESCAPE_SRC_DEMUSIC") or "").strip().lower()

    # --- son source
    segs = edl.get("src_audio")
    if segs is None:
        segs = derive_src_audio(edl)
        warnings.append("edl.src_audio absent : segments dérivés des shots")
    t_c = time.time()
    for seg in segs:
        clip = seg["clip"]
        a = clip_audio(clip)
        if a is None:
            warnings.append(f"audio introuvable pour {clip} (proxy et rush absents) : segment muet")
            continue
        y = render_segment(seg, a, warnings)
        if len(y) == 0:
            continue
        is_dialog = float(seg.get("gain_db", 0.0)) >= DIALOG_MIN_GAIN_DB
        # musique captée (sono, autoradio) : signalée, atténuée sur demande (src_fx demusic / variable d'env.)
        ms = round(music_score(y), 2)
        want = next((fx for fx in seg.get("src_fx") or [] if fx.get("type") == "demusic"), None)
        if want is None and demusic_mode == "auto" and ms >= MUSIC_WARN_SCORE:
            want = {"type": "demusic"}
        method = None
        if want is not None:
            y, method = demusic(y, float(want.get("amount_db", 12.0)), bool(want.get("demucs", True)))
        elif ms >= MUSIC_WARN_SCORE:
            warnings.append(f"musique probable dans le son source {clip} ({seg.get('shot')}, score {ms}) : "
                            f"ajouter src_fx demusic, baisser src_gain_db ou ESCAPE_SRC_DEMUSIC=auto")
        info["src_music"].append({"shot": seg.get("shot"), "clip": clip, "score": ms, "demusic": method})
        lvl = segment_level_gain(y, is_dialog) if level_sources else 0.0
        info["src_level_gain_db"].append({"shot": seg.get("shot"), "clip": clip, "t": round(float(seg["t"]), 3),
                                          "dialog": is_dialog, "level_db": round(lvl, 2)})
        fi = max(float(seg.get("fade_in") or 0.0), CUT_FADE_S)
        fo = max(float(seg.get("fade_out") or 0.0), CUT_FADE_S)
        y = dsp.fade(y, fi, fo)
        t = float(seg["t"])
        y = _apply_mute_ranges(y, t, seg.get("mute_ranges"))
        g = float(seg.get("gain_db", 0.0)) + lvl
        dsp.mix_at(buses["src"], y, t, g)
        if is_dialog:
            dsp.mix_at(dialog, y, t, g)
    log(f"[mixer] son source : {len(segs)} segments ({time.time() - t_c:.1f} s)")

    # --- voix off (clés séparées narrateur / entité pour le ducker d'intelligibilité)
    allowed = np.zeros((N, 2), np.float32)
    keys = {"narrator": np.zeros(N, np.float32), "entity": np.zeros(N, np.float32)}
    for v in edl.get("voices", []):
        p = _voice_path(v)
        if p is None:
            warnings.append(f"voix introuvable : {v.get('id')} ({v.get('file')})")
            continue
        y = dsp.load(p)
        t = float(v.get("t_file", v.get("t_speech", 0.0)))
        gdb = float(v.get("gain_db", 0.0))
        dsp.mix_at(allowed if v.get("allow_in_silence") else buses["vox"], y, t, gdb)
        kind = "entity" if (v.get("kind") == "entity" or str(v.get("id", "")).startswith("E_")) else "narrator"
        k0 = int(round(t * SR))
        a0, s0 = max(0, k0), max(0, -k0)
        m = min(N - a0, len(y) - s0)
        if m > 0:
            keys[kind][a0:a0 + m] += dsp.to_mono(y[s0:s0 + m]) * np.float32(dsp.lin(gdb))
    # présence des voix off (narrateur + entité) : clé de ducking des lits, du son source et des traînes SFX
    p_vox = dsp.activity(buses["vox"] + allowed, N, thr_db=-45.0, attack=DUCK_ATTACK, release=DUCK_RELEASE, hold=0.15)

    # --- cues (attaque intacte ; traîne baissée de SFX_TAIL_DUCK_DB sous les voix off)
    t_c = time.time()
    protect = np.zeros(N, np.float32)
    pre_sil: list[list[float]] = []
    for c in edl.get("cues", []):
        name = c.get("sfx")
        if name not in synth.CATALOG:
            warnings.append(f"SFX inconnu ignoré : {name}")
            continue
        params = dict(c.get("params") or {})
        gdb = float(c.get("gain_db", 0.0))
        t = float(c["t"])
        ps = params.pop("pre_silence", None)
        if ps:
            pre_sil.append([t - float(ps), t - 0.004])
        rtd = c.get("ramp_to_db", params.pop("ramp_to_db", None))
        if rtd is not None:
            params["ramp_to_db"] = float(rtd) - gdb  # absolu -> relatif au niveau de départ
        seed = int(params.pop("seed", _auto_seed(name, round(t, 3))))
        y = render_sfx(name, c.get("dur"), seed, params)
        if SFX_TAIL_DUCK_DB > 0 and len(y) > int(0.5 * SR):
            k0 = int(round(t * SR))
            idx = np.clip(np.arange(k0, k0 + len(y)), 0, N - 1)
            w = np.clip((np.arange(len(y)) / SR - 0.3) / 0.2, 0, 1)  # 0 avant 0,3 s, 1 après 0,5 s
            y = y * dsp.lin(-SFX_TAIL_DUCK_DB * w * p_vox[idx])[:, None].astype(np.float32)
        dsp.mix_at(buses["sfx"], y, t, gdb)
        a0, b0 = max(0, int(t * SR)), min(N, int((t + CUE_PROTECT_S) * SR))
        protect[a0:b0] = 1.0
    log(f"[mixer] cues : {len(edl.get('cues', []))} ({time.time() - t_c:.1f} s)")

    # --- clé de ducking (voix + répliques)
    # voix off propres : seuil bas ; répliques (prises iPhone bruitées, nivelées) : seuil plus haut pour
    # que le bruit de fond (voiture, salle) ne maintienne pas le ducking en permanence
    p_dlg = dsp.activity(dialog, N, thr_db=DIALOG_KEY_THR_DB, attack=DUCK_ATTACK, release=DUCK_RELEASE, hold=0.15)
    presence = np.maximum(p_vox, p_dlg)

    # --- lits musicaux
    t_c = time.time()
    for bed in edl.get("music", []):
        kind = bed.get("kind")
        if kind not in synth.CATALOG:
            warnings.append(f"lit inconnu ignoré : {kind}")
            continue
        t0, t1 = float(bed["t0"]), float(bed["t1"])
        if t1 <= t0:
            continue
        params = dict(bed.get("params") or {})
        seed = int(params.pop("seed", _auto_seed(kind, round(t0, 3))))
        n = int(round((t1 - t0) * SR))
        y = dsp.pad_to(render_sfx(kind, t1 - t0, seed, params), n)
        g = _bed_gain_curve(bed, t0, n)
        duck = float(bed.get("duck") or 0.0)
        if duck > 0:
            k0 = int(round(t0 * SR))
            pr = np.zeros(n)
            a0, b0 = max(0, k0), min(N, k0 + n)
            if b0 > a0:
                pr[a0 - k0:b0 - k0] = presence[a0:b0]
            g = g * dsp.lin(-duck * pr)
        dsp.mix_at(buses["music"], y * g[:, None].astype(np.float32), t0)
    log(f"[mixer] lits : {len(edl.get('music', []))} ({time.time() - t_c:.1f} s)")

    # --- son source baissé sous les voix off (le narrateur passe devant le son direct)
    if SRC_DUCK_UNDER_VO_DB > 0:
        buses["src"] *= dsp.lin(-SRC_DUCK_UNDER_VO_DB * p_vox)[:, None].astype(np.float32)

    # --- équilibre des bus, nettoyage infra-grave (< 28 Hz : inaudible, mange la marge du limiteur)
    for k in buses:
        if k != "vox":
            buses[k] = dsp.highpass(buses[k], SUBSONIC_HZ, order=2)
        if BUS_TRIM_DB.get(k):
            buses[k] *= np.float32(dsp.lin(BUS_TRIM_DB[k]))

    # --- ducker d'intelligibilité (bande parole) : le fond reste INTEL_TARGET_DB sous chaque voix
    t_c = time.time()
    p_ent = dsp.activity(keys["entity"], N, thr_db=-45.0, attack=DUCK_ATTACK, release=DUCK_RELEASE, hold=0.15)
    p_nar = dsp.activity(keys["narrator"], N, thr_db=-45.0, attack=DUCK_ATTACK, release=DUCK_RELEASE, hold=0.15)
    bg_vo = buses["music"] + buses["sfx"] + buses["src"]
    bg_dlg = buses["music"] + buses["sfx"]
    r_n = intel_reduction(keys["narrator"], bg_vo, p_nar, INTEL_TARGET_DB["narrator"])
    r_e = intel_reduction(keys["entity"], bg_vo, p_ent, INTEL_TARGET_DB["entity"])
    r_d = intel_reduction(dsp.to_mono(dialog), bg_dlg, p_dlg, INTEL_TARGET_DB["dialogue"])
    del bg_vo, bg_dlg
    nf = min(len(r_n), len(r_e), len(r_d))
    r_vo = _smooth_reduction(np.maximum(r_n[:nf], r_e[:nf]), N)
    r_all = np.maximum(r_vo, _smooth_reduction(r_d[:nf], N))
    buses["music"] = apply_split_gain(buses["music"], r_all)
    buses["sfx"] = apply_split_gain(buses["sfx"], r_all * (1.0 - protect))
    buses["src"] = apply_split_gain(buses["src"], r_vo)
    info["intel_duck"] = {"max_db": round(float(r_all.max()), 1),
                          "time_active_s": round(float(np.mean(r_all > 1.0) * dur), 1)}
    log(f"[mixer] intelligibilité : réduction max {r_all.max():.1f} dB, active {np.mean(r_all > 1.0) * dur:.1f} s "
        f"({time.time() - t_c:.1f} s)")

    # --- silences francs (après filtrage : le zéro reste absolu) ; trous « pre_silence » hors voix off
    mask = _silence_mask(N, edl.get("silence") or [])
    mask_ps = _silence_mask(N, pre_sil, ramp=0.003) if pre_sil else np.ones(N, np.float32)
    for k in buses:
        buses[k] *= (mask if k == "vox" else mask * mask_ps)[:, None]
    info["silence_non_voice_peak_db"] = [
        round(dsp.peak_db(sum(buses[k][int(float(a) * SR):int(float(b) * SR)] for k in buses)), 1)
        for a, b in (edl.get("silence") or [])]
    info["pre_silence"] = [[round(a, 3), round(b, 3), round(dsp.peak_db(
        sum(buses[k][int(a * SR):int(b * SR)] for k in ("src", "sfx", "music"))), 1)] for a, b in pre_sil]
    buses["vox"] += allowed
    info["presence_ratio"] = round(float(presence.mean()), 3)
    return buses, info


def _cut_summary(cut: np.ndarray) -> list[list[float]]:
    """Passages abaissés pour la marge du screamer : [début, fin, dB max]."""
    on = cut > 0.05
    out, k = [], 0
    edges = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    bounds = np.concatenate([[0], edges, [len(cut)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if on[a]:
            out.append([round(a / SR, 2), round(b / SR, 2), round(float(cut[a:b].max()), 2)])
    return out


def mix(edl: dict, quality: str = "final", stems: bool | None = None, level_sources: bool = True,
        log=print) -> Path:
    """Mixe et masterise l'EDL. Écrit RENDER_DIR/mix_{quality}.wav (float 32) + mix_{quality}.json."""
    t_all = time.time()
    config.ensure_dirs()
    buses, info = build_buses(edl, level_sources=level_sources, log=log)
    N = info["samples"]
    pre = buses["src"] + buses["vox"] + buses["sfx"] + buses["music"]
    ts = _screamer_time(edl)
    t_end, t_tail = _screamer_span(edl, ts) if ts is not None else (None, None)
    cut = np.zeros(N)
    ok, sd = True, {}
    g_master = None
    y = None
    for it in range(10):
        x = pre if not cut.any() else pre * dsp.lin(-cut)[:, None].astype(np.float32)
        if ts is not None:
            if g_master is None:  # gain de master provisoire pour caler la saturation sur le plafond
                dsp.master(x, MASTER_LUFS, MASTER_TP)
                g_master = dsp.last_master_gain()
            x = _densify(x, ts, t_end, t_tail, g_master)
        y = dsp.master(x, MASTER_LUFS, MASTER_TP, g0=g_master)
        g_master = dsp.last_master_gain()
        if ts is None:
            break
        sd = screamer_check(y, ts, t_tail)
        ok = (sd["margin_short_term_lu"] >= SCREAMER_MARGIN_S - 0.05
              and sd["margin_momentary_lu"] >= SCREAMER_MARGIN_M - 0.05)
        log(f"[mixer] master #{it + 1} : screamer S {sd['screamer_short_term_lufs']} / M {sd['screamer_momentary_lufs']}"
            f" ; marges S {sd['margin_short_term_lu']:+.2f} LU (2e pic {sd['other_max_short_term_t']} s),"
            f" M {sd['margin_momentary_lu']:+.2f} LU (2e pic {sd['other_max_momentary_t']} s)"
            f" -> {'OK' if ok else 'à corriger'}")
        if ok:
            break
        add = _excess_curve(N, sd, ts, t_tail)
        if not add.any():
            break
        cut = np.minimum(cut + add, MAX_SECTION_CUT_DB)
    sd.pop("_curves", None)
    y[: int(0.003 * SR)] *= dsp._ramp(int(0.003 * SR))[:, None].astype(np.float32)
    y[-int(0.01 * SR):] *= dsp._ramp(int(0.01 * SR))[::-1, None].astype(np.float32)
    out = config.RENDER_DIR / f"mix_{quality}.wav"
    dsp.save(out, y, subtype="FLOAT")
    report = {
        "file": str(out), "quality": quality, "duration": round(N / SR, 3),
        "lufs_integrated": round(dsp.lufs(y), 2), "true_peak_dbtp": round(dsp.true_peak_db(y), 2),
        "screamer_t": ts, "screamer_is_loudest": ok, "screamer_section": [ts, t_tail],
        "other_sections_cut_db": _cut_summary(cut), **sd,
        "bus_trim_db": BUS_TRIM_DB, "silence_non_voice_peak_db": info.get("silence_non_voice_peak_db"),
        "stems_lufs": {k: round(dsp.lufs(v), 2) for k, v in buses.items()},
        "intelligibility_duck": info.get("intel_duck"), "pre_silence": info.get("pre_silence"),
        "src_music": info.get("src_music"),
        "src_level_gain_db": info["src_level_gain_db"], "duck_presence_ratio": info["presence_ratio"],
        "pitch_backend": dsp.backend(), "warnings": info["warnings"], "render_s": round(time.time() - t_all, 1),
    }
    (config.RENDER_DIR / f"mix_{quality}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                                           encoding="utf-8")
    if stems if stems is not None else quality == "final":
        sd_dir = config.RENDER_DIR / f"stems_{quality}"
        for k, v in buses.items():
            dsp.save(sd_dir / f"{k}.wav", v, subtype="FLOAT")
    log(f"[mixer] {out.name} : {report['lufs_integrated']} LUFS, {report['true_peak_dbtp']} dBTP, "
        f"max S {report.get('max_short_term_lufs')} @ {report.get('max_short_term_t')} s, "
        f"screamer {'OK' if ok else 'NON'} ({report['render_s']} s)")
    for w in info["warnings"]:
        log(f"[mixer] ⚠ {w}")
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Mixage + master (−14 LUFS, −1 dBTP)")
    ap.add_argument("--edl", default=str(config.EDIT_DIR / "edl.json"))
    ap.add_argument("--quality", default="preview", choices=["preview", "final"])
    ap.add_argument("--stems", action="store_true")
    ap.add_argument("--no-level", action="store_true", help="ne pas niveler le son source par clip")
    a = ap.parse_args(argv)
    edl = json.loads(Path(a.edl).read_text(encoding="utf-8"))
    mix(edl, a.quality, stems=a.stems or None, level_sources=not a.no_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
