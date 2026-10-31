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
Extensions (intelligibilité du narrateur) : son source −6 dB et traînes SFX −5 dB sous les voix off
(l'attaque des SFX n'est jamais baissée).
Silences (`edl.silence`) : zéro absolu (rampes 5 ms) sur tous les bus, sauf voix allow_in_silence.
Master : −14 LUFS intégrés, −1,5 dBTP (marge AAC pour garantir ≤ −1 dBTP après encodage) ; le screamer doit porter le maximum de loudness court terme
(sinon sa section est relevée avant master, puis le reste est abaissé, et on re-masterise).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

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
SCREAMER_SPAN_S = 1.35         # durée de la section screamer relevée si nécessaire
MASTER_LUFS = -14.0
MASTER_TP = -1.5               # −1 dBTP exigé ; 0,5 dB de marge pour le dépassement du codage AAC (mesuré ~+0,3 dB)
SCREAMER_MAX_BOOST_DB = 4.0
MAX_SECTION_CUT_DB = 6.0       # abaissement maximal d'une section concurrente (avant master)    # au-delà, le limiteur sature : on abaisse plutôt les sections concurrentes
SUBSONIC_HZ = 28.0
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
    info: dict = {"duration": dur, "samples": N, "warnings": warnings, "src_level_gain_db": []}

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

    # --- voix off
    allowed = np.zeros((N, 2), np.float32)
    for v in edl.get("voices", []):
        p = _voice_path(v)
        if p is None:
            warnings.append(f"voix introuvable : {v.get('id')} ({v.get('file')})")
            continue
        y = dsp.load(p)
        t = float(v.get("t_file", v.get("t_speech", 0.0)))
        dsp.mix_at(allowed if v.get("allow_in_silence") else buses["vox"], y, t, float(v.get("gain_db", 0.0)))
    # présence des voix off (narrateur + entité) : clé de ducking des lits, du son source et des traînes SFX
    p_vox = dsp.activity(buses["vox"] + allowed, N, thr_db=-45.0, attack=DUCK_ATTACK, release=DUCK_RELEASE, hold=0.15)

    # --- cues (attaque intacte ; traîne baissée de SFX_TAIL_DUCK_DB sous les voix off)
    t_c = time.time()
    for c in edl.get("cues", []):
        name = c.get("sfx")
        if name not in synth.CATALOG:
            warnings.append(f"SFX inconnu ignoré : {name}")
            continue
        params = dict(c.get("params") or {})
        gdb = float(c.get("gain_db", 0.0))
        rtd = c.get("ramp_to_db", params.pop("ramp_to_db", None))
        if rtd is not None:
            params["ramp_to_db"] = float(rtd) - gdb  # absolu -> relatif au niveau de départ
        seed = int(params.pop("seed", _auto_seed(name, round(float(c["t"]), 3))))
        y = render_sfx(name, c.get("dur"), seed, params)
        t = float(c["t"])
        if SFX_TAIL_DUCK_DB > 0 and len(y) > int(0.5 * SR):
            k0 = int(round(t * SR))
            idx = np.clip(np.arange(k0, k0 + len(y)), 0, N - 1)
            w = np.clip((np.arange(len(y)) / SR - 0.3) / 0.2, 0, 1)  # 0 avant 0,3 s, 1 après 0,5 s
            y = y * dsp.lin(-SFX_TAIL_DUCK_DB * w * p_vox[idx])[:, None].astype(np.float32)
        dsp.mix_at(buses["sfx"], y, t, gdb)
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

    # --- équilibre des bus, nettoyage infra-grave (< 28 Hz : inaudible, mange la marge du limiteur),
    #     puis silences francs (après filtrage : le zéro reste absolu)
    for k in buses:
        if k != "vox":
            buses[k] = dsp.highpass(buses[k], SUBSONIC_HZ, order=2)
        if BUS_TRIM_DB.get(k):
            buses[k] *= np.float32(dsp.lin(BUS_TRIM_DB[k]))
    mask = _silence_mask(N, edl.get("silence") or [])
    for k in buses:
        buses[k] *= mask[:, None]
    info["silence_non_voice_peak_db"] = [
        round(dsp.peak_db(sum(buses[k][int(float(a) * SR):int(float(b) * SR)] for k in buses)), 1)
        for a, b in (edl.get("silence") or [])]
    buses["vox"] += allowed
    info["presence_ratio"] = round(float(presence.mean()), 3)
    return buses, info


def _screamer_ok(y: np.ndarray, ts: float) -> tuple[bool, dict]:
    t_end, S = dsp.short_term(y)
    tm, M = dsp.momentary(y)
    i = int(np.argmax(S))
    ov = np.clip(np.minimum(t_end, ts + SCREAMER_SPAN_S) - np.maximum(t_end - 3.0, ts), 0, None)
    contains = ov >= 0.6           # fenêtres qui contiennent l'essentiel du screamer
    other_mask = ov < 0.3          # fenêtres franchement ailleurs (les intermédiaires sont ignorées)
    s_scr = float(S[contains].max()) if contains.any() else -99.0
    others = S[other_mask]
    s_oth = float(others.max()) if len(others) else -99.0
    jm = int(np.argmax(M))
    m_in = bool(ts - 0.05 <= tm[jm] <= ts + SCREAMER_SPAN_S + 0.4)
    d = {"max_short_term_lufs": round(float(S[i]), 2), "max_short_term_t": round(float(t_end[i]), 2),
         "max_short_term_window": [round(float(t_end[i] - 3.0), 2), round(float(t_end[i]), 2)],
         "screamer_short_term_lufs": round(s_scr, 2), "other_max_short_term_lufs": round(s_oth, 2),
         "other_max_short_term_t": round(float(t_end[other_mask][np.argmax(others)]), 2) if len(others) else None,
         "max_momentary_lufs": round(float(M[jm]), 2), "max_momentary_t": round(float(tm[jm]), 2),
         "momentary_max_in_screamer": m_in}
    comp = other_mask & (S > s_scr - 0.3)  # toutes les sections concurrentes (fenêtres regroupées)
    spans: list[list[float]] = []
    for e, sv in zip(t_end[comp], S[comp]):
        if spans and e - 3.0 <= spans[-1][1]:
            spans[-1][1] = float(e)
            spans[-1][2] = max(spans[-1][2], float(sv))
        else:
            spans.append([float(e) - 3.0, float(e), float(sv)])
    d["competitors"] = [[round(a, 2), round(b, 2), round(v, 2)] for a, b, v in spans]
    return bool(s_scr > s_oth + 0.2), d


def _section_gain(n: int, a: float, b: float, db_: float, ramp: float = 0.01) -> np.ndarray:
    tau = np.arange(n) / SR
    w = np.clip(np.minimum((tau - a) / ramp + 1, (b - tau) / ramp + 1), 0, 1)
    return dsp.lin(db_ * w)


def mix(edl: dict, quality: str = "final", stems: bool | None = None, level_sources: bool = True,
        log=print) -> Path:
    """Mixe et masterise l'EDL. Écrit RENDER_DIR/mix_{quality}.wav (float 32) + mix_{quality}.json."""
    t_all = time.time()
    config.ensure_dirs()
    buses, info = build_buses(edl, level_sources=level_sources, log=log)
    N = info["samples"]
    pre = buses["src"] + buses["vox"] + buses["sfx"] + buses["music"]
    ts = _screamer_time(edl)
    boost = 0.0
    cut_regions: list[list[float]] = []  # [début, fin, dB]
    y = None
    ok, sd = True, {}
    prev_scr, last_step, boost_dead = None, 0.0, False
    for it in range(10):
        g = np.ones(N)
        if ts is not None and boost:
            g *= _section_gain(N, ts - 0.01, ts + SCREAMER_SPAN_S, boost)
        for a, b, cdb in cut_regions:
            g *= _section_gain(N, a, b, -cdb, 0.25)
        y = dsp.master(pre * g[:, None].astype(np.float32), MASTER_LUFS, MASTER_TP,
                       g0=dsp.last_master_gain() if it else None)
        if ts is None:
            break
        ok, sd = _screamer_ok(y, ts)
        log(f"[mixer] master #{it + 1} : screamer {sd['screamer_short_term_lufs']} LUFS (S) vs reste "
            f"{sd['other_max_short_term_lufs']} -> {'OK' if ok else 'à corriger'}")
        if ok:
            break
        deficit = sd["other_max_short_term_lufs"] - sd["screamer_short_term_lufs"] + 0.4
        scr = sd["screamer_short_term_lufs"]
        boost_dead = boost_dead or (prev_scr is not None and last_step > 0 and (scr - prev_scr) < 0.3 * last_step)
        prev_scr = scr
        if boost < SCREAMER_MAX_BOOST_DB and not boost_dead:
            last_step = min(SCREAMER_MAX_BOOST_DB - boost, max(1.0, deficit))
            boost += last_step
        else:  # le limiteur sature la section screamer : on abaisse toutes les sections concurrentes
            last_step = 0.0
            for a_cut, b_cut, sv in sd["competitors"]:
                need = 2 * (sv - scr + 0.4) + 0.5  # sections elles-mêmes limitées : coupe plus franche
                if b_cut > ts - 0.1 and a_cut < ts + SCREAMER_SPAN_S + 1.0:  # jamais sur le screamer
                    if a_cut < ts:
                        b_cut = ts - 0.1
                    else:
                        a_cut = ts + SCREAMER_SPAN_S + 1.0
                if b_cut - a_cut <= 0.3:
                    continue
                for r in cut_regions:
                    if a_cut < r[1] and b_cut > r[0]:  # recouvrement : on élargit et on approfondit
                        r[0], r[1] = min(r[0], a_cut), max(r[1], b_cut)
                        r[2] = min(r[2] + max(0.5, need), MAX_SECTION_CUT_DB)
                        break
                else:
                    cut_regions.append([a_cut, b_cut, min(max(0.5, need), MAX_SECTION_CUT_DB)])
    y[: int(0.003 * SR)] *= dsp._ramp(int(0.003 * SR))[:, None].astype(np.float32)
    y[-int(0.01 * SR):] *= dsp._ramp(int(0.01 * SR))[::-1, None].astype(np.float32)
    out = config.RENDER_DIR / f"mix_{quality}.wav"
    dsp.save(out, y, subtype="FLOAT")
    report = {
        "file": str(out), "quality": quality, "duration": round(N / SR, 3),
        "lufs_integrated": round(dsp.lufs(y), 2), "true_peak_dbtp": round(dsp.true_peak_db(y), 2),
        "screamer_t": ts, "screamer_is_loudest": ok, "screamer_boost_db": round(boost, 2),
        "other_sections_cut_db": [[round(a, 2), round(b, 2), round(c, 2)] for a, b, c in cut_regions], **sd,
        "bus_trim_db": BUS_TRIM_DB, "silence_non_voice_peak_db": info.get("silence_non_voice_peak_db"),
        "stems_lufs": {k: round(dsp.lufs(v), 2) for k, v in buses.items()},
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
