"""Traitement des voix off de « C'EST QUI LUI ? » (narrateur + entité).

Module autonome (numpy / scipy / soundfile ; pyloudnorm si disponible ; rubberband CLI
ou ffmpeg/librubberband pour la hauteur, repli numpy sinon). Portable Linux / Windows.

Conventions : audio float32, mono (N,) ou stéréo (N, 2). Sorties à 48 kHz stéréo.

API principale
--------------
narrator_chain(x, sr, **params) -> (y (N,2) 48 kHz, info)
entity_chain(x, sr, preset="name"|"ident"|"cql", seed=0, **overrides) -> (y (N,2) 48 kHz, info)
    info = {"onset_s", "speech_dur", "dur", "lufs", "peak_db", "layers": {...}}  (layers : entité seulement)
write_wav(path, y)  -> WAV 48 kHz stéréo PCM 24 bits
Briques réutilisables : pitch_shift, whisperize, plate_ir, dark_ir, reverb, reverse_swell, compress,
deess, limit, lufs, normalize_lufs, speech_bounds, eq (peaking / shelves / filtres).
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 48000
EPS = 1e-12


# =============================================================================
# E/S et utilitaires
# =============================================================================
def read(path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return x, sr


def write_wav(path, x: np.ndarray, sr: int = SR, subtype: str = "PCM_24") -> Path:
    """Écrit un WAV stéréo (PCM 24 bits par défaut)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = as_stereo(x)
    sf.write(str(path), np.clip(y, -1.0, 1.0).astype(np.float32), sr, subtype=subtype)
    return path


def mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def as_stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x[:, :2]


def pan(x: np.ndarray, position: float) -> np.ndarray:
    """Panoramique à puissance constante d'un signal mono ; position ∈ [-1 (gauche), 1 (droite)]."""
    x = mono(x)
    a = (np.clip(position, -1, 1) + 1) * math.pi / 4
    return np.stack([x * math.cos(a), x * math.sin(a)], axis=1) * math.sqrt(2)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    g = math.gcd(int(sr_in), int(sr_out))
    return signal.resample_poly(x, sr_out // g, sr_in // g, axis=0).astype(np.float32)


def db2lin(db: float) -> float:
    return float(10 ** (db / 20))


def lin2db(v) -> float:
    return float(20 * np.log10(max(float(v), EPS)))


def pad(x: np.ndarray, head: int = 0, tail: int = 0) -> np.ndarray:
    widths = [(head, tail)] + [(0, 0)] * (x.ndim - 1)
    return np.pad(x, widths)


def fade(x: np.ndarray, sr: int, fade_in: float = 0.0, fade_out: float = 0.0) -> np.ndarray:
    y = np.array(x, dtype=np.float32, copy=True)
    n_in, n_out = int(fade_in * sr), int(fade_out * sr)
    if n_in > 0:
        r = np.sin(np.linspace(0, math.pi / 2, n_in)) ** 2
        y[:n_in] *= r if y.ndim == 1 else r[:, None]
    if n_out > 0:
        r = np.cos(np.linspace(0, math.pi / 2, n_out)) ** 2
        y[-n_out:] *= r if y.ndim == 1 else r[:, None]
    return y


def mix_at(dst: np.ndarray, src: np.ndarray, offset: int, gain: float = 1.0) -> np.ndarray:
    """Ajoute src dans dst à partir de l'échantillon offset (agrandit dst si besoin)."""
    need = offset + len(src)
    if need > len(dst):
        dst = pad(dst, 0, need - len(dst))
    dst[offset:need] += gain * src
    return dst


# =============================================================================
# Filtres (biquads RBJ, en SOS)
# =============================================================================
def _biquad(kind: str, f0: float, sr: int, q: float = 0.7071, db: float = 0.0) -> np.ndarray:
    f0 = min(max(f0, 10.0), sr * 0.49)
    a_ = 10 ** (db / 40)
    w = 2 * math.pi * f0 / sr
    cw, sw = math.cos(w), math.sin(w)
    alpha = sw / (2 * q)
    if kind == "peak":
        b = [1 + alpha * a_, -2 * cw, 1 - alpha * a_]
        a = [1 + alpha / a_, -2 * cw, 1 - alpha / a_]
    elif kind in ("lowshelf", "highshelf"):
        sq = 2 * math.sqrt(a_) * alpha
        if kind == "lowshelf":
            b = [a_ * ((a_ + 1) - (a_ - 1) * cw + sq), 2 * a_ * ((a_ - 1) - (a_ + 1) * cw), a_ * ((a_ + 1) - (a_ - 1) * cw - sq)]
            a = [(a_ + 1) + (a_ - 1) * cw + sq, -2 * ((a_ - 1) + (a_ + 1) * cw), (a_ + 1) + (a_ - 1) * cw - sq]
        else:
            b = [a_ * ((a_ + 1) + (a_ - 1) * cw + sq), -2 * a_ * ((a_ - 1) + (a_ + 1) * cw), a_ * ((a_ + 1) + (a_ - 1) * cw - sq)]
            a = [(a_ + 1) - (a_ - 1) * cw + sq, 2 * ((a_ - 1) - (a_ + 1) * cw), (a_ + 1) - (a_ - 1) * cw - sq]
    elif kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    else:
        raise ValueError(kind)
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return np.concatenate([b, a])[None, :]


def _sos_apply(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def peaking(x, sr, f, db, q=1.0):
    return _sos_apply(x, _biquad("peak", f, sr, q, db))


def lowshelf(x, sr, f, db, q=0.7071):
    return _sos_apply(x, _biquad("lowshelf", f, sr, q, db))


def highshelf(x, sr, f, db, q=0.7071):
    return _sos_apply(x, _biquad("highshelf", f, sr, q, db))


def highpass(x, sr, f, order=2):
    return _sos_apply(x, signal.butter(order, f, "highpass", fs=sr, output="sos"))


def lowpass(x, sr, f, order=2):
    return _sos_apply(x, signal.butter(order, min(f, sr * 0.49), "lowpass", fs=sr, output="sos"))


def bandpass(x, sr, lo, hi, order=2):
    return _sos_apply(x, signal.butter(order, [lo, min(hi, sr * 0.49)], "bandpass", fs=sr, output="sos"))


def eq(x: np.ndarray, sr: int, bands) -> np.ndarray:
    """Applique une liste de bandes : ("hp", f) | ("lp", f) | ("peak", f, db, q) | ("lowshelf"|"highshelf", f, db)."""
    for band in bands:
        kind = band[0]
        if kind == "hp":
            x = highpass(x, sr, band[1], band[2] if len(band) > 2 else 2)
        elif kind == "lp":
            x = lowpass(x, sr, band[1], band[2] if len(band) > 2 else 2)
        elif kind == "peak":
            x = peaking(x, sr, band[1], band[2], band[3] if len(band) > 3 else 1.0)
        elif kind == "lowshelf":
            x = lowshelf(x, sr, band[1], band[2])
        elif kind == "highshelf":
            x = highshelf(x, sr, band[1], band[2])
        else:
            raise ValueError(f"bande EQ inconnue : {band}")
    return x


# =============================================================================
# Dynamique
# =============================================================================
def _envelope(level: np.ndarray, sr: int, attack_ms: float, release_ms: float, block: int = 48) -> np.ndarray:
    """Suiveur d'enveloppe attaque/relâchement (calculé par blocs d'1 ms puis interpolé)."""
    n = len(level)
    nb = max(1, int(math.ceil(n / block)))
    lv = np.pad(level, (0, nb * block - n)).reshape(nb, block).max(axis=1)
    ka = math.exp(-block / (sr * attack_ms / 1000.0))
    kr = math.exp(-block / (sr * release_ms / 1000.0))
    env = np.empty(nb, dtype=np.float64)
    e = 0.0
    for i in range(nb):
        v = lv[i]
        e = ka * e + (1 - ka) * v if v > e else kr * e + (1 - kr) * v
        env[i] = e
    t_blocks = (np.arange(nb) + 0.5) * block
    return np.interp(np.arange(n), t_blocks, env).astype(np.float32)


def compress(x: np.ndarray, sr: int, thr_db: float = -24.0, ratio: float = 2.5, attack_ms: float = 10.0,
             release_ms: float = 120.0, knee_db: float = 6.0, makeup_db: float = 0.0) -> np.ndarray:
    """Compresseur feed-forward à genou doux (détection RMS ~5 ms, liée en stéréo)."""
    m = mono(x).astype(np.float64)
    rms = np.sqrt(signal.lfilter([1 - math.exp(-1 / (0.005 * sr))], [1, -math.exp(-1 / (0.005 * sr))], m * m) + EPS)
    env = _envelope(rms, sr, attack_ms, release_ms)
    lvl = 20 * np.log10(env + EPS)
    over = lvl - thr_db
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    g = (10 ** ((makeup_db - gr) / 20)).astype(np.float32)
    return (x * (g if x.ndim == 1 else g[:, None])).astype(np.float32)


def deess(x: np.ndarray, sr: int, freq: float = 5500.0, thr_db: float = -30.0, ratio: float = 3.0,
          max_cut_db: float = 8.0, attack_ms: float = 1.0, release_ms: float = 60.0) -> np.ndarray:
    """De-esser à bande partagée : seule la bande au-dessus de `freq` est atténuée (reconstruction exacte)."""
    sos = signal.butter(2, freq, "highpass", fs=sr, output="sos")
    hi = signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=0).astype(np.float32)  # phase nulle
    lo = x - hi
    det = np.abs(mono(bandpass(x, sr, freq, min(freq * 2.2, sr * 0.45))))
    env = _envelope(det, sr, attack_ms, release_ms)
    lvl = 20 * np.log10(env + EPS)
    cut = np.clip((lvl - thr_db) * (1 - 1 / ratio), 0.0, max_cut_db)
    g = (10 ** (-cut / 20)).astype(np.float32)
    return (lo + hi * (g if x.ndim == 1 else g[:, None])).astype(np.float32)


def deess_relative(x: np.ndarray, sr: int, below_vowel_db: float = 9.0, freq: float = 3600.0,
                   max_cut_db: float = 14.0, release_ms: float = 40.0) -> np.ndarray:
    """De-esser RELATIF (indépendant du niveau d'entrée) : les trames de sifflantes (énergie 4–9 kHz
    dominante) sont ramenées à `below_vowel_db` dB sous le niveau des voyelles (p90 des trames où les
    basses 100 Hz–1 kHz dominent). Gain appliqué dans le domaine STFT (phase nulle) au-dessus de `freq`
    seulement : la consonne reste nette mais ne siffle plus plus fort que la voyelle (défaut typique
    des voix TTS soufflées)."""
    xm = np.asarray(x, dtype=np.float64)
    chans = [xm] if xm.ndim == 1 else [xm[:, c] for c in range(xm.shape[1])]
    n_fft, hop = 1024, 128
    m = mono(xm)
    X = _stft(m, n_fft, hop)
    P = np.abs(X) ** 2
    fr = np.fft.rfftfreq(n_fft, 1 / sr)
    hf = P[(fr >= 4000) & (fr < 9000)].sum(axis=0)
    lf = P[(fr >= 100) & (fr < 1000)].sum(axis=0)
    tot = P[fr >= 60].sum(axis=0) + EPS
    lvl = 10 * np.log10(tot)
    act = lvl > lvl.max() - 35
    vow = act & (lf > 10 * hf)
    if vow.sum() < 5:
        return np.asarray(x, dtype=np.float32)
    ref = np.percentile(lvl[vow], 90) - below_vowel_db
    hi_share = np.clip(P[fr >= freq].sum(axis=0) / tot, 1e-6, 1.0)
    need = np.clip(lvl - ref, 0.0, None)
    rem = 10 ** (-need / 10) - (1 - hi_share)  # énergie que doit garder la bande haute
    cut = -10 * np.log10(np.clip(rem / hi_share, 10 ** (-max_cut_db / 10), 1.0))
    cut = np.where((need > 0) & (hf > lf), cut, 0.0)
    # attaque immédiate (la trame est déjà centrée), relâche exponentielle
    kr = math.exp(-hop / (sr * release_ms / 1000.0))
    sm = np.empty_like(cut)
    e = 0.0
    for i, v in enumerate(cut):
        e = v if v > e else kr * e + (1 - kr) * v
        sm[i] = e
    w = np.clip((np.log2(np.maximum(fr, 1.0) / freq) + 0.5), 0.0, 1.0)[:, None]  # transition d'1/2 octave
    G = 10 ** (-(w * sm[None, :]) / 20)
    out = [_istft(_stft(c, n_fft, hop) * G, n_fft, hop, len(c)) for c in chans]
    return (out[0] if xm.ndim == 1 else np.stack(out, axis=1)).astype(np.float32)


def true_peak_db(x: np.ndarray, oversample: int = 4) -> float:
    y = signal.resample_poly(np.asarray(x, dtype=np.float64), oversample, 1, axis=0)
    return lin2db(np.max(np.abs(y)) if y.size else 0.0)


def limit(x: np.ndarray, sr: int, ceiling_db: float = -3.0, lookahead_ms: float = 4.0, release_ms: float = 80.0,
          oversample: int = 4) -> np.ndarray:
    """Limiteur à anticipation (crête vraie, suréchantillonnée) ; gain lissé, jamais au-dessus de 1."""
    ceil = db2lin(ceiling_db) * 0.995
    xs = as_stereo(x) if x.ndim == 2 else x[:, None]
    up = signal.resample_poly(xs.astype(np.float64), oversample, 1, axis=0)
    pk_up = np.max(np.abs(up), axis=1)
    pk = pk_up[: len(pk_up) // oversample * oversample].reshape(-1, oversample).max(axis=1)
    pk = np.pad(pk, (0, len(xs) - len(pk)), constant_values=0)
    need = np.minimum(1.0, ceil / np.maximum(pk, EPS))
    la = max(1, int(lookahead_ms * sr / 1000))
    # minimum glissant sur la fenêtre d'anticipation (le gain baisse AVANT la crête)
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    g = minimum_filter1d(need, size=2 * la + 1, mode="nearest")
    g = uniform_filter1d(g, size=la, mode="nearest")
    # relâchement exponentiel
    kr = math.exp(-1.0 / (sr * release_ms / 1000.0))
    out = np.empty_like(g)
    e = 1.0
    blk = 32
    for i in range(0, len(g), blk):
        seg = g[i:i + blk]
        m = seg.min()
        e = m if m < e else kr ** len(seg) * e + (1 - kr ** len(seg)) * m
        out[i:i + blk] = np.minimum(seg, e)
    y = xs * out[:, None]
    y = y.astype(np.float32)
    return y if x.ndim == 2 else y[:, 0]


def lufs(x: np.ndarray, sr: int = SR) -> float:
    """Sonie intégrée (ITU-R BS.1770). pyloudnorm si présent, sinon approximation K-pondérée."""
    xs = as_stereo(x).astype(np.float64)
    if len(xs) < int(0.45 * sr):
        xs = np.pad(xs, ((0, int(0.45 * sr) - len(xs)), (0, 0)))
    try:
        import pyloudnorm
        return float(pyloudnorm.Meter(sr).integrated_loudness(xs))
    except ImportError:
        k = np.concatenate([_biquad("highshelf", 1681.97, sr, 0.7072, 4.0), _biquad("highpass", 38.13, sr, 0.5003)])
        y = signal.sosfilt(k, xs, axis=0)
        blk, hop = int(0.4 * sr), int(0.1 * sr)
        ms = np.array([np.sum(np.mean(y[i:i + blk] ** 2, axis=0)) for i in range(0, len(y) - blk + 1, hop)])
        l_ = -0.691 + 10 * np.log10(ms + EPS)
        ms = ms[l_ > -70]
        rel = -0.691 + 10 * np.log10(np.mean(ms) + EPS) - 10
        ms = ms[-0.691 + 10 * np.log10(ms + EPS) > rel]
        return float(-0.691 + 10 * np.log10(np.mean(ms) + EPS))


def normalize_lufs(x: np.ndarray, sr: int, target: float = -20.0, peak_ceiling_db: float = -3.0) -> np.ndarray:
    """Normalise à `target` LUFS intégrés ; si la crête vraie dépasse `peak_ceiling_db`, limiteur puis
    renormalisation (jusqu'à 4 passes). Sortie garantie ≤ plafond (sonie alors au plus près de la cible)."""
    y = x.astype(np.float32)
    for _ in range(4):
        y = y * db2lin(target - lufs(y, sr))
        if true_peak_db(y) <= peak_ceiling_db:
            return y.astype(np.float32)
        y = limit(y, sr, ceiling_db=peak_ceiling_db - 0.15)
    return y.astype(np.float32)


# =============================================================================
# Hauteur / durée
# =============================================================================
def _which(name: str) -> str | None:
    return shutil.which(name) or shutil.which(name + ".exe")


def pitch_shift(x: np.ndarray, sr: int, semitones: float, formant: bool = True, tempo: float = 1.0,
                fine: bool = True) -> np.ndarray:
    """Transposition (± formants préservés) et changement de tempo (tempo > 1 = plus rapide).

    Ordre des moteurs : rubberband CLI (R3) → ffmpeg + librubberband → repli numpy (vocodeur de phase).
    """
    if abs(semitones) < 1e-3 and abs(tempo - 1.0) < 1e-4:
        return np.asarray(x, dtype=np.float32)
    for engine in (_pitch_rubberband_cli, _pitch_ffmpeg):
        try:
            y = engine(x, sr, semitones, formant, tempo, fine)
            if y is not None and len(y) > 0:
                return y
        except (OSError, subprocess.SubprocessError, RuntimeError):
            continue
    return _pitch_numpy(x, sr, semitones, formant, tempo)


def _tmp_wav_pair():
    d = tempfile.mkdtemp(prefix="voicefx_")
    return Path(d), Path(d) / "in.wav", Path(d) / "out.wav"


def _pitch_rubberband_cli(x, sr, semitones, formant, tempo, fine):
    exe = _which("rubberband")
    if exe is None:
        return None
    d, fin, fout = _tmp_wav_pair()
    try:
        sf.write(str(fin), np.asarray(x, dtype=np.float32), sr, subtype="FLOAT")
        cmd = [exe, "-q"]
        if fine:
            cmd.append("-3")
        if formant:
            cmd.append("-F")
        cmd += ["-p", f"{semitones:.4f}"]
        if abs(tempo - 1.0) > 1e-4:
            cmd += ["-T", f"{tempo:.5f}"]
        cmd += [str(fin), str(fout)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        y, _ = sf.read(str(fout), dtype="float32")
        return y
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _pitch_ffmpeg(x, sr, semitones, formant, tempo, fine):
    exe = _which("ffmpeg")
    if exe is None:
        return None
    d, fin, fout = _tmp_wav_pair()
    try:
        sf.write(str(fin), np.asarray(x, dtype=np.float32), sr, subtype="FLOAT")
        ratio = 2 ** (semitones / 12)
        af = (f"rubberband=pitch={ratio:.6f}:tempo={tempo:.6f}:formant={'preserved' if formant else 'shifted'}"
              f":pitchq=quality:transients=smooth")
        subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", "-i", str(fin), "-af", af,
                        "-c:a", "pcm_f32le", str(fout)], check=True, timeout=300, capture_output=True)
        y, _ = sf.read(str(fout), dtype="float32")
        return y
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _stft(x, n_fft, hop):
    return signal.stft(x, nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft, boundary="even", padded=True)[2]


def _istft(X, n_fft, hop, n):
    y = signal.istft(X, nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft, boundary=True)[1]
    return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)


def _cepstral_envelope(mag: np.ndarray, lifter: int) -> np.ndarray:
    """Enveloppe spectrale lissée par liftrage cepstral ; mag (F, T) → (F, T)."""
    logm = np.log(mag + 1e-7)
    cep = np.fft.irfft(logm, axis=0)
    cep[lifter:-lifter] = 0.0
    return np.exp(np.fft.rfft(cep, axis=0).real)


def _pitch_numpy(x, sr, semitones, formant, tempo):
    """Repli sans dépendance : vocodeur de phase + rééchantillonnage, correction d'enveloppe cepstrale."""
    xm = mono(x).astype(np.float64)
    ratio = 2 ** (semitones / 12)
    stretch = ratio / tempo  # étirement temporel avant rééchantillonnage
    n_fft, hop = 2048, 256
    X = _stft(xm, n_fft, hop)
    n_frames = X.shape[1]
    t_out = np.arange(0, n_frames - 1, 1 / stretch)
    phase = np.angle(X[:, 0])
    omega = 2 * np.pi * hop * np.arange(X.shape[0]) / n_fft
    Y = np.zeros((X.shape[0], len(t_out)), dtype=complex)
    for k, t in enumerate(t_out):
        i = int(t)
        a = t - i
        mag = (1 - a) * np.abs(X[:, i]) + a * np.abs(X[:, i + 1])
        Y[:, k] = mag * np.exp(1j * phase)
        dphi = np.angle(X[:, i + 1]) - np.angle(X[:, i]) - omega
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        phase += omega + dphi
    y = _istft(Y, n_fft, hop, int(len(xm) * stretch))
    y = signal.resample(y, max(1, int(round(len(y) / ratio)))).astype(np.float32)
    if formant and abs(semitones) > 0.05:
        Xo = _stft(xm, n_fft, hop)
        Yo = _stft(y, n_fft, hop)
        lif = int(0.0025 * sr)
        env_src = np.median(_cepstral_envelope(np.abs(Xo), lif), axis=1, keepdims=True)
        env_dst = np.median(_cepstral_envelope(np.abs(Yo), lif), axis=1, keepdims=True)
        Yo *= np.clip(env_src / (env_dst + 1e-9), 0.1, 10)
        y = _istft(Yo, n_fft, hop, len(y))
    return y.astype(np.float32)


def detune(x: np.ndarray, cents: float) -> np.ndarray:
    """Désaccord léger par rééchantillonnage (la durée varie d'autant : effet de doublage naturel)."""
    if abs(cents) < 0.01:
        return np.asarray(x, dtype=np.float32)
    ratio = 2 ** (cents / 1200)
    n_out = max(1, int(round(len(x) / ratio)))
    t = np.arange(n_out) * ratio
    return np.interp(t, np.arange(len(x)), mono(x)).astype(np.float32)


# =============================================================================
# Réverbérations (IR synthétiques déterministes)
# =============================================================================
def _decaying_noise_ir(sr, decay_s, predelay_ms, bands, seed, width, attack_ms=2.0):
    rng = np.random.default_rng(seed)
    n = int(sr * decay_s * 1.15)
    t = np.arange(n) / sr
    ir = np.zeros((n, 2))
    edges = [20.0] + [b[0] for b in bands[1:]] + [sr * 0.49]
    for (fc, t60_mult, gain_db), lo, hi in zip(bands, edges[:-1], edges[1:]):
        noise = rng.standard_normal((n, 2))
        mid = noise.mean(axis=1, keepdims=True)
        noise = mid * (1 - width) + noise * width  # décorrélation stéréo réglable
        sos = signal.butter(2, [max(lo, 20.0), min(hi, sr * 0.49)], "bandpass", fs=sr, output="sos")
        band = signal.sosfilt(sos, noise, axis=0)
        t60 = max(0.05, decay_s * t60_mult)
        band *= (10 ** (-3 * t / t60))[:, None] * db2lin(gain_db)
        ir += band
    a = int(attack_ms * sr / 1000)
    if a > 1:
        ir[:a] *= np.linspace(0, 1, a)[:, None]
    ir = np.pad(ir, ((int(predelay_ms * sr / 1000), 0), (0, 0)))
    ir /= np.sqrt(np.sum(ir ** 2) / 2) + EPS
    return ir.astype(np.float32)


def plate_ir(sr: int = SR, decay_s: float = 1.2, predelay_ms: float = 12.0, damping: float = 0.45,
             seed: int = 7) -> np.ndarray:
    """IR type « plate » : dense, sans réflexions discrètes, aigus qui s'éteignent plus vite (`damping`)."""
    bands = [(20, 1.05, -1.0), (250, 1.0, 0.0), (1000, 0.95, 0.0), (3000, 1 - 0.5 * damping, -1.0),
             (6000, 1 - 0.8 * damping, -3.0), (10000, 1 - 0.95 * damping, -6.0)]
    return _decaying_noise_ir(sr, decay_s, predelay_ms, bands, seed, width=0.9)


def dark_ir(sr: int = SR, decay_s: float = 2.8, predelay_ms: float = 25.0, seed: int = 11) -> np.ndarray:
    """IR sombre et longue (espace caverneux) pour la réverbe inversée de l'entité."""
    bands = [(20, 1.15, 0.0), (200, 1.05, 0.0), (800, 0.9, -1.0), (2000, 0.65, -4.0),
             (4500, 0.4, -9.0), (8000, 0.25, -16.0)]
    return _decaying_noise_ir(sr, decay_s, predelay_ms, bands, seed, width=1.0)


def reverb(x: np.ndarray, ir: np.ndarray, wet: float = 0.12, keep_tail: bool = True) -> np.ndarray:
    """Mix sec/humide ; la partie humide est normalisée à la même énergie que le sec. Sortie stéréo."""
    dry = as_stereo(x)
    w = np.stack([signal.oaconvolve(dry[:, c], ir[:, c]) for c in range(2)], axis=1)
    w *= np.sqrt(np.sum(dry ** 2) / (np.sum(w ** 2) + EPS))
    n = len(w) if keep_tail else len(dry)
    out = np.zeros((n, 2), dtype=np.float32)
    out[: len(dry)] += (1 - wet) * dry
    out += wet * w[:n]
    return out.astype(np.float32)


def reverse_swell(x: np.ndarray, ir: np.ndarray, length_s: float, sr: int = SR, curve: float = 1.6) -> np.ndarray:
    """Pré-swell en réverbe inversée : reverse(reverb(reverse(x))) ne gardant que la partie AVANT x.

    Retourne (L, 2), L = length_s * sr ; la montée finit exactement au début de x (à concaténer devant).
    """
    xs = as_stereo(x)[::-1]
    w = np.stack([signal.oaconvolve(xs[:, c], ir[:, c]) for c in range(2)], axis=1)[::-1]
    # w couvre [-(len(ir)-1), len(x)) en temps « avant x » : on garde les L échantillons qui précèdent x
    pre = w[: len(ir) - 1]
    L = min(int(length_s * sr), len(pre))
    pre = pre[len(pre) - L:]
    pre *= np.sqrt(np.sum(xs ** 2) / (np.sum(w ** 2) + EPS))
    env = np.linspace(0, 1, L) ** curve  # montée vers le mot
    return (pre * env[:, None]).astype(np.float32)


# =============================================================================
# Chuchotement : excitation remplacée par du bruit (enveloppe cepstrale × bruit blanc en STFT)
# =============================================================================
def voicing(X: np.ndarray, sr: int, n_fft: int, fmin: float = 60.0, fmax: float = 400.0) -> np.ndarray:
    """Degré de voisement par trame (0..1) : pic d'autocorrélation normalisée dans la plage de F0."""
    ac = np.fft.irfft(np.abs(X) ** 2, n=n_fft, axis=0)
    lo, hi = int(sr / fmax), min(int(sr / fmin), n_fft // 2 - 1)
    peak = ac[lo:hi].max(axis=0) / (ac[0] + EPS)
    v = np.clip((peak - 0.25) / 0.35, 0.0, 1.0)
    return np.convolve(v, np.ones(3) / 3, mode="same")  # lissage léger (≈ 3 trames)


def harmonic_enhance(x: np.ndarray, sr: int, amount_db: float = -6.0, fmax: float = 5000.0,
                     width: float = 0.17) -> np.ndarray:
    """Réduit le souffle d'une voix voilée : dans les trames voisées, atténue l'énergie entre les harmoniques
    (peigne gaussien sur k·F0, F0 par autocorrélation) sous `fmax`. Les consonnes ne sont pas touchées."""
    xm = mono(x).astype(np.float64)
    n_fft, hop = 2048, 256
    X = _stft(xm, n_fft, hop)
    ac = np.fft.irfft(np.abs(X) ** 2, n=n_fft, axis=0)
    lo, hi = int(sr / 400), int(sr / 60)
    seg = ac[lo:hi]
    lag = np.argmax(seg, axis=0)
    # interpolation parabolique du pic
    i = np.clip(lag, 1, len(seg) - 2)
    cols = np.arange(seg.shape[1])
    a_, b_, c_ = seg[i - 1, cols], seg[i, cols], seg[i + 1, cols]
    delta = 0.5 * (a_ - c_) / np.where(np.abs(a_ - 2 * b_ + c_) < EPS, EPS, a_ - 2 * b_ + c_)
    f0 = sr / (lo + i + np.clip(delta, -0.5, 0.5))
    v = voicing(X, sr, n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)[:, None]
    d = freqs / f0[None, :]
    dist = (d - np.round(d)) * f0[None, :]
    mask = np.exp(-0.5 * (dist / (width * f0[None, :])) ** 2)
    mask[(freqs[:, 0] < 0.6 * f0.min()), :] = 1.0
    g = mask + (1 - mask) * db2lin(amount_db)
    g[freqs[:, 0] > fmax, :] = 1.0
    G = v[None, :] * g + (1 - v[None, :])
    y = _istft(X * G, n_fft, hop, len(xm))
    return y.astype(np.float32)


def whisperize(x: np.ndarray, sr: int, seed: int = 0, lifter_ms: float = 2.2, n_fft: int | None = None,
               hop: int | None = None, keep_unvoiced: bool = True) -> np.ndarray:
    """Convertit une voix parlée en chuchotement.

    Trames voisées : enveloppe spectrale (liftrage cepstral, supprime les harmoniques) × bruit blanc
    de phase aléatoire, énergie de trame conservée. Trames non voisées (fricatives, explosions) :
    conservées telles quelles (`keep_unvoiced`), elles sont déjà du bruit -> consonnes nettes.
    """
    xm = mono(x).astype(np.float64)
    n_fft = n_fft or int(2 ** round(math.log2(0.032 * sr)))
    hop = hop or n_fft // 8
    X = _stft(xm, n_fft, hop)
    mag = np.abs(X)
    env = _cepstral_envelope(mag, max(4, int(lifter_ms * 1e-3 * sr)))
    rng = np.random.default_rng(seed)
    N = _stft(rng.standard_normal(len(xm)), n_fft, hop)
    N /= np.sqrt(np.mean(np.abs(N) ** 2, axis=0, keepdims=True)) + EPS
    Y = env * N
    e_x = np.sum(mag ** 2, axis=0, keepdims=True)
    e_y = np.sum(np.abs(Y) ** 2, axis=0, keepdims=True)
    Y *= np.sqrt(e_x / (e_y + EPS))  # conserve l'énergie de chaque trame
    if keep_unvoiced:
        v = voicing(X, sr, n_fft)[None, :]
        Y = v * Y + (1 - v) * X
    y = _istft(Y, n_fft, hop, len(xm))
    return y.astype(np.float32)


def breath(sr: int, dur: float = 0.35, seed: int = 0, inhale: bool = True) -> np.ndarray:
    """Petite respiration (inspiration) très proche : bruit filtré en formants, enveloppe montante."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    nz = rng.standard_normal(n)
    y = bandpass(nz, sr, 350, 5500, order=2)
    y = peaking(y, sr, 1300, 6, 1.2)
    y = peaking(y, sr, 2600, 4, 1.5)
    t = np.linspace(0, 1, n)
    env = (np.sin(np.pi * t) ** 1.5) * (t ** 0.6 if inhale else (1 - t) ** 0.6)
    return (y * env / (np.max(np.abs(y)) + EPS)).astype(np.float32)


def exciter(x: np.ndarray, sr: int, amount_db: float = -18.0, src_lo: float = 3500.0, src_hi: float = 9000.0,
            out_hp: float = 9000.0) -> np.ndarray:
    """Recrée de l'« air » au-dessus de la bande utile d'une voix 22 kHz (harmoniques de la bande 3,5–9 kHz)."""
    band = bandpass(x, sr, src_lo, src_hi, order=2)
    ref = np.sqrt(np.mean(band ** 2)) + EPS
    gen = np.tanh(band / ref * 1.5) * ref        # saturation douce -> harmoniques paires/impaires
    gen = np.abs(gen) * np.sign(band)            # + redressement partiel (harmoniques paires)
    gen = highpass(gen, sr, out_hp, order=4)
    gen = lowpass(gen, sr, min(12000.0, sr * 0.45), order=4)  # pente naturelle au-dessus de 12 kHz
    return (x + gen * db2lin(amount_db) * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(gen ** 2)) + EPS))).astype(np.float32)


# =============================================================================
# Analyse
# =============================================================================
def rms_env(x: np.ndarray, sr: int, win_ms: float = 10.0) -> np.ndarray:
    m = mono(x).astype(np.float64)
    w = max(1, int(win_ms * sr / 1000))
    return np.sqrt(np.convolve(m * m, np.ones(w) / w, mode="same") + EPS)


def speech_bounds(x: np.ndarray, sr: int, thr_db: float = -42.0, win_ms: float = 5.0,
                  abs_floor_db: float = -62.0) -> tuple[float, float]:
    """(début, fin) de la parole en secondes : enveloppe RMS au-dessus de max−|thr_db| (et d'un plancher absolu)."""
    env = rms_env(x, sr, win_ms)
    ref = np.max(env)
    thr = max(ref * db2lin(thr_db), db2lin(abs_floor_db))
    idx = np.where(env > thr)[0]
    if len(idx) == 0:
        return 0.0, len(env) / sr
    return float(idx[0] / sr), float((idx[-1] + 1) / sr)


def trim_to_speech(x: np.ndarray, sr: int, head_s: float = 0.0, tail_s: float = 0.0, thr_db: float = -42.0):
    a, b = speech_bounds(x, sr, thr_db)
    i0 = max(0, int((a - head_s) * sr))
    i1 = min(len(x), int((b + tail_s) * sr))
    return x[i0:i1], a - i0 / sr


# =============================================================================
# Chaîne NARRATEUR
# =============================================================================
NARRATOR_DEFAULTS = dict(
    semitones=-2.0,          # abaissement avec préservation des formants (grave mais naturel)
    tempo=1.0,               # > 1 accélère, < 1 ralentit (rubberband)
    formant=True,
    eq=[("hp", 65, 2), ("lowshelf", 160, 2.5), ("peak", 320, -2.0, 1.1), ("peak", 2800, 1.2, 0.9),
        ("highshelf", 9000, -1.5)],
    deess=dict(freq=5200, thr_db=-31, ratio=3.0, max_cut_db=7),
    sib_db=12.0,             # sifflantes ramenées à 12 dB sous les voyelles (de-esser relatif) ; None = off
    comp=dict(thr_db=-24, ratio=2.4, attack_ms=12, release_ms=140, knee_db=8, makeup_db=0),
    reverb=dict(decay_s=1.2, predelay_ms=14, damping=0.45, wet=0.12, seed=7),
    exciter_db=None,         # air recréé au-dessus de ~9 kHz (voix à bande étroite), ex. -24
    debreath_db=None,        # atténuation du souffle entre harmoniques (voix voilée), ex. -6
    head_s=0.06,             # silence avant le début de la parole
    tail_max_s=0.55,         # queue de réverbe gardée après la fin de la parole (fondu)
    tail_fade_s=0.12,
    lufs=-20.0,
    peak_db=-3.0,
)


def narrator_chain(x: np.ndarray, sr: int, **params) -> tuple[np.ndarray, dict]:
    p = {**NARRATOR_DEFAULTS, **params}
    y = resample(mono(x), sr, SR)
    y, _ = trim_to_speech(y, SR, 0.03, 0.03)
    if p["debreath_db"] is not None:
        y = harmonic_enhance(y, SR, p["debreath_db"])
    y = pitch_shift(y, SR, p["semitones"], formant=p["formant"], tempo=p["tempo"])
    y = eq(y, SR, p["eq"])
    if p["exciter_db"] is not None:
        y = exciter(y, SR, p["exciter_db"])
    y = deess(y, SR, **p["deess"])
    if p.get("sib_db") is not None:
        y = deess_relative(y, SR, below_vowel_db=p["sib_db"])
    y = compress(y, SR, **p["comp"])
    # recadrage précis sur la parole (sec) avant la réverbe
    a, b = speech_bounds(y, SR)
    i0 = max(0, int(a * SR) - int(0.004 * SR))
    dry = y[i0:int(b * SR) + int(0.004 * SR)]
    dry = fade(dry, SR, 0.004, 0.012)
    head = int(p["head_s"] * SR) - int(0.004 * SR)
    dry = pad(dry, max(0, head), 0)
    r = p["reverb"]
    out = reverb(dry, plate_ir(SR, r["decay_s"], r["predelay_ms"], r["damping"], r["seed"]), r["wet"])
    a, b = speech_bounds(dry, SR)
    end = min(len(out), int((b + p["tail_max_s"]) * SR))
    out = fade(out[:end], SR, 0.0, p["tail_fade_s"])
    out = normalize_lufs(out, SR, p["lufs"], p["peak_db"])
    info = dict(onset_s=round(a, 3), speech_dur=round(b - a, 3), dur=round(len(out) / SR, 3),
                lufs=round(lufs(out, SR), 2), peak_db=round(true_peak_db(out), 2))
    return out, info


# =============================================================================
# Chaîne ENTITÉ
# =============================================================================
ENTITY_PRESETS = {
    # layers = [(cents, retard ms, panoramique, gain)] des couches chuchotées transposées (formants vers le bas)
    # La couche sèche « collée à l'oreille » (ear) domine : c'est elle qui porte l'articulation (ASR) ;
    # les couches graves désaccordées et le sous-grave donnent la menace sans masquer les formants.
    # prénoms murmurés pendant les flashs de possession (~0,6 s) : menace, pré-swell court + micro-silence
    "name": dict(semitones=-5.0, tempo=1.0, layers=[(0.0, 0.0, 0.0, 0.3), (+20, 11, -0.45, 0.15), (-25, 18, +0.5, 0.15)],
                 low_layer=dict(semitones=-9.0, gain=0.12), ear=dict(semitones=-1.0, gain=2.5, pan=-0.2),
                 preswell_s=0.6, preswell_gain=0.75, preswell_gap_s=0.06, tail_s=0.8, tail_wet=0.16, breath=False),
    # « Identification terminée » : grave et doublée, mais la couche « oreille » domine -> intelligible
    "ident": dict(semitones=-4.5, tempo=1.0, layers=[(0.0, 0.0, 0.0, 0.5), (+18, 9, -0.4, 0.28), (-22, 16, +0.42, 0.25)],
                  low_layer=dict(semitones=-8.0, gain=0.18), ear=dict(semitones=-1.5, gain=1.8, pan=-0.18),
                  preswell_s=1.0, preswell_gain=0.8, preswell_gap_s=0.05, tail_s=1.0, tail_wet=0.14, breath=False),
    # « C'est qui lui ? » juste avant le screamer : ultra-intime, très sec, intelligible ; inspiration
    # très courte (onset ≤ 0,15 s) pour que le silence qui précède reste franc
    "cql": dict(semitones=-3.5, tempo=1.0, layers=[(0.0, 0.0, 0.0, 0.45), (+15, 8, +0.3, 0.25)],
                low_layer=dict(semitones=-7.0, gain=0.15), ear=dict(semitones=-1.0, gain=1.4, pan=-0.3),
                preswell_s=0.0, preswell_gain=0.0, tail_s=0.3, tail_wet=0.05, breath=True,
                head_s=0.015, breath_s=0.1, breath_gap_s=0.03),
}

ENTITY_EQ = [("hp", 55, 2), ("lowshelf", 150, 4.5), ("peak", 420, -2.5, 1.0), ("peak", 4000, 3.0, 0.8),
             ("peak", 7500, -3.0, 1.2), ("lp", 12500, 2)]


def _whisper_layer(y: np.ndarray, sr: int, semitones: float, formant: bool, seed: int) -> np.ndarray:
    z = pitch_shift(y, sr, semitones, formant=formant) if abs(semitones) > 1e-3 else y
    return whisperize(z, sr, seed=seed)


def entity_chain(x: np.ndarray, sr: int, preset: str = "name", seed: int = 0, lufs_target: float = -20.0,
                 peak_db: float = -3.0, **overrides) -> tuple[np.ndarray, dict]:
    """Chuchotement de l'entité : couches désaccordées, couche « à l'oreille », pré-swell en réverbe inversée."""
    p = {**ENTITY_PRESETS[preset], **overrides}
    y = resample(mono(x), sr, SR)
    y, _ = trim_to_speech(y, SR, 0.02, 0.02)
    if abs(p["tempo"] - 1.0) > 1e-4:
        y = pitch_shift(y, SR, 0.0, formant=True, tempo=p["tempo"])
    # couches principales : hauteur ET formants vers le bas (menace), chuchotées, désaccordées
    shifted = pitch_shift(y, SR, p["semitones"], formant=False)
    core = whisperize(shifted, SR, seed=seed + 1)
    bus = np.zeros((len(core) + int(0.05 * SR), 2), dtype=np.float32)
    for k, (cents, delay_ms, pos, gain) in enumerate(p["layers"]):
        lay = core if k == 0 else whisperize(detune(shifted, cents), SR, seed=seed + 10 + k)
        lay = compress(lay, SR, thr_db=-26, ratio=3.0, attack_ms=4, release_ms=90, knee_db=6)
        bus = mix_at(bus, pan(lay, pos), int(delay_ms * SR / 1000), gain)
    low = p.get("low_layer")
    if low and low["gain"] > 0:
        lay = _whisper_layer(y, SR, low["semitones"], formant=False, seed=seed + 30)
        lay = lowpass(lay, SR, 2500, 2)
        bus = mix_at(bus, pan(lay, 0.0), int(0.006 * SR), low["gain"])
    # couche sèche « collée à l'oreille » : enveloppe à peine abaissée -> intelligible
    # (pour un chuchotement, seule l'enveloppe spectrale compte : la transposition déplace les formants)
    e = p["ear"]
    ear = _whisper_layer(y, SR, e["semitones"], formant=False, seed=seed + 40)
    if e.get("voiced", 0.0) > 0:  # chuchotement « soufflé » : un peu de voix réelle sous le souffle (intelligibilité)
        z = pitch_shift(y, SR, e["semitones"], formant=False) if abs(e["semitones"]) > 1e-3 else y
        z = mono(z)[: len(ear)]
        z = np.pad(z, (0, len(ear) - len(z)))
        z = highpass(z, SR, 120, 2)
        rz = np.sqrt(np.mean(z ** 2)) + EPS
        re_ = np.sqrt(np.mean(ear ** 2)) + EPS
        ear = ((1 - e["voiced"]) * ear + e["voiced"] * z * (re_ / rz)).astype(np.float32)
    ear = compress(ear, SR, thr_db=-28, ratio=3.5, attack_ms=3, release_ms=80, knee_db=6)
    ear_st = pan(ear, e["pan"])
    itd = int(0.00035 * SR)  # léger retard inter-aural : la voix est d'un côté de la tête
    if e["pan"] < 0:
        ear_st[:, 1] = np.roll(ear_st[:, 1], itd) * 0.85
    else:
        ear_st[:, 0] = np.roll(ear_st[:, 0], itd) * 0.85
    bus = mix_at(bus, ear_st, 0, e["gain"])
    bus = eq(bus, SR, ENTITY_EQ)
    bus = deess(bus, SR, freq=5800, thr_db=-30, ratio=3.0, max_cut_db=6)
    bus = compress(bus, SR, thr_db=-24, ratio=2.5, attack_ms=6, release_ms=120, knee_db=6)
    a, b = speech_bounds(bus, SR, thr_db=-40)
    i0 = max(0, int(a * SR) - int(0.003 * SR))
    speech = fade(bus[i0:int(b * SR) + int(0.02 * SR)], SR, 0.003, 0.03)
    # queue sombre (réverbe directe discrète)
    if p["tail_s"] > 0:
        ir = dark_ir(SR, decay_s=max(0.4, p["tail_s"] * 1.6), predelay_ms=18, seed=seed + 50)
        speech = reverb(speech, ir, p["tail_wet"])
        _, sb = speech_bounds(speech, SR, thr_db=-40)
        speech = speech[: min(len(speech), int((sb + p["tail_s"]) * SR))]
        speech = fade(speech, SR, 0.0, min(0.25, p["tail_s"] / 2))
    head = np.zeros((int(p.get("head_s", 0.05) * SR), 2), dtype=np.float32)
    pre_len = 0
    if p["preswell_s"] > 0:
        swell = reverse_swell(bus[i0:], dark_ir(SR, decay_s=max(1.5, p["preswell_s"] * 2.2), predelay_ms=0,
                                                 seed=seed + 60), p["preswell_s"], SR)
        swell *= p["preswell_gain"]
        gap = int(p.get("preswell_gap_s", 0.0) * SR)
        if 0 < gap < len(swell):  # micro-silence entre l'aspiration et le mot (attaque nette, intelligible)
            swell[-gap:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, gap)))[:, None]
        head = np.concatenate([head, swell])
    if p.get("breath"):
        br = breath(SR, p.get("breath_s", 0.36), seed=seed + 70)
        br_st = pan(br, e["pan"]) * db2lin(-18)
        gap = int(p.get("breath_gap_s", 0.07) * SR)
        head = np.concatenate([head, br_st, np.zeros((gap, 2), np.float32)])
    pre_len = len(head)
    out = np.concatenate([head, speech]).astype(np.float32)
    # mesure : la parole commence là où commence `speech` (+ attaque mesurée sur le sec)
    sa, sb = speech_bounds(speech[: int((b - a + 0.03) * SR)], SR, thr_db=-40)
    out = normalize_lufs(out, SR, lufs_target, peak_db)
    info = dict(onset_s=round(pre_len / SR + sa, 3), speech_dur=round(sb - sa, 3), dur=round(len(out) / SR, 3),
                lufs=round(lufs(out, SR), 2), peak_db=round(true_peak_db(out), 2),
                layers=dict(ear=ear.astype(np.float32), core=core.astype(np.float32)))
    return out, info
