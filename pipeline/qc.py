"""Contrôle qualité du film livré : qc(final_mp4) -> dict (+ RENDER_DIR/qc/qc_report.md et qc_report.json).

Contrôles (statut pass / warn / fail ; le film passe s'il n'y a aucun fail) :
  format      ffprobe : H.264 High, yuv420p, 1080x1920 (540x960 en preview), 30/1, AAC LC 48 kHz stéréo
              ~320 kb/s, atome moov avant mdat (faststart), nombre d'images = round(durée EDL × 30)
  durée       85–95 s ; audio et vidéo de même durée
  noirs       blackdetect comparé aux noirs prévus de l'EDL (shots noires, cartons, fondus, début du titre ; ±0,1 s).
              Hors des noirs prévus, chaque image est classée : plan très sombre AVEC contenu (clip/gel/photo/
              montage, écart-type spatial et 99e centile au-dessus des seuils) → avertissement ; noir déjà
              présent dans le rush source → avertissement ; image quasi uniforme noire = TROU → échec
              (aussi pour les trous courts de 2 images que blackdetect ne voit pas).
  loudness    ebur128 : −14 ±1 LUFS intégrés, true peak ≤ −1 dBTP ; max court terme (3 s) dans la shot du
              screamer ; max momentané calé sur le screamer
  ASR         transcription du mix livré (pipeline.asr) : aucun mot de la liste noire, répliques présentes
  flashs      ≤ 3 flashs/s (paires de variations opposées de luminance ≥ 10 %, fenêtre glissante d'1 s)
  sous-titres zone sûre TikTok (mesurée en dessinant chaque sous-titre), ≤ 2 lignes de ≤ 34 caractères
  prénoms     affichés ⊂ {DOGAN, GABRIEL, HUGO, ARSÈNE}, aucune orthographe approchante dans les textes
  STUB        aucun module de repli utilisé (rendu final)
  planche     RENDER_DIR/qc/contact_{quality}.jpg : 1 image/s horodatée
"""
from __future__ import annotations

import difflib
import json
import math
import re
import struct
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from pipeline import render as R  # noqa: E402

ALLOWED_NAMES = ["DOGAN", "GABRIEL", "HUGO", "ARSÈNE"]
BLACK_TOL = 0.1           # tolérance (s) sur les noirs prévus
FLASH_DELTA = 0.10        # variation de luminance relative (linéaire) comptée comme transition de flash
FLASH_MAX_PER_S = 3
LUFS_TOL = 1.0
# Noir réel ou plan sombre ? Statistiques d'image en luma codée (0..1) sur une réduction 54x96 :
DARK_STD_MIN = 0.012      # écart-type spatial au-dessus duquel l'image a du contenu (plan de nuit : 0,017–0,035 ; noir : 0)
DARK_P99_MIN = 0.06       # et 99e centile (zones éclairées présentes ; plan de nuit : 0,17–0,23 ; noir : 0–0,02)
TRUE_BLACK_MEAN = 0.03    # image quasi uniforme ET moyenne sous ce seuil = noir
SRC_BLACK_LUMA = 0.04     # luminance moyenne du rush (analysis/<clip>.luma.json) sous laquelle la source est noire
HOLE_MIN_FRAMES = 2       # un trou = au moins 2 images noires consécutives non prévues
CONTENT_KINDS = ("clip", "freeze", "photo", "montage")


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
def norm_text(s: str) -> str:
    """minuscules, sans accents ni ponctuation, apostrophes -> espace."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[’'`´]", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _check(name: str, status: str, value=None, expected=None, detail: str = "") -> dict:
    return {"name": name, "status": status, "value": value, "expected": expected, "detail": detail}


def mp4_top_boxes(path: Path) -> list[tuple[str, int, int]]:
    """Boîtes de premier niveau d'un MP4 : [(type, offset, taille)]."""
    out = []
    with open(path, "rb") as f:
        f.seek(0, 2)
        end = f.tell()
        off = 0
        while off + 8 <= end:
            f.seek(off)
            hdr = f.read(8)
            size, typ = struct.unpack(">I4s", hdr)
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
            elif size == 0:
                size = end - off
            out.append((typ.decode("latin-1"), off, size))
            if size < 8:
                break
            off += size
    return out


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace")


def _screamer_shot(edl: dict) -> dict | None:
    ids = {c.get("shot") for c in edl.get("cues") or [] if c.get("sfx") == "screamer_stack"}
    for s in edl.get("shots", []):
        if s.get("id") in ids or any(c.get("sfx") == "screamer_stack" for c in s.get("cues") or []):
            return s
    for s in edl.get("shots", []):
        if "SCREAMER" in str(s.get("label", "")).upper():
            return s
    return None


def planned_dark(edl: dict) -> list[tuple[float, float, str]]:
    """Intervalles où un noir (ou quasi-noir) est voulu."""
    out = []
    try:
        reveal = float(json.loads((config.TITLE_DIR / "title_meta.json").read_text(encoding="utf-8"))["reveal"])
    except Exception:
        reveal = 1.2
    for s in edl.get("shots", []):
        st, du = float(s.get("start", 0)), float(s.get("dur", 0))
        kind = s.get("kind")
        if kind in ("black", "card"):
            out.append((st, st + du, f"{s['id']} ({kind})"))
        elif kind == "title":
            out.append((st, st + reveal + 0.4, f"{s['id']} (titre avant révélation)"))
        for fx in s.get("fx") or []:
            if not isinstance(fx, dict):
                continue
            t0 = R._num(fx.get("t0"), 0.0)
            t1 = R._num(fx.get("t1"), du)
            if fx.get("type") == "fade_to_black":
                out.append((st + t0 + 0.4 * (t1 - t0), st + du, f"{s['id']} (fondu au noir)"))
            elif fx.get("type") == "fade_from_black":
                out.append((st, st + t0 + 0.6 * (t1 - t0), f"{s['id']} (fondu du noir)"))
    out.sort()
    merged = []
    for a, b, why in out:
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b), merged[-1][2] + " + " + why)
        else:
            merged.append((a, b, why))
    return merged


# ---------------------------------------------------------------------------
# Contrôles
# ---------------------------------------------------------------------------
def check_format(mp4: Path, quality: str, n_expected: int | None) -> tuple[list[dict], dict]:
    checks = []
    info = R.ffprobe_json(mp4, "-show_streams", "-show_format")
    vs = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    as_ = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    W, H = R.SIZES[quality] if quality == "preview" else (config.W, config.H)
    if not vs:
        return [_check("flux vidéo", "fail", None, "h264")], info
    v = vs[0]
    checks.append(_check("codec vidéo", "pass" if v.get("codec_name") == "h264" else "fail", v.get("codec_name"),
                         "h264"))
    checks.append(_check("profil H.264", "pass" if v.get("profile") == "High" else "fail", v.get("profile"), "High"))
    checks.append(_check("pix_fmt", "pass" if v.get("pix_fmt") == "yuv420p" else "fail", v.get("pix_fmt"), "yuv420p"))
    checks.append(_check("résolution", "pass" if (v.get("width"), v.get("height")) == (W, H) else "fail",
                         f"{v.get('width')}x{v.get('height')}", f"{W}x{H}"))
    fr_ok = v.get("r_frame_rate") == "30/1" and v.get("avg_frame_rate") == "30/1"
    checks.append(_check("cadence", "pass" if fr_ok else "fail",
                         f"{v.get('r_frame_rate')} / {v.get('avg_frame_rate')}", "30/1 (CFR)"))
    n = R.count_frames(mp4)
    if n_expected is not None:
        checks.append(_check("nombre d'images", "pass" if n == n_expected else "fail", n, n_expected,
                             "= round(durée EDL × 30)"))
    if not as_:
        checks.append(_check("flux audio", "fail", None, "aac"))
    else:
        a = as_[0]
        checks.append(_check("codec audio", "pass" if a.get("codec_name") == "aac" and a.get("profile") == "LC"
                             else "fail", f"{a.get('codec_name')} {a.get('profile')}", "aac LC"))
        checks.append(_check("audio 48 kHz stéréo",
                             "pass" if a.get("sample_rate") == "48000" and int(a.get("channels", 0)) == 2 else "fail",
                             f"{a.get('sample_rate')} Hz, {a.get('channels')} canaux", "48000 Hz, 2 canaux"))
        br = int(a.get("bit_rate") or 0)
        st = "pass" if 280_000 <= br <= 345_000 else ("warn" if 128_000 <= br < 280_000 else "fail")
        checks.append(_check("débit audio", st, f"{br // 1000} kb/s", "320 kb/s",
                             "encodé à 320 kb/s ; débit mesuré plus bas sur un contenu simple" if st == "warn" else ""))
        vd, ad = float(v.get("duration", 0)), float(a.get("duration", 0))
        checks.append(_check("durées audio/vidéo", "pass" if abs(vd - ad) <= 0.05 else "fail",
                             f"vidéo {vd:.3f} s / audio {ad:.3f} s", "écart ≤ 0,05 s"))
    boxes = mp4_top_boxes(mp4)
    order = [b[0] for b in boxes]
    ok = "moov" in order and "mdat" in order and order.index("moov") < order.index("mdat")
    checks.append(_check("faststart (moov avant mdat)", "pass" if ok else "fail", " ".join(order), "moov < mdat"))
    dur = float(info.get("format", {}).get("duration", 0))
    checks.append(_check("durée", "pass" if config.DURATION_MIN <= dur <= config.DURATION_MAX else "fail",
                         round(dur, 2), f"{config.DURATION_MIN:.0f}–{config.DURATION_MAX:.0f} s"))
    return checks, info


def detect_black(mp4: Path) -> list[tuple[float, float]]:
    cmd = [R._ffmpeg(), "-hide_banner", "-nostats", "-nostdin", "-i", str(mp4), "-map", "0:v:0", "-an",
           "-vf", "scale=270:480,blackdetect=d=0.08:pic_th=0.98:pix_th=0.08", "-f", "null", "-"]
    r = _run(cmd)
    out = []
    for m in re.finditer(r"black_start:([\d.]+)\s+black_end:([\d.]+)", r.stderr):
        out.append((float(m.group(1)), float(m.group(2))))
    return out


def frame_stats(mp4: Path, w: int = 54, h: int = 96) -> dict:
    """Statistiques par image (un seul décodage réduit) : luminance linéaire moyenne (flashs), et en luma
    codée 0..1 : moyenne, écart-type spatial, 99e centile (noir réel vs plan sombre)."""
    cmd = [R._ffmpeg(), "-v", "error", "-nostdin", "-i", str(mp4), "-map", "0:v:0", "-vf",
           f"scale={w}:{h}:flags=area,format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    r = subprocess.run(cmd, capture_output=True)
    a = np.frombuffer(r.stdout, np.uint8)
    n = len(a) // (w * h * 3)
    rgb = a[: n * w * h * 3].reshape(n, h * w, 3).astype(np.float32) / 255.0
    coef = np.array([0.2126, 0.7152, 0.0722], np.float32)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = rgb @ coef
    return {"n": n, "Ylin": (lin @ coef).mean(axis=1), "mean": y.mean(axis=1), "std": y.std(axis=1),
            "p99": np.percentile(y, 99, axis=1)}


def _shot_at(edl: dict, t: float) -> dict | None:
    for s in edl.get("shots", []):
        st = float(s.get("start", 0))
        if st <= t < st + float(s.get("dur", 0)):
            return s
    return None


_LUMA_CACHE: dict[str, list | None] = {}


def _source_luma(clip: str | None, t_src: float) -> float | None:
    """Luminance moyenne du rush à l'instant source (analysis/<clip>.luma.json), ou None si inconnue."""
    if not clip:
        return None
    if clip not in _LUMA_CACHE:
        try:
            d = json.loads((config.ANALYSIS_DIR / f"{clip}.luma.json").read_text(encoding="utf-8"))
            _LUMA_CACHE[clip] = [float(x) for x in d.get("luma") or []]
        except Exception:
            _LUMA_CACHE[clip] = None
    lu = _LUMA_CACHE[clip]
    if not lu:
        return None
    return lu[min(len(lu) - 1, max(0, int(round(t_src * config.FPS))))]


def _source_of(shot: dict, t_local: float) -> tuple[str | None, float]:
    """(clip, temps source) de l'image d'une shot clip/gel (approximation suffisante pour le QC)."""
    if shot.get("kind") == "freeze":
        fs = shot.get("freeze_src") or {}
        return fs.get("clip") or shot.get("clip"), R._num(fs.get("t"), R._num(shot.get("src_in"), 0.0))
    try:
        ts = float(R.source_times(shot, np.array([max(0.0, t_local)]))[0])
    except Exception:
        ts = R._num(shot.get("src_in"), 0.0) + t_local
    return shot.get("clip"), ts


def _runs(idx: list[int]) -> list[tuple[int, int]]:
    """Plages d'indices consécutifs [(i0, i1 exclus)]."""
    out = []
    for i in idx:
        if out and i == out[-1][1]:
            out[-1] = (out[-1][0], i + 1)
        else:
            out.append((i, i + 1))
    return out


def _in_plan(t0: float, t1: float, plan) -> bool:
    return any(pa - BLACK_TOL <= t0 and t1 <= pb + BLACK_TOL for pa, pb, _ in plan)


def classify_dark(edl: dict | None, stats: dict, intervals: list[tuple[float, float]], plan) -> dict:
    """Classe les images des intervalles sombres NON prévus : trou (noir uniforme), plan très sombre avec contenu,
    ou noir présent dans le rush source. Ajoute les trous courts (≥ 2 images) que blackdetect ne voit pas."""
    fps = config.FPS
    n = stats["n"]
    content = (stats["std"] >= DARK_STD_MIN) & (stats["p99"] >= DARK_P99_MIN)
    black = (~content) & (stats["mean"] < TRUE_BLACK_MEAN)
    frames = set()
    for x, y in intervals:
        frames.update(range(max(0, int(round(x * fps))), min(n, int(round(y * fps)))))
    # trous courts hors blackdetect (2 images noires et plus, hors noirs prévus)
    for i0, i1 in _runs([int(i) for i in np.nonzero(black)[0]]):
        if i1 - i0 >= HOLE_MIN_FRAMES and not _in_plan(i0 / fps, i1 / fps, plan):
            frames.update(range(i0, i1))
    cls: dict[str, list[int]] = {"hole": [], "dark": [], "src_black": []}
    shots_of: dict[int, str] = {}
    for i in sorted(frames):
        t = (i + 0.5) / fps
        if _in_plan(t, t, plan):
            continue
        shot = _shot_at(edl, t) if edl else None
        kind = (shot or {}).get("kind")
        shots_of[i] = (shot or {}).get("id", "?")
        if content[i]:  # contenu visible (dans un clip/gel/photo/montage, ou un titre sombre) : pas un trou
            cls["dark"].append(i)
        elif kind in ("clip", "freeze"):
            clip, ts = _source_of(shot, t - float(shot.get("start", 0)))
            sl = _source_luma(clip, ts)
            cls["src_black" if sl is not None and sl < SRC_BLACK_LUMA else "hole"].append(i)
        else:
            cls["hole"].append(i)

    def describe(idx, min_len=1):
        out = []
        for i0, i1 in _runs(idx):
            if i1 - i0 < min_len:
                continue
            out.append({"t": [round(i0 / fps, 2), round(i1 / fps, 2)],
                        "shots": sorted({shots_of.get(i, "?") for i in range(i0, i1)}),
                        "luma": round(float(stats["mean"][i0:i1].mean()), 3),
                        "ecart_type": round(float(stats["std"][i0:i1].mean()), 4)})
        return out

    return {"holes": describe(cls["hole"], HOLE_MIN_FRAMES), "dark": describe(cls["dark"]),
            "src_black": describe(cls["src_black"])}


def check_black(mp4: Path, edl: dict | None, stats: dict | None = None) -> tuple[list[dict], dict]:
    blacks = detect_black(mp4)
    plan = planned_dark(edl) if edl else []
    unplanned = []
    for a, b in blacks:
        covered = any(pa - BLACK_TOL <= a and b <= pb + BLACK_TOL for pa, pb, _ in plan)
        if not covered:
            # couverture partielle : on mesure la partie non prévue
            rest = [(a, b)]
            for pa, pb, _ in plan:
                nxt = []
                for x, y in rest:
                    if pb + BLACK_TOL <= x or y <= pa - BLACK_TOL:
                        nxt.append((x, y))
                        continue
                    if x < pa - BLACK_TOL:
                        nxt.append((x, pa - BLACK_TOL))
                    if y > pb + BLACK_TOL:
                        nxt.append((pb + BLACK_TOL, y))
                rest = nxt
            unplanned += [(x, y) for x, y in rest if y - x > 0.05]
    stats = stats if stats is not None else frame_stats(mp4)
    cl = classify_dark(edl, stats, unplanned, plan)
    planned_black_shots = [(float(s["start"]), float(s["start"]) + float(s["dur"]), s["id"])
                           for s in (edl or {}).get("shots", []) if s.get("kind") == "black"]
    missing = []
    for a, b, sid in planned_black_shots:
        if b - a < 0.1:
            continue
        cov = sum(max(0.0, min(b, y) - max(a, x)) for x, y in blacks)
        if cov < 0.7 * (b - a):  # couverture par les plages noires détectées
            missing.append(sid)
    checks = [_check("noirs involontaires", "fail" if cl["holes"] else "pass", cl["holes"] or "aucun",
                     "seulement les noirs prévus (±0,1 s)",
                     f"{len(blacks)} plage(s) noire(s) détectée(s), {len(plan)} prévue(s) ; trou = image quasi "
                     f"uniforme noire (écart-type < {DARK_STD_MIN}) hors noirs prévus")]
    if cl["dark"]:
        checks.append(_check("plans très sombres", "warn", cl["dark"], "lisibles",
                             "contenu présent (pas un trou) : vérifier sur la planche que l'image reste lisible"))
    if cl["src_black"]:
        checks.append(_check("noir présent dans le rush", "warn", cl["src_black"], "voulu par le montage ?",
                             "le rush source est lui-même noir à cet endroit"))
    if missing:
        checks.append(_check("noirs prévus présents", "warn", missing, "noirs francs",
                             "shots noires non détectées comme noires (sous-titres ? grain ?)"))
    return checks, {"detected": [[round(a, 3), round(b, 3)] for a, b in blacks],
                    "planned": [[round(a, 3), round(b, 3), w] for a, b, w in plan],
                    "unplanned": [[round(a, 3), round(b, 3)] for a, b in unplanned], **cl}


def measure_loudness(mp4: Path) -> dict:
    cmd = [R._ffmpeg(), "-hide_banner", "-nostats", "-nostdin", "-v", "verbose", "-i", str(mp4), "-map", "0:a:0",
           "-af", "ebur128=peak=true:framelog=verbose", "-f", "null", "-"]
    r = _run(cmd)
    ts, M, S = [], [], []
    for m in re.finditer(r"t:\s*([\d.]+)\s+TARGET:\S+ LUFS\s+M:\s*(-?[\d.]+|-inf|nan)\s+S:\s*(-?[\d.]+|-inf|nan)",
                         r.stderr):
        ts.append(float(m.group(1)))
        M.append(float(m.group(2)) if m.group(2) not in ("-inf", "nan") else -120.0)
        S.append(float(m.group(3)) if m.group(3) not in ("-inf", "nan") else -120.0)
    summ = r.stderr[r.stderr.rfind("Summary:"):]
    gi = re.search(r"I:\s*(-?[\d.]+) LUFS", summ)
    gp = re.search(r"Peak:\s*(-?[\d.]+|-inf) dBFS", summ)
    lra = re.search(r"LRA:\s*([\d.]+) LU", summ)
    return {"I": float(gi.group(1)) if gi else None,
            "TP": float(gp.group(1)) if gp and gp.group(1) != "-inf" else None,
            "LRA": float(lra.group(1)) if lra else None, "t": ts, "M": M, "S": S}


def check_loudness(mp4: Path, edl: dict | None) -> tuple[list[dict], dict]:
    L = measure_loudness(mp4)
    checks = []
    I, TP = L["I"], L["TP"]
    checks.append(_check("loudness intégré", "pass" if I is not None and abs(I - config.TARGET_LUFS) <= LUFS_TOL
                         else "fail", I, f"{config.TARGET_LUFS:.0f} ±{LUFS_TOL:.0f} LUFS"))
    checks.append(_check("true peak", "pass" if TP is not None and TP <= config.TARGET_TP + 1e-6 else "fail", TP,
                         f"≤ {config.TARGET_TP:.0f} dBTP"))
    data = {"I": I, "TP": TP, "LRA": L["LRA"]}
    if L["t"]:
        t = np.asarray(L["t"])
        S, M = np.asarray(L["S"]), np.asarray(L["M"])
        ks, km = int(np.argmax(S)), int(np.argmax(M))
        data.update(max_S=float(S[ks]), max_S_t=float(t[ks]), max_M=float(M[km]), max_M_t=float(t[km]))
        scr = _screamer_shot(edl) if edl else None
        if scr:
            a, b = float(scr["start"]), float(scr["start"]) + float(scr["dur"])
            win = (float(t[ks]) - 3.0, float(t[ks]))
            ok = win[0] <= b and win[1] >= a
            checks.append(_check("max court terme = screamer", "pass" if ok else "fail",
                                 f"{S[ks]:.1f} LUFS (S) sur {win[0]:.1f}–{win[1]:.1f} s",
                                 f"fenêtre contenant {scr['id']} ({a:.2f}–{b:.2f} s)"))
            okm = a - 0.05 <= float(t[km]) <= b + 0.45
            checks.append(_check("screamer calé (max momentané)", "pass" if okm else "warn",
                                 f"{M[km]:.1f} LUFS (M) à {t[km]:.1f} s", f"dans {scr['id']} ({a:.2f}–{b:.2f} s)"))
            data["screamer"] = [a, b]
        else:
            checks.append(_check("max court terme = screamer", "warn", f"{S[ks]:.1f} LUFS à {t[ks]:.1f} s",
                                 "shot du screamer", "shot du screamer introuvable dans l'EDL"))
    return checks, data


def luminance_series(mp4: Path) -> np.ndarray:
    """Luminance relative (linéaire) moyenne de chaque image."""
    w, h = 54, 96
    cmd = [R._ffmpeg(), "-v", "error", "-nostdin", "-i", str(mp4), "-map", "0:v:0", "-vf",
           f"scale={w}:{h}:flags=area,format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    r = subprocess.run(cmd, capture_output=True)
    a = np.frombuffer(r.stdout, np.uint8)
    n = len(a) // (w * h * 3)
    rgb = a[: n * w * h * 3].reshape(n, h, w, 3).astype(np.float32) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return Y.reshape(n, -1).mean(axis=1)


def count_flashes(Y: np.ndarray, fps: int = config.FPS, delta: float = FLASH_DELTA) -> tuple[int, float, list]:
    """Flashs = paires de variations opposées ≥ delta (la plus sombre < 0,8). Renvoie (max/s, t, transitions)."""
    if len(Y) < 3:
        return 0, 0.0, []
    # extrema successifs
    ext = [0]
    for i in range(1, len(Y) - 1):
        if (Y[i] - Y[ext[-1]]) * (Y[i + 1] - Y[i]) < 0 and abs(Y[i] - Y[ext[-1]]) > 1e-4:
            ext.append(i)
    ext.append(len(Y) - 1)
    trans = []  # (index de fin, signe)
    for a, b in zip(ext[:-1], ext[1:]):
        d = float(Y[b] - Y[a])
        if abs(d) >= delta and min(Y[a], Y[b]) < 0.8:
            trans.append((b, 1 if d > 0 else -1))
    best, best_t = 0, 0.0
    for i, (b, _) in enumerate(trans):
        win = [x for x in trans if b - fps < x[0] <= b]
        flashes = len(win) // 2
        if flashes > best:
            best, best_t = flashes, b / fps
    return best, best_t, [(round(b / fps, 2), s) for b, s in trans]


def check_flashes(mp4: Path, stats: dict | None = None) -> tuple[list[dict], dict]:
    Y = stats["Ylin"] if stats is not None else luminance_series(mp4)
    best, t, trans = count_flashes(Y)
    return ([_check("flashs", "pass" if best <= FLASH_MAX_PER_S else "fail", f"{best} flash(s)/s max (t≈{t:.1f} s)",
                    f"≤ {FLASH_MAX_PER_S}/s", f"{len(trans)} variation(s) ≥ {int(FLASH_DELTA * 100)} %")],
            {"max_per_s": best, "t": t, "transitions": trans})


def check_subtitles(edl: dict) -> tuple[list[dict], dict]:
    G = R.gfx()
    try:
        x0s, y0s, x1s, y1s = G.safe_zone()
    except Exception:
        x0s, y0s, x1s, y1s = 60, 160, config.W - 180, int(0.8 * config.H)
    y1s = min(y1s, int(0.8 * config.H))
    bad, boxes = [], []
    cache = {}
    for sb in edl.get("subtitles") or []:
        text, kind = sb.get("text", ""), sb.get("kind", "dialogue")
        lines = text.split("\n")
        if len(lines) > 2 or any(len(ln) > 34 for ln in lines):
            bad.append({"text": text, "probleme": "plus de 2 lignes ou plus de 34 caractères par ligne"})
        key = (text, kind)
        if key not in cache:
            img = np.zeros((config.H, config.W, 3), np.float32)
            try:
                out = G.draw_subtitle(img, text, kind, alpha=1.0)
            except TypeError:
                out = G.draw_subtitle(img, text, kind)
            m = np.asarray(out).max(axis=2) > 0.12
            ys, xs = np.nonzero(m)
            cache[key] = None if len(ys) == 0 else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        bb = cache[key]
        if bb is None:
            bad.append({"text": text, "probleme": "rien n'est dessiné"})
            continue
        boxes.append({"text": text, "bbox": bb})
        tol = 4
        if bb[0] < x0s - tol or bb[2] > x1s + tol or bb[1] < y0s - tol or bb[3] > y1s + tol:
            bad.append({"text": text, "bbox": bb, "probleme": f"hors zone sûre {x0s},{y0s}–{x1s},{y1s}"})
    n = len(edl.get("subtitles") or [])
    return ([_check("sous-titres (zone sûre, ≤ 2 lignes)", "fail" if bad else ("pass" if n else "warn"),
                    bad or f"{n} sous-titre(s) conformes", f"x∈[{x0s},{x1s}], y∈[{y0s},{y1s}]")],
            {"safe_zone": [x0s, y0s, x1s, y1s], "boxes": boxes})


def check_names(edl: dict) -> tuple[list[dict], dict]:
    shown, bad = [], []
    texts = []
    for s in edl.get("shots", []):
        for ov in s.get("overlays") or []:
            if ov.get("type") == "name":
                if ov.get("show"):
                    nm = str(ov.get("name", ""))
                    shown.append((s["id"], nm))
                    if nm not in ALLOWED_NAMES:
                        bad.append(f"{s['id']} : prénom affiché « {nm} » hors liste")
            elif ov.get("type") == "hud_faces" and isinstance(ov.get("order"), (list, tuple)):
                for nm in ov["order"]:
                    shown.append((s["id"], f"HUD {nm}"))
                    if str(nm).upper() not in ALLOWED_NAMES:
                        bad.append(f"{s['id']} : prénom HUD « {nm} » hors liste")
            elif ov.get("type") in ("card", "title_sub"):
                texts.append((s["id"], str(ov.get("text", ""))))
    texts += [("sous-titres", sb.get("text", "")) for sb in edl.get("subtitles") or []]
    refs = {norm_text(n): n for n in ALLOWED_NAMES}
    for where, txt in texts:
        for tok in re.findall(r"[A-Za-zÀ-ÿ]+", txt):
            nt = norm_text(tok)
            if len(nt) < 4:
                continue
            for rn, real in refs.items():
                close = nt == rn or difflib.SequenceMatcher(None, nt, rn).ratio() >= 0.83
                if close and tok.upper() != real:
                    bad.append(f"{where} : « {tok} » ressemble à {real.capitalize()} (orthographe ?)")
    return ([_check("prénoms", "fail" if bad else "pass", bad or [n for _, n in shown],
                    "⊂ {Dogan, Gabriel, Hugo, Arsène}, orthographe exacte")], {"shown": shown})


def _fuzzy_find(words: list[str], phrase: str) -> tuple[float, int]:
    pw = phrase.split()
    if not pw or not words:
        return 0.0, -1
    best, at = 0.0, -1
    target = " ".join(pw)
    for L in range(max(1, len(pw) - 2), len(pw) + 3):
        for i in range(0, max(1, len(words) - L + 1)):
            r = difflib.SequenceMatcher(None, " ".join(words[i:i + L]), target).ratio()
            if r > best:
                best, at = r, i
    return best, at


def check_asr(mp4: Path, edl: dict | None, log=print) -> tuple[list[dict], dict]:
    try:
        from pipeline import asr
    except Exception as e:
        return [_check("ASR (liste noire)", "warn", None, None, f"pipeline.asr indisponible : {e}")], {}
    t0 = time.time()
    try:
        tr = asr.transcribe_file(str(mp4), language="fr")
    except Exception as e:
        return [_check("ASR (liste noire)", "fail", None, None, f"transcription impossible : {e}")], {}
    words = [w for w in tr.get("words", []) if w.get("w")]
    toks, ttimes = [], []
    for w in words:
        for t in norm_text(w["w"]).split():
            toks.append(t)
            ttimes.append(float(w.get("start", 0)))
    text = " ".join(toks)
    log(f"[qc] ASR ({tr.get('backend')}, {time.time() - t0:.0f} s) : {len(toks)} mots")
    found_bl = []
    for ph in config.INSTRUCTION_BLACKLIST:
        p = norm_text(ph)
        pw = p.split()
        hit = None
        for i in range(len(toks) - len(pw) + 1):
            if toks[i:i + len(pw)] == pw:
                hit = i
                break
        if hit is None and len(pw) >= 2:
            r, i = _fuzzy_find(toks, p)
            if r >= 0.88:
                hit = i
        if hit is not None:
            found_bl.append({"consigne": ph, "t": round(ttimes[hit], 2) if hit < len(ttimes) else None,
                             "contexte": " ".join(toks[max(0, hit - 4):hit + len(pw) + 4])})
    checks = [_check("ASR : aucune consigne de tournage", "fail" if found_bl else "pass", found_bl or "aucune",
                     "aucun mot de la liste noire", f"moteur {tr.get('backend')}")]
    lines = []
    for sb in (edl or {}).get("subtitles") or []:
        kind = sb.get("kind", "dialogue")
        if kind == "entity":
            continue
        p = norm_text(sb.get("text", "").replace("\n", " "))
        # recherche locale (±3 s) puis globale
        a, b = float(sb.get("start", 0)) - 3, float(sb.get("end", 0)) + 3
        loc = [t for t, tt in zip(toks, ttimes) if a <= tt <= b]
        r, _ = _fuzzy_find(loc, p)
        if r < 0.7:
            r = max(r, _fuzzy_find(toks, p)[0])
        lines.append({"text": sb.get("text", "").replace("\n", " "), "kind": kind, "score": round(r, 2),
                      "found": r >= 0.7})
    dl = [x for x in lines if x["kind"] == "dialogue"]
    nr = [x for x in lines if x["kind"] == "narrator"]
    miss_d = [x["text"] for x in dl if not x["found"]]
    miss_n = [x["text"] for x in nr if not x["found"]]
    if dl:
        checks.append(_check("ASR : répliques présentes", "fail" if miss_d else "pass",
                             f"{len(dl) - len(miss_d)}/{len(dl)}", "toutes",
                             ("manquantes : " + " | ".join(miss_d)) if miss_d else ""))
    if nr:
        checks.append(_check("ASR : voix off reconnue", "warn" if len(miss_n) > len(nr) // 4 else "pass",
                             f"{len(nr) - len(miss_n)}/{len(nr)}", "≥ 75 % (indicatif)",
                             ("non reconnues : " + " | ".join(miss_n)) if miss_n else ""))
    return checks, {"backend": tr.get("backend"), "text": text, "lines": lines, "blacklist_hits": found_bl}


def contact_sheet(mp4: Path, out: Path, cols: int = 12, tw: int = 180) -> Path:
    """Planche 1 image/s (à t = 0,5 ; 1,5 ; …) horodatée."""
    from PIL import Image, ImageDraw
    th = int(tw * 16 / 9)
    cmd = [R._ffmpeg(), "-v", "error", "-nostdin", "-ss", "0.5", "-i", str(mp4), "-map", "0:v:0",
           "-vf", f"fps=1,scale={tw}:{th}:flags=area,format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    r = subprocess.run(cmd, capture_output=True)
    a = np.frombuffer(r.stdout, np.uint8)
    n = len(a) // (tw * th * 3)
    frames = a[: n * tw * th * 3].reshape(n, th, tw, 3)
    lab = 22
    rows = max(1, math.ceil(n / cols))
    sheet = Image.new("RGB", (cols * tw, rows * (th + lab)), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    try:
        f = R.gfx().font("plexmono", 15)
    except Exception:
        f = R._StubGfx.font("plexmono", 15)
    for i in range(n):
        x, y = (i % cols) * tw, (i // cols) * (th + lab)
        sheet.paste(Image.fromarray(frames[i]), (x, y + lab))
        t = i + 0.5
        d.text((x + 4, y + 3), f"{int(t // 60):02d}:{t % 60:04.1f}", fill=(230, 230, 230), font=f)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return out


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, (list, dict)):
        s = json.dumps(v, ensure_ascii=False)
        return s if len(s) < 300 else s[:297] + "…"
    return str(v)


def write_report(res: dict, out_md: Path) -> None:
    icon = {"pass": "OK", "warn": "ATTENTION", "fail": "ÉCHEC", "skip": "—"}
    L = [f"# Contrôle qualité — {Path(res['file']).name}", "",
         f"- Qualité : **{res['quality']}** — verdict : **{'VALIDÉ' if res['pass'] else 'À CORRIGER'}**",
         f"- Date : {res['date']} — durée du contrôle : {res['qc_s']} s",
         f"- Planche 1 i/s : `{res.get('contact_sheet')}`", "",
         "| Contrôle | Statut | Mesure | Attendu | Détail |", "|---|---|---|---|---|"]
    for c in res["checks"]:
        L.append(f"| {c['name']} | {icon.get(c['status'], c['status'])} | {_fmt(c['value'])} | "
                 f"{_fmt(c['expected']) if c['expected'] is not None else ''} | {c['detail']} |".replace("\n", " "))
    d = res.get("data", {})
    if d.get("black"):
        L += ["", "## Noirs", f"- détectés : {d['black']['detected']}", f"- prévus : {d['black']['planned']}"]
        for key, lab in (("holes", "TROUS (noir uniforme non prévu)"), ("dark", "plans très sombres (contenu présent)"),
                         ("src_black", "noir déjà présent dans le rush")):
            if d["black"].get(key):
                L.append(f"- {lab} : " + "; ".join(f"{x['t'][0]:.2f}–{x['t'][1]:.2f} s {'/'.join(x['shots'])} "
                                                    f"(luma {x['luma']}, σ {x['ecart_type']})"
                                                    for x in d["black"][key]))
    if d.get("loudness"):
        lo = d["loudness"]
        L += ["", "## Loudness", f"- intégré {lo.get('I')} LUFS, true peak {lo.get('TP')} dBTP, LRA {lo.get('LRA')} LU",
              f"- max court terme {lo.get('max_S')} LUFS à {lo.get('max_S_t')} s ; max momentané {lo.get('max_M')} "
              f"LUFS à {lo.get('max_M_t')} s"]
    if d.get("asr", {}).get("lines"):
        L += ["", "## ASR — répliques et voix off", "| texte | type | score | trouvé |", "|---|---|---|---|"]
        for x in d["asr"]["lines"]:
            L.append(f"| {x['text']} | {x['kind']} | {x['score']} | {'oui' if x['found'] else 'NON'} |")
    if d.get("flashes", {}).get("transitions") is not None:
        L += ["", "## Flashs", f"- maximum {d['flashes']['max_per_s']} flash(s) par seconde (t≈{d['flashes']['t']} s)"]
    if res.get("render"):
        rr = res["render"]
        L += ["", "## Rendu", f"- STUB utilisés : {rr.get('stubs') or 'aucun'}",
              f"- garde-fou flashs : {[f for f in rr.get('flashes', []) if f.get('action') != 'gardé'] or 'rien sauté'}"]
        if rr.get("warnings"):
            L += ["- avertissements :"] + [f"  - {w}" for w in rr["warnings"][:60]]
    if res.get("mix"):
        L += ["", "## Mixage", f"- {res['mix'].get('lufs_integrated')} LUFS, {res['mix'].get('true_peak_dbtp')} dBTP "
              f"(WAV), screamer le plus fort : {res['mix'].get('screamer_is_loudest')}"]
        for w in res["mix"].get("warnings", [])[:30]:
            L.append(f"  - {w}")
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")


def qc(final_mp4, edl: dict | None = None, quality: str | None = None, run_asr: bool = True, log=print) -> dict:
    """Contrôle qualité complet du MP4 livré. Écrit RENDER_DIR/qc/qc_report.md + qc_report.json."""
    t_all = time.time()
    mp4 = Path(final_mp4)
    if not mp4.exists():
        raise FileNotFoundError(f"fichier à contrôler absent : {mp4}")
    if edl is None:
        try:
            edl = R.load_edl()
        except Exception:
            edl = None
    try:
        w, _ = R.video_size(mp4)
    except Exception:
        w = config.W
    quality = quality or ("preview" if w < config.W else "final")
    qdir = config.RENDER_DIR / "qc"
    qdir.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    data: dict = {}
    n_exp = R.fidx(R.edl_duration(edl)) if edl else None

    stats: dict = {}

    def _stats():
        if not stats:
            stats.update(frame_stats(mp4))
        return stats

    steps = [("format", lambda: check_format(mp4, quality, n_exp)),
             ("black", lambda: check_black(mp4, edl, _stats())),
             ("loudness", lambda: check_loudness(mp4, edl)),
             ("flashes", lambda: check_flashes(mp4, _stats()))]
    if edl:
        steps += [("subtitles", lambda: check_subtitles(edl)), ("names", lambda: check_names(edl))]
    if run_asr:
        steps.append(("asr", lambda: check_asr(mp4, edl, log)))
    for key, fn in steps:
        t0 = time.time()
        try:
            c, d = fn()
        except Exception as e:  # un contrôle qui plante = échec explicite
            c, d = [_check(key, "fail", None, None, f"contrôle impossible : {type(e).__name__}: {e}")], {}
        checks += c
        data[key] = d
        log(f"[qc] {key:9s} " + ", ".join(f"{x['name']}={x['status']}" for x in c) + f" ({time.time() - t0:.0f} s)")
    if not run_asr:
        checks.append(_check("ASR (liste noire)", "skip", None, None, "désactivé"))
    rep_p = config.RENDER_DIR / f"video_{quality}.json"
    render_rep = None
    if rep_p.exists():
        render_rep = json.loads(rep_p.read_text(encoding="utf-8"))
        st = render_rep.get("stubs") or []
        checks.append(_check("modules réels (pas de STUB)", ("fail" if quality == "final" else "warn") if st else "pass",
                             st or "aucun STUB", "aucun STUB"))
        skipped = [f for f in render_rep.get("flashes", []) if f.get("action") != "gardé"]
        if skipped:
            checks.append(_check("garde-fou flashs (rendu)", "warn", skipped, "rien à sauter",
                                 "effets de flash sautés au rendu"))
    mix_p = config.RENDER_DIR / f"mix_{quality}.json"
    mix_rep = json.loads(mix_p.read_text(encoding="utf-8")) if mix_p.exists() else None
    sheet = None
    try:
        sheet = contact_sheet(mp4, qdir / f"contact_{quality}.jpg")
    except Exception as e:
        checks.append(_check("planche contact", "warn", None, None, str(e)))
    res = {"file": str(mp4), "quality": quality, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "pass": not any(c["status"] == "fail" for c in checks), "checks": checks, "data": data,
           "contact_sheet": str(sheet) if sheet else None, "render": render_rep and {
               k: render_rep.get(k) for k in ("stubs", "flashes", "warnings", "frames")},
           "mix": mix_rep and {k: mix_rep.get(k) for k in ("lufs_integrated", "true_peak_dbtp", "screamer_is_loudest",
                                                           "warnings")},
           "qc_s": round(time.time() - t_all, 1)}
    (qdir / "qc_report.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    write_report(res, qdir / "qc_report.md")
    if quality == "final":
        _update_lisezmoi(res)
    nf = sum(c["status"] == "fail" for c in checks)
    nw = sum(c["status"] == "warn" for c in checks)
    log(f"[qc] {'VALIDÉ' if res['pass'] else 'À CORRIGER'} : {nf} échec(s), {nw} avertissement(s) -> "
        f"{qdir / 'qc_report.md'}")
    return res


def _update_lisezmoi(res: dict) -> None:
    """Dernière ligne du LISEZMOI = verdict réel du contrôle qualité."""
    p = config.OUT_DIR / config.README_TXT
    if not p.exists():
        return
    lines = p.read_text(encoding="utf-8").splitlines()[:5]
    fails = [c["name"] for c in res["checks"] if c["status"] == "fail"]
    last = ("Contrôle qualité automatique validé (durée, format, loudness, ASR, noirs, flashs, sous-titres, prénoms) ; "
            "originaux intacts." if not fails else
            "Contrôle qualité automatique : À CORRIGER (" + ", ".join(fails[:4]) + ") ; originaux intacts.")
    if len(lines) >= 5:
        lines[4] = last
    else:
        lines.append(last)
    p.write_text("\n".join(lines[:5]) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Contrôle qualité du film livré")
    ap.add_argument("mp4", nargs="?", default=None)
    ap.add_argument("--quality", default=None, choices=["preview", "final"])
    ap.add_argument("--edl", default=None)
    ap.add_argument("--no-asr", action="store_true")
    a = ap.parse_args(argv)
    mp4 = a.mp4 or (config.OUT_DIR / config.FINAL_MP4)
    edl = R.load_edl(a.edl) if a.edl else None
    res = qc(mp4, edl, a.quality, run_asr=not a.no_asr)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
