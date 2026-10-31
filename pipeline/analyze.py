"""Analyse des proxies : transcriptions mot à mot, planches contact, visages, personnes, enveloppe audio,
luminance par image, consignes de tournage, rapport de revue `analysis/report.md`.

Tout est écrit dans config.ANALYSIS_DIR (jamais dans le dépôt). Les fonctions de lecture (`find_phrase`,
`resolve_auto`, `detect_instructions_in_words`...) ne dépendent d'aucun modèle lourd : `edit/build_edl.py`
les utilise et retombe sur ses valeurs de repli si un fichier d'analyse manque.

CLI : python -m pipeline.analyze [--force] [--only IMG_8346 ...] [--jobs 2] [--skip asr,faces,...]
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config  # noqa: E402

FFMPEG = os.environ.get("ESCAPE_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("ESCAPE_FFPROBE", "ffprobe")
STEP_S = 0.2           # visages / personnes à 5 i/s
FACE_TILE = 400        # tuiles carrées pour les petits visages (détecteur courte portée 128x128)
THUMB_W, THUMB_H = 216, 384
SHEET_COLS = 8


def log(msg: str) -> None:
    """print robuste : une console Windows en cp1252 ne doit jamais faire planter le pipeline (→, ≠, …)."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(msg.encode(enc, "replace").decode(enc, "replace"), flush=True)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def _adir(analysis_dir=None) -> Path:
    return Path(analysis_dir) if analysis_dir else config.ANALYSIS_DIR


def _read_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(p: Path, obj) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, p)
    return p


def _proxy(clip: str) -> Path:
    p = config.proxy_path(clip)
    if not p.exists():
        raise FileNotFoundError(f"proxy absent : {p} (lancer d'abord python -m pipeline.ingest)")
    return p


def clip_duration(clip: str, analysis_dir=None) -> float | None:
    """Durée du proxy (inventaire, sinon ffprobe), None si inconnue."""
    inv = _read_json(_adir(analysis_dir) / "inventory.json") or {}
    d = ((inv.get("clips") or {}).get(clip) or {}).get("duration")
    if d:
        return float(d)
    p = config.proxy_path(clip)
    if p.exists() and p.suffix == ".mp4":
        r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        try:
            return float(r.stdout.strip())
        except ValueError:
            return None
    return None


def _ffmpeg_frames(path: Path, vf: str, w: int, h: int, pix: str = "rgb24"):
    """Générateur d'images (numpy) décodées par ffmpeg avec le filtre `vf` (dimensions de sortie w x h)."""
    ch = 3 if pix == "rgb24" else 1
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-i", str(path), "-vf", vf, "-an"]
    if "select=" in vf:
        cmd += ["-fps_mode", "passthrough"]  # uniquement les images sélectionnées, sans duplication
    cmd += ["-f", "rawvideo", "-pix_fmt", pix, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = w * h * ch
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, ch) if ch == 3 else np.frombuffer(buf, np.uint8).reshape(h, w)
    finally:
        proc.stdout.close()
        proc.wait()


def _load_audio(path: Path, sr: int = config.SR) -> np.ndarray:
    r = subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-i", str(path), "-vn", "-map", "0:a:0?", "-ac", "1",
                        "-ar", str(sr), "-f", "f32le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.float32).copy()


# ---------------------------------------------------------------------------
# Texte : normalisation et recherche floue
# ---------------------------------------------------------------------------
def norm_text(s: str) -> str:
    """minuscules, sans accents, apostrophes/traits d'union/ponctuation → espaces."""
    s = (s or "").lower().replace("œ", "oe").replace("æ", "ae")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _word_tokens(words: list[dict]) -> list[tuple[str, int]]:
    out = []
    for i, w in enumerate(words):
        for t in norm_text(w.get("w", "")).split():
            out.append((t, i))
    return out


def match_phrase(words: list[dict], phrase: str, window=None, min_score: float | None = None) -> list[dict]:
    """Toutes les occurrences (non chevauchantes, triées dans le temps) d'une phrase dans une liste de mots.
    Similarité caractère (difflib) entre la phrase normalisée et des fenêtres de jetons de longueur voisine."""
    ptoks = norm_text(phrase).split()
    if not ptoks or not words:
        return []
    if window:
        lo, hi = float(window[0]), float(window[1])
        idx = [i for i, w in enumerate(words) if lo - 0.05 <= w["start"] <= hi]
        if not idx:
            return []
        sub = [words[i] for i in idx]
    else:
        idx, sub = list(range(len(words))), words
    toks = _word_tokens(sub)
    if not toks:
        return []
    n = len(ptoks)
    if min_score is None:
        min_score = 0.86 if n <= 1 else (0.8 if n == 2 else 0.7)
    target = " ".join(ptoks)
    cands = []
    for i in range(len(toks)):
        for L in range(max(1, n - 2), n + 3):
            if i + L > len(toks):
                break
            win = [t for t, _ in toks[i:i + L]]
            sm = difflib.SequenceMatcher(None, target, " ".join(win), autojunk=False)
            score = sm.ratio()
            if win[0] == ptoks[0]:
                score += 0.02  # départage : préférer une fenêtre qui commence sur le 1er mot
            if win[-1] == ptoks[-1]:
                score += 0.02
            if score >= min_score:
                cands.append((score, toks[i][1], toks[i + L - 1][1]))
    cands.sort(key=lambda c: (-c[0], c[1]))
    if cands:  # occurrences : seulement celles proches de la meilleure (évite les correspondances partielles)
        best = cands[0][0]
        cands = [c for c in cands if c[0] >= best - 0.15]
    taken: list[tuple[int, int]] = []
    out = []
    for sc, a, b in cands:
        if any(not (b < x or a > y) for x, y in taken):
            continue
        taken.append((a, b))
        out.append({"start": float(sub[a]["start"]), "end": float(sub[b]["end"]), "score": round(min(sc, 1.0), 3),
                    "text": " ".join(sub[k]["w"] for k in range(a, b + 1)), "i0": idx[a], "i1": idx[b]})
    out.sort(key=lambda m: m["start"])
    return out


def transcript_path(clip: str, analysis_dir=None) -> Path:
    return _adir(analysis_dir) / f"{clip}.transcript.json"


def load_transcript(clip: str, analysis_dir=None) -> dict | None:
    return _read_json(transcript_path(clip, analysis_dir))


def find_phrase_match(clip: str, phrase: str, occ: int = 0, window=None, analysis_dir=None) -> dict | None:
    tr = load_transcript(clip, analysis_dir)
    if not tr or not tr.get("words"):
        return None
    ms = match_phrase(tr["words"], phrase, window)
    if not ms:
        return None
    try:
        return ms[occ]
    except IndexError:
        return ms[-1] if occ > 0 else None


def find_phrase(clip, phrase, edge="start", occ=0, window=None, analysis_dir=None) -> float | None:
    """Instant (s, temps du proxy) du début (`edge='start'`) ou de la fin (`'end'`) de la phrase, ou None."""
    m = find_phrase_match(clip, phrase, occ, window, analysis_dir)
    if m is None:
        return None
    return round(m["start"] if edge == "start" else m["end"], 3)


def transcribe(clip, force=False, backend=None, analysis_dir=None) -> dict:
    """Transcription mot à mot du proxy → analysis/<clip>.transcript.json (format du contrat)."""
    out = transcript_path(clip, analysis_dir)
    src = _proxy(clip)
    if out.exists() and not force:
        old = _read_json(out)
        if old and old.get("proxy_mtime") == int(src.stat().st_mtime):
            return old
    from pipeline import asr
    t0 = time.time()
    r = asr.transcribe_file(src, backend=backend)
    res = {"clip": clip, "backend": r["backend"], "language": "fr", "words": r["words"], "segments": r["segments"],
           "duration": r.get("duration"), "errors": r.get("errors", []), "proxy_mtime": int(src.stat().st_mtime)}
    _write_json(out, res)
    log(f"[analyze] {clip}: transcription {r['backend']} ({len(r['words'])} mots, {time.time() - t0:.1f} s) "
        f"« {' '.join(s['text'] for s in r['segments'])[:110]} »")
    return res


# ---------------------------------------------------------------------------
# Consignes de tournage
# ---------------------------------------------------------------------------
def _utterance_bounds(words: list[dict], i0: int, i1: int, gap: float = 0.35) -> tuple[int, int]:
    a, b = i0, i1
    while a > 0 and words[a]["start"] - words[a - 1]["end"] < gap:
        a -= 1
    while b + 1 < len(words) and words[b + 1]["start"] - words[b]["end"] < gap:
        b += 1
    return a, b


def _tok_eq(t: str, p: str) -> bool:
    """Égalité de mots entiers, tolérante à une lettre près pour les mots de 4 lettres et plus
    (filme/filmes, possede/possedes ; 4 lettres : terminaison seulement, fais/fait, crie/criez) ; les mots courts
    (y, le, la, c, est, as…) doivent être identiques."""
    if t == p:
        return True
    if len(p) < 4 or t[:1] != p[:1] or abs(len(t) - len(p)) > 1:
        return False
    if len(p) == 4:  # mots de 4 lettres : seule la terminaison varie (fais/fait, crie/criez)
        return t[:3] == p[:3] and len(t) <= 5
    if len(t) == len(p):
        return sum(x != y for x, y in zip(t, p)) <= 1
    short, long_ = (t, p) if len(t) < len(p) else (p, t)
    return any(long_[:k] + long_[k + 1:] == short for k in range(len(long_)))


def match_blacklist(tokens: list[str], blacklist=None) -> list[tuple[int, int, str]]:
    """Consignes trouvées dans une suite de jetons normalisés (norm_text) : [(i0, i1 inclus, consigne)].
    Recherche par MOTS ENTIERS (même nombre de mots, chacun égal à une lettre près s'il est long)."""
    blacklist = blacklist if blacklist is not None else config.INSTRUCTION_BLACKLIST
    out = []
    for ph in blacklist:
        pt = norm_text(ph).split()
        n = len(pt)
        if not n:
            continue
        for i in range(len(tokens) - n + 1):
            if all(_tok_eq(tokens[i + k], pt[k]) for k in range(n)):
                out.append((i, i + n - 1, ph))
    return sorted(out)


def detect_instructions_in_words(words: list[dict], blacklist=None, pad: float = 0.12) -> list[dict]:
    """Consignes (config.INSTRUCTION_BLACKLIST, mots entiers) trouvées dans des mots horodatés. Chaque repère couvre
    l'énoncé entier qui contient la consigne (mots séparés par < 0,35 s), avec une marge `pad`."""
    toks = _word_tokens(words)
    hits = []
    words_only = set(getattr(config, "INSTRUCTION_WORDS_ONLY", []))
    for i0, i1, ph in match_blacklist([t for t, _ in toks], blacklist):
        if ph in words_only:  # mot courant (« attends », « c'est bon ») : le mot seul
            a, b = toks[i0][1], toks[i1][1]
        else:  # vraie consigne : tout l'énoncé
            a, b = _utterance_bounds(words, toks[i0][1], toks[i1][1])
        hits.append({"phrase": ph, "start": round(max(0.0, words[a]["start"] - pad), 3),
                     "end": round(words[b]["end"] + pad, 3), "score": 1.0,
                     "text": " ".join(words[k]["w"] for k in range(a, b + 1))})
    hits.sort(key=lambda h: h["start"])
    merged: list[dict] = []
    for h in hits:  # fusion des repères qui se recouvrent
        if merged and h["start"] <= merged[-1]["end"]:
            m = merged[-1]
            m["end"] = max(m["end"], h["end"])
            if h["phrase"] not in m["phrase"]:
                m["phrase"] += " | " + h["phrase"]
            if len(h["text"]) > len(m["text"]):
                m["text"] = h["text"]
        else:
            merged.append(dict(h))
    return merged


def detect_instructions(clip, analysis_dir=None) -> list[dict]:
    tr = load_transcript(clip, analysis_dir)
    return detect_instructions_in_words(tr["words"]) if tr and tr.get("words") else []


# ---------------------------------------------------------------------------
# Visages et personnes (MediaPipe Tasks)
# ---------------------------------------------------------------------------
MODEL_URLS = {
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/"
                                     "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "efficientdet_lite2.tflite": "https://storage.googleapis.com/mediapipe-models/object_detector/"
                                 "efficientdet_lite2/float32/latest/efficientdet_lite2.tflite",
}


def ensure_model(name: str) -> Path:
    """Chemin d'un modèle MediaPipe ; téléchargé (storage.googleapis.com) s'il manque dans MODELS_DIR."""
    p = config.model_path(name)
    if p.exists():
        return p
    url = MODEL_URLS.get(name)
    if not url:
        raise FileNotFoundError(p)
    import urllib.request
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    log(f"[analyze] téléchargement de {name} …")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, p)
    return p


def _mp_detectors():
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    fd = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(ensure_model("blaze_face_short_range.tflite"))),
        running_mode=vision.RunningMode.IMAGE, min_detection_confidence=0.5))
    od = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(ensure_model("efficientdet_lite2.tflite"))),
        running_mode=vision.RunningMode.IMAGE, max_results=8, score_threshold=0.3, category_allowlist=["person"]))
    return mp, fd, od


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-6, min(aa, ab))  # intersection / plus petite boîte (fusion des doublons de tuiles)


def _nms(boxes: list[dict], thr: float = 0.5) -> list[dict]:
    out: list[dict] = []
    for b in sorted(boxes, key=lambda d: -d["score"]):
        if all(_iou(b["bbox"], o["bbox"]) < thr for o in out):
            out.append(b)
    return out


def _positions(total: int, size: int, step: float) -> list[int]:
    if total <= size:
        return [0]
    ps = list(range(0, total - size + 1, max(1, int(step))))
    if ps[-1] != total - size:
        ps.append(total - size)
    return ps


def _detect_faces_img(mp, fd, rgb: np.ndarray, tiles: bool = True, tile_thr: float = 0.72) -> list[dict]:
    """Visages (BlazeFace courte portée) sur l'image entière + tuiles carrées (grandes et 400 px) pour
    retrouver les petits visages des plans larges ; fusion des doublons."""
    H, W = rgb.shape[:2]
    regions = [(0, 0, W, H, 0.5)]
    if tiles:
        s = min(W, H)
        for y in _positions(H, s, s * 0.6):
            for x in _positions(W, s, s * 0.6):
                regions.append((x, y, s, s, tile_thr))
        for y in _positions(H, FACE_TILE, FACE_TILE * 0.8):
            for x in _positions(W, FACE_TILE, FACE_TILE * 0.8):
                regions.append((x, y, FACE_TILE, FACE_TILE, tile_thr + 0.03))
    found = []
    for x, y, w, h, thr in regions:
        crop = np.ascontiguousarray(rgb[y:y + h, x:x + w])
        res = fd.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
        for d in res.detections:
            sc = float(d.categories[0].score)
            if sc < thr:
                continue
            bb = d.bounding_box
            x0, y0 = x + bb.origin_x, y + bb.origin_y
            box = [max(0, x0), max(0, y0), min(W, x0 + bb.width), min(H, y0 + bb.height)]
            if box[2] - box[0] < 12 or box[3] - box[1] < 12:
                continue
            face = {"bbox": [int(v) for v in box], "score": round(sc, 3)}
            kp = d.keypoints or []
            if len(kp) >= 4:  # yeux (0, 1) et bouche (3) : inclinaison du visage (0° = droit, bouche sous les yeux)
                ex, ey = (kp[0].x + kp[1].x) / 2 * w, (kp[0].y + kp[1].y) / 2 * h
                face["roll"] = round(math.degrees(math.atan2(kp[3].x * w - ex, kp[3].y * h - ey)), 1)
            found.append(face)
    return _nms(found, 0.45)


def _detect_persons_img(mp, od, rgb: np.ndarray) -> list[dict]:
    H, W = rgb.shape[:2]
    res = od.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    out = []
    for d in res.detections:
        bb = d.bounding_box
        box = [max(0, bb.origin_x), max(0, bb.origin_y), min(W, bb.origin_x + bb.width), min(H, bb.origin_y + bb.height)]
        out.append({"bbox": [int(v) for v in box], "score": round(float(d.categories[0].score), 3)})
    return _nms(out, 0.7)


def _orientation_suspect(mp, fd, samples: list[np.ndarray], n_upright: int, faces=None) -> int | None:
    """Contrôle d'orientation du proxy : visages couchés (inclinaison médiane > 50° d'après les points clés),
    ou visages qui n'apparaissent qu'en tournant l'image. Renvoie l'angle suspect (90/180/270) ou None."""
    rolls = [f["roll"] for fs in (faces or []) for f in fs if f.get("score", 0) > 0.8 and "roll" in f]
    if len(rolls) >= 3:
        med = float(np.median(rolls))
        if abs(med) > 50:
            return int(round(med / 90.0)) * 90 % 360 or None
        return None
    if n_upright >= 3 or len(samples) < 2:
        return None
    hits = {90: 0, 180: 0, 270: 0}
    for im in samples[:8]:
        for ang, k in ((90, 1), (180, 2), (270, 3)):
            res = fd.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.rot90(im, k))))
            if any(d.categories[0].score > 0.85 for d in res.detections):
                hits[ang] += 1
    ang, n = max(hits.items(), key=lambda kv: kv[1])
    return ang if n >= 2 and n > n_upright else None


def _vision_job(clip: str, step_s: float, analysis_dir: str | None) -> tuple[str, int, float]:
    """Une passe de décodage à 1/step_s i/s : visages + personnes → <clip>.faces.json / .persons.json."""
    t_start = time.time()
    mp, fd, od = _mp_detectors()
    try:
        src = _proxy(clip)
        every = max(1, int(round(step_s * config.FPS)))  # proxies CFR 30 i/s : images exactes n = 0, 6, 12...
        step_s = every / config.FPS
        ts, faces, persons, samples = [], [], [], []
        for k, rgb in enumerate(_ffmpeg_frames(src, f"select=not(mod(n\\,{every}))", config.W, config.H)):
            ts.append(round(k * step_s, 3))
            faces.append(_detect_faces_img(mp, fd, rgb))
            persons.append(_detect_persons_img(mp, od, rgb))
            if k % 5 == 0 and not faces[-1]:
                samples.append(np.ascontiguousarray(rgb[::2, ::2]))
        meta = {"clip": clip, "step_s": step_s, "w": config.W, "h": config.H,
                "proxy_mtime": int(src.stat().st_mtime),
                "orientation_suspect": _orientation_suspect(mp, fd, samples, sum(1 for f in faces if f), faces)}
        a = _adir(analysis_dir)
        _write_json(a / f"{clip}.faces.json", {**meta, "t": ts, "faces": faces})
        _write_json(a / f"{clip}.persons.json", {**meta, "t": ts, "persons": persons})
        return clip, len(ts), time.time() - t_start
    finally:
        fd.close()
        od.close()


def _vision_fresh(clip: str, kind: str, analysis_dir=None) -> bool:
    d = _read_json(_adir(analysis_dir) / f"{clip}.{kind}.json")
    p = config.proxy_path(clip)
    return bool(d) and p.exists() and d.get("proxy_mtime") == int(p.stat().st_mtime)


def detect_faces(clip, step_s=STEP_S, force=False, analysis_dir=None) -> dict:
    """Visages à 5 i/s → analysis/<clip>.faces.json {"t":[...], "faces":[[{"bbox":[x0,y0,x1,y1],"score"}]]}."""
    if force or not _vision_fresh(clip, "faces", analysis_dir):
        _vision_job(clip, step_s, str(_adir(analysis_dir)))
    return _read_json(_adir(analysis_dir) / f"{clip}.faces.json")


def detect_persons(clip, step_s=STEP_S, force=False, analysis_dir=None) -> dict:
    """Personnes (efficientdet_lite2, catégorie person) → analysis/<clip>.persons.json."""
    if force or not _vision_fresh(clip, "persons", analysis_dir):
        _vision_job(clip, step_s, str(_adir(analysis_dir)))
    return _read_json(_adir(analysis_dir) / f"{clip}.persons.json")


def photo_faces(photo: str, force=False, analysis_dir=None) -> dict:
    """Visages d'une photo (PNG proxy) → analysis/<photo>.faces.json (t=[0]) + netteté (variance du laplacien)."""
    out = _adir(analysis_dir) / f"{photo}.faces.json"
    src = _proxy(photo)
    if out.exists() and not force:
        d = _read_json(out)
        if d and d.get("proxy_mtime") == int(src.stat().st_mtime):
            return d
    from PIL import Image
    import cv2
    rgb = np.asarray(Image.open(src).convert("RGB"))
    mp, fd, od = _mp_detectors()
    try:
        faces = _detect_faces_img(mp, fd, rgb, tile_thr=0.6)  # photo unique et nette : seuil plus permissif
        persons = _detect_persons_img(mp, od, rgb)
    finally:
        fd.close()
        od.close()
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    d = {"clip": photo, "w": rgb.shape[1], "h": rgb.shape[0], "t": [0.0], "faces": [faces], "persons": [persons],
         "sharpness": round(sharp, 1), "proxy_mtime": int(src.stat().st_mtime)}
    _write_json(out, d)
    return d


# ---------------------------------------------------------------------------
# Audio et luminance
# ---------------------------------------------------------------------------
def audio_peaks(clip, force=False, analysis_dir=None) -> dict:
    """Enveloppe RMS (pas 10 ms, fenêtre 20 ms, dBFS) + pics → analysis/<clip>.audio.json."""
    out = _adir(analysis_dir) / f"{clip}.audio.json"
    src = _proxy(clip)
    if out.exists() and not force:
        d = _read_json(out)
        if d and d.get("proxy_mtime") == int(src.stat().st_mtime):
            return d
    from scipy.signal import find_peaks
    sr = config.SR
    x = _load_audio(src, sr)
    hop, win = int(0.01 * sr), int(0.02 * sr)
    n = max(0, (len(x) - win) // hop + 1)
    if n:
        idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
        rms = np.sqrt((x[idx] ** 2).mean(axis=1))
    else:
        rms = np.zeros(0)
    db = 20 * np.log10(rms + 1e-9)
    peaks = []
    if n > 3:
        pk, prop = find_peaks(db, prominence=6.0, distance=25)
        order = np.argsort(-db[pk])[:20]
        peaks = sorted(({"t": round((pk[i] * hop + win / 2) / sr, 3), "db": round(float(db[pk[i]]), 1),
                         "prominence": round(float(prop["prominences"][i]), 1)} for i in order), key=lambda p: p["t"])
    d = {"clip": clip, "hop": 0.01, "win": 0.02, "t0": round(win / 2 / sr, 3),
         "rms_db": [round(float(v), 1) for v in db], "peaks": peaks,
         "max_db": round(float(db.max()), 1) if n else None,
         "t_max": round((int(np.argmax(db)) * hop + win / 2) / sr, 3) if n else None,
         "proxy_mtime": int(src.stat().st_mtime)}
    _write_json(out, d)
    return d


def flash_frames(clip, force=False, analysis_dir=None) -> dict:
    """Luminance moyenne (0..1) de chaque image à 30 i/s + éclairs détectés → analysis/<clip>.luma.json."""
    out = _adir(analysis_dir) / f"{clip}.luma.json"
    src = _proxy(clip)
    if out.exists() and not force:
        d = _read_json(out)
        if d and d.get("proxy_mtime") == int(src.stat().st_mtime):
            return d
    w, h = 54, 96
    luma = [float(f.mean()) / 255.0 for f in _ffmpeg_frames(src, f"scale={w}:{h}:flags=area", w, h, "gray")]
    L = np.array(luma)
    flashes = []
    if len(L) > 5:
        from scipy.ndimage import median_filter
        base = median_filter(L, size=31, mode="nearest")
        delta = L - base
        hot = delta > max(0.08, 3 * float(np.std(delta)))
        i = 0
        while i < len(L):
            if hot[i]:
                j = i
                while j < len(L) and hot[j]:
                    j += 1
                k = i + int(np.argmax(L[i:j]))
                flashes.append({"t": round(k / config.FPS, 3), "luma": round(float(L[k]), 3),
                                "delta": round(float(delta[k]), 3), "frames": j - i})
                i = j
            else:
                i += 1
    d = {"clip": clip, "fps": config.FPS, "luma": [round(v, 4) for v in luma], "flashes": flashes,
         "proxy_mtime": int(src.stat().st_mtime)}
    _write_json(out, d)
    return d


# ---------------------------------------------------------------------------
# Planches contact
# ---------------------------------------------------------------------------
def _font(size: int):
    from PIL import ImageFont
    for name in ("IBMPlexMono-Regular.ttf", "IBMPlexMono-Medium.ttf", "Inter-Regular.ttf"):
        for p in config.FONTS_DIR.rglob(name):
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                pass
    for p in sorted(config.FONTS_DIR.rglob("*.ttf")):
        try:
            return ImageFont.truetype(str(p), size)
        except OSError:
            pass
    return ImageFont.load_default()


def _nearest(ts: list[float], t: float) -> int | None:
    if not ts:
        return None
    k = int(np.argmin(np.abs(np.asarray(ts) - t)))
    return k if abs(ts[k] - t) <= 0.15 else None


def contact_sheet(clip, analysis_dir=None) -> Path:
    """Planche contact 1 i/s horodatée (visages en vert, personnes en orange, transcription et consignes en bas)
    → analysis/contact/<clip>.jpg."""
    from PIL import Image, ImageDraw
    a = _adir(analysis_dir)
    src = _proxy(clip)
    frames = list(_ffmpeg_frames(src, f"select=not(mod(n\\,{config.FPS})),scale={THUMB_W}:{THUMB_H}:flags=area",
                                 THUMB_W, THUMB_H))
    faces = _read_json(a / f"{clip}.faces.json") or {}
    persons = _read_json(a / f"{clip}.persons.json") or {}
    tr = load_transcript(clip, a) or {}
    instr = detect_instructions(clip, a)
    sx = THUMB_W / config.W
    f_small, f_mid, f_big = _font(15), _font(17), _font(26)
    cols = SHEET_COLS
    rows = max(1, math.ceil(len(frames) / cols))
    pad, head = 6, 46
    lines = []
    for s in tr.get("segments", []):
        bad = any(h["start"] - 0.05 <= s["start"] <= h["end"] for h in instr)
        lines.append((f"[{s['start']:6.2f}–{s['end']:6.2f}] {s['text']}", bad))
    for h in instr:
        lines.append((f"!! CONSIGNE {h['start']:.2f}–{h['end']:.2f} s : « {h['text']} » ({h['phrase']})", True))
    wrap = []
    max_chars = int((cols * (THUMB_W + pad)) / 9.2)
    for txt, bad in lines:
        while len(txt) > max_chars:
            cut = txt.rfind(" ", 0, max_chars)
            cut = cut if cut > 20 else max_chars
            wrap.append((txt[:cut], bad))
            txt = "      " + txt[cut:].lstrip()
        wrap.append((txt, bad))
    foot = 16 + 21 * max(1, len(wrap))
    W = cols * (THUMB_W + pad) + pad
    H = head + rows * (THUMB_H + pad) + pad + foot
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    dr = ImageDraw.Draw(sheet)
    dur = clip_duration(clip, a) or len(frames)
    dr.text((pad + 2, 8), f"{clip}   {dur:.2f} s   ASR : {tr.get('backend', '—')}   "
                          f"(vert = visages, orange = personnes, rouge = consigne)", font=f_big, fill=(235, 235, 235))
    for k, fr in enumerate(frames):
        im = Image.fromarray(fr)
        d2 = ImageDraw.Draw(im)
        t = float(k)
        j = _nearest(faces.get("t", []), t)
        if j is not None:
            for f in faces["faces"][j]:
                b = [v * sx for v in f["bbox"]]
                d2.rectangle(b, outline=(40, 230, 90), width=2)
        j = _nearest(persons.get("t", []), t)
        if j is not None:
            for p in persons["persons"][j]:
                b = [v * sx for v in p["bbox"]]
                d2.rectangle(b, outline=(255, 150, 30), width=2)
        if any(h["start"] <= t + 0.5 and h["end"] >= t - 0.5 for h in instr):
            d2.rectangle([0, 0, THUMB_W - 1, THUMB_H - 1], outline=(230, 30, 30), width=4)
        d2.rectangle([0, THUMB_H - 24, 76, THUMB_H], fill=(0, 0, 0))
        d2.text((5, THUMB_H - 22), f"{t:5.1f} s", font=f_mid, fill=(255, 230, 80))
        x = pad + (k % cols) * (THUMB_W + pad)
        y = head + (k // cols) * (THUMB_H + pad)
        sheet.paste(im, (x, y))
    y = head + rows * (THUMB_H + pad) + 8
    for txt, bad in (wrap or [("(aucune parole reconnue)", False)]):
        dr.text((pad + 2, y), txt, font=f_small, fill=(255, 90, 90) if bad else (210, 210, 210))
        y += 21
    out = a / "contact" / f"{clip}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=86)
    return out


def photo_sheet(photos: list[str], analysis_dir=None) -> Path | None:
    from PIL import Image, ImageDraw
    a = _adir(analysis_dir)
    tiles = []
    for p in photos:
        if not config.proxy_path(p).exists():
            continue
        im = Image.open(config.proxy_path(p)).convert("RGB")
        s = 640 / im.height
        th = im.resize((max(1, int(im.width * s)), 640), Image.LANCZOS)
        dr = ImageDraw.Draw(th)
        fj = _read_json(a / f"{p}.faces.json") or {}
        for f in (fj.get("faces") or [[]])[0]:
            dr.rectangle([v * s for v in f["bbox"]], outline=(40, 230, 90), width=2)
        for q in (fj.get("persons") or [[]])[0]:
            dr.rectangle([v * s for v in q["bbox"]], outline=(255, 150, 30), width=2)
        dr.rectangle([0, 0, th.width, 26], fill=(0, 0, 0))
        dr.text((6, 3), f"{p}  {len((fj.get('faces') or [[]])[0])} visages  net={fj.get('sharpness', '?')}",
                font=_font(16), fill=(255, 230, 80))
        tiles.append(th)
    if not tiles:
        return None
    W = sum(t.width for t in tiles) + 6 * (len(tiles) + 1)
    sheet = Image.new("RGB", (W, 652), (18, 18, 20))
    x = 6
    for t in tiles:
        sheet.paste(t, (x, 6))
        x += t.width + 6
    out = a / "contact" / "photos.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=86)
    return out


# ---------------------------------------------------------------------------
# Résolveurs automatiques (utilisés par edit/build_edl.py et par le rapport)
# ---------------------------------------------------------------------------
_RANGE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_auto(spec: str) -> dict:
    """'audio_peak:IMG_8338:4.3-7.6:-0.12' → {"kind","clip","t0","t1","offset"}."""
    parts = spec.split(":")
    if len(parts) < 3:
        raise ValueError(f"résolveur auto mal formé : {spec}")
    m = _RANGE.match(parts[2])
    if not m:
        raise ValueError(f"plage mal formée : {spec}")
    off = float(parts[3]) if len(parts) > 3 and parts[3].strip() else 0.0
    return {"kind": parts[0].strip(), "clip": parts[1].strip(), "t0": float(m.group(1)), "t1": float(m.group(2)),
            "offset": off}


def _in_range(ts, t0, t1):
    return [i for i, t in enumerate(ts) if t0 - 1e-6 <= t <= t1 + 1e-6]


def resolve_auto(spec: str, analysis_dir=None) -> tuple[object | None, str]:
    """Valeur d'un résolveur automatique, ou (None, raison) si l'analyse manque / ne trouve rien.
    person_peak / face_largest / flash_peak / audio_peak / person_absent_after → temps (s, proxy) ;
    person_center → [x, y] normalisés."""
    a = _adir(analysis_dir)
    try:
        p = parse_auto(spec)
    except ValueError as e:
        return None, str(e)
    kind, clip, t0, t1, off = p["kind"], p["clip"], p["t0"], p["t1"], p["offset"]
    if kind in ("person_peak", "person_center"):
        d = _read_json(a / f"{clip}.persons.json")
        if not d:
            return None, "personnes non analysées"
        best = None
        cands = []
        for i in _in_range(d["t"], t0, t1):
            for q in d["persons"][i]:
                x0, y0, x1, y1 = q["bbox"]
                area = (x1 - x0) * (y1 - y0) / (d["w"] * d["h"])
                v = q["score"] * (0.6 + 0.4 * min(1.0, area / 0.05))
                cands.append((v, d["t"][i], q))
                if best is None or v > best[0]:
                    best = (v, d["t"][i], q)
        if best is None:
            return None, f"aucune personne détectée entre {t0} et {t1} s"
        if kind == "person_peak":
            return round(best[1] + off, 3), f"personne la plus nette à {best[1]:.2f} s (score {best[2]['score']})"
        top = sorted(cands, key=lambda c: -c[0])[:max(3, len(cands) // 3)]
        bx = np.median([[(q["bbox"][0] + q["bbox"][2]) / 2, (q["bbox"][1] + q["bbox"][3]) / 2] for _, _, q in top], axis=0)
        return [round(float(bx[0]) / d["w"], 3), round(float(bx[1]) / d["h"], 3)], \
            f"centre médian de {len(top)} détections"
    if kind == "person_absent_after":
        # premier instant de la fenêtre à partir duquel plus AUCUNE personne n'est détectée (jusqu'à t1)
        d = _read_json(a / f"{clip}.persons.json")
        if not d:
            return None, "personnes non analysées"
        def present(i):
            return any(q.get("score", 0) >= 0.3 for q in d["persons"][i])
        idx = _in_range(d["t"], t0, t1)
        if not idx:
            return None, f"pas d'analyse entre {t0} et {t1} s"
        before = _in_range(d["t"], t0 - 3.0, t1)
        if not any(present(i) for i in before):
            return None, f"aucune personne détectée autour de {t0}–{t1} s (détecteur aveugle ?)"
        if present(idx[-1]):
            return None, f"une personne est encore détectée à {d['t'][idx[-1]]:.2f} s"
        k = len(idx) - 1
        while k > 0 and not present(idx[k - 1]):
            k -= 1
        t = d["t"][idx[k]]
        return round(t + off, 3), f"plus personne à partir de {t:.2f} s"
    if kind == "face_largest":
        d = _read_json(a / f"{clip}.faces.json")
        if not d:
            return None, "visages non analysés"
        best = None
        for i in _in_range(d["t"], t0, t1):
            for f in d["faces"][i]:
                x0, y0, x1, y1 = f["bbox"]
                v = (x1 - x0) * (y1 - y0) * f["score"]
                if best is None or v > best[0]:
                    best = (v, d["t"][i], f)
        if best is None:
            return None, f"aucun visage entre {t0} et {t1} s"
        w = best[2]["bbox"][2] - best[2]["bbox"][0]
        return round(best[1] + off, 3), f"plus grand visage à {best[1]:.2f} s ({w} px de large)"
    if kind == "flash_peak":
        d = _read_json(a / f"{clip}.luma.json")
        if not d or not d.get("luma"):
            return None, "luminance non analysée"
        L = np.asarray(d["luma"])
        i0, i1 = max(0, int(math.floor(t0 * d["fps"]))), min(len(L) - 1, int(math.ceil(t1 * d["fps"])))
        if i1 < i0:
            return None, "plage hors du clip"
        k = i0 + int(np.argmax(L[i0:i1 + 1]))
        seg = L[i0:i1 + 1]
        if float(seg.max() - np.median(seg)) < 0.05:
            return None, f"pas d'éclair net entre {t0} et {t1} s"
        return round(k / d["fps"] + off, 3), f"image la plus lumineuse à {k / d['fps']:.2f} s (luma {L[k]:.2f})"
    if kind == "audio_peak":
        d = _read_json(a / f"{clip}.audio.json")
        if not d or not d.get("rms_db"):
            return None, "audio non analysé"
        R = np.asarray(d["rms_db"])
        i0 = max(0, int((t0 - d["t0"]) / d["hop"]))
        i1 = min(len(R) - 1, int((t1 - d["t0"]) / d["hop"]))
        if i1 < i0:
            return None, "plage hors du clip"
        k = i0 + int(np.argmax(R[i0:i1 + 1]))
        t = d["t0"] + k * d["hop"]
        return round(t + off, 3), f"pic audio {R[k]:.1f} dBFS à {t:.2f} s"
    return None, f"résolveur inconnu : {kind}"


# ---------------------------------------------------------------------------
# Orchestration + rapport
# ---------------------------------------------------------------------------
def _spec_anchors() -> list[dict]:
    """Ancres A(...) de edit/edl_spec.py (clip, phrase, bord, repli) pour le rapport."""
    try:
        from edit import edl_spec
    except Exception:
        return []
    out, seen = [], set()

    def visit(obj, clip):
        if isinstance(obj, dict):
            if "anchor" in obj and "fallback" in obj:
                key = (clip, obj["anchor"], obj["edge"], obj.get("occ", 0), obj.get("offset", 0.0))
                if key not in seen and clip and not str(clip).startswith("decision:"):
                    seen.add(key)
                    out.append({"clip": clip, **obj})
                return
            for v in obj.values():
                visit(v, clip)
        elif isinstance(obj, (list, tuple)):
            if len(obj) == 3 and obj[0] == "clip":
                visit(obj[2], obj[1])
                return
            for v in obj:
                visit(v, clip)
    for sh in edl_spec.SHOTS:
        visit(sh, sh.get("clip"))
    return out


def _decision_autos(decisions: dict) -> list[tuple[str, str, object]]:
    out = []

    def visit(obj, path):
        if isinstance(obj, dict):
            if "auto" in obj:
                out.append((path, obj["auto"], obj.get("fallback")))
                return
            for k, v in obj.items():
                if not str(k).startswith("_"):
                    visit(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                visit(v, f"{path}[{i}]")
    visit(decisions, "")
    return out


REVIEW_HINTS = {
    "name_hugo": "Qui est à l'image ? Hugo = chignon, t-shirt noir, veste en cuir. Passer `confirmed: true` sur le bon candidat.",
    "name_arsene": "Arsène = lunettes, cheveux bouclés longs, haut noir WÜRTH, chapeau melon. Confirmer un candidat ou laisser sans prénom.",
    "hud_order_8347": "Ordre gauche→droite des 4 visages sur 8347 (planche contact) ; null = étiquettes sans prénom.",
    "possess_1": "8340 : visage qui émerge / yeux fixes ; vérifier QUI c'est (le chuchotement doit correspondre).",
    "possess_3": "8339 : selfie (probablement Hugo) — vérifier.",
    "possess_4": "8337 : passager éclairé en rouge — identité et `whisper` à vérifier.",
    "group_photo": "Choisir la photo de groupe la plus nette avec les 4 visages (voir contact/photos.jpg).",
    "silhouette_slots": "Positions d'incrustation (x, foot_y, height normalisés) : zones sombres crédibles, pas sur un visage.",
    "bodybag_8348_t": "Instant où le sac mortuaire est le mieux cadré.",
    "room_wide_8349_t": "Plan large de la salle (portraits visibles).",
    "gabriel_terror_8349_t": "Gros plan de Gabriel terrifié (7–13 s).",
    "bk_8355_t": "Burger King : rires, visages lisibles.",
    "bk_8356_t": "Burger King : moment où les rires peuvent ralentir.",
    "silhouette_center_8346": "Centre de la silhouette dans 8346 (poussée caméra).",
    "wall_empty_8346_t": "Première image de 8346 où la silhouette a COMPLÈTEMENT disparu derrière le mur (gel S3b).",
    "group_photo_faces": "Si la photo de groupe n'a pas 4 visages détectés : centres [[x, y, rayon]] normalisés, gauche→droite.",
    "privacy_zones": "Inconnus reconnaissables (BK, parking) non floutés automatiquement : {\"IMG_8357\": [[x0,y0,x1,y1,t0,t1]]}.",
    "hud_boxes_8347": "Si le HUD de 8347 ne trouve pas les 4 visages : 4 boîtes [[x0,y0,x1,y1]] normalisées, gauche→droite.",
    "portrait_eyes_8349": "Yeux des TABLEAUX de 8349 [[x, y, r]] normalisés si la détection automatique s'abstient.",
    "face_boxes": "Repli manuel par shot pour les effets visages (ex. screamer S6e) : {\"S6e\": [x0,y0,x1,y1]} normalisé.",
}


def write_report(clips: list[str], photos: list[str], analysis_dir=None, timings: dict | None = None) -> Path:
    a = _adir(analysis_dir)
    inv = _read_json(a / "inventory.json") or {}
    decisions = _read_json(config.EDIT_DIR / "decisions.json") or {}
    L = ["# Rapport d'analyse — C'EST QUI LUI ?", ""]
    if (config.RUSH_DIR / "FIXTURES_NOT_REAL.txt").exists():
        L += ["> **ATTENTION : FAUX RUSHS SYNTHÉTIQUES (tests techniques). Aucune décision créative ne doit en être tirée.**", ""]
    L += [f"- Rushs : `{config.RUSH_DIR}`", f"- Proxies : `{config.PROXY_DIR}`", f"- Analyse : `{a}`",
          f"- Planches contact : `{a / 'contact'}` (1 image/s, visages en vert, personnes en orange, consignes en rouge)", ""]
    L += ["## Comment utiliser ce rapport (étape de revue OBLIGATOIRE avant le rendu final)",
          "1. Regarder chaque planche `analysis/contact/IMG_xxxx.jpg` et `contact/photos.jpg`.",
          "2. Vérifier ci-dessous les répliques retrouvées (sinon le montage utilise les timecodes approximatifs du brief).",
          "3. Compléter `edit/decisions.json` : clés de `_review_required` (identités `confirmed`, ordre HUD, instants, "
          "positions d'incrustation), puis `\"_reviewed\": true`.",
          "4. Relancer `python -m edit.build_edl` et lire `edl_report.md` (durée, chevauchements, avertissements).", ""]
    # inventaire
    L += ["## Inventaire", "", "| clip | durée proxy | source | fps (r / moy.) | rotation | HDR | audio | remarques |",
          "|---|---|---|---|---|---|---|---|"]
    for c in clips:
        e = (inv.get("clips") or {}).get(c) or {}
        pr = e.get("probe") or {}
        v = pr.get("video") or {}
        au = [x for x in pr.get("audio", []) if x.get("index") == pr.get("audio_index")]
        L.append(f"| {c} | {e.get('duration', '—')} s | {v.get('codec', '?')} {v.get('width', '?')}x{v.get('height', '?')} "
                 f"| {v.get('r_frame_rate', '?')} / {round(v.get('fps_avg') or 0, 2)} | {v.get('rotation', '?')}° "
                 f"| {'oui' if v.get('hdr') else 'non'} | {(au[0]['codec'] + ' ' + str(au[0]['channels']) + 'ch') if au else 'aucun'} "
                 f"| {'; '.join(e.get('notes') or []) or ''}{' ERREUR ' + e['error'] if e.get('error') else ''} |")
    for p in photos:
        e = (inv.get("photos") or {}).get(p) or {}
        L.append(f"| {p} | photo | {e.get('raw_size')} EXIF {e.get('exif_orientation')} | — | — | — | — | "
                 f"{e.get('size')} {e.get('color', '')} {e.get('datetime') or ''} |")
    if inv.get("missing"):
        L += ["", f"**Rushs manquants** : {', '.join(inv['missing'])}"]
    if inv.get("unexpected"):
        L += ["", f"Fichiers non prévus dans le dossier des rushs : {', '.join(inv['unexpected'])}"]
    # répliques
    L += ["", "## Répliques du brief / ancres du montage (`edit/edl_spec.py`)", "",
          "Temps dans le proxy. « brief » = timecode approximatif utilisé en repli si la phrase est introuvable.", "",
          "| clip | phrase cherchée | début | fin | brief | score | texte reconnu |", "|---|---|---|---|---|---|---|"]
    seen = {}
    for an in _spec_anchors():
        k = (an["clip"], an["anchor"], an.get("occ", 0), str(an.get("window")))
        hint = an["fallback"] if (an["edge"] == "start" and not an.get("offset")) else None
        if k not in seen:
            seen[k] = dict(an, hint=hint)
        elif seen[k]["hint"] is None:
            seen[k]["hint"] = hint
    for (clip, phrase, occ, _), an in seen.items():
        m = find_phrase_match(clip, phrase, occ, an.get("window"), a)
        hint = "—" if an["hint"] is None else f"~{an['hint']}"
        if m:
            far = an["hint"] is not None and abs(m["start"] - an["hint"]) > 2.0
            L.append(f"| {clip} | {phrase} | {m['start']:.2f}{' **(loin du brief)**' if far else ''} | {m['end']:.2f} | "
                     f"{hint} | {m['score']:.2f} | {m['text']} |")
        else:
            L.append(f"| {clip} | {phrase} | **INTROUVABLE** | — | {hint} | — | — |")
    # consignes
    L += ["", "## Consignes de tournage repérées (coupées automatiquement si la plage est utilisée)", ""]
    any_hit = False
    for c in clips:
        for h in detect_instructions(c, a):
            any_hit = True
            L.append(f"- **{c}** {h['start']:.2f}–{h['end']:.2f} s : « {h['text']} » (motif : {h['phrase']})")
    if not any_hit:
        L.append("- aucune (vérifier quand même à l'écoute : la transcription peut en manquer)")
    # décisions
    L += ["", "## `edit/decisions.json` : valeurs suggérées par l'analyse", "",
          "| clé | résolveur | suggestion | repli | détail |", "|---|---|---|---|---|"]
    for path, spec, fb in _decision_autos(decisions):
        v, why = resolve_auto(spec, a)
        L.append(f"| {path} | `{spec}` | {v if v is not None else '—'} | {fb} | {why} |")
    L += ["", "### Clés à vérifier (`_review_required`)", ""]
    for k in decisions.get("_review_required", []):
        L.append(f"- `{k}` = `{json.dumps(decisions.get(k), ensure_ascii=False)[:160]}` — {REVIEW_HINTS.get(k, '')}")
    L.append(f"- `_reviewed` = {decisions.get('_reviewed')}")
    # photos
    if photos:
        L += ["", "## Photos (8350–8353)", "", "| photo | visages | personnes | netteté |", "|---|---|---|---|"]
        best = None
        for p in photos:
            d = _read_json(a / f"{p}.faces.json") or {}
            nf = len((d.get("faces") or [[]])[0])
            npers = len((d.get("persons") or [[]])[0])
            L.append(f"| {p} | {nf} | {npers} | {d.get('sharpness', '?')} |")
            key = (min(nf, 4), d.get("sharpness") or 0)
            if best is None or key > best[0]:
                best = (key, p)
        if best:
            L.append(f"\nSuggestion `group_photo` : **{best[1]}** (le plus de visages, puis la plus nette).")
    # détail par clip
    L += ["", "## Détail par clip", ""]
    for c in clips:
        tr = load_transcript(c, a) or {}
        fa = _read_json(a / f"{c}.faces.json") or {}
        pe = _read_json(a / f"{c}.persons.json") or {}
        au = _read_json(a / f"{c}.audio.json") or {}
        lu = _read_json(a / f"{c}.luma.json") or {}
        L += [f"### {c}", f"![{c}](contact/{c}.jpg)", ""]
        if tr.get("segments"):
            for s in tr["segments"]:
                L.append(f"- `{s['start']:6.2f}–{s['end']:6.2f}` {s['text']}")
        else:
            L.append("- (aucune parole reconnue)")
        if fa.get("orientation_suspect"):
            L.append(f"- **ORIENTATION SUSPECTE** : des visages n'apparaissent qu'après rotation de "
                     f"{fa['orientation_suspect']}° — vérifier la planche et la rotation du rush")
        if fa.get("t"):
            counts = [len(f) for f in fa["faces"]]
            big = max(((max((f["bbox"][2] - f["bbox"][0]) for f in fs), t) for fs, t in zip(fa["faces"], fa["t"]) if fs),
                      default=None)
            L.append(f"- visages : max {max(counts)} simultanés, présents sur {sum(1 for k in counts if k)}/{len(counts)} "
                     f"échantillons" + (f", plus grand {big[0]} px à {big[1]:.1f} s" if big else ""))
        if pe.get("t"):
            pc = [len(p) for p in pe["persons"]]
            L.append(f"- personnes : max {max(pc)} simultanées, présentes sur {sum(1 for k in pc if k)}/{len(pc)} échantillons")
        if au.get("peaks"):
            top = sorted(au["peaks"], key=lambda p: -p["db"])[:5]
            L.append("- pics audio : " + ", ".join(f"{p['t']:.2f} s ({p['db']} dB)" for p in sorted(top, key=lambda p: p["t"])))
        if lu.get("flashes"):
            L.append("- éclairs : " + ", ".join(f"{f['t']:.2f} s (+{f['delta']:.2f})" for f in lu["flashes"][:12]))
        L.append("")
    if timings:
        L += ["## Temps de calcul", "", ", ".join(f"{k} {v:.0f} s" for k, v in timings.items()), ""]
    out = a / "report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def run_analysis(force=False, clips=None, jobs=2, skip=()) -> Path:
    """Analyse complète (transcriptions, visages/personnes, audio, luminance, planches) + analysis/report.md."""
    config.ensure_dirs()
    a = config.ANALYSIS_DIR
    clips = [c for c in (clips or config.CLIPS) if c in config.CLIPS and config.proxy_path(c).exists()]
    photos = [p for p in config.PHOTOS if config.proxy_path(p).exists()]
    timings = {}
    t = time.time()
    if "asr" not in skip:
        for c in clips:
            try:
                transcribe(c, force=force)
            except Exception as e:
                log(f"[analyze] {c}: transcription impossible ({e})")
        timings["transcriptions"] = time.time() - t
    t = time.time()
    if "vision" not in skip:
        todo = [c for c in clips if force or not (_vision_fresh(c, "faces") and _vision_fresh(c, "persons"))]
        if todo:
            with ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
                futs = {ex.submit(_vision_job, c, STEP_S, str(a)): c for c in todo}
                for fu in as_completed(futs):
                    try:
                        c, n, dt = fu.result()
                        log(f"[analyze] {c}: visages + personnes sur {n} images ({dt:.0f} s)")
                    except Exception as e:
                        log(f"[analyze] {futs[fu]}: détection impossible ({e})")
        for p in photos:
            try:
                photo_faces(p, force=force)
            except Exception as e:
                log(f"[analyze] {p}: visages photo impossibles ({e})")
        timings["visages/personnes"] = time.time() - t
    t = time.time()
    for c in clips:
        try:
            if "audio" not in skip:
                audio_peaks(c, force=force)
            if "luma" not in skip:
                flash_frames(c, force=force)
        except Exception as e:
            log(f"[analyze] {c}: audio/luminance impossibles ({e})")
    timings["audio/luminance"] = time.time() - t
    t = time.time()
    if "contact" not in skip:
        for c in clips:
            try:
                contact_sheet(c)
            except Exception as e:
                log(f"[analyze] {c}: planche contact impossible ({e})")
        photo_sheet(photos)
    timings["planches"] = time.time() - t
    instr = {c: detect_instructions(c) for c in clips}
    _write_json(a / "instructions.json", {c: h for c, h in instr.items() if h})
    rep = write_report(clips, photos, a, timings)
    log(f"[analyze] rapport : {rep}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyse des proxies (transcription, visages, personnes, audio, planches)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--skip", default="", help="étapes à sauter : asr,vision,audio,luma,contact")
    ap.add_argument("--report-only", action="store_true", help="réécrire seulement report.md")
    a = ap.parse_args(argv)
    if a.report_only:
        clips = [c for c in (a.only or config.CLIPS) if config.proxy_path(c).exists()]
        log(str(write_report(clips, [p for p in config.PHOTOS if config.proxy_path(p).exists()])))
        return 0
    run_analysis(force=a.force, clips=a.only, jobs=a.jobs, skip=tuple(s for s in a.skip.split(",") if s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
