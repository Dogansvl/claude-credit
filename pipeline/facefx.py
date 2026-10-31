"""VFX visages (MediaPipe FaceLandmarker, 478 points) pour « C'EST QUI LUI ? ».

API (voir docs/ARCHITECTURE.md) :
  FaceTrack.compute(video_path, t0, t1, fps=30) -> FaceTrack   (mode VIDEO + découverte des petits visages)
  FaceTrack.from_image(img_uint8) -> FaceTrack                  (mode IMAGE, tuiles pour les petits visages)
  FaceTrack.at(i) -> list[Face]   (i = index d'image relatif à t0 ; visages triés de gauche à droite)
  Face : .lm (478, 2) px float32, .bbox (x0, y0, x1, y1), .score
  Effets (img float32 RGB [0,1] -> img) :
    black_eyes, stretch_smile, melt, widen_eyes, demon_face, head_tic, portrait_eyes

Principes : chaque effet travaille sur une ROI locale autour du visage, les déformations sont des champs
de déplacement lisses (RBF gaussiennes) appliqués par `cv2.remap` (cartographie inverse), les bords sont
adoucis et le grain d'origine est conservé. Sans visage, un effet renvoie l'image inchangée.
Tout l'aléatoire est déterministe (graines fixes ou dérivées de frame_idx).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

try:
    import config
except ModuleNotFoundError:  # exécution directe
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config  # noqa: E402

VERSION = "facefx-5"
FFMPEG = os.environ.get("ESCAPE_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("ESCAPE_FFPROBE", "ffprobe")

# ---------------------------------------------------------------------------
# Indices du maillage MediaPipe (478 points, iris raffinés 468-477)
# « LEFT » = l'œil à gauche de l'image (côté des points 33/133, iris 468).
# ---------------------------------------------------------------------------
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LEFT_CORNERS = (33, 133)    # (externe, interne)
RIGHT_CORNERS = (263, 362)
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_UPPER_LID = [246, 161, 160, 159, 158, 157, 173]
LEFT_LOWER_LID = [7, 163, 144, 145, 153, 154, 155]
RIGHT_UPPER_LID = [466, 388, 387, 386, 385, 384, 398]
RIGHT_LOWER_LID = [249, 390, 373, 374, 380, 381, 382]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
INNER_LOWER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
OUTER_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
CHIN_OVAL = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]
CHEEKS = [205, 425, 50, 280, 187, 411, 216, 436, 206, 426, 92, 322, 57, 287, 186, 410, 212, 432]
ANCHORS_UPPER = [1, 2, 4, 5, 6, 98, 327, 168, 197, 195, 33, 133, 263, 362, 159, 145, 386, 374,
                 70, 105, 107, 300, 334, 336, 10, 234, 454, 127, 356, 93, 323, 116, 345, 123, 352]
NOSE_WINGS = (129, 358)
FOREHEAD, CHIN = 10, 152


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------
@dataclass
class Face:
    """Un visage à une image donnée. lm : (478, 2) px float32 dans l'image d'origine."""
    lm: np.ndarray
    score: float = 1.0
    track_id: int = -1
    frame_size: tuple[int, int] | None = None   # (w, h) de l'image où les points ont été mesurés

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        x0, y0 = np.nanmin(self.lm, axis=0)
        x1, y1 = np.nanmax(self.lm, axis=0)
        return float(x0), float(y0), float(x1), float(y1)

    @property
    def center(self) -> np.ndarray:
        x0, y0, x1, y1 = self.bbox
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)

    @property
    def size(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return float(max(x1 - x0, y1 - y0))

    def scaled(self, sx: float, sy: float | None = None) -> "Face":
        sy = sx if sy is None else sy
        fs = None if self.frame_size is None else (round(self.frame_size[0] * sx), round(self.frame_size[1] * sy))
        return Face((self.lm * np.float32([sx, sy])).astype(np.float32), self.score, self.track_id, fs)

    def transformed(self, M: np.ndarray, frame_size: tuple[int, int] | None = None) -> "Face":
        """Applique une transformation affine 2x3 (ex. recadrage/zoom caméra du rendu)."""
        M = np.asarray(M, np.float32).reshape(2, 3)
        lm = np.stack([self.lm[:, 0] * M[0, 0] + self.lm[:, 1] * M[0, 1] + M[0, 2],
                       self.lm[:, 0] * M[1, 0] + self.lm[:, 1] * M[1, 1] + M[1, 2]], 1)
        return Face(lm.astype(np.float32), self.score, self.track_id, frame_size)


def _face_from_lm(lm: np.ndarray, score: float, tid: int, wh) -> Face:
    return Face(np.ascontiguousarray(lm, dtype=np.float32), float(score), int(tid), wh)


# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------
_MP_LOCK = threading.Lock()
_IMAGE_LMK: dict[int, object] = {}


def _make_landmarker(video: bool, num_faces: int, conf: float = 0.35):
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.model_path("face_landmarker.task"))),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_faces=num_faces,
        min_face_detection_confidence=conf,
        min_face_presence_confidence=conf,
        min_tracking_confidence=conf,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def _image_landmarker(num_faces: int):
    lmk = _IMAGE_LMK.get(num_faces)
    if lmk is None:
        lmk = _make_landmarker(False, num_faces)
        _IMAGE_LMK[num_faces] = lmk
    return lmk


def close_models() -> None:
    """Libère les modèles MediaPipe (évite un avertissement à la sortie de l'interpréteur)."""
    with _MP_LOCK:
        for lmk in _IMAGE_LMK.values():
            try:
                lmk.close()
            except Exception:
                pass
        _IMAGE_LMK.clear()


import atexit  # noqa: E402
atexit.register(close_models)


def _mp_image(rgb_u8: np.ndarray):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_u8))


def _result_lms(res, w: int, h: int, ox: float = 0.0, oy: float = 0.0) -> list[np.ndarray]:
    out = []
    for lms in res.face_landmarks:
        a = np.array([(p.x, p.y) for p in lms], np.float32)
        a[:, 0] = a[:, 0] * w + ox
        a[:, 1] = a[:, 1] * h + oy
        out.append(a)
    return out


def _detect_image(rgb: np.ndarray, num_faces: int = 6) -> list[np.ndarray]:
    with _MP_LOCK:
        res = _image_landmarker(num_faces).detect(_mp_image(rgb))
    h, w = rgb.shape[:2]
    return _result_lms(res, w, h)


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance_for_detection(rgb: np.ndarray) -> np.ndarray:
    """Image servant UNIQUEMENT à la détection (les effets s'appliquent à l'original) : les rushs sont de
    nuit (habitacle rouge, noir + éclairs). Dominante rouge extrême -> canal max en niveaux de gris ;
    image sombre -> gamma ; puis CLAHE sur la luminance. Images normales : inchangées."""
    small = rgb[::8, ::8].astype(np.float32)
    m = small.mean((0, 1)) / 255.0
    lum = float(m[0] * 0.2126 + m[1] * 0.7152 + m[2] * 0.0722)
    red = m[0] > 2.2 * max(m[1], 1e-3) and m[0] > 2.2 * max(m[2], 1e-3)
    if lum > 0.28 and not red:
        return rgb
    img = rgb
    if red:
        g = rgb.max(axis=2)
        img = np.dstack([g, g, g])
    lv = float(img[::8, ::8].mean()) / 255.0
    if lv < 0.35:
        gamma = float(np.clip(math.log(0.42) / math.log(max(lv, 1e-3)), 0.35, 1.0))
        lut = (np.power(np.arange(256) / 255.0, gamma) * 255 + 0.5).astype(np.uint8)
        img = cv2.LUT(img, lut)
    lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2LAB)
    lab[..., 0] = _CLAHE.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _bbox(lm: np.ndarray) -> np.ndarray:
    return np.array([lm[:, 0].min(), lm[:, 1].min(), lm[:, 0].max(), lm[:, 1].max()], np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _plausible(lm: np.ndarray) -> bool:
    """Rejette les maillages dégénérés (ratio interoculaire/visage hors norme, yeux sous la bouche)."""
    b = _bbox(lm)
    fw, fh = b[2] - b[0], b[3] - b[1]
    if fw < 6 or fh < 6 or not np.isfinite(lm).all():
        return False
    eyes = (lm[LEFT_EYE].mean(0), lm[RIGHT_EYE].mean(0))
    iod = float(np.linalg.norm(eyes[0] - eyes[1]))
    mouth = (lm[61] + lm[291]) / 2
    down = lm[CHIN] - lm[FOREHEAD]
    if np.linalg.norm(down) < 1e-3:
        return False
    eye_mid = (eyes[0] + eyes[1]) / 2
    return 0.15 < iod / max(fw, fh) < 0.75 and float(np.dot(mouth - eye_mid, down)) > 0


def _tiles(w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Tuiles carrées chevauchantes (côté = petit côté, puis moitié), pour les petits visages."""
    tiles = []
    side = min(w, h)
    for s in (side, side // 2):
        if s < 192:
            break
        step = max(1, s // 2)
        xs = list(range(0, max(w - s, 0) + 1, step))
        ys = list(range(0, max(h - s, 0) + 1, step))
        if xs[-1] != max(w - s, 0):
            xs.append(max(w - s, 0))
        if ys[-1] != max(h - s, 0):
            ys.append(max(h - s, 0))
        for y in ys:
            for x in xs:
                t = (x, y, min(s, w), min(s, h))
                if t != (0, 0, w, h):
                    tiles.append(t)
    return tiles


def _refine_roi(rgb: np.ndarray, lm_prev: np.ndarray, expand: float = 2.0) -> np.ndarray | None:
    """Relance le landmarker sur un recadrage carré autour d'un visage connu (petits visages / suivi)."""
    h, w = rgb.shape[:2]
    b = _bbox(lm_prev)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    side = max(b[2] - b[0], b[3] - b[1]) * expand
    side = max(side, 48.0)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad = [max(0, -x0), max(0, -y0), max(0, x1 - w), max(0, y1 - h)]
    crop = rgb[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if crop.size == 0:
        return None
    if any(pad):
        crop = cv2.copyMakeBorder(crop, pad[1], pad[3], pad[0], pad[2], cv2.BORDER_REPLICATE)
    cs = crop.shape[0]
    scale = 1.0
    if cs < 256:  # agrandit les petits visages pour le détecteur
        scale = 256.0 / cs
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
    lms = _detect_image(crop, 2)
    if not lms:
        return None
    best, bd = None, 1e9
    for lm in lms:
        lm = lm / scale + np.float32([x0, y0])
        d = float(np.linalg.norm(_bbox(lm)[:2] + _bbox(lm)[2:] - (b[:2] + b[2:])))
        if d < bd:
            best, bd = lm, d
    if best is None or not _plausible(best) or _iou(_bbox(best), b) < 0.25:
        return None
    return best


def detect_faces_image(rgb: np.ndarray, num_faces: int = 6, small_faces: bool = True) -> list[np.ndarray]:
    """Détection IMAGE : image entière + tuiles (petits visages), fusion NMS, raffinement par ROI."""
    h, w = rgb.shape[:2]
    cands: list[tuple[np.ndarray, int]] = [(lm, 0) for lm in _detect_image(rgb, num_faces)]
    if small_faces:
        for (x, y, tw, th) in _tiles(w, h):
            for lm in _detect_image(rgb[y:y + th, x:x + tw], num_faces):
                b = _bbox(lm)
                # visage coupé par le bord de la tuile (et pas par le bord de l'image) : ignoré
                m = 2.0
                if (b[0] < m and x > 0) or (b[1] < m and y > 0) or (b[2] > tw - m and x + tw < w) or \
                        (b[3] > th - m and y + th < h):
                    continue
                cands.append((lm + np.float32([x, y]), 1))
    keep: list[np.ndarray] = []
    for lm, src in cands:
        if not _plausible(lm):
            continue
        b = _bbox(lm)
        if any(_iou(b, _bbox(k)) > 0.3 for k in keep):
            continue
        if src == 1:  # confirmation par une seconde passe recentrée (élimine les faux positifs des tuiles)
            lm2 = _refine_roi(rgb, lm)
            if lm2 is None:
                continue
            lm = lm2
        keep.append(lm)
    keep.sort(key=lambda a: float(_bbox(a)[0] + _bbox(a)[2]))
    return keep[:num_faces]


# ---------------------------------------------------------------------------
# Lecture vidéo (ffmpeg -> rawvideo RGB)
# ---------------------------------------------------------------------------
def probe_size(video_path) -> tuple[int, int]:
    """(w, h) affichés (rotation de la matrice d'affichage appliquée, comme le fait ffmpeg au décodage)."""
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:stream_side_data=rotation:stream_tags=rotate", "-of", "json",
                        str(video_path)], capture_output=True, text=True, check=True)
    st = json.loads(r.stdout)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if not rot and "tags" in st and "rotate" in st["tags"]:
        rot = int(float(st["tags"]["rotate"]))
    if abs(rot) % 180 == 90:
        w, h = h, w
    return w, h


def read_frames(video_path, t0: float, t1: float, fps: float = 30.0) -> Iterator[np.ndarray]:
    """Images uint8 RGB de t0 à t1 (exclu) à `fps` : image i = instant t0 + i/fps (seek précis)."""
    w, h = probe_size(video_path)
    n = max(1, int(round((t1 - t0) * fps)))
    cmd = [FFMPEG, "-v", "error", "-nostdin", "-ss", f"{max(0.0, t0):.6f}", "-i", str(video_path),
           "-vf", f"fps={fps}", "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = w * h * 3
    try:
        for _ in range(n):
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


# ---------------------------------------------------------------------------
# Lissage one-euro (aller-retour pour annuler le retard) et interpolation des trous
# ---------------------------------------------------------------------------
def _one_euro_pass(x: np.ndarray, fps: float, scale: float, min_cutoff: float, beta: float,
                   d_cutoff: float) -> np.ndarray:
    def alpha(cut):
        tau = 1.0 / (2 * math.pi * cut)
        return 1.0 / (1.0 + tau * fps)

    out = np.empty_like(x)
    out[0] = x[0]
    dx_hat = 0.0
    a_d = alpha(d_cutoff)
    for t in range(1, len(x)):
        # vitesse globale du visage (unités : tailles de visage par seconde) -> coupure commune à tous les points
        dx = float(np.mean(np.abs(x[t] - out[t - 1]))) * fps / max(scale, 1e-3)
        dx_hat = a_d * dx + (1 - a_d) * dx_hat
        a = alpha(min_cutoff + beta * dx_hat)
        out[t] = a * x[t] + (1 - a) * out[t - 1]
    return out


def _smooth_track(lm: np.ndarray, valid: np.ndarray, fps: float, min_cutoff=1.0, beta=1.5, d_cutoff=1.0):
    """lm (T, 478, 2) ; lisse chaque segment contigu valide (one-euro avant + arrière, moyenne)."""
    out = lm.copy()
    T = len(lm)
    t = 0
    while t < T:
        if not valid[t]:
            t += 1
            continue
        s = t
        while t < T and valid[t]:
            t += 1
        seg = lm[s:t]
        if len(seg) >= 3:
            scale = float(np.median([_bbox(f)[2] - _bbox(f)[0] for f in seg]))
            fw = _one_euro_pass(seg, fps, scale, min_cutoff, beta, d_cutoff)
            bw = _one_euro_pass(seg[::-1], fps, scale, min_cutoff, beta, d_cutoff)[::-1]
            out[s:t] = 0.5 * (fw + bw)
    return out


def _fill_gaps(lm: np.ndarray, valid: np.ndarray, max_gap: int = 5) -> tuple[np.ndarray, np.ndarray]:
    lm = lm.copy()
    filled = valid.copy()
    idx = np.flatnonzero(valid)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in range(1, gap + 1):
                u = k / (gap + 1)
                lm[a + k] = (1 - u) * lm[a] + u * lm[b]
                filled[a + k] = True
    return lm, filled


# ---------------------------------------------------------------------------
# FaceTrack
# ---------------------------------------------------------------------------
class FaceTrack:
    """Pistes de visages sur une plage d'images. lm : (T, K, 478, 2) (NaN = absent), score : (T, K)."""

    def __init__(self, lm: np.ndarray, score: np.ndarray, frame_size: tuple[int, int], fps: float = 30.0,
                 t0: float = 0.0, meta: dict | None = None):
        self.lm = lm.astype(np.float32)
        self.score = score.astype(np.float32)
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.fps = float(fps)
        self.t0 = float(t0)
        self.meta = meta or {}

    def __len__(self) -> int:
        return int(self.lm.shape[0])

    @property
    def n_tracks(self) -> int:
        return int(self.lm.shape[1])

    def at(self, i: int, scale: float | None = None) -> list[Face]:
        """Visages à l'image i (relative à t0, bornée à la plage), triés de gauche à droite."""
        if len(self) == 0 or self.n_tracks == 0:
            return []
        i = int(min(max(int(i), 0), len(self) - 1))
        faces = []
        for k in range(self.n_tracks):
            lm = self.lm[i, k]
            if np.isnan(lm[0, 0]):
                continue
            f = _face_from_lm(lm, self.score[i, k], k, self.frame_size)
            faces.append(f.scaled(scale) if scale else f)
        faces.sort(key=lambda f: f.center[0])
        return faces

    def at_time(self, t: float) -> list[Face]:
        return self.at(int(round((t - self.t0) * self.fps)))

    def largest(self, i: int) -> Face | None:
        fs = self.at(i)
        return max(fs, key=lambda f: f.size) if fs else None

    # ------------------------------------------------------------------
    @classmethod
    def from_image(cls, img, num_faces: int = 6, small_faces: bool = True, use_cache: bool = True) -> "FaceTrack":
        """Visages d'une image fixe (uint8 RGB ; float [0,1] accepté). Une seule image : .at(0).
        Cache npz (CACHE_DIR/faces/img_<hash>.npz) : la détection par tuiles coûte ~1–2 s."""
        rgb = _to_u8(img)
        h, w = rgb.shape[:2]
        key = hashlib.sha1(np.ascontiguousarray(rgb[::7, ::5]).tobytes() + f"{w}x{h}{num_faces}{small_faces}{VERSION}"
                           .encode()).hexdigest()[:20]
        cache = config.CACHE_DIR / "faces" / f"img_{key}.npz"
        if use_cache and cache.exists():
            try:
                z = np.load(cache, allow_pickle=False)
                return cls(z["lm"], z["score"], (w, h), 30.0, 0.0, {"source": "image", "cache": str(cache)})
            except Exception:
                pass
        lms = detect_faces_image(rgb, num_faces, small_faces)
        enh = enhance_for_detection(rgb)
        if enh is not rgb and len(lms) < num_faces:   # image sombre / dominante rouge : seconde chance
            lms2 = detect_faces_image(enh, num_faces, small_faces)
            if len(lms2) > len(lms):
                lms = lms2
        K = len(lms)
        lm = np.full((1, K, 478, 2), np.nan, np.float32)
        for k, a in enumerate(lms):
            lm[0, k, :a.shape[0]] = a[:478]
        score = np.full((1, K), 0.9, np.float32)
        if use_cache:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, lm=lm, score=score)
                os.replace(tmp, cache)
            except OSError:
                pass
        return cls(lm, score, (w, h), 30.0, 0.0, {"source": "image"})

    @classmethod
    def compute(cls, video_path, t0: float, t1: float, fps: float = 30, max_faces: int = 6,
                small_faces: bool = True, use_cache: bool = True, key_every: int = 5) -> "FaceTrack":
        """Suivi des visages de t0 à t1 : MediaPipe en mode VIDEO sur l'image entière, + découverte des
        petits visages par tuiles toutes les `key_every` images et suivi par ROI ; association des
        détections en pistes, interpolation des trous ≤ 5 images, lissage one-euro ; cache npz."""
        video_path = Path(video_path)
        st = video_path.stat()
        key = json.dumps([str(video_path.resolve()), st.st_size, st.st_mtime_ns, round(t0, 4), round(t1, 4),
                          fps, max_faces, small_faces, key_every, VERSION])
        h = hashlib.sha1(key.encode()).hexdigest()[:20]
        cache = config.CACHE_DIR / "faces" / f"{h}.npz"
        if use_cache and cache.exists():
            try:
                z = np.load(cache, allow_pickle=False)
                meta = json.loads(str(z["meta"]))
                return cls(z["lm"], z["score"], tuple(meta["frame_size"]), meta["fps"], meta["t0"], meta)
            except Exception:
                pass
        w, hh = probe_size(video_path)
        obs = _track_video(video_path, t0, t1, fps, max_faces, small_faces, key_every, (w, hh))
        lm, score = obs
        meta = {"frame_size": [w, hh], "fps": fps, "t0": t0, "t1": t1, "video": str(video_path),
                "version": VERSION, "n_frames": int(lm.shape[0]), "n_tracks": int(lm.shape[1])}
        tr = cls(lm, score, (w, hh), fps, t0, meta)
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, lm=tr.lm, score=tr.score, meta=json.dumps(meta))
            os.replace(tmp, cache)
        return tr


def _track_video(video_path, t0, t1, fps, max_faces, small_faces, key_every, wh):
    w, h = wh
    vid = _make_landmarker(True, max_faces)
    tracks: list[dict] = []   # {"obs": {i: lm}, "last": i, "lm": lm}
    ring: list[np.ndarray] = []
    n = 0
    try:
        use_enh = None
        for i, raw in enumerate(read_frames(video_path, t0, t1, fps)):
            n = i + 1
            # représentation pour la détection (originale ou rehaussée), choisie sur la 1re image puis
            # réévaluée aux images clés tant qu'aucun visage n'est suivi
            enh = enhance_for_detection(raw)
            if enh is not raw and (use_enh is None or (i % key_every == 0 and not any(
                    i - tr["last"] <= key_every for tr in tracks))):
                n_o = len(detect_faces_image(raw, max_faces, False))
                n_e = len(detect_faces_image(enh, max_faces, False))
                use_enh = n_e > n_o if (n_e != n_o) else bool(use_enh)
            rgb = enh if use_enh else raw
            ring.append(rgb)
            if len(ring) > key_every:
                ring.pop(0)
            ts = int(round(i * 1000.0 / fps)) + 1
            res = vid.detect_for_video(_mp_image(rgb), ts)
            dets = [lm for lm in _result_lms(res, w, h) if _plausible(lm)]
            matched = _associate(tracks, dets, i)
            # pistes actives non retrouvées : suivi par ROI (visages petits ou partiellement masqués)
            for k, tr in enumerate(tracks):
                if k in matched or i - tr["last"] > key_every:
                    continue
                lm2 = _refine_roi(rgb, tr["lm"])
                if lm2 is not None:
                    _add_obs(tr, i, lm2, 0.85)
                    matched.add(k)
            if small_faces and (i % key_every == 0):
                known = [tr["lm"] for tr in tracks if tr["last"] == i]
                for lm in detect_faces_image(rgb, max_faces, True):
                    if any(_iou(_bbox(lm), _bbox(k2)) > 0.2 for k2 in known):
                        continue
                    tr = {"obs": {}, "last": i, "lm": lm, "q": {}}
                    _add_obs(tr, i, lm, 0.85)
                    # rattrapage arrière jusqu'à la découverte précédente
                    prev = lm
                    for j in range(1, len(ring)):
                        fr = ring[-1 - j]
                        lmb = _refine_roi(fr, prev)
                        if lmb is None:
                            break
                        tr["obs"][i - j] = lmb
                        tr["q"][i - j] = 0.85
                        prev = lmb
                    tracks.append(tr)
                    known.append(lm)
    finally:
        try:
            vid.close()
        except Exception:
            pass
    return _finalize_tracks(tracks, n, fps, max_faces)


def _add_obs(tr: dict, i: int, lm: np.ndarray, q: float) -> None:
    tr["obs"][i] = lm
    tr.setdefault("q", {})[i] = q
    tr["last"] = i
    tr["lm"] = lm


def _associate(tracks: list[dict], dets: list[np.ndarray], i: int) -> set:
    """Association glouton par IoU / distance des centres ; nouvelles pistes pour le reste."""
    matched: set[int] = set()
    pairs = []
    for d, lm in enumerate(dets):
        b = _bbox(lm)
        for k, tr in enumerate(tracks):
            if i - tr["last"] > 8:
                continue
            bp = _bbox(tr["lm"])
            iou = _iou(b, bp)
            dist = np.linalg.norm((b[:2] + b[2:]) / 2 - (bp[:2] + bp[2:]) / 2) / max(bp[2] - bp[0], 1)
            if iou > 0.15 or dist < 0.5:
                pairs.append((iou - 0.2 * dist, d, k))
    pairs.sort(reverse=True)
    used_d: set[int] = set()
    for _, d, k in pairs:
        if d in used_d or k in matched:
            continue
        _add_obs(tracks[k], i, dets[d], 0.95)
        used_d.add(d)
        matched.add(k)
    for d, lm in enumerate(dets):
        if d not in used_d:
            tr = {"obs": {}, "last": i, "lm": lm, "q": {}}
            _add_obs(tr, i, lm, 0.95)
            tracks.append(tr)
            matched.add(len(tracks) - 1)
    return matched


def _finalize_tracks(tracks: list[dict], n: int, fps: float, max_faces: int):
    min_len = max(1, min(4, n // 4))
    good = [tr for tr in tracks if len(tr["obs"]) >= min_len]
    good.sort(key=lambda tr: -len(tr["obs"]))
    good = good[:max_faces]
    K = len(good)
    lm = np.full((n, K, 478, 2), np.nan, np.float32)
    score = np.zeros((n, K), np.float32)
    for k, tr in enumerate(good):
        arr = np.zeros((n, 478, 2), np.float32)
        valid = np.zeros(n, bool)
        for i, a in tr["obs"].items():
            if 0 <= i < n:
                arr[i] = a[:478]
                valid[i] = True
                score[i, k] = tr["q"].get(i, 0.9)
        arr, filled = _fill_gaps(arr, valid, 5)
        score[filled & ~valid, k] = 0.5
        arr = _smooth_track(arr, filled, fps)
        arr[~filled] = np.nan
        lm[:, k] = arr
    # ordre stable : gauche -> droite à la première apparition
    if K > 1:
        first_x = []
        for k in range(K):
            ok = np.flatnonzero(~np.isnan(lm[:, k, 0, 0]))
            first_x.append(float(np.nanmean(lm[ok[0], k, :, 0])) if len(ok) else 1e9)
        order = np.argsort(first_x)
        lm, score = lm[:, order], score[:, order]
    return lm, score


# ---------------------------------------------------------------------------
# Outils image
# ---------------------------------------------------------------------------
def _to_u8(img) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a[..., :3]
    return (np.clip(a[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)


def _as_float(img) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    return a.astype(np.float32, copy=False)


def _fit_face(face: Face | None, img: np.ndarray) -> Face | None:
    """Remet les points à l'échelle si l'image a été redimensionnée (même cadrage, ex. aperçu 540x960)."""
    if face is None:
        return None
    if face.frame_size is None:
        return face
    w, h = face.frame_size
    H, W = img.shape[:2]
    if (w, h) == (W, H) or w <= 0 or h <= 0:
        return face
    if abs(W / w - H / h) < 0.02 * max(W / w, H / h):
        return face.scaled(W / w, H / h)
    return face


def _faces(face) -> list[Face]:
    if face is None:
        return []
    if isinstance(face, Face):
        return [face]
    return [f for f in face if f is not None]


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _proj(xs: np.ndarray, ys: np.ndarray, o, v) -> np.ndarray:
    """Projection (h, w) de la grille (xs en ligne, ys en colonne) sur l'axe v depuis l'origine o."""
    return (xs[None, :] - o[0]) * v[0] + (ys[:, None] - o[1]) * v[1]


def _axes(roi):
    x0, y0, x1, y1 = roi
    return np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32)


def _lum(img: np.ndarray) -> np.ndarray:
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _clip_roi(x0, y0, x1, y1, W, H):
    x0, y0 = int(max(0, math.floor(x0))), int(max(0, math.floor(y0)))
    x1, y1 = int(min(W, math.ceil(x1))), int(min(H, math.ceil(y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def _soft_poly(poly: np.ndarray, roi, dilate: float, feather: float, ss: int = 1) -> np.ndarray:
    """Masque doux d'un polygone dans la ROI (x0,y0,x1,y1) : dilatation + fondu (px), sur-échantillonné."""
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    W, H = w * ss, h * ss
    pts = np.round((np.asarray(poly, np.float32) - [x0, y0]) * ss * 16).astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [pts], 255, cv2.LINE_AA, shift=4)
    inside = (m > 127).astype(np.uint8)
    if inside.any() and (dilate != 0 or feather > 0):
        d_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
        d_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
        sd = (d_out - d_in) / ss  # distance signée en px de la ROI
        f = max(feather, 0.35)
        a = _smoothstep(dilate + f / 2, dilate - f / 2, sd).astype(np.float32)
    else:
        a = m.astype(np.float32) / 255.0
    if ss > 1:
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    return a


def _noise_sigma(img: np.ndarray) -> float:
    """Écart-type du bruit (haute fréquence de la luminance), mesuré sur une fenêtre centrale ≤ 192 px."""
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    sub = img[max(0, cy - 96): cy + 96, max(0, cx - 96): cx + 96]
    l = _lum(sub)
    hf = l - cv2.blur(l, (3, 3))
    return float(1.2533 * np.mean(np.abs(hf)) * 1.22)


_NOISE_TILE: dict[int, np.ndarray] = {}


def _noise_tile(seed: int) -> np.ndarray:
    """Tuile de bruit 256x256 (grain légèrement corrélé), mise en cache."""
    t = _NOISE_TILE.get(seed % 16)
    if t is None:
        rng = np.random.default_rng(seed % 16)
        t = rng.standard_normal((256, 256), dtype=np.float32)
        t = cv2.GaussianBlur(t, (0, 0), 0.6)
        t /= max(float(t.std()), 1e-6)
        _NOISE_TILE[seed % 16] = t
    return t


def _restore_grain(before: np.ndarray, after: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
    """Le rééchantillonnage lisse le grain : on rajoute un bruit équivalent à la perte, dans le masque."""
    s0, s1 = _noise_sigma(before), _noise_sigma(after)
    miss = math.sqrt(max(s0 * s0 - s1 * s1, 0.0))
    if miss < 2e-3:
        return after
    h, w = after.shape[:2]
    t = _noise_tile(seed)
    ox, oy = (seed * 37) % 256, (seed * 91) % 256
    n = np.tile(np.roll(t, (oy, ox), (0, 1)), (h // 256 + 1, w // 256 + 1))[:h, :w]
    return after + (n * (miss * mask))[..., None]


# ---------------------------------------------------------------------------
# Champs de déplacement (RBF gaussiennes) et remap
# ---------------------------------------------------------------------------
# NB : pas de BLAS/LAPACK dans les chemins chauds (numpy.linalg / @) : sous charge, les threads
# OpenBLAS en attente active rendent une résolution 110x110 1000x plus lente. cv2.solve + einsum.
def _rbf_weights(ctrl: np.ndarray, disp: np.ndarray, sigma: float, lam: float = 0.03) -> np.ndarray:
    c = ctrl.astype(np.float64)
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * sigma * sigma)) + lam * np.eye(len(c))
    ok, w = cv2.solve(K, disp.astype(np.float64), flags=cv2.DECOMP_CHOLESKY)
    if not ok:
        ok, w = cv2.solve(K, disp.astype(np.float64), flags=cv2.DECOMP_SVD)
    return w.astype(np.float32)


def _rbf_eval(pts: np.ndarray, ctrl: np.ndarray, wts: np.ndarray, sigma: float) -> np.ndarray:
    dx = pts[:, 0:1].astype(np.float32) - ctrl[None, :, 0].astype(np.float32)
    dy = pts[:, 1:2].astype(np.float32) - ctrl[None, :, 1].astype(np.float32)
    k = np.exp((dx * dx + dy * dy) * np.float32(-0.5 / (sigma * sigma)))
    return np.einsum("mn,nk->mk", k, wts.astype(np.float32))


def _grid(roi, step):
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    gw, gh = max(2, int(math.ceil(w / step))), max(2, int(math.ceil(h / step)))
    xs = (np.arange(gw) + 0.5) * (w / gw) - 0.5 + x0
    ys = (np.arange(gh) + 0.5) * (h / gh) - 0.5 + y0
    gx, gy = np.meshgrid(xs, ys)
    return gx.astype(np.float32), gy.astype(np.float32)


def _upsample(f: np.ndarray, roi) -> np.ndarray:
    x0, y0, x1, y1 = roi
    return cv2.resize(f.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)


def _edge_taper(roi, margin: float = 0.12) -> np.ndarray:
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    mx, my = max(2.0, w * margin), max(2.0, h * margin)
    tx = _smoothstep(0, mx, np.arange(w, dtype=np.float32)) * _smoothstep(0, mx, (w - 1 - np.arange(w)).astype(np.float32))
    ty = _smoothstep(0, my, np.arange(h, dtype=np.float32)) * _smoothstep(0, my, (h - 1 - np.arange(h)).astype(np.float32))
    return (ty[:, None] * tx[None, :]).astype(np.float32)


def _limit_fold(fx: np.ndarray, fy: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Réduit le champ si sa jacobienne risque de se replier (det(I - J) < 0,25)."""
    for _ in range(6):
        gxx = np.gradient(fx, axis=1) / step
        gxy = np.gradient(fx, axis=0) / step
        gyx = np.gradient(fy, axis=1) / step
        gyy = np.gradient(fy, axis=0) / step
        det = (1 - gxx) * (1 - gyy) - gxy * gyx
        if det.min() > 0.25:
            break
        fx, fy = fx * 0.8, fy * 0.8
    return fx, fy


def _remap_roi(img: np.ndarray, roi, dx: np.ndarray, dy: np.ndarray, interp=cv2.INTER_CUBIC) -> np.ndarray:
    """Sortie ROI : out(q) = img(q - d(q)) (d = déplacement « vers l'avant » défini en destination)."""
    x0, y0, x1, y1 = roi
    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    mx = xs[None, :] - dx
    my = ys[:, None] - dy
    return cv2.remap(img, mx.astype(np.float32), my.astype(np.float32), interp, borderMode=cv2.BORDER_REFLECT101)


# ---------------------------------------------------------------------------
# Géométrie du visage
# ---------------------------------------------------------------------------
class _Geo:
    def __init__(self, lm: np.ndarray):
        self.lm = lm
        self.has_iris = lm.shape[0] >= 478 and np.isfinite(lm[468:478]).all()
        self.eye_c = [lm[LEFT_EYE].mean(0), lm[RIGHT_EYE].mean(0)]
        self.iod = float(np.linalg.norm(self.eye_c[1] - self.eye_c[0]))
        self.eye_mid = (self.eye_c[0] + self.eye_c[1]) / 2
        down = lm[CHIN] - lm[FOREHEAD]
        self.fh = float(np.linalg.norm(down))
        self.down = down / max(self.fh, 1e-6)
        self.right = np.array([-self.down[1], self.down[0]], np.float32)   # perpendiculaire
        if np.dot(self.right, self.eye_c[1] - self.eye_c[0]) < 0:
            self.right = -self.right
        self.fw = float(np.linalg.norm(lm[454] - lm[234]))
        self.mouth_c = (lm[61] + lm[291]) / 2
        self.mw = float(np.linalg.norm(lm[291] - lm[61]))
        b = _bbox(lm)
        self.bbox = b
        self.size = float(max(b[2] - b[0], b[3] - b[1]))

    def eye(self, side: int):
        lm = self.lm
        idx = LEFT_EYE if side == 0 else RIGHT_EYE
        co = LEFT_CORNERS if side == 0 else RIGHT_CORNERS
        outer, inner = lm[co[0]], lm[co[1]]
        ew = float(np.linalg.norm(outer - inner))
        ax = (outer - inner) / max(ew, 1e-6)           # vers la tempe
        ay = np.array([-ax[1], ax[0]], np.float32)
        if np.dot(ay, self.down) < 0:
            ay = -ay                                     # vers le bas du visage
        c = (outer + inner) / 2
        if self.has_iris:
            ir = lm[LEFT_IRIS] if side == 0 else lm[RIGHT_IRIS]
            # l'iris 468 appartient à l'œil 33/133 dans MediaPipe ; on vérifie quand même
            other = lm[RIGHT_IRIS] if side == 0 else lm[LEFT_IRIS]
            if np.linalg.norm(other[0] - c) < np.linalg.norm(ir[0] - c):
                ir = other
            iris_c = ir[0]
            iris_r = float(np.mean(np.linalg.norm(ir[1:] - ir[0], axis=1)))
        else:
            iris_c, iris_r = lm[idx].mean(0), 0.2 * ew
        iris_r = float(np.clip(iris_r, 0.12 * ew, 0.32 * ew))
        return dict(poly=lm[idx], c=c, ew=ew, ax=ax, ay=ay, iris=iris_c, iris_r=iris_r,
                    outer=outer, inner=inner)


# ---------------------------------------------------------------------------
# Veines (géométrie déterministe dans le repère de l'œil)
# ---------------------------------------------------------------------------
_VEIN_CACHE: dict[tuple, list] = {}


def _veins_local(seed: int, n: int, reach: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Polylignes (repère œil : x vers la tempe, y vers le bas, unité = largeur de l'œil) + épaisseurs."""
    key = (seed, n, round(reach, 3))
    if key in _VEIN_CACHE:
        return _VEIN_CACHE[key]
    rng = np.random.default_rng(seed)
    out = []

    def walk(p, ang, length, th, depth):
        nseg = 9
        pts, ths = [p.copy()], [th]
        for s in range(nseg):
            ang += rng.normal(0, 0.22)
            step = length / nseg
            p = p + step * np.array([math.cos(ang), math.sin(ang)])
            pts.append(p.copy())
            ths.append(th * (1 - 0.75 * (s + 1) / nseg))
            if depth < 2 and s in (2, 4, 6) and rng.random() < 0.45:
                walk(p.copy(), ang + rng.choice([-1, 1]) * rng.uniform(0.45, 0.95),
                     length * rng.uniform(0.3, 0.55), ths[-1] * 0.8, depth + 1)
        out.append((np.array(pts, np.float32), np.array(ths, np.float32)))

    for k in range(n):
        # départs sur une ellipse juste hors de l'ouverture, surtout côté tempe et sous l'œil (pas côté nez)
        th = rng.uniform(-0.55, 1.0) * math.pi * (0.85 if rng.random() < 0.5 else -0.85)
        if math.cos(th) < -0.6:
            th = rng.uniform(-0.4, 0.9) * math.pi * 0.5
        p0 = np.array([0.56 * math.cos(th), 0.30 * math.sin(th)], np.float32)
        ang = th + rng.normal(0, 0.35)
        walk(p0, ang, reach * rng.uniform(0.35, 1.0), rng.uniform(0.008, 0.017), 0)
    _VEIN_CACHE[key] = out
    return out


def _vein_mask(eye: dict, roi, seed: int, n: int, reach: float, ss: int) -> np.ndarray:
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    m = np.zeros((h * ss, w * ss), np.float32)
    ew = eye["ew"]
    for pts, ths in _veins_local(seed, n, reach):
        P = (pts[:, :1] * ew) * eye["ax"] + (pts[:, 1:] * ew) * eye["ay"] + eye["c"]
        P = np.round((P - [x0, y0]) * ss * 16).astype(np.int32)
        n_p = len(P)
        for c0, c1 in ((0, n_p // 3 + 1), (n_p // 3, 2 * n_p // 3 + 1), (2 * n_p // 3, n_p)):  # effilée en 3 tronçons
            if c1 - c0 < 2:
                continue
            t = max(1, int(round(float(ths[c0]) * ew * ss)))
            fade = 1.0 - 0.6 * c0 / n_p
            cv2.polylines(m, [P[c0:c1]], False, float(fade), t, cv2.LINE_AA, shift=4)
    m = cv2.GaussianBlur(m, (0, 0), 0.55 * ss)
    if ss > 1:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    m = np.clip(m, 0, 1)
    # halo diffus : la veine est SOUS la peau
    halo = cv2.GaussianBlur(m, (0, 0), max(1.2, 0.035 * ew))
    return np.clip(0.75 * m + 1.2 * halo, 0, 1)


# ---------------------------------------------------------------------------
# black_eyes
# ---------------------------------------------------------------------------
def _black_eyes_one(out: np.ndarray, face: Face, amount: float, veins: float = 1.0, socket: float = 0.18,
                    gloss: float = 1.0) -> None:
    a = float(np.clip(amount, 0.0, 1.5))
    if a <= 0.0:
        return
    g = _Geo(face.lm)
    H, W = out.shape[:2]
    for side in (0, 1):
        e = g.eye(side)
        ew = e["ew"]
        if ew < 2.0:
            continue
        r = 1.7 * ew
        roi = _clip_roi(e["c"][0] - r, e["c"][1] - r, e["c"][0] + r, e["c"][1] + r, W, H)
        if roi is None:
            continue
        x0, y0, x1, y1 = roi
        src = out[y0:y1, x0:x1].copy()
        ss = int(min(6, max(1, math.ceil(48.0 / ew))))
        xs, ys = _axes(roi)
        u = _proj(xs, ys, e["c"], e["ax"]) / ew      # le long de l'œil (vers la tempe)
        v = _proj(xs, ys, e["c"], e["ay"]) / ew      # vers le bas
        # ouverture de l'œil, légèrement dilatée (couvre la ligne des cils), bord adouci
        small = float(_smoothstep(22.0, 8.0, ew))          # petits yeux (plans larges) : plus lisible
        dil = max(0.45, 0.045 * ew) + 0.6 * small
        fea = max(0.6, 0.07 * ew)
        socket = socket + 0.2 * small
        m_eye = _soft_poly(e["poly"], roi, dil, fea, ss)
        # remplissage radial depuis la pupille : l'encre se répand avec amount
        ri = np.sqrt((xs[None, :] - e["iris"][0]) ** 2 + (ys[:, None] - e["iris"][1]) ** 2) / (0.55 * ew)
        R = a * 1.45 - 0.08
        fill = _smoothstep(R + 0.18, R - 0.18, ri) if a < 1.0 else np.ones_like(ri)
        m = m_eye * fill
        # orbites assombries (ecchymose violacée), plus marquées dessous
        if socket > 0:
            so = np.exp(-((u / 0.78) ** 2 + ((v - 0.10) / 0.48) ** 2) * 1.6) * socket * min(a, 1.0)
            tint = np.float32([0.85, 1.0, 0.78])
            src = src * (1 - so[..., None] * tint)
        # veines sombres autour de l'œil
        if veins > 0 and ew >= 9:
            vm = _vein_mask(e, roi, 1009 + 31 * side + int(face.track_id % 7), 11, 0.75, ss)
            vm = vm * (1 - m_eye) * veins * _smoothstep(0.25, 1.0, a) * 0.8
            vt = np.float32([0.60, 0.70, 0.55])   # violet-noir sous la peau
            src = src * (1 - vm[..., None] * vt)
        # noir profond, texture et grain d'origine conservés
        lum0 = _lum(src)
        hf = lum0 - cv2.GaussianBlur(lum0, (0, 0), max(0.7, 0.03 * ew))
        base = np.float32([0.010, 0.011, 0.014])
        black = base + 0.035 * cv2.GaussianBlur(lum0, (0, 0), max(1.0, 0.15 * ew))[..., None] + 0.45 * hf[..., None]
        # reflet humide : liseré clair le long de la paupière inférieure + reflet spéculaire
        local = float(np.clip(np.percentile(lum0, 75), 0.02, 1.0))
        rim = np.exp(-((v - 0.11) / 0.07) ** 2) * np.exp(-(u / 0.33) ** 2) * 0.05 * local
        black = black + rim[..., None] * np.float32([0.9, 0.95, 1.0])
        spec = _catchlight(src, m_eye, e, xs[None, :], ys[:, None], local) * gloss
        black = black + spec[..., None] * np.float32([0.97, 0.98, 1.0])
        res = src * (1 - m[..., None]) + black * m[..., None]
        out[y0:y1, x0:x1] = res


def _catchlight(src, m_eye, e, xx, yy, local):
    """Conserve le vrai reflet de l'œil s'il existe (lumière de la scène), sinon un reflet synthétique."""
    ew = e["ew"]
    lum = cv2.GaussianBlur(_lum(src), (0, 0), max(0.5, 0.03 * ew))
    inside = m_eye > 0.6
    rad = max(0.55, 0.05 * ew)
    if inside.sum() >= 4:
        vals = lum[inside]
        med, mx = float(np.median(vals)), float(vals.max())
        if mx - med > 0.12 and mx > 0.3:
            k = np.argmax(np.where(inside, lum, -1))
            py, px = np.unravel_index(k, lum.shape)
            cx, cy = float(xx[0, px]), float(yy[py, 0])
            inten = min(0.95, mx * 0.95)
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            return inten * np.exp(-d2 / (2 * rad * rad)) * m_eye
    c = e["iris"] - 0.20 * ew * e["ax"] * np.sign(np.dot(e["ax"], [1, 0]) + 1e-6) - 0.16 * ew * e["ay"]
    d2 = (xx - c[0]) ** 2 + (yy - c[1]) ** 2
    inten = min(0.8, 0.12 + 0.75 * local)
    return inten * (np.exp(-d2 / (2 * rad * rad)) + 0.25 * np.exp(-d2 / (2 * (2.6 * rad) ** 2))) * m_eye


def black_eyes(img, face, amount: float = 1.0):
    """Yeux entièrement noirs (ouverture des paupières), bord adouci, reflet humide, fines veines.
    `face` : Face, liste de Face ou None. amount < 1 : le noir se répand depuis la pupille."""
    img = _as_float(img)
    fs = [_fit_face(f, img) for f in _faces(face)]
    if not fs or amount <= 0:
        return img
    out = img.copy()
    for f in fs:
        _black_eyes_one(out, f, amount)
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# Déformations de la bouche (stretch_smile / demon_face)
# ---------------------------------------------------------------------------
def _mouth_controls(g: _Geo, a: float, open_v: float = 0.0, out_k: float = 0.30, up_k: float = 0.16):
    """Points de contrôle (source) et déplacements pour un sourire étiré (+ ouverture verticale)."""
    lm = g.lm
    m, mw = g.mouth_c, g.mw
    u = (lm[291] - lm[61]) / max(mw, 1e-6)
    vup = -g.down  # vers les yeux
    vup = vup - np.dot(vup, u) * u
    vup = vup / max(np.linalg.norm(vup), 1e-6)
    pts, disp = [], []

    def add(i, d, p=None):
        pts.append(lm[i] if p is None else p)
        disp.append(d)

    lips = sorted(set(LIPS_OUTER + LIPS_INNER))
    for i in lips:
        s = float(np.dot(lm[i] - m, u) / (mw / 2))
        s = float(np.clip(s, -1.2, 1.2))
        hx = np.sign(s) * abs(s) ** 1.4 * out_k * mw * a
        hy = (abs(s) ** 2.2) * up_k * mw * a
        d = u * hx + vup * hy
        if open_v > 0:
            lower = i in INNER_LOWER or i in OUTER_LOWER
            upper = i in INNER_UPPER or i in LIPS_OUTER[11:] + [61, 291]
            w_c = 1 - min(1.0, abs(s)) ** 2
            if lower and i not in (61, 291, 78, 308):
                d = d - vup * open_v * mw * a * (0.35 + 0.65 * w_c)
            elif upper and i not in (61, 291, 78, 308):
                d = d + vup * 0.10 * open_v * mw * a * w_c
        add(i, d)
    for i in CHEEKS:
        s = float(np.clip(np.dot(lm[i] - m, u) / (mw / 2), -1.6, 1.6))
        d = u * np.sign(s) * 0.10 * mw * a + vup * 0.09 * mw * a
        add(i, d)
    if open_v > 0:  # mâchoire qui tombe
        for i in CHIN_OVAL:
            s = float(np.clip(np.dot(lm[i] - m, u) / (mw / 2), -2, 2))
            w_c = max(0.0, 1 - (abs(s) / 2.2) ** 2)
            add(i, -vup * open_v * mw * a * 0.85 * w_c)
    else:
        for i in CHIN_OVAL:
            add(i, np.zeros(2, np.float32))
    for i in ANCHORS_UPPER:
        add(i, np.zeros(2, np.float32))
    # anneau d'ancres fixes autour du visage : rien ne bouge au-delà (décor, cheveux)
    fc = lm[FACE_OVAL].mean(0)
    for i in FACE_OVAL[::2]:
        q = fc + (lm[i] - fc) * 1.35
        add(None, np.zeros(2, np.float32), q)
    return np.array(pts, np.float32), np.array(disp, np.float32)


def _mouth_field(g: _Geo, roi, a: float, open_v: float):
    src, disp = _mouth_controls(g, a, open_v)
    dst = src + disp            # le champ est défini à la destination (cartographie inverse)
    sigma = 0.42 * g.mw
    wts = _rbf_weights(dst, disp, sigma)
    step = max(2.0, min(roi[2] - roi[0], roi[3] - roi[1]) / 60.0)
    gx, gy = _grid(roi, step)
    f = _rbf_eval(np.stack([gx.ravel(), gy.ravel()], 1), dst, wts, sigma)
    fx, fy = f[:, 0].reshape(gx.shape), f[:, 1].reshape(gx.shape)
    fx, fy = _limit_fold(fx, fy, step)
    return fx, fy, (src, disp)


def _mouth_roi(g: _Geo, W, H, extra_down: float = 0.0):
    c = g.mouth_c
    r = 1.45 * g.mw + 0.25 * g.size
    return _clip_roi(c[0] - r, c[1] - 0.9 * r, c[0] + r, c[1] + r * (1.0 + extra_down), W, H)


def _darken_mouth(out: np.ndarray, roi, inner_pts: np.ndarray, g: _Geo, a: float, depth: float,
                  seam: float, teeth: float = 0.6) -> None:
    """Intérieur de la bouche assombri (dents gardées en partie, `teeth`), ou fente sombre si fermée."""
    mw = g.mw
    b = _bbox(np.asarray(inner_pts, np.float32))
    mg = 0.12 * mw + 3
    sub = _clip_roi(max(b[0] - mg, roi[0]), max(b[1] - mg, roi[1]), min(b[2] + mg, roi[2]), min(b[3] + mg, roi[3]),
                    out.shape[1], out.shape[0])
    if sub is None:
        return
    roi = sub
    x0, y0, x1, y1 = roi
    area = abs(cv2.contourArea(inner_pts.astype(np.float32)))
    ss = int(min(4, max(1, math.ceil(40.0 / max(mw, 1)))))
    reg = out[y0:y1, x0:x1]
    if area > 0.004 * mw * mw:
        m = _soft_poly(inner_pts, roi, -0.01 * mw, max(0.6, 0.05 * mw), ss)
        xs, ys = _axes(roi)
        c = inner_pts.mean(0)
        dd = np.sqrt((xs[None, :] - c[0]) ** 2 + (ys[:, None] - c[1]) ** 2) / (0.35 * mw)
        lum = _lum(reg)
        keep = 0.10 + teeth * _smoothstep(0.30, 0.85, lum)          # dents : gardées, gorge : noire
        keep = keep * (1 - 0.45 * _smoothstep(1.0, 0.0, dd))           # plus profond au centre
        dark = reg * keep[..., None] + np.float32([0.010, 0.005, 0.005])
        k = (m * min(1.0, depth * a))[..., None]
        reg[:] = reg * (1 - k) + dark * k
    if seam > 0:
        line = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.float32)
        P = (np.asarray(inner_pts, np.float32) - [x0, y0]) * ss
        cv2.polylines(line, [np.round(P * 16).astype(np.int32)], True, 1.0,
                      max(1, int(round(0.03 * mw * ss))), cv2.LINE_AA, shift=4)
        line = cv2.GaussianBlur(line, (0, 0), max(0.6, 0.015 * mw) * ss)
        if ss > 1:
            line = cv2.resize(line, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        k = np.clip(line * seam * a, 0, 0.85)
        reg[:] = reg * (1 - k[..., None])


def _crease(out: np.ndarray, roi, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, width: float,
            strength: float) -> None:
    """Pli naso-génien : courbe sombre douce (p0 aile du nez -> p1 coin -> p2 sous le coin)."""
    x0, y0, x1, y1 = roi
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    ts = np.linspace(0, 1, 24)[:, None]
    curve = (1 - ts) ** 2 * p0 + 2 * (1 - ts) * ts * p1 + ts ** 2 * p2
    P = np.round((curve - [x0, y0]) * 16).astype(np.int32)
    cv2.polylines(m, [P], False, 1.0, max(1, int(round(width))), cv2.LINE_AA, shift=4)
    m = cv2.GaussianBlur(m, (0, 0), max(0.8, width * 0.9))
    reg = out[y0:y1, x0:x1]
    reg[:] = reg * (1 - (m * strength)[..., None] * np.float32([0.9, 1.0, 1.0]))


def stretch_smile(img, face, amount: float = 1.0):
    """Coins de la bouche tirés vers l'extérieur et le haut (sourire anormalement large, glaçant),
    lèvres et joues suivent (champ RBF lisse limité au bas du visage), intérieur assombri."""
    img = _as_float(img)
    fs = [_fit_face(f, img) for f in _faces(face)]
    a = float(np.clip(amount, 0.0, 1.5))
    if not fs or a <= 0:
        return img
    out = img.copy()
    H, W = img.shape[:2]
    for f in fs:
        g = _Geo(f.lm)
        if g.mw < 4:
            continue
        roi = _mouth_roi(g, W, H)
        if roi is None:
            continue
        fx, fy, (src, disp) = _mouth_field(g, roi, a, open_v=0.06)
        dx, dy = _upsample(fx, roi), _upsample(fy, roi)
        tap = _edge_taper(roi)
        dx, dy = dx * tap, dy * tap
        x0, y0, x1, y1 = roi
        before = out[y0:y1, x0:x1].copy()
        warped = _remap_roi(out, roi, dx, dy)
        mag = np.sqrt(dx * dx + dy * dy)
        out[y0:y1, x0:x1] = _restore_grain(before, warped, np.clip(mag / 2.0, 0, 1), 7 + f.track_id)
        lmw = g.lm.copy()
        idx_map = {i: k for k, i in enumerate(sorted(set(LIPS_OUTER + LIPS_INNER)))}
        for i, k in idx_map.items():
            lmw[i] = src[k] + disp[k]
        inner = lmw[LIPS_INNER]
        _darken_mouth(out, roi, inner, g, a, depth=0.9, seam=0.55)
        # plis du sourire (naso-géniens) creusés
        for wing, corner in ((NOSE_WINGS[0], 61), (NOSE_WINGS[1], 291)):
            p0 = g.lm[wing] + 0.05 * g.mw * (g.lm[wing] - g.mouth_c) / max(np.linalg.norm(g.lm[wing] - g.mouth_c), 1)
            p1 = lmw[corner] + (lmw[corner] - g.mouth_c) * 0.12 - g.down * 0.05 * g.mw
            p2 = lmw[corner] + g.down * 0.35 * g.mw + (lmw[corner] - g.mouth_c) * 0.05
            _crease(out, roi, p0, p1, p2, max(1.0, 0.045 * g.mw), 0.22 * min(a, 1.0))
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# melt
# ---------------------------------------------------------------------------
def _melt_field(g: _Geo, roi, a: float, t: float, step: float):
    """Champ (grille grossière) : glissement vers le bas, nul au-dessus des yeux, maximal au menton puis
    s'éteignant sur le cou ; coulures (couloirs étroits) dont la longueur dérive avec t."""
    gx, gy = _grid(roi, step)
    ar = (gx - g.eye_mid[0]) * g.right[0] + (gy - g.eye_mid[1]) * g.right[1]   # latéral
    ad = (gx - g.eye_mid[0]) * g.down[0] + (gy - g.eye_mid[1]) * g.down[1]     # vers le menton
    chin_d = max(float(np.dot(g.lm[CHIN] - g.eye_mid, g.down)), 1e-3)
    s = ad / chin_d
    hw = 0.5 * g.fw
    xr = ar / max(hw, 1e-3)
    hx = _smoothstep(1.05, 0.62, np.abs(xr))
    # profil : démarre un peu au-dessus des yeux (paupières qui tombent), max au menton, décroît sur le cou
    up = _smoothstep(-0.18, 1.0, s) ** 1.25
    down = np.exp(-np.clip(s - 1.0, 0, None) ** 2 / (2 * 0.55 ** 2))
    prof = np.where(s <= 1.0, up, down)
    # bruit basse fréquence qui dérive dans le temps
    n1 = (0.55 * np.sin(2 * math.pi * (0.9 * xr + 0.13 * t) + 1.3)
          + 0.45 * np.sin(2 * math.pi * (1.7 * xr - 0.19 * t + 0.35 * s) + 0.4))
    drift = 1.0 + 0.08 * math.sin(2 * math.pi * 0.21 * t) + 0.05 * math.sin(2 * math.pi * 0.47 * t + 1)
    base = a * 0.095 * g.fh * drift * prof * hx * (1.0 + 0.30 * n1)
    # coulures : couloirs à positions fixes, longueur qui pousse et respire lentement
    drip = np.zeros_like(base)
    lanes = ((-0.58, 0.16, 0.9, 0.0), (-0.22, 0.13, 0.6, 1.7), (0.10, 0.15, 1.0, 3.1),
             (0.42, 0.12, 0.7, 4.4), (0.68, 0.15, 0.8, 0.9))
    for cx, wdt, ln, ph in lanes:
        grow = 0.75 + 0.25 * math.sin(2 * math.pi * 0.17 * t + ph)
        lane = np.exp(-((xr - cx) / wdt) ** 2)
        reach = _smoothstep(0.30, 1.0, s) * np.exp(-np.clip(s - 1.0 - 0.40 * ln * grow, 0, None) ** 2 / 0.25)
        drip += lane * reach * ln * grow
    drip = a * 0.06 * g.fh * drip * hx
    dd = base + drip
    dl = a * 0.018 * g.fw * prof * hx * np.sin(2 * math.pi * (0.23 * t + 0.8 * s))
    fx = dd * g.down[0] + dl * g.right[0]
    fy = dd * g.down[1] + dl * g.right[1]
    # étirement (dérivée le long du visage) pour l'ombrage, calculé sur la grille
    st = np.gradient(dd, axis=0) * g.down[1] + np.gradient(dd, axis=1) * g.down[0]
    return fx.astype(np.float32), fy.astype(np.float32), (st / step).astype(np.float32)


def melt(img, face, amount: float = 1.0, t: float = 0.0):
    """Visage qui fond : déplacement vers le bas croissant vers le menton, coulures qui dérivent avec t,
    léger flou directionnel et ombrage des zones étirées."""
    img = _as_float(img)
    fs = [_fit_face(f, img) for f in _faces(face)]
    a = float(np.clip(amount, 0.0, 1.5))
    if not fs or a <= 0:
        return img
    out = img.copy()
    H, W = img.shape[:2]
    for f in fs:
        g = _Geo(f.lm)
        if g.fh < 8:
            continue
        c = g.eye_mid
        roi = _clip_roi(c[0] - 0.75 * g.fw, c[1] - 0.45 * g.fh, c[0] + 0.75 * g.fw, c[1] + 1.75 * g.fh, W, H)
        if roi is None:
            continue
        step = max(2.0, min(roi[2] - roi[0], roi[3] - roi[1]) / 90.0)
        fx, fy, st = _melt_field(g, roi, a, t, step)
        fx, fy = _limit_fold(fx, fy, step)
        tap = _edge_taper(roi, 0.06)
        dx, dy = _upsample(fx, roi) * tap, _upsample(fy, roi) * tap
        x0, y0, x1, y1 = roi
        before = out[y0:y1, x0:x1].copy()
        warped = _remap_roi(out, roi, dx, dy)
        mag = np.sqrt(dx * dx + dy * dy)
        # léger flou directionnel le long de la coulure (proportionnel au déplacement)
        L = 0.022 * g.fh * a
        if L >= 1.0:
            acc = np.zeros_like(warped)
            K = 4
            for k in range(K):
                o = (k / (K - 1)) * L
                acc += _remap_roi(out, roi, dx + o * g.down[0], dy + o * g.down[1], cv2.INTER_LINEAR)
            wb = (np.clip(mag / (0.10 * g.fh), 0, 1) * 0.6)[..., None]
            warped = warped * (1 - wb) + (acc / K) * wb
        # zones étirées légèrement assombries (cire qui s'affaisse)
        shade = cv2.resize(np.clip(st, 0, None), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        shade = 1 - np.clip(shade * 0.55, 0, 0.22) * min(a, 1.0) * tap
        warped = warped * shade[..., None]
        warped = _restore_grain(before, warped, np.clip(mag / 2.0, 0, 1), 13 + f.track_id)
        out[y0:y1, x0:x1] = warped
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# widen_eyes
# ---------------------------------------------------------------------------
def widen_eyes(img, face, amount: float = 1.0):
    """Yeux écarquillés : grossissement local (bulge) anisotrope de chaque œil."""
    img = _as_float(img)
    fs = [_fit_face(f, img) for f in _faces(face)]
    a = float(np.clip(amount, 0.0, 1.5))
    if not fs or a <= 0:
        return img
    out = img.copy()
    H, W = img.shape[:2]
    for f in fs:
        g = _Geo(f.lm)
        for side in (0, 1):
            e = g.eye(side)
            ew = e["ew"]
            if ew < 3:
                continue
            c = e["iris"] * 0.6 + e["c"] * 0.4
            Rx, Ry = 0.95 * ew, 0.85 * ew
            roi = _clip_roi(c[0] - 1.3 * Rx, c[1] - 1.3 * Rx, c[0] + 1.3 * Rx, c[1] + 1.3 * Rx, W, H)
            if roi is None:
                continue
            x0, y0, x1, y1 = roi
            xs, ys = _axes(roi)
            u = _proj(xs, ys, c, e["ax"])
            v = _proj(xs, ys, c, e["ay"])
            rho2 = (u / Rx) ** 2 + (v / Ry) ** 2
            fall = np.clip(1 - rho2, 0, 1) ** 2
            kx, ky = 0.26 * a, 0.40 * a
            su = u * (1 - kx * fall)
            sv = v * (1 - ky * fall)
            mx = c[0] + su * e["ax"][0] + sv * e["ay"][0]
            my = c[1] + su * e["ax"][1] + sv * e["ay"][1]
            before = out[y0:y1, x0:x1].copy()
            warped = cv2.remap(out, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT101)
            out[y0:y1, x0:x1] = _restore_grain(before, warped, fall, 17 + side)
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# demon_face
# ---------------------------------------------------------------------------
def demon_face(img, face, amount: float = 1.0, t: float = 0.0):
    """Visage possédé (screamer) : bouche étirée et ouverte verticalement, léger melt, orbites creusées,
    yeux noirs, veines, peau désaturée verdâtre."""
    img = _as_float(img)
    fs = [_fit_face(f, img) for f in _faces(face)]
    a = float(np.clip(amount, 0.0, 1.5))
    if not fs or a <= 0:
        return img
    out = img.copy()
    H, W = img.shape[:2]
    for f in fs:
        g = _Geo(f.lm)
        if g.mw < 4:
            continue
        c = g.eye_mid
        r = 0.95 * max(g.fw, g.fh)
        roi = _clip_roi(c[0] - r, c[1] - 0.75 * r, c[0] + r, c[1] + 1.55 * r, W, H)
        if roi is None:
            continue
        x0, y0, x1, y1 = roi
        # 1) peau : désaturée, verdâtre, contraste dur, bords du visage creusés (masque elliptique très doux)
        reg = out[y0:y1, x0:x1]
        step = max(2.0, min(x1 - x0, y1 - y0) / 90.0)
        gx, gy = _grid(roi, step)
        hc = 0.55 * g.eye_mid + 0.45 * g.mouth_c
        ur = ((gx - hc[0]) * g.right[0] + (gy - hc[1]) * g.right[1]) / (0.58 * g.fw)
        vd = ((gx - hc[0]) * g.down[0] + (gy - hc[1]) * g.down[1]) / (0.60 * g.fh)
        rho = np.sqrt(ur * ur + vd * vd)
        # masque peau = ellipse douce ∩ ovale du visage élargi (pas de halo vert sur le décor)
        # ovale du visage rastérisé directement à la résolution de la grille (rapide)
        gh_, gw_ = gx.shape
        sxg, syg = gw_ / (x1 - x0), gh_ / (y1 - y0)
        polyg = (g.lm[FACE_OVAL] - [x0, y0]) * [sxg, syg] + [x0, y0]
        oval = _soft_poly(polyg, (x0, y0, x0 + gw_, y0 + gh_), 0.06 * g.fh * sxg, 0.24 * g.fh * sxg, 2)
        fm = _upsample(_smoothstep(1.30, 0.70, rho) * oval, roi)
        ring = _upsample((_smoothstep(0.55, 0.95, rho) * _smoothstep(1.35, 0.95, rho)) * 0.20 * oval, roi)
        lum = _lum(reg)[..., None]
        sick = (0.80 * lum + 0.20 * reg) * np.float32([0.90, 0.97, 0.90])
        mu = float(lum.mean())
        sick = np.clip((sick - 0.5 * mu) * 1.12 + 0.5 * mu, 0, 1)
        k = (fm * min(a, 1.0) * 0.7)[..., None]
        reg[:] = (reg * (1 - k) + sick * k) * (1 - (ring * min(a, 1.0))[..., None])
        # 2) déformation : bouche étirée + mâchoire qui tombe + léger melt (un seul rééchantillonnage)
        mfx, mfy, (src, disp) = _mouth_field(g, roi, a * 0.75, open_v=0.42)
        lfx, lfy, _ = _melt_field(g, roi, 0.28 * a, t, step)
        if lfx.shape != mfx.shape:
            lfx = cv2.resize(lfx, mfx.shape[::-1])
            lfy = cv2.resize(lfy, mfx.shape[::-1])
        fx, fy = _limit_fold(mfx + lfx, mfy + lfy, (x1 - x0) / mfx.shape[1])
        tap = _edge_taper(roi, 0.08)
        dx, dy = _upsample(fx, roi) * tap, _upsample(fy, roi) * tap
        before = out[y0:y1, x0:x1].copy()
        warped = _remap_roi(out, roi, dx, dy)
        mag = np.sqrt(dx * dx + dy * dy)
        out[y0:y1, x0:x1] = _restore_grain(before, warped, np.clip(mag / 2.0, 0, 1), 23 + f.track_id)
        # positions déformées des points utiles (bouche) pour la suite
        lmw = g.lm.copy()
        lips = sorted(set(LIPS_OUTER + LIPS_INNER))
        for kk, i in enumerate(lips):
            lmw[i] = src[kk] + disp[kk]
        # 3) gorge noire
        _darken_mouth(out, roi, lmw[LIPS_INNER], g, min(a, 1.0), depth=1.0, seam=0.7, teeth=0.35)
        # lèvres mortes (bleutées, désaturées)
        lb = _bbox(lmw[LIPS_OUTER])
        lroi = _clip_roi(lb[0] - 4, lb[1] - 4, lb[2] + 4, lb[3] + 4, W, H)
        if lroi is None:
            continue
        lipm = _soft_poly(lmw[LIPS_OUTER], lroi, 0.0, max(0.8, 0.05 * g.mw), 1) - \
            _soft_poly(lmw[LIPS_INNER], lroi, 0.0, max(0.8, 0.05 * g.mw), 1)
        lipm = np.clip(lipm, 0, 1) * 0.7 * min(a, 1.0)
        reg = out[lroi[1]:lroi[3], lroi[0]:lroi[2]]
        dead = _lum(reg)[..., None] * np.float32([0.62, 0.66, 0.78])
        reg[:] = reg * (1 - lipm[..., None]) + dead * lipm[..., None]
        # 4) orbites creusées + yeux noirs + veines marquées
        fit = Face(g.lm, f.score, f.track_id, f.frame_size)
        _black_eyes_one(out, fit, 1.0 * min(a / 0.8, 1.0) if a < 0.8 else 1.0, veins=1.35, socket=0.55)
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# head_tic
# ---------------------------------------------------------------------------
_TIC_WIN = 6


def _tic_event(frame_idx: int, amount: float, seed: int = 4242):
    """Décrit l'à-coup actif à cette image : (phase 0..2, angle°, dx, dy, hold) ou None. Déterministe."""
    a = float(np.clip(amount, 0, 1.5))
    if a <= 0:
        return None
    w = frame_idx // _TIC_WIN
    rng = np.random.default_rng([seed, w])
    p_evt = 0.30 + 0.55 * min(a, 1.0)
    happens = rng.random() < p_evt
    start = int(rng.integers(0, _TIC_WIN - 3))
    length = int(rng.integers(2, 4))
    ang = rng.uniform(-1, 1) * (3.0 + 9.0 * a)
    mag = rng.uniform(0.35, 1.0) * (0.02 + 0.06 * a)
    th = rng.uniform(0, 2 * math.pi)
    hold = rng.random() < 0.5
    ph = frame_idx - w * _TIC_WIN - start
    if not happens or ph < 0 or ph >= length:
        return None
    return ph, ang, mag * math.cos(th), mag * math.sin(th) * 0.6, hold


def head_tic_hold(frame_idx: int, amount: float = 1.0) -> int:
    """Extension (optionnelle pour le rendu) : nombre d'images à « répéter » (0 = aucune) pour imiter un
    saut d'image pendant un tic : utiliser la source frame_idx - head_tic_hold(...)."""
    ev = _tic_event(frame_idx, amount)
    if ev is None:
        return 0
    ph, *_rest, hold = ev
    return int(ph) if hold else 0


def head_tic(img, face, frame_idx: int, amount: float = 1.0):
    """À-coups saccadés de la tête (2–3 images) : rotation/décalage rigide de la tête (champ lisse, le cou
    s'étire) avec flou de bougé ; sans visage : à-coup de toute l'image. Déterministe par frame_idx."""
    img = _as_float(img)
    ev = _tic_event(int(frame_idx), amount)
    if ev is None:
        return img
    ph, ang, mx, my, hold = ev
    k = [1.0, 0.85 if hold else 0.55, 0.35][min(ph, 2)] if not (hold and ph == 1) else 1.0
    H, W = img.shape[:2]
    fs = [_fit_face(f, img) for f in _faces(face)]
    if not fs:
        # toute l'image : rotation + décalage, léger zoom pour ne pas montrer de bord
        A = math.radians(ang * 0.35 * k)
        tx, ty = mx * k * 0.5 * W, my * k * 0.5 * H
        zoom = 1.0 + 1.2 * (abs(mx) + abs(my)) * k + abs(A) * 0.6
        M = cv2.getRotationMatrix2D((W / 2, H / 2), math.degrees(A), zoom)
        M[:, 2] += (tx, ty)
        out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        if ph == 0:   # flou de bougé de l'à-coup
            M2 = cv2.getRotationMatrix2D((W / 2, H / 2), math.degrees(A) * 0.5, (1 + zoom) / 2)
            M2[:, 2] += (tx * 0.5, ty * 0.5)
            out = 0.6 * out + 0.4 * cv2.warpAffine(img, M2, (W, H), flags=cv2.INTER_LINEAR,
                                                   borderMode=cv2.BORDER_REFLECT101)
        return np.clip(out, 0, 1).astype(np.float32)
    out = img.copy()
    for f in fs:
        g = _Geo(f.lm)
        if g.fh < 8:
            continue
        hc = (g.eye_mid + g.mouth_c) / 2
        pivot = g.lm[CHIN] + g.down * 0.35 * g.fh
        rx, ry = 0.75 * g.fw, 0.80 * g.fh
        r = 1.6 * max(rx, ry) + 0.1 * g.fh
        roi = _clip_roi(hc[0] - r, hc[1] - r, hc[0] + r, hc[1] + r, W, H)
        if roi is None:
            continue
        x0, y0, x1, y1 = roi
        step = max(2.0, min(x1 - x0, y1 - y0) / 80.0)
        gx, gy = _grid(roi, step)
        u = ((gx - hc[0]) * g.right[0] + (gy - hc[1]) * g.right[1]) / rx
        v = ((gx - hc[0]) * g.down[0] + (gy - hc[1]) * g.down[1]) / ry
        wg = _smoothstep(1.55, 0.90, np.sqrt(u * u + v * v))
        tap = _edge_taper(roi, 0.05)
        wgt = _upsample(wg, roi) * tap

        def field(kk):
            # déplacement rigide (rotation autour du cou + translation), pondéré par le masque tête
            A = math.radians(ang * kk)
            ca, sa = math.cos(A), math.sin(A)
            px, py = gx - pivot[0], gy - pivot[1]
            ddx = (ca * px - sa * py + pivot[0] - gx + mx * kk * g.fh) * wg
            ddy = (sa * px + ca * py + pivot[1] - gy + my * kk * g.fh) * wg
            return _upsample(ddx, roi) * tap, _upsample(ddy, roi) * tap

        before = out[y0:y1, x0:x1].copy()
        if ph == 0 or (not hold and ph == 1):   # flou de bougé de l'à-coup (exposition pendant le mouvement)
            ddx, ddy = field(k)
            warped = 0.5 * _remap_roi(img, roi, ddx, ddy)
            xs, ys = _axes(roi)
            for q, wq in ((0.78, 0.2), (0.89, 0.3)):   # échantillons flous calculés à mi-résolution
                mxh = (xs[None, ::2] - ddx[::2, ::2] * q).astype(np.float32)
                myh = (ys[::2, None] - ddy[::2, ::2] * q).astype(np.float32)
                half = cv2.remap(img, mxh, myh, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
                warped += wq * cv2.resize(half, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        else:
            ddx, ddy = field(k)
            warped = _remap_roi(img, roi, ddx, ddy)
        out[y0:y1, x0:x1] = _restore_grain(before, warped, wgt, 29 + int(frame_idx))
    if min(1.0, amount) > 0.7 and ph == 0:  # secousse globale d'1–2 px sur l'attaque
        sh = 0.0025 * W * (1 if mx > 0 else -1)
        M = np.float32([[1, 0, sh], [0, 1, -sh * 0.5]])
        out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# portrait_eyes
# ---------------------------------------------------------------------------
def _gaze_offset(t: float) -> tuple[float, float]:
    """Trajectoire du regard (unités : demi-largeur d'œil) : se tourne vers la caméra puis la suit."""
    e1 = float(_smoothstep(0.10, 0.55, t))
    e2 = float(_smoothstep(0.55, 2.2, t))
    x = -0.42 * e1 + 0.80 * e2 + 0.06 * math.sin(2 * math.pi * 0.35 * t)
    y = 0.05 * math.sin(2 * math.pi * 0.21 * t + 0.5)
    return x, y


def portrait_eyes(img, faces, t: float, amount: float = 1.0):
    """Tableaux : déplace iris/pupilles des visages peints détectés pour qu'ils suivent la caméra dans le
    temps (regard qui se tourne puis glisse). Sans visage : image inchangée."""
    img = _as_float(img)
    fs = [_fit_face(f, img) for f in _faces(faces)]
    a = float(np.clip(amount, 0.0, 1.5))
    if not fs or a <= 0:
        return img
    out = img.copy()
    H, W = img.shape[:2]
    gx, gy = _gaze_offset(float(t))
    for f in fs:
        g = _Geo(f.lm)
        for side in (0, 1):
            e = g.eye(side)
            ew = e["ew"]
            if ew < 4:
                continue
            r = 1.0 * ew
            roi = _clip_roi(e["c"][0] - r, e["c"][1] - r, e["c"][0] + r, e["c"][1] + r, W, H)
            if roi is None:
                continue
            x0, y0, x1, y1 = roi
            ss = int(min(6, max(1, math.ceil(64.0 / ew))))
            reg = out[y0:y1, x0:x1]
            big = cv2.resize(reg, ((x1 - x0) * ss, (y1 - y0) * ss), interpolation=cv2.INTER_CUBIC) if ss > 1 else reg.copy()
            S = lambda p: (np.asarray(p, np.float32) - [x0, y0]) * ss  # noqa: E731
            eye_m = _soft_poly(S(e["poly"]) / ss + [x0, y0], roi, -0.02 * ew, max(0.5, 0.06 * ew), ss)
            eye_m_big = cv2.resize(eye_m, big.shape[1::-1], interpolation=cv2.INTER_LINEAR) if ss > 1 else eye_m
            ic = S(e["iris"])
            ir = e["iris_r"] * ss
            hh, ww = big.shape[:2]
            yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
            d = np.sqrt((xx - ic[0]) ** 2 + (yy - ic[1]) ** 2)
            iris_m = _smoothstep(ir * 1.18, ir * 0.85, d)
            # « blanc » de l'œil peint : couleur de l'ouverture hors iris (sinon pâle par défaut)
            scl = (eye_m_big > 0.5) & (d > ir * 1.3)
            if scl.sum() >= 3:
                sclera = np.median(big[scl], axis=0)
            else:
                sclera = np.percentile(big.reshape(-1, 3), 80, axis=0)
            hole = ((iris_m > 0.05) & (eye_m_big > 0.05)).astype(np.uint8)
            u8 = (np.clip(big, 0, 1) * 255).astype(np.uint8)
            clean = cv2.inpaint(u8, hole * 255, max(3, int(ir * 0.6)), cv2.INPAINT_TELEA).astype(np.float32) / 255
            # l'inpainting bave la couleur de l'iris : on la tire vers le blanc de l'œil
            hs = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), max(0.8, 0.15 * ir))[..., None]
            clean = clean * (1 - 0.45 * hs) + sclera[None, None] * 0.45 * hs
            # nouveau centre : décalage le long de l'axe de l'œil (vers la caméra), borné par l'ouverture
            off = (gx * e["ax"] * (0.5 * ew) + gy * e["ay"] * (0.5 * ew)) * a * ss
            # l'œil gauche et le droit regardent dans la même direction (axe image, pas axe œil)
            if np.dot(e["ax"], [1, 0]) < 0:
                off = (-gx * e["ax"] * (0.5 * ew) + gy * e["ay"] * (0.5 * ew)) * a * ss
            M = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
            iris_layer = cv2.warpAffine(big, M, (ww, hh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
            iris_mm = cv2.warpAffine(iris_m.astype(np.float32), M, (ww, hh), flags=cv2.INTER_LINEAR)
            m_new = (iris_mm * eye_m_big)[..., None]
            res = clean * (1 - m_new) + iris_layer * m_new
            # mélange limité à l'ouverture de l'œil
            em = eye_m_big[..., None]
            res = big * (1 - em) + res * em
            if ss > 1:
                res = cv2.resize(res, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
            reg[:] = res
    return np.clip(out, 0, 1)


__all__ = ["Face", "FaceTrack", "black_eyes", "stretch_smile", "melt", "widen_eyes", "demon_face",
           "head_tic", "head_tic_hold", "portrait_eyes", "read_frames", "probe_size", "detect_faces_image"]
