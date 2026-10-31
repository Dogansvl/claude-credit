"""Silhouette de « lui » : détourage depuis IMG_8346 et incrustation raccord dans d'autres plans.

API (docs/ARCHITECTURE.md) :
  extract(video_path, t_range, out_dir, hint_center=None) -> dict
      {"still": path, "bbox": [x0,y0,x1,y1] (px source), "foot": [x,y], "height_px": h, "method": ...}
      + extensions : "t" (instant de la meilleure image), "frames" (dossier), "frames_fps", "score", "candidates".
  composite(img, sil_rgba, x, foot_y, height, opacity=0.9, blur=1.5, darkness=0.9, blend='normal',
            frame_idx=0, occluder_mask=None) -> img
      x, foot_y, height : normalisés (fractions de W, H, H) si ≤ 2, sinon en pixels.
      sil_rgba : ndarray RGBA (uint8 ou float [0,1]) ou chemin d'un PNG RGBA.
  person_mask(img) -> (H, W) float32 : personnes au premier plan (pour passer la silhouette DERRIÈRE).

Chaîne d'extraction : boîte personne (MediaPipe ObjectDetector efficientdet_lite2) -> détourage
u2net_human_seg (onnxruntime direct, modèle local) -> secours « sombre sur fond clair » (contraste local, le
vrai plan : forme sombre devant un mur blanc éclairé par les phares) -> secours MediaPipe selfie_multiclass ;
affinage des bords (filtre guidé), décontamination des couleurs de bord ; choix de la meilleure image
(personne complète, nette, masque franc). Échec total : silhouette procédurale (method="procedural").
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

try:
    import config
except ModuleNotFoundError:  # exécution directe
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config  # noqa: E402

try:
    from pipeline import facefx as _fx
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline import facefx as _fx  # noqa: E402

VERSION = "silhouette-2"
ORT_THREADS = int(os.environ.get("ESCAPE_ORT_THREADS", "2"))
_LOCK = threading.Lock()
_SESS: dict[str, object] = {}
_MP: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------
def _ort(name: str):
    s = _SESS.get(name)
    if s is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = ORT_THREADS
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        s = ort.InferenceSession(str(config.model_path(name)), so, providers=["CPUExecutionProvider"])
        _SESS[name] = s
    return s


def _detector():
    d = _MP.get("det")
    if d is None:
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(config.model_path("efficientdet_lite2.tflite"))),
            running_mode=vision.RunningMode.IMAGE, score_threshold=0.15, max_results=12,
            category_allowlist=["person"])
        d = vision.ObjectDetector.create_from_options(opts)
        _MP["det"] = d
    return d


def _segmenter():
    s = _MP.get("seg")
    if s is None:
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(config.model_path("selfie_multiclass.tflite"))),
            running_mode=vision.RunningMode.IMAGE, output_confidence_masks=True, output_category_mask=False)
        s = vision.ImageSegmenter.create_from_options(opts)
        _MP["seg"] = s
    return s


def close_models() -> None:
    for k in list(_MP):
        try:
            _MP.pop(k).close()
        except Exception:
            pass


import atexit  # noqa: E402
atexit.register(close_models)


def detect_persons(rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Boîtes personne [x0,y0,x1,y1] (px) + score, efficientdet_lite2."""
    import mediapipe as mp
    with _LOCK:
        res = _detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    out = []
    for d in res.detections:
        b = d.bounding_box
        sc = d.categories[0].score if d.categories else 0.0
        out.append((np.array([b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height], np.float32),
                    float(sc)))
    return out


def _u2net(rgb: np.ndarray, model: str = "u2net_human_seg.onnx") -> np.ndarray:
    """Masque [0,1] (même taille que rgb) — u2net / u2net_human_seg (320) ou isnet (1024)."""
    sess = _ort(model)
    inp = sess.get_inputs()[0]
    S = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 320
    x = cv2.resize(rgb, (S, S), interpolation=cv2.INTER_AREA if rgb.shape[0] > S else cv2.INTER_LINEAR)
    x = x.astype(np.float32)
    x = x / max(float(x.max()), 1e-6)
    if "isnet" in model:
        mean, std = np.float32([0.5, 0.5, 0.5]), np.float32([1.0, 1.0, 1.0])
    else:
        mean, std = np.float32([0.485, 0.456, 0.406]), np.float32([0.229, 0.224, 0.225])
    x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    with _LOCK:
        y = sess.run(None, {inp.name: x})[0][0, 0]
    y = (y - y.min()) / max(float(y.max() - y.min()), 1e-6)
    return cv2.resize(y.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


def _selfie(rgb: np.ndarray) -> np.ndarray:
    import mediapipe as mp
    with _LOCK:
        res = _segmenter().segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    bg = res.confidence_masks[0].numpy_view().astype(np.float32)
    return np.clip(1.0 - bg, 0, 1)


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
def _u8(img) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a[..., :3])
    return (np.clip(a[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)


def _f32(img) -> np.ndarray:
    a = np.asarray(img)
    return a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32, copy=False)


def _lum(a: np.ndarray) -> np.ndarray:
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / u) if u > 0 else 0.0


def _guided(guide_u8: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Filtre guidé (bords du masque collés aux contours de l'image)."""
    try:
        return cv2.ximgproc.guidedFilter(guide_u8, src.astype(np.float32), r, eps * 255 * 255)
    except Exception:  # implémentation de secours (boîtes), guide en luminance
        I = _lum(guide_u8.astype(np.float32) / 255.0)
        p = src.astype(np.float32)
        k = (2 * r + 1, 2 * r + 1)
        mI, mp_ = cv2.blur(I, k), cv2.blur(p, k)
        cov = cv2.blur(I * p, k) - mI * mp_
        var = cv2.blur(I * I, k) - mI * mI
        a = cov / (var + eps)
        b = mp_ - a * mI
        return cv2.blur(a, k) * I + cv2.blur(b, k)


def _largest_components(mask_bin: np.ndarray, ref_box=None, keep_frac: float = 0.15) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask_bin.astype(np.uint8), 8)
    if n <= 1:
        return mask_bin.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    keep = np.zeros(n, bool)
    keep[best] = True
    for i in range(1, n):
        if areas[i - 1] >= keep_frac * areas[best - 1]:
            keep[i] = True
    return keep[lab]


def _decontaminate(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Remplace la couleur des pixels de bord (mélangés au fond) par la couleur du sujet voisine :
    évite le liseré clair d'un sujet sombre détouré sur un mur blanc."""
    hard = (alpha > 0.92).astype(np.float32)
    if hard.sum() < 30:
        return rgb
    est = rgb.copy()
    filled = hard.copy()
    for s in (2, 4, 8, 16, 32):
        num = cv2.GaussianBlur(rgb * hard[..., None], (0, 0), s)
        den = cv2.GaussianBlur(hard, (0, 0), s)
        ok = (den > 0.02) & (filled < 0.5)
        est[ok] = num[ok] / den[ok, None]
        filled[ok] = 1.0
    w = _smoothstep(0.55, 0.95, alpha)[..., None]
    return rgb * w + est * (1 - w)


def _crop_box(box, W, H, pad_x=0.18, pad_y=0.08, min_aspect=0.62):
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    bw2 = max(bw * (1 + 2 * pad_x), bh * min_aspect)
    X0, X1 = int(max(0, cx - bw2 / 2)), int(min(W, cx + bw2 / 2))
    Y0, Y1 = int(max(0, y0 - pad_y * bh)), int(min(H, y1 + pad_y * bh))
    return X0, Y0, X1, Y1


# ---------------------------------------------------------------------------
# Secours « sombre sur fond clair »
# ---------------------------------------------------------------------------
def dark_on_light(rgb: np.ndarray, hint=None, box=None) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Forme sombre et haute devant un fond clair (mur éclairé par les phares).
    Renvoie (bbox, alpha plein cadre, score) ou (None, None, 0)."""
    H, W = rgb.shape[:2]
    L = _lum(rgb.astype(np.float32) / 255.0)
    k = max(15, int(W / 9) | 1)
    # fermeture morphologique : efface les objets sombres plus étroits que k -> « fond » local
    bg = cv2.morphologyEx(L, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 4)
    dark = np.clip(bg - L, 0, 1)
    rel = dark / np.maximum(bg, 0.05)
    cand = (rel > 0.35) & (bg > 0.18)
    if box is not None:
        x0, y0, x1, y1 = [int(v) for v in box]
        lim = np.zeros_like(cand)
        lim[max(0, y0):y1, max(0, x0):x1] = True
        cand &= lim
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(cand, 8)
    best, bs = None, 0.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 0.06 * H or area < 0.0004 * W * H:
            continue
        aspect = h / max(w, 1)
        s_aspect = float(_smoothstep(1.2, 2.4, aspect)) * float(_smoothstep(8.0, 5.0, aspect))
        fill = area / max(w * h, 1)
        comp = lab[y:y + h, x:x + w] == i
        top = comp[: max(1, h // 8)].sum(0)
        mid = comp[h // 5: h // 3 + 1].sum(0)
        wt, wm = float((top > 0).sum()), float((mid > 0).sum())
        s_head = float(_smoothstep(0.95, 0.75, wt / max(wm, 1.0)))   # tête plus étroite que les épaules
        s = s_aspect * s_head * (0.5 + 0.5 * min(1.0, fill / 0.45)) * min(1.0, h / (0.25 * H) + 0.3)
        if hint is not None:
            hx, hy = hint[0] * W, hint[1] * H
            d = math.hypot(cent[i][0] - hx, cent[i][1] - hy) / (0.35 * H)
            s *= math.exp(-d * d)
        if s > bs:
            best, bs = i, s
    if best is None:
        return None, None, 0.0
    x, y, w, h, _ = stats[best]
    comp = (lab == best).astype(np.uint8)
    comp = cv2.dilate(comp, np.ones((5, 5), np.uint8))
    alpha = _smoothstep(0.22, 0.55, rel) * comp
    return np.array([x, y, x + w, y + h], np.float32), alpha.astype(np.float32), float(bs)


# ---------------------------------------------------------------------------
# Détourage d'une image
# ---------------------------------------------------------------------------
def _matte_frame(rgb: np.ndarray, hint=None, prev_box=None) -> dict | None:
    """Meilleur détourage de la personne visée dans une image. Renvoie dict ou None."""
    H, W = rgb.shape[:2]
    dets = detect_persons(rgb)
    choice, best = None, -1.0
    for box, sc in dets:
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bh < 0.05 * H or bw <= 0:
            continue
        s = sc
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if hint is not None:
            d = math.hypot(cx - hint[0] * W, cy - hint[1] * H) / (0.3 * H)
            s *= math.exp(-d * d) + 0.05
        if prev_box is not None:
            s *= 0.4 + _iou(box, prev_box)
        s *= 0.6 + 0.4 * min(1.0, bh / (0.3 * H))
        if s > best:
            choice, best = (box, sc), s
    method = "rembg_u2net_human_seg"
    if choice is None:
        dbox, dal, dsc = dark_on_light(rgb, hint)
        if dbox is None:
            return None
        return _finish(rgb, dbox, dal, "dark_on_light", 0.5 * dsc + 0.2)
    box, det_sc = choice
    X0, Y0, X1, Y1 = _crop_box(box, W, H)
    crop = rgb[Y0:Y1, X0:X1]
    m = _u2net(crop)
    full = np.zeros((H, W), np.float32)
    full[Y0:Y1, X0:X1] = m
    # contrôle de qualité du masque dans la boîte ; sinon « sombre sur fond clair », puis selfie
    q = _mask_quality(full, box)
    if q < 0.35:
        dbox, dal, dsc = dark_on_light(rgb, hint, box=_crop_box(box, W, H, 0.3, 0.15))
        if dbox is not None and dsc > 0.3:
            qd = _mask_quality(dal, box)
            if qd > q + 0.15:
                full, method, q = dal, "dark_on_light", qd
    if q < 0.3:
        sm = np.zeros((H, W), np.float32)
        sm[Y0:Y1, X0:X1] = _selfie(crop)
        qs = _mask_quality(sm, box)
        if qs > q:
            full, method, q = sm, "mediapipe_selfie", qs
    return _finish(rgb, box, full, method, 0.5 * det_sc + 0.5 * q)


def _mask_quality(mask: np.ndarray, box) -> float:
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    sub = mask[max(0, y0):y1, max(0, x0):x1]
    if sub.size == 0:
        return 0.0
    cover = float((sub > 0.5).mean())                  # une personne remplit ~35–65 % de sa boîte
    s_cover = float(_smoothstep(0.08, 0.35, cover) * _smoothstep(0.92, 0.75, cover))
    decisive = float(np.mean(np.abs(sub * 2 - 1)))
    rows = np.flatnonzero((sub > 0.5).any(1))
    vext = (rows[-1] - rows[0] + 1) / max(sub.shape[0], 1) if len(rows) else 0.0
    return s_cover * (0.4 + 0.6 * decisive) * float(_smoothstep(0.4, 0.85, vext))


def _finish(rgb, box, full_mask, method, score) -> dict | None:
    H, W = rgb.shape[:2]
    X0, Y0, X1, Y1 = _crop_box(box, W, H, 0.3, 0.12, 0.5)
    crop = rgb[Y0:Y1, X0:X1]
    m = full_mask[Y0:Y1, X0:X1].astype(np.float32)
    if m.max() < 0.3:
        return None
    hb = Y1 - Y0
    r = max(2, int(hb / 160))
    ref = _guided(crop, m, r, 1e-3)
    ref = _guided(crop, np.clip(ref, 0, 1), max(1, r // 2), 2e-4)
    alpha = _smoothstep(0.12, 0.88, np.clip(ref, 0, 1)).astype(np.float32)
    keep = _largest_components(alpha > 0.5)
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    alpha = alpha * keep
    ys, xs = np.nonzero(alpha > 0.5)
    if len(ys) < 50:
        return None
    by0, by1, bx0, bx1 = ys.min(), ys.max(), xs.min(), xs.max()
    # complétude : pénalise un sujet coupé par le bord de l'image
    touch = 0.0
    if Y0 + by0 <= 1:
        touch += 0.5
    if Y0 + by1 >= H - 2:
        touch += 0.5
    if X0 + bx0 <= 1 or X0 + bx1 >= W - 2:
        touch += 0.25
    lap = cv2.Laplacian(_lum(crop.astype(np.float32) / 255.0), cv2.CV_32F)
    band = cv2.dilate((alpha > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) - \
        cv2.erode((alpha > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8))
    sharp = float(np.var(lap[band > 0])) if band.any() else 0.0
    # pied : centre des pixels du bas de la silhouette
    bottom = alpha[max(by0, by1 - max(2, (by1 - by0) // 30)): by1 + 1]
    fx = float(np.average(np.arange(bottom.shape[1]), weights=bottom.sum(0) + 1e-6)) if bottom.sum() > 0 else (bx0 + bx1) / 2
    return {"crop_rgb": crop, "alpha": alpha, "origin": (X0, Y0), "method": method, "score": float(score),
            "touch": touch, "sharp": sharp, "bbox": [X0 + bx0, Y0 + by0, X0 + bx1 + 1, Y0 + by1 + 1],
            "foot": [X0 + fx, Y0 + by1 + 1.0], "height_px": float(by1 - by0 + 1)}


def _to_rgba(res: dict, margin: float = 0.04) -> np.ndarray:
    a = res["alpha"]
    rgb = res["crop_rgb"].astype(np.float32) / 255.0
    rgb = _decontaminate(rgb, a)
    X0, Y0 = res["origin"]
    bx0, by0, bx1, by1 = [int(v) for v in res["bbox"]]
    bx0, bx1, by0, by1 = bx0 - X0, bx1 - X0, by0 - Y0, by1 - Y0
    m = int(margin * (by1 - by0)) + 2
    ys, xs = slice(max(0, by0 - m), min(a.shape[0], by1 + m)), slice(max(0, bx0 - m), min(a.shape[1], bx1 + m))
    out = np.dstack([rgb[ys, xs], a[ys, xs]])
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def _save_rgba(path: Path, rgba_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgra = cv2.cvtColor(rgba_u8, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        raise RuntimeError(f"PNG impossible : {path}")
    path.write_bytes(buf.tobytes())   # (imwrite échoue avec les chemins non ASCII sous Windows)


def load_rgba(path) -> np.ndarray:
    """PNG RGBA -> uint8 (h, w, 4) RGBA (lecture compatible chemins non ASCII)."""
    data = np.fromfile(str(path), np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    if im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    return cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------
def extract(video_path, t_range, out_dir, hint_center=None, sample_fps: float = 10.0, force: bool = False) -> dict:
    """Détoure la silhouette sur t_range (s) de la vidéo ; écrit still.png (RGBA, meilleure image),
    frames/%03d.png (RGBA, à sample_fps) et meta.json dans out_dir. Cache : meta.json si paramètres identiques."""
    out_dir = Path(out_dir)
    video_path = Path(video_path)
    t0, t1 = float(t_range[0]), float(t_range[1])
    st = video_path.stat() if video_path.exists() else None
    key = hashlib.sha1(json.dumps([str(video_path.resolve()), st.st_size if st else 0, st.st_mtime_ns if st else 0,
                                   t0, t1, hint_center, sample_fps, VERSION]).encode()).hexdigest()[:16]
    meta_p = out_dir / "meta.json"
    if not force and meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("key") == key and Path(meta["still"]).exists():
                return meta
        except Exception:
            pass
    out_dir.mkdir(parents=True, exist_ok=True)
    fr_dir = out_dir / "frames"
    if fr_dir.exists():
        for p in fr_dir.glob("*.png"):
            p.unlink()
    results = []
    prev_box = None
    hint = tuple(hint_center) if hint_center is not None else None
    if st is not None:
        for i, rgb in enumerate(_fx.read_frames(video_path, t0, t1, sample_fps)):
            r = _matte_frame(rgb, hint, prev_box)
            if r is None:
                continue
            r["t"] = t0 + i / sample_fps
            r["idx"] = i
            prev_box = np.array(r["bbox"], np.float32)
            results.append(r)
    if not results:
        return _procedural_result(out_dir, key, reason="aucune personne détectée")
    # score global : détection/masque, complétude, netteté relative, taille
    sharp_ref = max(np.percentile([r["sharp"] for r in results], 90), 1e-9)
    h_ref = max(r["height_px"] for r in results)
    for r in results:
        r["total"] = (0.45 * r["score"] + 0.25 * min(1.0, r["sharp"] / sharp_ref)
                      + 0.15 * (r["height_px"] / h_ref) + 0.15 * (1.0 - min(1.0, r["touch"])))
    best = max(results, key=lambda r: r["total"])
    if best["total"] < 0.25:
        return _procedural_result(out_dir, key, reason=f"détourage trop faible ({best['total']:.2f})")
    for k, r in enumerate(results):
        _save_rgba(fr_dir / f"{k:03d}.png", _to_rgba(r))
    still = out_dir / "still.png"
    _save_rgba(still, _to_rgba(best))
    meta = {"still": str(still), "bbox": [float(v) for v in best["bbox"]], "foot": [float(v) for v in best["foot"]],
            "height_px": float(best["height_px"]), "method": best["method"], "t": float(best["t"]),
            "score": float(best["total"]), "frames": str(fr_dir), "frames_fps": sample_fps,
            "n_frames": len(results), "key": key, "version": VERSION,
            "candidates": [{"t": round(r["t"], 3), "method": r["method"], "total": round(r["total"], 3)} for r in results]}
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# Silhouette procédurale (secours)
# ---------------------------------------------------------------------------
def _catmull(pts: np.ndarray, n: int = 12) -> np.ndarray:
    P = np.vstack([pts[-1:], pts, pts[:2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out, np.float32)


def procedural(height_px: int = 1800, seed: int = 7) -> np.ndarray:
    """Grande silhouette mince (~8 têtes) : manteau long légèrement évasé, épaules tombantes, bras le long du
    corps (jour entre bras et buste), mains pendantes, tête légèrement penchée, aucun visage. RGBA uint8."""
    rng = np.random.default_rng(seed)
    Hh = int(height_px)
    Ww = int(Hh * 0.30)
    ss = 2
    u = Hh * ss / 100.0          # unité : 1 % de la hauteur
    cx = Ww * ss / 2
    m = np.zeros((Hh * ss, Ww * ss), np.uint8)

    def poly(pts, closed=True):
        P = np.array(pts, np.float32)
        P[:, 0] += rng.normal(0, 0.06, len(P))
        P = np.stack([cx + P[:, 0] * u, P[:, 1] * u], 1)
        P = _catmull(P, 8) if closed else P
        cv2.fillPoly(m, [np.round(P * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)

    # tête (penchée de ~9°, légèrement en avant) et cou
    cv2.ellipse(m, (int(cx + 0.8 * u), int(6.2 * u)), (int(3.45 * u), int(4.5 * u)), -9, 0, 360, 255, -1, cv2.LINE_AA)
    poly([(-2.0, 8.6), (2.4, 8.6), (2.4, 12.2), (3.4, 14.0), (-3.2, 14.0), (-2.1, 12.2)])
    # buste + manteau (évasé vers l'ourlet, ourlet irrégulier)
    poly([(-2.2, 13.4), (2.4, 13.4), (6.0, 14.6), (8.0, 16.0), (7.6, 21.0), (6.9, 30.0), (6.6, 38.0),
          (7.1, 48.0), (7.9, 60.0), (8.6, 72.0), (9.1, 82.5), (6.0, 84.0), (3.0, 83.6), (0.3, 84.2),
          (-3.0, 83.4), (-6.2, 84.1), (-8.9, 82.6), (-8.4, 72.0), (-7.7, 60.0), (-6.9, 48.0), (-6.4, 38.0),
          (-6.7, 30.0), (-7.4, 21.0), (-7.8, 16.0), (-5.8, 14.6)])
    # bras dans les manches, légèrement écartés du buste ; mains pendantes
    for sgn in (1, -1):
        poly([(sgn * 7.4, 15.4), (sgn * 9.3, 17.0), (sgn * 9.9, 22.0), (sgn * 10.1, 31.0), (sgn * 9.9, 40.0),
              (sgn * 9.6, 45.5), (sgn * 9.7, 47.5), (sgn * 9.4, 51.0), (sgn * 8.7, 52.6), (sgn * 8.0, 51.4),
              (sgn * 7.8, 47.5), (sgn * 7.9, 45.0), (sgn * 7.6, 40.0), (sgn * 7.5, 31.0), (sgn * 7.3, 22.0)])
    # jambes (pantalon) et chaussures
    for sgn, dx in ((1, 0.0), (-1, 0.15)):
        poly([(sgn * 0.5, 82.0), (sgn * 3.1, 82.0), (sgn * 3.0, 90.0), (sgn * 2.8, 96.6), (sgn * 1.0, 96.6),
              (sgn * 0.7, 90.0)])
        poly([(sgn * 0.6 + dx, 96.0), (sgn * 3.0 + dx, 96.0), (sgn * 3.4 + dx, 98.4), (sgn * 3.9 + dx, 99.6),
              (sgn * 0.5 + dx, 99.8), (sgn * 0.4 + dx, 97.6)])
    # bords irréguliers (tissu, cheveux) : bruit sur le contour
    a = m.astype(np.float32) / 255
    noise = cv2.GaussianBlur(rng.standard_normal(a.shape).astype(np.float32), (0, 0), 2.0 * ss) * 0.22
    a = np.clip((cv2.GaussianBlur(a, (0, 0), 1.0 * ss) - 0.5 + noise) * 3 + 0.5, 0, 1)
    a = cv2.resize(a, (Ww, Hh), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (0, 0), 0.7)
    # matière : noir bleuté, plis verticaux du manteau à peine visibles
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    folds = 0.012 * np.sin(xx / Ww * 31 + np.sin(yy / Hh * 7) * 2) * _smoothstep(0.2, 0.5, yy / Hh)
    tex = cv2.GaussianBlur(rng.standard_normal((Hh, Ww)).astype(np.float32), (0, 0), 1.2) * 0.006
    base = 0.035 + folds + tex
    rgb = np.dstack([base * 0.92, base * 0.96, base * 1.08])
    rgba = np.dstack([np.clip(rgb, 0, 1), a])
    return (rgba * 255 + 0.5).astype(np.uint8)


def _procedural_result(out_dir: Path, key: str, reason: str) -> dict:
    rgba = procedural(1800)
    still = out_dir / "still.png"
    _save_rgba(still, rgba)
    fr = out_dir / "frames"
    _save_rgba(fr / "000.png", rgba)
    a = rgba[..., 3] > 127
    ys, xs = np.nonzero(a)
    meta = {"still": str(still), "bbox": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
            "foot": [float(rgba.shape[1] / 2), float(ys.max() + 1)], "height_px": float(ys.max() - ys.min() + 1),
            "method": "procedural", "reason": reason, "t": None, "score": 0.0, "frames": str(fr),
            "frames_fps": 0, "n_frames": 1, "key": key, "version": VERSION}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# composite
# ---------------------------------------------------------------------------
_RGBA_CACHE: dict[str, tuple[np.ndarray, tuple]] = {}


def _prep_sil(sil_rgba) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """RGBA float32 [0,1] recadré sur l'alpha + (bbox alpha)."""
    if isinstance(sil_rgba, (str, Path)):
        k = str(sil_rgba)
        if k in _RGBA_CACHE:
            return _RGBA_CACHE[k]
        arr = load_rgba(k)
    else:
        k = None
        arr = np.asarray(sil_rgba)
    a = arr.astype(np.float32) / 255.0 if arr.dtype == np.uint8 else arr.astype(np.float32)
    if a.shape[2] == 3:
        a = np.dstack([a, np.ones(a.shape[:2], np.float32)])
    ys, xs = np.nonzero(a[..., 3] > 0.04)
    if len(ys) == 0:
        res = (a, (0, 0, a.shape[1], a.shape[0]))
    else:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        res = (a[y0:y1, x0:x1], (x0, y0, x1, y1))
    if k is not None:
        _RGBA_CACHE[k] = res
    return res


def _foot_x(a: np.ndarray) -> float:
    h = a.shape[0]
    bottom = a[int(h * 0.9):, :, 3]
    w = bottom.sum(0)
    return float(np.average(np.arange(a.shape[1]), weights=w + 1e-6)) if w.sum() > 0 else a.shape[1] / 2


def composite(img, sil_rgba, x, foot_y, height, opacity=0.9, blur=1.5, darkness=0.9, blend="normal",
              frame_idx=0, occluder_mask=None):
    """Incruste la silhouette debout, pieds en (x, foot_y), hauteur `height`. Raccord : assombrie et
    désaturée selon la luminance de la plaque autour du point d'insertion, flou de profondeur, grain de la
    plaque, respiration ±0,5 % à 0,3 Hz et micro-balancement ; occluder_mask (H, W) la passe derrière."""
    plate = _f32(img)
    H, W = plate.shape[:2]
    if opacity <= 0 or height <= 0:
        return plate
    sil, _ = _prep_sil(sil_rgba)
    sh, sw = sil.shape[:2]
    xp = x * W if x <= 2 else float(x)
    fy = foot_y * H if foot_y <= 2 else float(foot_y)
    hp = height * H if height <= 2 else float(height)
    t = frame_idx / float(config.FPS)
    breath = 1.0 + 0.005 * math.sin(2 * math.pi * 0.3 * t)
    sway = 0.35 * math.sin(2 * math.pi * 0.17 * t + 0.7) + 0.12 * math.sin(2 * math.pi * 0.41 * t + 2.0)
    s = hp / sh
    # pré-réduction (anti-crénelage) puis affine résiduelle
    if s < 0.7:
        pre = max(s * 1.15, 1e-3)
        sil_s = cv2.resize(sil * np.dstack([sil[..., 3:]] * 3 + [np.ones_like(sil[..., 3:])]),
                           (max(2, int(round(sw * pre))), max(2, int(round(sh * pre)))), interpolation=cv2.INTER_AREA)
        k = sil_s.shape[0] / sh
    else:
        sil_s = sil * np.dstack([sil[..., 3:]] * 3 + [np.ones_like(sil[..., 3:])])   # prémultiplié
        k = 1.0
    sx = s / k
    sy = s * breath / k
    fxs = _foot_x(sil) * k
    fys = float(sil_s.shape[0])
    A = math.radians(sway)
    ca, sa = math.cos(A), math.sin(A)
    # matrice : (u, v) silhouette -> image ; pieds fixes, rotation autour des pieds
    M = np.float32([[ca * sx, -sa * sy, 0], [sa * sx, ca * sy, 0]])
    M[0, 2] = xp - (M[0, 0] * fxs + M[0, 1] * fys)
    M[1, 2] = fy - (M[1, 0] * fxs + M[1, 1] * fys)
    corners = np.float32([[0, 0], [sil_s.shape[1], 0], [0, fys], [sil_s.shape[1], fys]])
    pc = np.stack([corners[:, 0] * M[0, 0] + corners[:, 1] * M[0, 1] + M[0, 2],
                   corners[:, 0] * M[1, 0] + corners[:, 1] * M[1, 1] + M[1, 2]], 1)
    pad = int(3 * max(blur, 1) * W / 1080) + 8
    x0, y0 = int(max(0, pc[:, 0].min() - pad)), int(max(0, pc[:, 1].min() - pad))
    x1, y1 = int(min(W, pc[:, 0].max() + pad)), int(min(H, pc[:, 1].max() + pad))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return plate
    M2 = M.copy()
    M2[:, 2] -= (x0, y0)
    lay = cv2.warpAffine(sil_s, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = np.clip(lay[..., 3], 0, 1)
    rgb = lay[..., :3] / np.maximum(a[..., None], 1e-4)
    roi = plate[y0:y1, x0:x1]
    # statistiques de la plaque autour du point d'insertion (hors silhouette)
    ex = int(0.5 * (x1 - x0)) + 4
    X0, Y0, X1, Y1 = max(0, x0 - ex), max(0, y0 - ex // 2), min(W, x1 + ex), min(H, y1 + ex // 2)
    around = plate[Y0:Y1, X0:X1]
    amask = np.ones(around.shape[:2], bool)
    amask[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] &= a < 0.05
    Lp = _lum(around)[amask] if amask.any() else _lum(around).ravel()
    Lp = Lp[::max(1, Lp.size // 20000)]
    # référence : la plaque sous la silhouette et ses ombres voisines (pas les zones claires du cadre)
    under = _lum(roi)[a > 0.5]
    L_bg = float(np.percentile(Lp, 40))
    if under.size > 20:
        L_bg = min(L_bg, float(np.median(under)))
    black = float(np.percentile(Lp, 3))
    C_bg = around[amask].mean(0) if amask.any() else around.reshape(-1, 3).mean(0)
    tint = C_bg / max(float(_lum(C_bg[None])[0]), 1e-3)
    tint = 0.55 * tint + 0.45                       # teinte ambiante partielle
    # couleur de la silhouette : tonalités internes atténuées, niveau = plaque assombrie
    sl = _lum(rgb)
    wsum = float(a.sum())
    sl_mean = float((sl * a).sum() / wsum) if wsum > 1 else 0.1
    rel = np.clip(sl / max(sl_mean, 1e-3), 0.5, 2.0) ** 0.35
    # toujours plus sombre que la plaque qu'elle masque : (1 - darkness) × lumière locale (plaque floutée,
    # bornée par la référence d'ombre), plancher = noirs de la plaque (légèrement plus profonds)
    k_l = max(5.0, 0.08 * (y1 - y0))
    L_loc = np.minimum(cv2.GaussianBlur(_lum(roi), (0, 0), k_l), max(L_bg, black))
    dk = 1.0 - float(np.clip(darkness, 0, 1))
    fig_l = np.maximum(0.6 * black, L_loc * dk) + (L_bg - black) * dk * 0.15
    fig = (fig_l * rel)[..., None] * tint[None, None, :]
    # flou de profondeur (prémultiplié) + light wrap discret
    sig = max(0.0, float(blur)) * W / 1080.0
    if sig > 0.05:
        pm = cv2.GaussianBlur(fig * a[..., None], (0, 0), sig)
        a = cv2.GaussianBlur(a, (0, 0), sig)
        fig = pm / np.maximum(a[..., None], 1e-4)
    a_in = cv2.erode(a, np.ones((3, 3), np.uint8)) if a.shape[0] > 4 else a
    wrap = cv2.GaussianBlur(roi, (0, 0), max(2.0, 4.0 * W / 1080))
    edge = np.clip(a - cv2.GaussianBlur(a_in, (0, 0), max(1.5, 2.5 * W / 1080)), 0, 1)
    fig = fig + 0.22 * np.clip(wrap - fig, 0, None) * edge[..., None]
    # grain : même écart-type que la plaque
    sig_n = _fx._noise_sigma(around)
    if sig_n > 1e-4:
        h_, w_ = fig.shape[:2]
        tile = _fx._noise_tile(1000 + int(frame_idx))
        n = np.tile(np.roll(tile, (int(frame_idx) * 37 % 256, int(frame_idx) * 71 % 256), (0, 1)),
                    (h_ // 256 + 1, w_ // 256 + 1))[:h_, :w_]
        fig = fig + (n * sig_n)[..., None]
    A_ = np.clip(a * float(opacity), 0, 1)
    if occluder_mask is not None:
        occ = np.asarray(occluder_mask, np.float32)
        if occ.shape[:2] != (H, W):
            occ = cv2.resize(occ, (W, H), interpolation=cv2.INTER_LINEAR)
        A_ = A_ * (1.0 - np.clip(occ[y0:y1, x0:x1], 0, 1))
    A3 = A_[..., None]
    if blend == "multiply":
        factor = np.clip(fig / max(L_bg, 0.02), 0, 1)
        res = roi * (1 - A3 * (1 - factor))
    else:
        res = roi * (1 - A3) + fig * A3
    out = plate.copy()
    out[y0:y1, x0:x1] = np.clip(res, 0, 1)
    return out


# ---------------------------------------------------------------------------
# person_mask
# ---------------------------------------------------------------------------
_PM_CACHE: dict[str, np.ndarray] = {}


def person_mask(img, min_score: float = 0.3, min_height: float = 0.10, max_boxes: int = 3) -> np.ndarray:
    """Masque (H, W) float32 des personnes au premier plan : boîtes efficientdet (les `max_boxes` plus
    grandes, hauteur ≥ min_height·H) + u2net_human_seg dans chaque boîte, bords affinés au filtre guidé ;
    secours selfie_multiclass sur l'image entière. Cache mémoire par image."""
    rgb = _u8(img)
    H, W = rgb.shape[:2]
    k = hashlib.sha1(cv2.resize(rgb, (64, 112), interpolation=cv2.INTER_AREA).tobytes()).hexdigest()
    if k in _PM_CACHE:
        return _PM_CACHE[k].copy()
    mask = np.zeros((H, W), np.float32)
    boxes = [b for b, s in detect_persons(rgb) if s >= min_score and (b[3] - b[1]) > min_height * H]
    boxes = sorted(boxes, key=lambda b: -(b[3] - b[1]) * (b[2] - b[0]))[:max_boxes]
    for b in boxes:
        X0, Y0, X1, Y1 = _crop_box(b, W, H, 0.15, 0.06)
        if X1 - X0 < 8 or Y1 - Y0 < 8:
            continue
        crop = rgb[Y0:Y1, X0:X1]
        m = _u2net(crop)
        # limite au voisinage de la boîte (bords doux)
        yy = np.arange(Y0, Y1, dtype=np.float32)[:, None]
        xx = np.arange(X0, X1, dtype=np.float32)[None, :]
        bw, bh = b[2] - b[0], b[3] - b[1]
        lim = _smoothstep(b[0] - 0.12 * bw, b[0] + 0.02 * bw, xx) * _smoothstep(b[2] + 0.12 * bw, b[2] - 0.02 * bw, xx) * \
            _smoothstep(b[1] - 0.06 * bh, b[1] + 0.01 * bh, yy) * _smoothstep(b[3] + 0.06 * bh, b[3] - 0.01 * bh, yy)
        r = max(2, int((Y1 - Y0) / 120))
        m = np.clip(_guided(crop, m, r, 1e-3), 0, 1)
        mask[Y0:Y1, X0:X1] = np.maximum(mask[Y0:Y1, X0:X1], _smoothstep(0.15, 0.85, m) * lim)
    if not boxes:
        sm = _selfie(rgb)
        mask = np.where(sm > 0.5, sm, 0).astype(np.float32)
        mask = _smoothstep(0.3, 0.8, mask).astype(np.float32)
    if len(_PM_CACHE) > 64:
        _PM_CACHE.clear()
    _PM_CACHE[k] = mask
    return mask.copy()


__all__ = ["extract", "composite", "person_mask", "procedural", "load_rgba", "detect_persons", "dark_on_light"]
