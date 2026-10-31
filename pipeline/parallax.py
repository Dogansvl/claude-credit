"""Parallaxe 2,5D sur les photos (carte de profondeur MiDaS small ONNX).

API (docs/ARCHITECTURE.md) :
  depth(img) -> (H, W) float32, 1 = proche   (MiDaS small 256x256, profondeur inverse, lissée, cache disque)
  render_parallax(img, depth, t, dur, zoom=(1.0, 1.16), center=(0.5, 0.45), amount=1.0, out_size=(1080, 1920)) -> img
Extension : project_points(points_xy, img_shape, depth, t, dur, ...) -> (N, 2) px de sortie
            (pour que les overlays — cercles rouges, silhouette — suivent la caméra).

Rendu en deux couches : premier plan (les personnes, profondeur > seuil d'Otsu, bords recalés au filtre
guidé) et fond dont la zone masquée par le premier plan est comblée (inpainting) — les désocclusions
révèlent donc un fond plausible, jamais une déchirure. Chaque couche suit sa propre carte de profondeur
lissée (« feuille élastique » sans pli), et la caméra pousse vers `center` : les plans proches grossissent
plus que le fond, avec une légère dérive latérale.
"""
from __future__ import annotations

import hashlib
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

VERSION = "parallax-2"
ORT_THREADS = int(os.environ.get("ESCAPE_ORT_THREADS", "2"))
K_PAR = 0.11          # grossissement supplémentaire du plan le plus proche à amount=1 (fin du mouvement)
DRIFT = 0.012         # dérive latérale (fraction de la largeur de sortie) par unité de profondeur
_LOCK = threading.Lock()
_SESS = {}
_SCENES: dict[str, dict] = {}


def _f32(img) -> np.ndarray:
    a = np.asarray(img)
    return a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32, copy=False)


def _u8(img) -> np.ndarray:
    a = np.asarray(img)
    return a if a.dtype == np.uint8 else (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _midas():
    s = _SESS.get("midas")
    if s is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = ORT_THREADS
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        s = ort.InferenceSession(str(config.model_path("midas_small.onnx")), so, providers=["CPUExecutionProvider"])
        _SESS["midas"] = s
    return s


def _guided(guide_u8: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    try:
        return cv2.ximgproc.guidedFilter(guide_u8, src.astype(np.float32), r, eps * 255 * 255)
    except Exception:
        I = cv2.cvtColor(guide_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        p = src.astype(np.float32)
        k = (2 * r + 1, 2 * r + 1)
        mI, mp_ = cv2.blur(I, k), cv2.blur(p, k)
        a = (cv2.blur(I * p, k) - mI * mp_) / (cv2.blur(I * I, k) - mI * mI + eps)
        b = mp_ - a * mI
        return cv2.blur(a, k) * I + cv2.blur(b, k)


# ---------------------------------------------------------------------------
# depth
# ---------------------------------------------------------------------------
def _key(img: np.ndarray, extra: str = "") -> str:
    small = cv2.resize(_u8(img), (96, 128), interpolation=cv2.INTER_AREA)
    return hashlib.sha1(small.tobytes() + str(img.shape).encode() + extra.encode()).hexdigest()[:20]


def depth(img, use_cache: bool = True) -> np.ndarray:
    """Profondeur relative (H, W) float32 dans [0, 1], 1 = proche. MiDaS small (+ miroir), normalisation
    robuste (centiles 2–98), suréchantillonnage recalé sur les contours (filtre guidé), cache npy."""
    rgb = _f32(img)[..., :3]
    H, W = rgb.shape[:2]
    k = _key(rgb, VERSION)
    cache = config.CACHE_DIR / "depth" / f"{k}.npy"
    if use_cache and cache.exists():
        try:
            d = np.load(cache)
            if d.shape == (H, W):
                return d
        except Exception:
            pass
    sess = _midas()
    inp = sess.get_inputs()[0]
    x = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    x = (x - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])
    batch = np.stack([x, x[:, ::-1]]).transpose(0, 3, 1, 2).astype(np.float32)
    with _LOCK:
        y0 = sess.run(None, {inp.name: batch[:1]})[0][0]
        y1 = sess.run(None, {inp.name: batch[1:]})[0][0][:, ::-1]
    d = 0.5 * (y0 + y1)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
    r = max(4, int(W / 90))
    d = _guided(_u8(rgb), d, r, 2e-3)
    d = cv2.GaussianBlur(d, (0, 0), max(1.0, W / 500))
    d = np.clip(d, 0, 1).astype(np.float32)
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp.npy")
        np.save(tmp, d)
        os.replace(tmp, cache)
    return d


# ---------------------------------------------------------------------------
# Préparation de la scène (couches, comblement) — mise en cache mémoire par image
# ---------------------------------------------------------------------------
def _otsu(vals: np.ndarray) -> float:
    h, e = np.histogram(vals, 64, (0, 1))
    p = h.astype(np.float64) / max(h.sum(), 1)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(p)
    m0 = np.cumsum(p * c)
    mt = m0[-1]
    var = (mt * w0 - m0) ** 2 / np.maximum(w0 * (1 - w0), 1e-12)
    return float(c[int(np.argmax(var))])


def _extend(values: np.ndarray, valid: np.ndarray, down: int = 4) -> np.ndarray:
    """Prolonge `values` hors de `valid` (inpainting basse résolution) et lisse : carte continue."""
    H, W = values.shape
    small = cv2.resize(values, (max(8, W // down), max(8, H // down)), interpolation=cv2.INTER_AREA)
    vm = cv2.resize(valid.astype(np.float32), small.shape[::-1], interpolation=cv2.INTER_AREA) > 0.5
    u8 = (np.clip(small, 0, 1) * 255).astype(np.uint8)
    hole = (~vm).astype(np.uint8) * 255
    if hole.any() and vm.any():
        u8 = cv2.inpaint(u8, hole, 5, cv2.INPAINT_TELEA)
    out = cv2.resize(u8.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_CUBIC)
    return np.where(valid, values, out).astype(np.float32)


def _push_pull(rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Comble `hole` par moyenne pondérée multi-échelle (pyramide push-pull) : lisse, sans traînées."""
    w = (~hole).astype(np.float32)
    levels = [(rgb * w[..., None], w)]
    while min(levels[-1][1].shape) > 4:
        c, ww = levels[-1]
        levels.append((cv2.pyrDown(c), cv2.pyrDown(ww)))
    c, ww = levels[-1]
    est = c / np.maximum(ww[..., None], 1e-6)
    for c, ww in reversed(levels[:-1]):
        up = cv2.pyrUp(est, dstsize=(ww.shape[1], ww.shape[0]))
        a = np.clip(ww * 4.0, 0, 1)[..., None]
        est = (c / np.maximum(ww[..., None], 1e-6)) * a + up * (1 - a)
    return est.astype(np.float32)


def _fast_key(img: np.ndarray, dep: np.ndarray) -> str:
    """Clé bon marché (appelée à chaque image) : forme + échantillon clairsemé des pixels."""
    a = np.ascontiguousarray(np.asarray(img)[::53, ::47])
    b = np.ascontiguousarray(np.asarray(dep)[::53, ::47])
    return hashlib.sha1(a.tobytes() + b.tobytes() + str(np.shape(img)).encode()).hexdigest()[:24]


def _scene(img: np.ndarray, dep: np.ndarray, ws: float = 1.0) -> dict:
    """Couches de la scène à l'échelle de travail ws (≤ 1 : réduite à la résolution utile de la sortie)."""
    k = _fast_key(img, dep) + f"{ws:.3f}"
    sc = _SCENES.get(k)
    if sc is not None:
        return sc
    rgb = _f32(img)[..., :3]
    H0, W0 = rgb.shape[:2]
    d = cv2.resize(dep.astype(np.float32), (W0, H0)) if dep.shape != (H0, W0) else dep.astype(np.float32)
    if ws < 0.999:
        sz = (max(8, int(round(W0 * ws))), max(8, int(round(H0 * ws))))
        rgb = cv2.resize(rgb, sz, interpolation=cv2.INTER_AREA)
        d = cv2.resize(d, sz, interpolation=cv2.INTER_AREA)
    H, W = rgb.shape[:2]
    thr = float(np.clip(_otsu(d.ravel()[::7]), 0.35, 0.85))
    near = _smoothstep(thr - 0.035, thr + 0.035, d)
    # bords du premier plan recalés sur l'image
    near = np.clip(_guided(_u8(rgb), near, max(3, int(W / 160)), 1e-3), 0, 1)
    near = _smoothstep(0.2, 0.8, near).astype(np.float32)
    frac = float(near.mean())
    two = 0.03 < frac < 0.75
    sc = {"rgb": rgb, "depth": d, "thr": thr, "two_layers": two, "shape": (H, W)}
    if two:
        # fond : premier plan effacé (remplissage pyramidal « push-pull », grain réinjecté)
        grow = cv2.dilate((near > 0.1).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                  (int(W / 60) | 1,) * 2))
        fill = _push_pull(rgb, grow.astype(bool))
        rng = np.random.default_rng(5)
        fill += rng.standard_normal((H, W, 1)).astype(np.float32) * 0.010
        # le fond ne diffère de l'original que sous le premier plan (quasi) opaque : pas de voile au repos
        gm = _smoothstep(0.5, 0.95, near)[..., None]
        sc["bg"] = rgb * (1 - gm) + fill * gm
        sc["bg_depth"] = cv2.GaussianBlur(_extend(np.minimum(d, thr), ~grow.astype(bool)), (0, 0), W / 120)
        sc["fg_alpha"] = near
        sc["fg_depth"] = cv2.GaussianBlur(_extend(d, near > 0.5), (0, 0), W / 200)
    else:
        sc["bg"] = rgb
        sc["bg_depth"] = cv2.GaussianBlur(d, (0, 0), W / 150)
    if len(_SCENES) > 6:
        _SCENES.clear()
    _SCENES[k] = sc
    return sc


# ---------------------------------------------------------------------------
# Caméra
# ---------------------------------------------------------------------------
def _ease(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    s = p * p * (3 - 2 * p)
    return 0.35 * p + 0.65 * s


def _camera(shape, t, dur, zoom, center, amount, out_size):
    H, W = shape
    Wo, Ho = int(out_size[0]), int(out_size[1])
    p = _ease(t / max(dur, 1e-6))
    z = zoom[0] + (zoom[1] - zoom[0]) * p
    margin = 1.0 + K_PAR * amount * 0.6 + DRIFT * amount * 1.5
    S = max(Wo / W, Ho / H) * margin * z        # px source -> px sortie
    # centre clampé pour que le cadre reste dans l'image
    cw, ch = Wo / S, Ho / S
    cx = min(max(center[0] * W, cw / 2), W - cw / 2)
    cy = min(max(center[1] * H, ch / 2), H - ch / 2)
    return dict(p=p, S=S, cx=cx, cy=cy, Wo=Wo, Ho=Ho, amount=amount)


def _drift(cam, dd):
    """Dérive (px de sortie) d'un point de profondeur relative dd : glissement latéral + léger vertical."""
    p, a, Wo = cam["p"], cam["amount"], cam["Wo"]
    return DRIFT * Wo * a * dd * (2.2 * p - 0.6), -0.4 * DRIFT * Wo * a * dd * p


def _layer_maps(cam, dmap: np.ndarray, d_ref: float, step: int = 4):
    """Cartes de rééchantillonnage (sortie -> source) d'une couche, point fixe sur la profondeur."""
    Wo, Ho, S, cx, cy, p, a = cam["Wo"], cam["Ho"], cam["S"], cam["cx"], cam["cy"], cam["p"], cam["amount"]
    gw, gh = -(-Wo // step), -(-Ho // step)
    # positions alignées sur la convention de cv2.resize (centres de pixels)
    qx = ((np.arange(gw, dtype=np.float32) + 0.5) * step - 0.5)[None, :] - Wo / 2
    qy = ((np.arange(gh, dtype=np.float32) + 0.5) * step - 0.5)[:, None] - Ho / 2
    sx = cx + qx / S + np.zeros_like(qy)
    sy = cy + qy / S + np.zeros_like(qx)
    for _ in range(3):
        dd = cv2.remap(dmap, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE) - d_ref
        f = 1.0 + K_PAR * a * dd * p
        ox, oy = _drift(cam, dd)
        sx = cx + (qx - ox) / (S * f)
        sy = cy + (qy - oy) / (S * f)
    def big(m):
        return cv2.resize(m.astype(np.float32), (gw * step, gh * step), interpolation=cv2.INTER_LINEAR)[:Ho, :Wo]
    return big(sx), big(sy)


def render_parallax(img, depth, t: float, dur: float, zoom=(1.0, 1.16), center=(0.5, 0.45), amount=1.0,
                    out_size=(1080, 1920)):
    """Une image (out_size = (W, H)) de la poussée 2,5D à l'instant t ∈ [0, dur]."""
    H0, W0 = np.shape(img)[:2]
    s_max = max(_camera((H0, W0), x, 1.0, zoom, center, float(amount), out_size)["S"] for x in (0.0, 1.0))
    s_max *= 1.0 + K_PAR * float(amount) * 0.5
    ws = 1.0 if s_max >= 0.9 else float(np.ceil(s_max * 1.1 * 20) / 20)   # résolution utile, par pas de 5 %
    sc = _scene(img, depth, ws)
    H, W = sc["shape"]
    cam = _camera((H, W), float(t), float(dur), zoom, center, float(amount), out_size)
    d_ref = float(np.median(sc["depth"][int(cam["cy"]) - 8:int(cam["cy"]) + 8, int(cam["cx"]) - 8:int(cam["cx"]) + 8]))
    mx, my = _layer_maps(cam, sc["bg_depth"], d_ref)
    out = cv2.remap(sc["bg"], mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
    if sc["two_layers"]:
        fx, fy = _layer_maps(cam, sc["fg_depth"], d_ref)
        fg = cv2.remap(sc["rgb"], fx, fy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
        al = cv2.remap(sc["fg_alpha"], fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None]
        fg -= out
        fg *= al
        out += fg
    return np.clip(out, 0, 1, out=out)


def project_points(points_xy, img_shape, depth, t: float, dur: float, zoom=(1.0, 1.16), center=(0.5, 0.45),
                   amount=1.0, out_size=(1080, 1920)) -> np.ndarray:
    """Position (px de sortie) de points de la photo (px source) à l'instant t (même caméra que le rendu)."""
    H, W = img_shape[:2]
    cam = _camera((H, W), float(t), float(dur), zoom, center, float(amount), out_size)
    d = depth if depth.shape == (H, W) else cv2.resize(depth, (W, H))
    d_ref = float(np.median(d[int(cam["cy"]) - 8:int(cam["cy"]) + 8, int(cam["cx"]) - 8:int(cam["cx"]) + 8]))
    pts = np.asarray(points_xy, np.float32).reshape(-1, 2)
    out = []
    for x, y in pts:
        dd = float(d[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))]) - d_ref
        f = 1.0 + K_PAR * cam["amount"] * dd * cam["p"]
        ox, oy = _drift(cam, dd)
        out.append(((x - cam["cx"]) * cam["S"] * f + ox + cam["Wo"] / 2,
                    (y - cam["cy"]) * cam["S"] * f + oy + cam["Ho"] / 2))
    return np.array(out, np.float32)


__all__ = ["depth", "render_parallax", "project_points"]
