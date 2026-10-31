"""Motion design + étalonnage du film « C'EST QUI LUI ? » (contrat : docs/ARCHITECTURE.md, pipeline/gfx.py).

Système graphique, cohérent sur tout le film :
  * cartons, prénoms, ligne sous le titre : Cormorant Garamond (serif élégante, raccord avec le titre 3D en
    Bodoni), CAPITALES, approche très large (+300 à +360), blanc cassé, apparition flou -> net avec un léger
    scintillement d'exposition ;
  * HUD et timecode : IBM Plex Mono ;
  * sous-titres : Inter (répliques), Inter italique blanc cassé (narrateur), Cormorant italique rouge
    désaturé (l'entité), contour sombre fin + ombre douce, dans la zone sûre TikTok.

Conventions : images float32 RGB (H, W, 3) dans [0, 1], de n'importe quelle taille. Les mesures sont données
pour 1080x1920 et mises à l'échelle par W/1080 (l'aperçu est en 540x960). Aucune fonction ne modifie l'image
d'entrée (elles renvoient une copie). Tout l'aléatoire est déterministe (graine, frame_idx ou texte).
Les textes sont pré-rendus en sprites mis en cache ; aucune boucle Python par pixel.
"""
from __future__ import annotations

import functools
import json
import math
import sys
import zlib
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:  # config.py est à la racine du dépôt
    import config
except ModuleNotFoundError:  # exécution directe
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config  # noqa: E402

__all__ = [
    "font", "grade", "film_finish", "glitch", "rgb_split", "negative", "darken",
    "draw_card", "draw_name", "draw_timecode", "draw_subtitle", "draw_hud_face", "draw_red_circle",
    "draw_title_sub", "safe_zone", "exposure_gain", "GFX_VERSION",
]

GFX_VERSION = "gfx-2.0"          # à inclure dans les clés de cache des rendus de shots

REF_W, REF_H = 1080, 1920
SAFE_REF = (60, 160, 900, 1536)  # x0, y0, x1, y1 à 1080x1920 (1536 = début des 20 % du bas)
SUB_BASELINE = 0.765             # dernière ligne de base des sous-titres (fraction de H, < 0,78)

# ----------------------------------------------------------------------------- palette
OFF_WHITE = np.array([0.925, 0.905, 0.862], np.float32)    # cartons, prénoms
SUB_WHITE = np.array([0.965, 0.958, 0.940], np.float32)    # répliques
NARR_WHITE = np.array([0.905, 0.878, 0.818], np.float32)   # narrateur (blanc cassé chaud)
ENTITY_RED = np.array([0.745, 0.315, 0.285], np.float32)   # l'entité (rouge désaturé)
HUD_WHITE = np.array([0.870, 0.950, 0.972], np.float32)
HUD_RED = np.array([0.945, 0.140, 0.110], np.float32)
HUD_RED_FILL = np.array([0.560, 0.035, 0.035], np.float32)
REC_RED = np.array([0.910, 0.165, 0.125], np.float32)
MARKER_RED = np.array([0.835, 0.055, 0.050], np.float32)
MARKER_DARK = np.array([0.470, 0.012, 0.025], np.float32)
INK_BLACK = np.array([0.010, 0.011, 0.014], np.float32)
_LUMA = np.array([[0.2126, 0.7152, 0.0722]], np.float32)


# ============================================================================= petits outils
def _f32(img) -> np.ndarray:
    a = np.asarray(img)
    return a if a.dtype == np.float32 else a.astype(np.float32)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _lerp(a, b, t):
    return a + (b - a) * t


def _sstep(e0: float, e1: float, x):
    t = np.clip((np.asarray(x, np.float32) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ease_out_cubic(p: float) -> float:
    p = _clamp(p)
    return 1.0 - (1.0 - p) ** 3


def _ease_in_out(p: float) -> float:
    p = _clamp(p)
    return p * p * (3.0 - 2.0 * p)


def _ease_out_expo(p: float) -> float:
    p = _clamp(p)
    return 1.0 if p >= 1.0 else 1.0 - 2.0 ** (-10.0 * p)


def _ease_out_back(p: float, k: float = 1.6) -> float:
    p = _clamp(p)
    q = p - 1.0
    return 1.0 + (k + 1.0) * q ** 3 + k * q ** 2


_M64 = (1 << 64) - 1


def _hash(*vals) -> int:
    """Hachage entier déterministe (splitmix64), indépendant de PYTHONHASHSEED."""
    h = 0x9E3779B97F4A7C15
    for v in vals:
        h = (h ^ (int(v) & _M64)) * 0xBF58476D1CE4E5B9 & _M64
        h ^= h >> 31
        h = (h + 0x9E3779B97F4A7C15) & _M64
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9 & _M64
        h = (h ^ (h >> 27)) * 0x94D049BB133111EB & _M64
        h ^= h >> 31
    return h


def _rand(*vals) -> float:
    return (_hash(*vals) >> 11) / float(1 << 53)


def _str_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def _flicker(t: float, seed: int, amp: float, rate: float = 24.0) -> float:
    """Scintillement d'exposition façon projecteur : valeur aléatoire tenue à `rate` i/s."""
    return 1.0 + amp * (2.0 * _rand(seed, int(math.floor(t * rate))) - 1.0)


def safe_zone(w: int | None = None, h: int | None = None) -> tuple[int, int, int, int]:
    """Zone sûre TikTok (x0, y0, x1, y1) en px ; 1080x1920 par défaut, mise à l'échelle si (w, h) est donné."""
    if w is None:
        return SAFE_REF
    h = h if h is not None else int(round(w * REF_H / REF_W))
    sx, sy = w / REF_W, h / REF_H
    x0, y0, x1, y1 = SAFE_REF
    return int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy))


# ============================================================================= polices
_FAMILIES = {
    "inter": ("Interopsz_wght.ttf", None),
    "interitalic": ("Inter-Italicopsz_wght.ttf", None),
    "archivo": ("Archivowdth_wght.ttf", None),
    "archivonarrow": ("ArchivoNarrowwght.ttf", None),
    "bodoni": ("BodoniModaopsz_wght.ttf", None),
    "cormorant": ("CormorantGaramondwght.ttf", None),
    "cormorantitalic": ("CormorantGaramond-Italicwght.ttf", None),
    "playfair": ("PlayfairDisplaywght.ttf", None),
    "cinzel": ("Cinzelwght.ttf", None),
    "oswald": ("Oswaldwght.ttf", None),
    "montserrat": ("Montserratwght.ttf", None),
    "spacegrotesk": ("SpaceGroteskwght.ttf", None),
    "bebas": ("BebasNeue-Regular.ttf", None),
    "plexmono": ("IBMPlexMono-Regular.ttf", "IBMPlexMono-Medium.ttf"),  # (normal, >= 500)
    "sharetech": ("ShareTechMono-Regular.ttf", None),
}
_ALIASES = {
    "ibmplexmono": "plexmono", "plex": "plexmono", "mono": "plexmono", "cormorantgaramond": "cormorant",
    "cormorantgaramonditalic": "cormorantitalic", "bodonimoda": "bodoni", "playfairdisplay": "playfair",
    "sharetechmono": "sharetech", "bebasneue": "bebas", "serif": "cormorant", "sans": "inter",
}


def _family_key(name: str) -> str:
    key = "".join(ch for ch in name.lower() if ch.isalnum())
    key = _ALIASES.get(key, key)
    if key not in _FAMILIES:
        raise ValueError(f"police inconnue : {name!r} (disponibles : {sorted(_FAMILIES)})")
    return key


@functools.lru_cache(maxsize=256)
def _font_cached(key: str, px: int, weight: int | None, opsz: float | None):
    regular, medium = _FAMILIES[key]
    fname = medium if (medium and weight is not None and weight >= 500) else regular
    path = config.FONTS_DIR / fname
    f = ImageFont.truetype(str(path), max(1, int(px)))
    try:
        axes = f.get_variation_axes()
    except (OSError, AttributeError):  # police statique
        return f
    vals = []
    for ax in axes:
        nm = ax.get("name", b"")
        nm = nm.decode("ascii", "ignore") if isinstance(nm, bytes) else str(nm)
        lo, hi, dflt = float(ax["minimum"]), float(ax["maximum"]), float(ax.get("default", ax["minimum"]))
        if nm.lower().startswith("weight"):
            v = float(weight) if weight is not None else dflt
        elif nm.lower().startswith("optical"):
            v = float(opsz) if opsz is not None else px * 0.5
        else:
            v = dflt
        vals.append(min(hi, max(lo, v)))
    f.set_variation_by_axes(vals)
    return f


def font(name: str, size: int, weight: int | None = None):
    """PIL.ImageFont d'une famille d'assets/fonts ("inter", "inter_italic", "archivo", "bodoni",
    "cormorant", "cormorant_italic", "plexmono", "oswald", "playfair", "montserrat", "spacegrotesk",
    "cinzel", "bebas", "sharetech", "archivo_narrow") ; `weight` règle l'axe variable wght (100–900)."""
    return _font_cached(_family_key(name), int(round(size)), None if weight is None else int(weight), None)


# ============================================================================= sprites de texte
# chiffres alignés (Cormorant a des chiffres elzéviriens par défaut : « 3o » au lieu de « 30 »)
try:
    from PIL import features as _pil_features
    _FEATURES = ["lnum", "kern"] if _pil_features.check("raqm") else None
except Exception:  # pragma: no cover
    _FEATURES = None

class _Spr(NamedTuple):
    a: np.ndarray      # couverture (h, w) float32 en lecture seule
    ox: float          # x de l'origine du tracé (gauche du 1er glyphe)
    base: float        # y de la (première) ligne de base
    adv: float         # chasse totale (approche comprise, sans approche finale)
    cap: float         # hauteur de capitale (px)
    ink: tuple         # boîte d'encre (x0, y0, x1, y1)


def _ink_box(a: np.ndarray) -> tuple:
    ys = np.flatnonzero(a.max(axis=1) > 0.02)
    xs = np.flatnonzero(a.max(axis=0) > 0.02)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)


def _layout(f, text: str, tracking_px: float):
    """Positions des glyphes (None = composition normale) et chasse totale, avec approche et crénage."""
    feat = _FEATURES

    def length(t_):
        return f.getlength(t_, features=feat)

    if abs(tracking_px) < 1e-6:  # composition normale (crénage/ligatures du moteur de rendu)
        return None, (length(text) if text else 0.0)
    xs, x = [], 0.0
    for i, ch in enumerate(text):
        xs.append(x)
        if i + 1 < len(text):
            nx = text[i + 1]
            x += length(ch + nx) - length(nx) + tracking_px  # chasse + crénage de la paire
    return xs, x + (length(text[-1]) if text else 0.0)


def _line_raw(text: str, family: str, size: float, weight: int | None = None, tracking: float = 0.0,
              ss: int = 2) -> _Spr:
    """Une ligne de texte avec approche (tracking, en em), rendue à `ss`x puis réduite (placement sub-pixel)."""
    fam = _family_key(family)
    px = max(1, int(round(size * ss)))
    f = _font_cached(fam, px, weight, max(6.0, size * 0.5))
    asc, desc = f.getmetrics()
    pad = int(math.ceil(0.3 * px)) + 2 * ss
    feat = _FEATURES
    xs, adv = _layout(f, text, tracking * px)
    w = int(math.ceil(adv)) + 2 * pad
    h = asc + desc + 2 * pad
    w += (-w) % ss
    h += (-h) % ss
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    if xs is None:
        if text:
            d.text((pad, pad + asc), text, font=f, fill=255, anchor="ls", features=feat)
    else:
        for ch, xx in zip(text, xs):
            if not ch.isspace():
                d.text((pad + xx, pad + asc), ch, font=f, fill=255, anchor="ls", features=feat)
    a = np.asarray(im, np.float32) * np.float32(1.0 / 255.0)
    if ss > 1:
        a = cv2.resize(a, (w // ss, h // ss), interpolation=cv2.INTER_AREA)
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    cap = -f.getbbox("H", anchor="ls")[1] / ss
    return _Spr(a, pad / ss, (pad + asc) / ss, adv / ss, cap, _ink_box(a))


_line = functools.lru_cache(maxsize=1024)(_line_raw)   # sprites fixes (les animés passent par _line_raw)


@functools.lru_cache(maxsize=96)
def _block(lines: tuple, family: str, size: float, weight: int | None, tracking: float, pitch: float,
           align: str = "center", ss: int = 2) -> _Spr:
    """Bloc multi-lignes (pitch = interligne en px entre lignes de base)."""
    sprs = [_line(l, family, size, weight, tracking, ss) for l in lines]
    if len(sprs) == 1:
        return sprs[0]
    width = max(s.adv for s in sprs)
    pad = int(math.ceil(max(s.ox for s in sprs))) + 1
    top = int(math.ceil(max(s.base for s in sprs)))
    bottom = int(math.ceil(max(s.a.shape[0] - s.base for s in sprs)))
    h = top + int(math.ceil(pitch * (len(sprs) - 1))) + bottom
    w = int(math.ceil(width)) + 2 * pad
    a = np.zeros((h, w), np.float32)
    for i, s in enumerate(sprs):
        lx = pad + (0.0 if align == "left" else (width - s.adv) * (0.5 if align == "center" else 1.0))
        x = int(round(lx - s.ox))
        y = int(round(top + i * pitch - s.base))
        hh, ww = s.a.shape
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + ww, w), min(y + hh, h)
        np.maximum(a[y0:y1, x0:x1], s.a[y0 - y:y1 - y, x0 - x:x1 - x], out=a[y0:y1, x0:x1])
    a.flags.writeable = False
    return _Spr(a, float(pad), float(top), float(width), sprs[0].cap, _ink_box(a))


_NO_NNBSP = {"plexmono", "sharetech", "bebas"}


def _typo(text: str, family: str = "inter") -> str:
    """Typographie française : apostrophe courbe, espace fine insécable avant ? ! ; : et dans « », points de
    suspension."""
    nb = "\u00a0" if _family_key(family) in _NO_NNBSP else "\u202f"
    t = text.replace("'", "\u2019").replace("...", "\u2026")
    for p_ in ("?", "!", ";", ":"):
        t = t.replace(" " + p_, nb + p_).replace("\u00a0" + p_, nb + p_)
    t = t.replace("« ", "«" + nb).replace(" »", nb + "»")
    return t


@functools.lru_cache(maxsize=4096)
def _text_width(text: str, family: str, size: float, weight, tracking: float) -> float:
    """Chasse d'une ligne (px) sans la rastériser."""
    if not text:
        return 0.0
    px = max(1, int(round(size)))
    f = _font_cached(_family_key(family), px, weight, max(6.0, size * 0.5))
    return _layout(f, text, tracking * px)[1]


@functools.lru_cache(maxsize=1024)
def _fit(text: str, family: str, size: float, weight, tracking: float, max_w: float, max_lines: int = 2,
         min_scale: float = 0.78) -> tuple[tuple, float]:
    """Taille et coupure de lignes pour tenir dans max_w : réduction douce, puis coupure équilibrée."""
    paras = [p.strip() for p in text.split("\n") if p.strip()] or [""]
    if len(paras) > max_lines:  # trop de lignes imposées : on recompose en au plus max_lines lignes
        paras = [" ".join(paras)]

    def widest(lines, sz):
        return max(_text_width(l, family, sz, weight, tracking) for l in lines)

    sz = size
    while widest(paras, sz) > max_w and sz > size * min_scale:
        sz *= 0.97
    if widest(paras, sz) <= max_w or len(paras) >= max_lines:
        while widest(paras, sz) > max_w and sz > 4:
            sz *= 0.97
        return tuple(paras), sz
    # coupure équilibrée du paragraphe le plus long
    lines: list[str] = []
    for p in paras:
        if _text_width(p, family, size, weight, tracking) <= max_w or len(paras) + len(lines) >= max_lines + 1:
            lines.append(p)
            continue
        words = p.split(" ")  # (les espaces insécables ne coupent pas)
        best = None
        for k in range(1, len(words)):
            a, b = " ".join(words[:k]), " ".join(words[k:])
            wa, wb = _text_width(a, family, size, weight, tracking), _text_width(b, family, size, weight, tracking)
            score = max(wa, wb) + 0.02 * max(0.0, wa - wb)  # préfère une 1re ligne plus courte
            if best is None or score < best[0]:
                best = (score, a, b)
        lines.extend([best[1], best[2]] if best else [p])
    sz = size
    while widest(lines, sz) > max_w and sz > 4:
        sz *= 0.97
    return tuple(lines), sz


# ============================================================================= composition
def _over(dst: np.ndarray, a: np.ndarray, color, x: int, y: int, opacity: float = 1.0) -> None:
    """Compose en place un masque de couverture (h, w) d'une couleur (3,) ou (h, w, 3) au point entier (x, y)."""
    if opacity <= 0.0:
        return
    H, W = dst.shape[:2]
    h, w = a.shape[:2]
    X0, Y0, X1, Y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    m = a[Y0 - y:Y1 - y, X0 - x:X1 - x]
    if opacity != 1.0:
        m = m * np.float32(opacity)
    col = np.asarray(color, np.float32)
    if col.ndim == 3:
        col = col[Y0 - y:Y1 - y, X0 - x:X1 - x]
    roi = dst[Y0:Y1, X0:X1]
    roi += (col - roi) * m[..., None]


def _add(dst: np.ndarray, a: np.ndarray, color, x: int, y: int, gain: float = 1.0) -> None:
    """Addition (lumière) d'un masque coloré."""
    if gain <= 0.0:
        return
    H, W = dst.shape[:2]
    h, w = a.shape[:2]
    X0, Y0, X1, Y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    m = a[Y0 - y:Y1 - y, X0 - x:X1 - x]
    dst[Y0:Y1, X0:X1] += m[..., None] * (np.asarray(color, np.float32) * np.float32(gain))


def _fast_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Flou gaussien large et peu coûteux (réduction, flou, agrandissement) pour ombres et halos."""
    if sigma < 2.5:
        return cv2.GaussianBlur(a, (0, 0), max(0.3, sigma))
    f = int(min(8, max(2, sigma // 2)))
    h, w = a.shape[:2]
    small = cv2.resize(a, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _composite(dst: np.ndarray, x: int, y: int, layers: list) -> None:
    """Compose en place et en une passe des couches de même taille placées en (x, y).
    layers : [(masque (h, w), couleur (3,), opacité, 'over' | 'add')], de dessous vers dessus."""
    H, W = dst.shape[:2]
    h, w = layers[0][0].shape[:2]
    X0, Y0, X1, Y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    sl = (slice(Y0 - y, Y1 - y), slice(X0 - x, X1 - x))
    T, ws, cols = None, [], []
    for m, col, op, mode in layers:
        if op <= 0:
            continue
        a = m[sl] * np.float32(op)
        if mode == "over":
            inv = 1.0 - a
            ws = [wi * inv for wi in ws]
            T = inv if T is None else T * inv
        ws.append(a)
        cols.append(np.asarray(col, np.float32))
    if not ws:
        return
    roi = dst[Y0:Y1, X0:X1]
    if T is not None:
        roi *= T[..., None]
    for i in range(0, len(ws), 3):
        wk, ck = ws[i:i + 3], cols[i:i + 3]
        while len(wk) < 3:
            wk.append(np.zeros_like(ws[0]))
            ck.append(np.zeros(3, np.float32))
        roi += cv2.transform(cv2.merge(wk), np.stack(ck, 1))
    np.minimum(roi, 1.0, out=roi)


def _warp(a: np.ndarray, ax: float, ay: float, cx: float, cy: float, scale: float = 1.0,
          blur: float = 0.0) -> tuple[np.ndarray, int, int]:
    """Place le sprite pour que son point (ax, ay) tombe en (cx, cy) (sub-pixel), avec échelle et flou."""
    h, w = a.shape
    m = int(math.ceil(3.0 * blur)) + 2
    X0 = int(math.floor(cx - ax * scale)) - m
    Y0 = int(math.floor(cy - ay * scale)) - m
    ow = int(math.ceil(w * scale)) + 2 * m + 2
    oh = int(math.ceil(h * scale)) + 2 * m + 2
    M = np.float32([[scale, 0.0, cx - X0 - ax * scale], [0.0, scale, cy - Y0 - ay * scale]])
    out = cv2.warpAffine(a, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if blur > 0.3:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    return out, X0, Y0


_SOFT_CACHE: dict = {}


def _soft(spr_a: np.ndarray, sigma: float, pad: int) -> np.ndarray:
    """Version floutée (ombre/halo) d'un sprite, avec marge `pad`, mise en cache par identité du sprite."""
    key = (id(spr_a), round(sigma, 2), pad)
    hit = _SOFT_CACHE.get(key)
    if hit is not None and hit[0] is spr_a:
        return hit[1]
    big = cv2.copyMakeBorder(np.ascontiguousarray(spr_a), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    out = cv2.GaussianBlur(big, (0, 0), max(0.3, sigma))
    out.flags.writeable = False
    if len(_SOFT_CACHE) > 512:
        _SOFT_CACHE.clear()
    _SOFT_CACHE[key] = (spr_a, out)
    return out


def _hline(dst: np.ndarray, x0: float, x1: float, y: float, thick: float, color, opacity: float) -> None:
    """Filet horizontal anti-crénelé (extrémités et épaisseur sub-pixel)."""
    if x1 - x0 <= 0.05 or opacity <= 0:
        return
    H, W = dst.shape[:2]
    xa, xb = int(math.floor(x0)), int(math.ceil(x1))
    ya, yb = int(math.floor(y - thick / 2)), int(math.ceil(y + thick / 2))
    xs = np.arange(xa, xb, dtype=np.float32)
    cov_x = np.clip(np.minimum(xs + 1, x1) - np.maximum(xs, x0), 0, 1)
    ys = np.arange(ya, yb, dtype=np.float32)
    cov_y = np.clip(np.minimum(ys + 1, y + thick / 2) - np.maximum(ys, y - thick / 2), 0, 1)
    _over(dst, np.outer(cov_y, cov_x).astype(np.float32), color, xa, ya, opacity)


@functools.lru_cache(maxsize=64)
def _disc(r: float, soft: float = 0.75) -> np.ndarray:
    n = int(math.ceil(r + 2 * soft)) + 1
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1].astype(np.float32)
    a = np.clip((r - np.sqrt(xx * xx + yy * yy)) / soft + 0.5, 0, 1).astype(np.float32)
    a.flags.writeable = False
    return a


def _reveal(t: float, dur: float, t_in: float, t_out: float, seed: int, flick: float = 1.0):
    """Enveloppe des cartons : opacité, flou (px @1080), halo ; flou -> net + scintillement d'exposition."""
    fi = max(1e-3, min(t_in, 0.5 * dur))
    fo = max(1e-3, min(t_out, 0.4 * dur))
    p_in = _clamp(t / fi)
    p_out = _clamp((dur - t) / fo)
    a_in = _ease_out_cubic(p_in)
    a_out = _ease_in_out(p_out)
    amp = flick * (0.32 * (1.0 - p_in) ** 0.7 + 0.035)
    fl = _flicker(t, seed, amp)
    alpha = _clamp(a_in * a_out * fl)
    blur = 9.0 * (1.0 - a_in) ** 1.3 + 5.0 * (1.0 - a_out)
    glow = 0.55 * math.sin(math.pi * p_in) * (fl - 1.0 + 0.6) + 0.06
    return alpha, blur, max(0.0, glow)


# ============================================================================= cartons
CARD_FAMILY, CARD_WEIGHT, CARD_SIZE, CARD_TRACK = "cormorant", 500, 58.0, 0.34


def draw_card(img, text: str, t: float, dur: float, style: str = "card", size: tuple | None = None):
    """Carton de cinéma sobre : capitales serif blanc cassé, approche très large, centré.

    t : temps depuis le début du carton ; dur : durée du carton. style='card' (sur noir) ou 'card_over'
    (sur l'image : bande sombre douce et ombre portée). img=None -> fond noir de taille `size` (1080x1920).
    """
    if img is None:
        w, h = size if size else (REF_W, REF_H)
        out = np.zeros((int(h), int(w), 3), np.float32)
    else:
        out = _f32(img).copy()
    if dur <= 0 or not (0.0 <= t <= dur) or not text.strip():
        return out
    H, W = out.shape[:2]
    s = W / REF_W
    seed = _str_seed(text)
    alpha, blur, glow = _reveal(t, dur, 0.8, 0.5, seed)
    if alpha <= 0.003:
        return out
    # largeur max 0,64·W centrée (poussée lente comprise) : reste dans la zone sûre x ≤ 900/1080
    lines, px = _fit(_typo(text.upper(), CARD_FAMILY), CARD_FAMILY, CARD_SIZE * s, CARD_WEIGHT, CARD_TRACK, 0.64 * W,
                     2, 0.86)
    spr = _block(lines, CARD_FAMILY, px, CARD_WEIGHT, CARD_TRACK, 1.62 * px)
    x0, y0, x1, y1 = spr.ink
    cx, cy = W * 0.5, H * (0.478 if style == "card" else 0.47)
    scale = 1.0 + 0.032 * _clamp(t / dur)                       # lente poussée
    m, X, Y = _warp(spr.a, (x0 + x1) / 2, (y0 + y1) / 2, cx, cy, scale, blur * s)
    if style != "card":
        # bande sombre très douce derrière le texte + ombre portée : lisible sur n'importe quelle image
        bh = (y1 - y0) * 4.2
        ys = np.arange(H, dtype=np.float32)
        prof = np.exp(-0.5 * ((ys - cy) / (bh * 0.5)) ** 2).astype(np.float32) * np.float32(0.42 * alpha)
        r0, r1 = int(max(0, cy - 2.2 * bh)), int(min(H, cy + 2.2 * bh))
        out[r0:r1] *= (1.0 - prof[r0:r1])[:, None, None]
    soft = _fast_blur(m, 6.0 * s)     # sert d'ombre portée (sur image) et de halo (scintillement)
    _composite(out, X, Y, [(soft, INK_BLACK, 0.65 * alpha if style != "card" else 0.0, "over"),
                           (soft, OFF_WHITE, 0.55 * glow * alpha if glow > 0.01 else 0.0, "add"),
                           (m, OFF_WHITE, alpha, "over")])
    return out


# ============================================================================= prénoms
NAME_FAMILY, NAME_WEIGHT, NAME_SIZE = "cormorant", 500, 116.0
NAME_TRACK0, NAME_TRACK1 = 0.50, 0.30
NAME_BASE_Y = 0.655   # ligne de base du prénom (fraction de H) : bien au-dessus des 20 % du bas
NAME_SUB_DEFAULT = "30.10.2026"


@functools.lru_cache(maxsize=64)
def _name_px(txt: str, W: int) -> float:
    """Taille du prénom : même au plus large de l'animation (approche de départ), il reste dans la zone sûre
    (largeur ≤ 2·(900 − 540)/1080·W, centré)."""
    s = W / REF_W
    return _fit(txt, NAME_FAMILY, NAME_SIZE * s, NAME_WEIGHT, NAME_TRACK0, 0.66 * W, 1, 0.5)[1]


def draw_name(img, name: str, t: float, sub: str | None = None):
    """Prénom en arrêt sur image : grande capitale à approche large qui se resserre, filet fin, petit texte
    secondaire en mono (`sub`, par défaut la date « 30.10.2026 » ; "" pour aucun). t = temps depuis le gel."""
    out = _f32(img).copy()
    if t < 0 or not name:
        return out
    H, W = out.shape[:2]
    s = W / REF_W
    txt = name.upper()
    seed = _str_seed(txt)
    px = _name_px(txt, W)
    k = _ease_out_expo(t / 0.62)
    tracking = round(_lerp(NAME_TRACK0, NAME_TRACK1, k) / 0.004) * 0.004
    # pendant le resserrement, sprite recalculé à chaque image (pas de cache : mémoire bornée)
    spr = (_line if k >= 0.999 else _line_raw)(txt, NAME_FAMILY, px, NAME_WEIGHT, tracking, 2)
    te = t + 1.0 / 30.0   # visible dès la 1re image du gel (calé sur l'impact sonore)
    alpha = _clamp(_ease_out_cubic(te / 0.16) * _flicker(t, seed, 0.30 * (1 - _clamp(t / 0.5)) + 0.03))
    blur = 8.0 * s * (1.0 - _ease_out_cubic(te / 0.26))
    cx, by = W * 0.5, H * NAME_BASE_Y
    m, X, Y = _warp(spr.a, spr.ox + spr.adv / 2, spr.base, cx, by, 1.0, blur)
    soft = _fast_blur(m, 8.5 * s)
    _composite(out, X, Y, [(soft, INK_BLACK, 0.55 * alpha, "over"),
                           (soft, OFF_WHITE, 0.35 * (1 - t / 0.4) * alpha if t < 0.4 else 0.0, "add"),
                           (m, OFF_WHITE, alpha, "over")])
    # filet fin qui s'ouvre depuis le centre
    final_w = _text_width(txt, NAME_FAMILY, px, NAME_WEIGHT, NAME_TRACK1)
    rw = 0.5 * final_w * _ease_out_cubic((t - 0.05) / 0.45)
    ry = by + 0.40 * px
    if rw > 0.5:
        _hline(out, cx - rw / 2, cx + rw / 2, ry + 1.5 * s, 3.0 * s, INK_BLACK, 0.35 * alpha)
        _hline(out, cx - rw / 2, cx + rw / 2, ry, max(1.0, 1.5 * s), OFF_WHITE, 0.85 * alpha)
    sub = NAME_SUB_DEFAULT if sub is None else sub
    if sub:
        sa = _clamp((t - 0.16) / 0.3) * alpha
        if sa > 0.01:
            ss_ = _line(sub.upper(), "plexmono", 22.0 * s, 400, 0.42, 2)
            sx = int(round(cx - ss_.adv / 2 - ss_.ox))
            sy = int(round(ry + 38.0 * s - ss_.base))
            _over(out, _soft(ss_.a, 3.0 * s, 0), INK_BLACK, sx, sy, 0.6 * sa)
            _over(out, ss_.a, OFF_WHITE, sx, sy, 0.78 * sa)
    return out


# ============================================================================= timecode
TC_X, TC_BASE1, TC_BASE2 = 74.0, 199.0, 230.0
TC_SIZE1, TC_SIZE2 = 34.0, 20.0


def draw_timecode(img, hhmm: str, date: str, t: float):
    """Horodatage caméscope sobre en haut à gauche (zone sûre, y≈170–235) : « ● 20:48 » puis la date en
    dessous, en mono. Le point rouge clignote (1 Hz)."""
    out = _f32(img).copy()
    H, W = out.shape[:2]
    s, sy = W / REF_W, H / REF_H
    st = _line(hhmm, "plexmono", TC_SIZE1 * s, 400, 0.06, 1)
    sd = _line(date, "plexmono", TC_SIZE2 * s, 400, 0.26, 1)
    x_dot = TC_X * s
    x_txt = x_dot + 26.0 * s
    b1, b2 = TC_BASE1 * sy, TC_BASE2 * sy
    tx, ty = int(round(x_txt - st.ox)), int(round(b1 - st.base))
    dx, dy = int(round(x_txt - sd.ox)), int(round(b2 - sd.base))
    pad = int(math.ceil(6 * s)) + 1
    _over(out, _soft(st.a, 2.5 * s, pad), INK_BLACK, tx - pad, ty - pad, 0.55)
    _over(out, _soft(sd.a, 2.5 * s, pad), INK_BLACK, dx - pad, dy - pad, 0.5)
    _over(out, st.a, OFF_WHITE, tx, ty, 0.93)
    _over(out, sd.a, OFF_WHITE, dx, dy, 0.70)
    on = (t % 1.0) < 0.62
    disc = _disc(6.0 * s, 0.75)
    n = disc.shape[0] // 2
    cxd, cyd = int(round(x_dot + 6 * s)), int(round(b1 - st.cap * 0.5))
    if on:
        _add(out, _soft(disc, 5.0 * s, 0), REC_RED, cxd - n, cyd - n, 0.45)
    _over(out, disc, REC_RED, cxd - n, cyd - n, 0.95 if on else 0.18)
    return out


# ============================================================================= sous-titres
_SUB_STYLES = {
    "dialogue": dict(family="inter", weight=500, size=44.0, tracking=0.0, color=SUB_WHITE, ss=1),
    "narrator": dict(family="interitalic", weight=420, size=44.0, tracking=0.0, color=NARR_WHITE, ss=1),
    "entity": dict(family="cormorantitalic", weight=600, size=56.0, tracking=0.035, color=ENTITY_RED, ss=2),
}
_SUB_KIND_ALIASES = {"narration": "narrator", "narrateur": "narrator", "voice": "narrator", "replique": "dialogue",
                     "réplique": "dialogue", "real": "dialogue", "entite": "entity", "entité": "entity"}


@functools.lru_cache(maxsize=48)
def _sub_sprite(text: str, kind: str, W: int, H: int):
    """Sprite prémultiplié (couleur (h,w,3), alpha (h,w)), position (x, y) du coin dans l'image."""
    st = _SUB_STYLES[kind]
    s = W / REF_W
    x0s, _, x1s, _ = safe_zone(W, H)
    max_w = (x1s - x0s) - 2 * 14 * s
    lines, px = _fit(_typo(text.strip(), st["family"]), st["family"], st["size"] * s, st["weight"], st["tracking"],
                     max_w, 2, 0.8)
    pitch = px * (1.24 if kind != "entity" else 1.08)
    spr = _block(lines, st["family"], px, st["weight"], st["tracking"], pitch, "center", st["ss"])
    pad = int(math.ceil(10 * s)) + 2
    a = cv2.copyMakeBorder(np.ascontiguousarray(spr.a), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    r = max(1, int(round(1.6 * s)))
    stroke = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    stroke = cv2.GaussianBlur(stroke, (0, 0), 0.6 * s + 0.3)
    shadow = cv2.GaussianBlur(a, (0, 0), 6.0 * s)
    sh = int(round(3 * s))
    shadow = np.roll(shadow, sh, axis=0)
    col = np.zeros(a.shape + (3,), np.float32)
    al = np.zeros(a.shape, np.float32)
    for m, c, o in ((shadow, INK_BLACK, 0.62), (stroke, INK_BLACK, 0.58), (a, st["color"], 1.0)):
        mm = m * np.float32(o)
        col = col * (1.0 - mm[..., None]) + np.asarray(c, np.float32) * mm[..., None]
        al = al * (1.0 - mm) + mm
    # placement : centré dans la zone sûre, dernière ligne de base à SUB_BASELINE·H
    n = len(lines)
    last_base = spr.base + (n - 1) * pitch + pad
    cx = (x0s + x1s) / 2.0
    ink_x0, _, ink_x1, _ = spr.ink
    x = int(round(cx - (ink_x0 + ink_x1) / 2.0 - pad))   # centré dans la zone sûre (largeur ≤ max_w)
    y = int(round(SUB_BASELINE * H - last_base))
    col.flags.writeable = False
    al.flags.writeable = False
    return col, al, x, y


def draw_subtitle(img, text: str, kind: str = "dialogue", alpha: float = 1.0):
    """Sous-titre discret dans la zone sûre TikTok (x∈[60, 900]·W/1080, base ≤ 0,78·H, ≤ 2 lignes).
    kind : 'dialogue' (vraies répliques), 'narrator' (italique blanc cassé), 'entity' (italique rouge)."""
    out = _f32(img).copy()
    if not text or not text.strip() or alpha <= 0.0:
        return out
    k = _SUB_KIND_ALIASES.get(kind, kind)
    k = k if k in _SUB_STYLES else "dialogue"
    H, W = out.shape[:2]
    col, al, x, y = _sub_sprite(text, k, W, H)
    h, w = al.shape
    X0, Y0, X1, Y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if X1 <= X0 or Y1 <= Y0:
        return out
    a = al[Y0 - y:Y1 - y, X0 - x:X1 - x]
    c = col[Y0 - y:Y1 - y, X0 - x:X1 - x]
    roi = out[Y0:Y1, X0:X1]
    g = np.float32(_clamp(alpha))
    roi *= (1.0 - a * g)[..., None]
    roi += c * g
    return out


# ============================================================================= HUD de reconnaissance faciale
HUD_LOCK_S = 0.30
_HUD_LM_IDX = None


def _hud_landmark_subset(n: int) -> np.ndarray:
    """Sous-ensemble régulier et stable des 478 points (≈ 1 sur 3, + contour des yeux/lèvres)."""
    global _HUD_LM_IDX
    if _HUD_LM_IDX is None or _HUD_LM_IDX[0] != n:
        idx = np.arange(0, n, 3)
        _HUD_LM_IDX = (n, idx)
    return _HUD_LM_IDX[1]


def _split_label(label: str, family: str, size: float, max_w: float) -> list[str]:
    if _text_width(label, family, size, 500, 0.06) <= max_w:
        return [label]
    parts = [p.strip() for p in label.split("—")]
    if len(parts) < 2:
        return [label]
    lines = [parts[0]]
    for p in parts[1:]:
        cand = lines[-1] + " — " + p
        if _text_width(cand, family, size, 500, 0.06) <= max_w:
            lines[-1] = cand
        else:
            lines.append(p)
    return lines


def _auto_sublabel(label: str, victim: bool) -> str:
    if victim:
        return "CIBLE VERROUILLÉE"
    v = 96.0 + 3.9 * _rand(_str_seed(label), 7)
    return f"CONFIANCE {v:.1f} %".replace(".", ",")


def draw_hud_face(img, bbox, t_since_lock: float, label: str, sublabel: str | None, frame_idx: int,
                  mode: str = "identify", landmarks=None, label_side: str | None = None):
    """HUD de reconnaissance faciale autour d'un visage.

    bbox : (x0, y0, x1, y1) en px de `img` ; t_since_lock : temps depuis le début du verrouillage (s ; entre
    −0,3 et 0 : recherche) ; label : ex. « SUJET 01 — DOGAN — IDENTIFIÉ » (s'écrit lettre par lettre) ;
    sublabel : petite ligne sous l'étiquette (None -> automatique, "" -> aucune) ; mode 'identify' (blanc
    froid) ou 'victim' (rouge, glitch) ; landmarks : (N, 2) px (MediaPipe 478 points) -> points du maillage.
    label_side (extension) : 'below' | 'above' | None (auto : dessous s'il y a la place) — utile pour
    alterner les étiquettes de visages voisins.
    """
    out = _f32(img).copy()
    t = float(t_since_lock)
    if bbox is None or t < -0.3:
        return out
    H, W = out.shape[:2]
    s = W / REF_W
    victim = mode == "victim"
    acc = HUD_RED if victim else HUD_WHITE
    fi = int(frame_idx)
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return out
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    pad = 0.17 * max(bw, bh)
    tw, th = bw + 2 * pad, bh + 2 * pad
    lock = 0.18 if victim else HUD_LOCK_S
    glitchy = victim and (t < 0.16 or _rand(0x7A, fi) < 0.22)

    # --- cadre : se resserre en ~0,3 s (léger dépassement), clignote pendant le verrouillage
    if t < 0:
        k = 1.75 + 0.06 * math.sin(fi * 1.7)
        box_a = 0.30 + 0.15 * (fi % 2)
    else:
        p = _clamp(t / lock)
        k = 1.0 + (0.55 if not victim else 0.22) * (1.0 - _ease_out_back(p, 2.2))
        box_a = 1.0 if p >= 1.0 else (1.0 if (fi // 2) % 2 == 0 else 0.55)
    jx = jy = 0.0
    if glitchy:
        jx = (2 * _rand(0x31, fi) - 1) * 7 * s
        jy = (2 * _rand(0x32, fi) - 1) * 3 * s
    hw, hh = tw * k / 2.0, th * k / 2.0
    bx0, by0, bx1, by1 = cx - hw + jx, cy - hh + jy, cx + hw + jx, cy + hh + jy

    # --- étiquette (écriture lettre par lettre) -----------------------------------------------
    lab_size = 19.0 * s
    lab_fam = "plexmono"
    max_lab_w = max(1.05 * tw, 165 * s)        # étroite : les visages voisins ne se chevauchent pas
    lab_lines = _split_label(label.upper(), lab_fam, lab_size, max_lab_w) if label else []
    total_chars = sum(len(l) for l in lab_lines)
    t_type0 = 0.03 if victim else 0.16
    cps = 75.0 if victim else 58.0
    n_typed = int(max(0.0, t - t_type0) * cps) if t >= 0 else 0
    n_typed = min(n_typed, total_chars)
    typing_done_t = t_type0 + total_chars / cps
    sub_txt = _auto_sublabel(label or "", victim) if sublabel is None else sublabel
    sub_size = 14.5 * s
    line_h = 25.0 * s
    padx, pady = 9.0 * s, 6.0 * s

    # --- zone de travail locale (ROI) ---------------------------------------------------------
    full_lab_w = max([_text_width(l, lab_fam, lab_size, 500, 0.06) for l in lab_lines] +
                     [_text_width(sub_txt.upper(), lab_fam, sub_size, 400, 0.08) if sub_txt else 0.0])
    pill_w_max = full_lab_w + 2 * padx + 12 * s
    pill_h_full = len(lab_lines) * line_h + (line_h * 0.8 if sub_txt else 0.0) + 2 * pady
    fx0, fy0 = cx - tw / 2, cy - th / 2
    fy1 = cy + th / 2
    sx0, sy0, sx1, sy1 = safe_zone(W, H)
    below_ok = fy1 + 10 * s + pill_h_full <= min(sy1, H - 2)
    side = label_side
    if side is None and bw < 0.13 * W:  # petits visages voisins (groupe) : étiquettes paires au-dessus
        num = "".join(ch if ch.isdigit() else " " for ch in (label or "")).split()
        side = "above" if num and int(num[0]) % 2 == 0 else None
    if side == "above" and fy0 - 10 * s - pill_h_full >= sy0:
        below_ok = False
    elif side == "below" and fy1 + 10 * s + pill_h_full <= H - 2:
        below_ok = True
    pill_y = fy1 + 10 * s if below_ok else fy0 - 10 * s - pill_h_full
    pill_x = min(max(fx0, sx0), max(sx0, sx1 - pill_w_max))
    rx0 = int(math.floor(min(bx0, pill_x) - 16 * s))
    ry0 = int(math.floor(min(by0, pill_y) - 16 * s))
    rx1 = int(math.ceil(max(bx1, pill_x + pill_w_max) + 16 * s))
    ry1 = int(math.ceil(max(by1, pill_y + pill_h_full) + 16 * s))
    rx0, ry0, rx1, ry1 = max(rx0, 0), max(ry0, 0), min(rx1, W), min(ry1, H)
    if rx1 - rx0 < 2 or ry1 - ry0 < 2:
        return out
    rw, rh = rx1 - rx0, ry1 - ry0
    a_acc = np.zeros((rh, rw), np.uint8)   # traits (couleur d'accent)
    a_thin = np.zeros((rh, rw), np.uint8)  # rectangle fin, points du maillage
    SH = 4
    F = float(1 << SH)

    def P(x, y):
        return (int(round((x - rx0) * F)), int(round((y - ry0) * F)))

    thick = max(1, int(round(2.0 * s)))
    cl = min(max(0.22 * min(bw, bh) * k, 12 * s), 58 * s)
    v = int(255 * box_a)
    for (px_, py_, dx_, dy_) in ((bx0, by0, 1, 1), (bx1, by0, -1, 1), (bx0, by1, 1, -1), (bx1, by1, -1, -1)):
        pts = np.array([P(px_ + dx_ * cl, py_), P(px_, py_), P(px_, py_ + dy_ * cl)], np.int32)
        cv2.polylines(a_acc, [pts], False, v, thick, cv2.LINE_AA, SH)
    if t >= 0:
        # petites graduations au milieu des côtés + rectangle fin
        tk = 7 * s
        mx, my = (bx0 + bx1) / 2, (by0 + by1) / 2
        for a_, b_ in (((mx, by0), (mx, by0 + tk)), ((mx, by1), (mx, by1 - tk)),
                       ((bx0, my), (bx0 + tk, my)), ((bx1, my), (bx1 - tk, my))):
            cv2.line(a_acc, P(*a_), P(*b_), int(v * 0.8), max(1, thick - 1), cv2.LINE_AA, SH)
        cv2.rectangle(a_thin, P(bx0, by0), P(bx1, by1), int(60 * box_a), 1, cv2.LINE_AA, SH)

    # --- ligne de balayage avec traîne ---------------------------------------------------------
    scan_y = None
    if t >= 0:
        ts = (t - 0.04) / (0.55 if not victim else 0.32)
        if 0.0 <= ts <= 1.0:
            scan_y = by0 + (by1 - by0) * _ease_in_out(ts)
        elif not victim and t > 1.2 and ((t - 1.2) % 1.8) < 0.6:  # balayages lents ensuite
            scan_y = by0 + (by1 - by0) * _ease_in_out(((t - 1.2) % 1.8) / 0.6)
        if scan_y is not None:
            yi = int(round(scan_y - ry0))
            xa, xb = int(round(bx0 - rx0 + thick)), int(round(bx1 - rx0 - thick))
            trail = int(max(4, 0.22 * (by1 - by0)))
            ya = max(0, yi - trail)
            if 0 <= yi < rh and xb > xa:
                grad = (np.linspace(0, 1, yi - ya + 1, dtype=np.float32) ** 2.2 * 46).astype(np.uint8)
                xa_, xb_ = max(xa, 0), min(xb, rw)
                np.maximum(a_thin[ya:yi + 1, xa_:xb_], grad[:, None], out=a_thin[ya:yi + 1, xa_:xb_])
                cv2.line(a_acc, P(bx0 + thick, scan_y), P(bx1 - thick, scan_y), 200, max(1, int(round(1.4 * s))),
                         cv2.LINE_AA, SH)

    # --- points du maillage --------------------------------------------------------------------
    if landmarks is not None and t >= 0:
        lm = np.asarray(landmarks, np.float32).reshape(-1, 2)
        if lm.shape[0] > 0:
            sub = lm[_hud_landmark_subset(lm.shape[0])]
            vis_base = 0.85 if t < 0.9 else max(0.35, 0.85 - (t - 0.9) * 0.8)
            r_dot = max(0.9, 1.35 * s)
            for (lx, ly) in sub:
                if scan_y is not None and ly > scan_y and t < 0.7:
                    continue
                val = int(255 * vis_base * (0.75 + 0.25 * _rand(int(lx * 7 + ly * 13), fi // 3)))
                cv2.circle(a_thin, P(lx + jx, ly + jy), int(r_dot * F), val, -1, cv2.LINE_AA, SH)

    # --- pastille ------------------------------------------------------------------------------
    a_bg = np.zeros((rh, rw), np.float32)
    a_txt = np.zeros((rh, rw), np.float32)
    if t >= 0 and lab_lines and (n_typed > 0 or (t > t_type0 - 0.05)):
        typed, left = [], n_typed
        for l in lab_lines:
            typed.append(l[:max(0, left)])
            left -= len(l)
        widths = [_text_width(tl, lab_fam, lab_size, 500, 0.06) if tl else 0.0 for tl in typed]
        cur_line = max(0, min(len(lab_lines) - 1, sum(1 for tl in typed if tl) - 1))
        show_sub = bool(sub_txt) and t > typing_done_t + 0.08
        sub_a = _clamp((t - typing_done_t - 0.08) / 0.2) if show_sub else 0.0
        sub_w = _text_width(sub_txt.upper(), lab_fam, sub_size, 400, 0.08) if show_sub else 0.0
        cur_w = 9.0 * s
        pw = max(max(widths) + cur_w + 4 * s, sub_w) + 2 * padx
        nl = max(1, sum(1 for tl in typed if tl))
        ph = nl * line_h + (line_h * 0.8 if show_sub else 0.0) + 2 * pady
        px0, py0 = pill_x - rx0, pill_y - ry0
        if not below_ok:  # pastille au-dessus : elle grandit vers le haut
            py0 = (pill_y + pill_h_full - ph) - ry0
        a_bg_u8 = np.zeros((rh, rw), np.uint8)
        cv2.rectangle(a_bg_u8, P(px0 + rx0, py0 + ry0), P(px0 + rx0 + pw, py0 + ry0 + ph), 255, -1, cv2.LINE_AA, SH)
        a_bg = a_bg_u8.astype(np.float32) * np.float32(1 / 255)
        cv2.rectangle(a_thin, P(px0 + rx0, py0 + ry0), P(px0 + rx0 + pw, py0 + ry0 + ph), 110, 1, cv2.LINE_AA, SH)
        # liseré vertical d'accent à gauche de la pastille
        cv2.line(a_acc, P(px0 + rx0, py0 + ry0), P(px0 + rx0, py0 + ry0 + ph), 255, max(1, thick), cv2.LINE_AA, SH)
        for i, tl in enumerate(typed):
            if not tl:
                continue
            sp = _line(tl, lab_fam, lab_size, 500, 0.06, 1)
            bx = int(round(px0 + padx - sp.ox))
            byl = int(round(py0 + pady + line_h * (i + 0.78) - sp.base))
            _paste_max(a_txt, sp.a, bx, byl)
        # curseur bloc clignotant
        if t < typing_done_t + 0.45 and ((t * 8.0) % 2.0) < 1.25:
            cw = widths[cur_line]
            cxl = px0 + padx + cw + 2 * s
            cyb = py0 + pady + line_h * (cur_line + 0.78)
            x_a, x_b = int(round(cxl)), int(round(cxl + 8 * s))
            y_a, y_b = int(round(cyb - 14 * s)), int(round(cyb + 2 * s))
            a_txt[max(0, y_a):max(0, y_b), max(0, x_a):max(0, x_b)] = 0.9
        if show_sub:
            sp = _line(sub_txt.upper(), lab_fam, sub_size, 400, 0.08, 1)
            bx = int(round(px0 + padx - sp.ox))
            byl = int(round(py0 + pady + line_h * nl + line_h * 0.55 - sp.base))
            _paste_max(a_txt, sp.a * np.float32((0.85 if victim else 0.7) * sub_a), bx, byl)

    acc_f = a_acc.astype(np.float32) * np.float32(1 / 255)
    thin_f = a_thin.astype(np.float32) * np.float32(1 / 255)

    # --- glitch (victime) : décalage de bandes horizontales + fantôme décalé -----------------
    if glitchy:
        rng = np.random.default_rng(_hash(0x91, fi) & 0xFFFFFFFF)
        for _ in range(int(rng.integers(2, 6))):
            y_a = int(rng.integers(0, rh))
            y_b = min(rh, y_a + int(rng.integers(2, max(3, int(26 * s)))))
            d = int(rng.normal(0, 14 * s))
            for arr in (acc_f, thin_f, a_bg, a_txt):
                arr[y_a:y_b] = np.roll(arr[y_a:y_b], d, axis=1)
    # --- composition en une passe : fond de pastille, halo et traits, fantôme, texte ------------
    #   out = roi·T + Σ couleur_i·w_i  (T = transmission), les w_i combinés par cv2.transform
    bg_col = HUD_RED_FILL if victim else INK_BLACK
    a0 = a_bg * np.float32(0.86 if victim else 0.62)
    a1 = np.maximum(acc_f, thin_f * np.float32(0.9))
    a2 = a_txt
    glow = _fast_blur(acc_f, 3.0 * s + 0.5) * np.float32(0.35 if victim else 0.22)
    gh = (np.roll(acc_f, int(5 * s) + 1, axis=1) * np.float32(0.16)) if glitchy else None
    t12 = (1.0 - a1) * (1.0 - a2)
    w_bg = a0 * t12
    w_acc = a1 * (1.0 - a2) + glow
    T = (1.0 - a0) * t12
    txt_col = np.array([1.0, 0.94, 0.92], np.float32) if victim else OFF_WHITE
    cols = [bg_col, acc, txt_col]
    ws = [w_bg, w_acc, a2]
    if gh is not None:  # fantôme cyan décalé du tracé (glitch)
        T = T * (1.0 - gh)
        cols.append(np.array([0.1, 0.9, 0.95], np.float32))
        ws.append(gh)
    C = cv2.transform(cv2.merge(ws[:3]), np.stack(cols[:3], 1).astype(np.float32))
    if len(ws) > 3:
        C += ws[3][..., None] * cols[3]
    roi = out[ry0:ry1, rx0:rx1]
    roi *= T[..., None]
    roi += C
    np.minimum(roi, 1.0, out=roi)
    return out


def _paste_max(dst: np.ndarray, a: np.ndarray, x: int, y: int) -> None:
    H, W = dst.shape
    h, w = a.shape
    X0, Y0, X1, Y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if X1 > X0 and Y1 > Y0:
        np.maximum(dst[Y0:Y1, X0:X1], a[Y0 - y:Y1 - y, X0 - x:X1 - x], out=dst[Y0:Y1, X0:X1])


# ============================================================================= cercles rouges (feutre)
class _Path(NamedTuple):
    xy: np.ndarray     # (N, 2) points en unités normalisées (rayon ~1)
    u: np.ndarray      # (N,) abscisse curviligne normalisée 0..1
    w: np.ndarray      # (N,) facteur d'épaisseur
    ink: np.ndarray    # (N,) densité d'encre 0..1
    loop: np.ndarray   # (N,) index de passage (les recouvrements entre passages foncent l'encre)
    rot: float


@functools.lru_cache(maxsize=128)
def _marker_path(seed: int, frantic: bool) -> _Path:
    rng = np.random.default_rng((seed * 7919 + (1 if frantic else 0)) & 0xFFFFFFFF)
    if frantic:
        turns, n = rng.uniform(3.9, 4.7), 2600
    else:
        turns, n = rng.uniform(1.86, 2.12), 1400
    direction = -1.0 if rng.random() < 0.75 else 1.0          # surtout anti-horaire (droitier)
    th0 = rng.uniform(-0.80, -0.35) * math.pi                  # départ en haut
    q = np.linspace(0.0, 1.0, n, dtype=np.float64)
    th = th0 + direction * 2.0 * math.pi * turns * q
    # ellipse irrégulière : ondulations basses fréquences différentes à chaque boucle
    amp = 0.05 if frantic else 0.038
    r = np.ones(n)
    for kf in (2, 3):
        r += amp / kf ** 0.5 * np.sin(kf * th + rng.uniform(0, 2 * math.pi))
    r += (0.045 if frantic else 0.035) * np.sin(2 * math.pi * turns * q * rng.uniform(0.45, 0.6) + rng.uniform(0, 6.3))
    # dérive du centre et des axes d'une boucle à l'autre
    knots = int(math.ceil(turns)) + 2
    kq = np.linspace(0, 1, knots)
    spread = 0.10 if frantic else 0.035
    cxk, cyk = rng.normal(0, spread, knots), rng.normal(0, spread, knots)
    sxk, syk = 1 + rng.normal(0, spread * 1.1, knots), 1 + rng.normal(0, spread * 1.1, knots)
    cxq, cyq = np.interp(q, kq, cxk), np.interp(q, kq, cyk)
    sxq, syq = np.interp(q, kq, sxk), np.interp(q, kq, syk)
    tilt = np.interp(q, kq, rng.normal(0, 0.16 if frantic else 0.0, knots))
    # attaque (le feutre se pose un peu à l'intérieur) et dépassement final vers l'extérieur
    r *= 1.0 - 0.07 * (1.0 - _sstep(0.0, 0.04, q))
    tail = _sstep(0.90 if not frantic else 0.94, 1.0, q)
    r *= 1.0 + (0.16 if not frantic else 0.30) * tail ** 1.6
    ex, ey = r * sxq * np.cos(th), r * syq * np.sin(th)
    x = cxq + ex * np.cos(tilt) - ey * np.sin(tilt)
    y = cyq + ex * np.sin(tilt) + ey * np.cos(tilt)
    xy = np.stack([x, y], 1)
    seg = np.hypot(np.diff(x), np.diff(y))
    u = np.concatenate([[0.0], np.cumsum(seg)])
    u /= u[-1]
    # pression : pose, plein, levée ; encre : sèche un peu sur la fin, plus sale en frénétique
    w = (0.55 + 0.45 * _sstep(0.0, 0.035, u)) * (1.0 - 0.5 * _sstep(0.9, 1.0, u))
    w *= 1.0 + 0.10 * np.sin(2 * math.pi * (u * rng.uniform(5, 9) + rng.uniform(0, 1)))
    w *= 1.0 + 0.05 * np.sin(2 * math.pi * (u * rng.uniform(55, 80) + rng.uniform(0, 1)))  # bord irrégulier
    ink = 0.95 - 0.10 * u + 0.05 * np.sin(2 * math.pi * (u * rng.uniform(3, 6) + rng.uniform(0, 1)))
    ink *= 1.0 - 0.35 * _sstep(0.93, 1.0, u)
    if frantic:  # plus fin et plus pâle là où la main va vite (côtés longs), encre qui sèche
        speed = np.abs(np.sin(th + rng.uniform(0, 6.3)))
        w *= 0.72 + 0.28 * (1.0 - speed) + 0.08 * np.sin(2 * math.pi * u * rng.uniform(9, 14))
        ink *= 0.80 + 0.20 * (1.0 - speed)
    loop = np.floor((th - th0) * direction / (2 * math.pi) + 1e-9).astype(np.int32)
    rot = float(rng.uniform(-0.22, 0.22))
    return _Path(xy.astype(np.float32), u.astype(np.float32), w.astype(np.float32),
                 np.clip(ink, 0, 1).astype(np.float32), loop, rot)


@functools.lru_cache(maxsize=32)
def _marker_texture(seed: int, h: int, w: int, s: float) -> np.ndarray:
    """Texture périodique (tuile h x w, raccord invisible) : grain fin + taches d'encre."""
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f2 = fx * fx + fy * fy

    def band(sigma):
        n = np.fft.fft2(rng.standard_normal((h, w)))
        out = np.real(np.fft.ifft2(n * np.exp(-2.0 * (np.pi * sigma) ** 2 * f2)))
        return out / (out.std() + 1e-9)

    tex = np.clip(0.93 - 0.07 * band(0.7 * s + 0.3) - 0.06 * band(7.0 * s + 1.0), 0.6, 1.0).astype(np.float32)
    tex.flags.writeable = False
    return tex


def draw_red_circle(img, center_xy, axes_xy, progress: float, seed: int, frantic: bool = False):
    """Cercle rouge tracé à la main au feutre autour de center_xy (px), demi-axes axes_xy (px).
    progress 0→1 anime le tracé ; ellipse irrégulière à ~2 boucles avec dépassement (frantic : 4 boucles
    rapides et sales). Épaisseur/opacité variables, texture, recouvrements plus denses."""
    out = _f32(img).copy()
    prog = _clamp(float(progress))
    if prog <= 0.0:
        return out
    H, W = out.shape[:2]
    s = W / REF_W
    cx, cy = float(center_xy[0]), float(center_xy[1])
    ax, ay = max(2.0, float(axes_xy[0])), max(2.0, float(axes_xy[1]))
    P = _marker_path(int(seed), bool(frantic))
    # le feutre démarre un peu lentement puis accélère (frénétique : vitesse constante)
    ue = prog if frantic else (0.35 * prog * prog + 0.65 * prog) if prog < 1 else 1.0
    n = int(np.searchsorted(P.u, ue, side="right"))
    if n < 2:
        return out
    cr, sr = math.cos(P.rot), math.sin(P.rot)
    X = P.xy[:n, 0] * ax
    Y = P.xy[:n, 1] * ay
    pts = np.stack([cx + X * cr - Y * sr, cy + X * sr + Y * cr], 1)
    size_k = math.sqrt(_clamp(((ax + ay) / 2) / (115.0 * s), 0.6, 1.5))
    base_w = (6.4 if frantic else 8.6) * s * size_k
    wid = P.w[:n] * base_w
    ink = P.ink[:n]
    loops = P.loop[:n]
    # ROI sur-échantillonnée (rendu sans couture puis réduction INTER_AREA)
    m = float(wid.max()) + 4.0
    rx0, ry0 = int(math.floor(pts[:, 0].min() - m)), int(math.floor(pts[:, 1].min() - m))
    rx1, ry1 = int(math.ceil(pts[:, 0].max() + m)), int(math.ceil(pts[:, 1].max() + m))
    rx0c, ry0c, rx1c, ry1c = max(rx0, 0), max(ry0, 0), min(rx1, W), min(ry1, H)
    if rx1c <= rx0c or ry1c <= ry0c:
        return out
    rw, rh = rx1 - rx0, ry1 - ry0
    SS = 3 if max(rw, rh) < 700 else 2
    SH = 3
    F = SS * (1 << SH)
    tang = np.gradient(pts, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    left = ((pts + nrm * (wid[:, None] / 2) - [rx0, ry0]) * F).astype(np.int32)
    right = ((pts - nrm * (wid[:, None] / 2) - [rx0, ry0]) * F).astype(np.int32)
    centers = ((pts - [rx0, ry0]) * F).astype(np.int32)
    passes = []
    for lp in np.unique(loops):
        idx = np.flatnonzero(loops == lp)
        mk = np.zeros((rh * SS, rw * SS), np.uint8)
        step = 10
        for i0 in range(idx[0], idx[-1], step):
            i1 = min(i0 + step, idx[-1])
            poly = np.concatenate([left[i0:i1 + 1], right[i0:i1 + 1][::-1]], 0)
            cv2.fillPoly(mk, [poly], int(255 * float(ink[i0:i1 + 1].mean())), cv2.LINE_8, SH)
        for i in (idx[0], idx[-1]):  # bouts ronds
            cv2.circle(mk, (int(centers[i, 0]), int(centers[i, 1])), int(wid[i] / 2 * F), int(255 * float(ink[i])),
                       -1, cv2.LINE_8, SH)
        passes.append(cv2.resize(mk, (rw, rh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
    keep = np.ones((rh, rw), np.float32)
    for a in passes:
        keep *= 1.0 - a * np.float32(0.92)
    A = 1.0 - keep                                 # couverture cumulée (recouvrements plus denses)
    dens = sum(passes) if len(passes) > 1 else passes[0]
    # texture du feutre : tuile fixe (graine), indexée en coordonnées image -> stable pendant le tracé
    tex = _marker_texture(int(seed) * 31 + (7 if frantic else 0), 256, 256, s)
    ty = np.arange(ry0, ry1) % 256
    tx = np.arange(rx0, rx1) % 256
    tex = tex[ty[:, None], tx[None, :]]
    edge = np.clip(A - cv2.GaussianBlur(A, (0, 0), 1.1 * s + 0.4), 0, 1)
    A2 = np.clip(A * tex * np.float32(0.95) + edge * np.float32(0.9), 0, 0.97)
    mixd = np.clip((dens - 1.0) * 1.4, 0, 1) * np.float32(0.8) + edge * np.float32(0.5)
    sl = (slice(ry0c - ry0, ry1c - ry0), slice(rx0c - rx0, rx1c - rx0))
    a_ = A2[sl]
    w_dark = a_ * mixd[sl]
    w_red = a_ - w_dark
    roi = out[ry0c:ry1c, rx0c:rx1c]
    roi *= (1.0 - a_)[..., None]
    roi += cv2.transform(cv2.merge([w_red, w_dark, np.zeros_like(a_)]),
                         np.stack([MARKER_RED, MARKER_DARK, np.zeros(3, np.float32)], 1))
    return out


# ============================================================================= ligne sous le titre
@functools.lru_cache(maxsize=1)
def _title_meta() -> dict:
    p = config.TITLE_DIR / "title_meta.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def draw_title_sub(img, text: str, t: float, dur: float):
    """Ligne sous le titre 3D (« INSPIRÉ DE FAITS RÉELS — 30.10.2026 ») : serif capitales + date en mono,
    séparées par un filet ; ancrée sous la boîte du titre (assets/title/title_meta.json, sub_anchor)."""
    out = _f32(img).copy()
    if dur <= 0 or not (0.0 <= t <= dur) or not text.strip():
        return out
    H, W = out.shape[:2]
    s = W / REF_W
    meta = _title_meta()
    mw, mh = (meta.get("size") or [REF_W, REF_H])[:2]
    anc = meta.get("sub_anchor") or [REF_W / 2, 1015]
    cx = W / 2.0 if abs(anc[0] * W / mw - W / 2.0) < 0.06 * W else anc[0] * W / mw
    top = float(anc[1]) * H / mh
    alpha, blur, glow = _reveal(t, dur, 0.9, 0.45, _str_seed(text), 0.8)
    if alpha <= 0.003:
        return out
    parts = [_typo(p.strip(), CARD_FAMILY) for p in text.upper().replace(" - ", " — ").split("—")]
    head, tail = parts[0], " — ".join(parts[1:]) if len(parts) > 1 else ""
    lines, px = _fit(head, CARD_FAMILY, 36.0 * s, CARD_WEIGHT, 0.32, 0.64 * W)
    spr = _block(lines, CARD_FAMILY, px, CARD_WEIGHT, 0.32, 1.45 * px)
    base1 = top + spr.cap
    m, X, Y = _warp(spr.a, spr.ox + spr.adv / 2, spr.base, cx, base1, 1.0 + 0.02 * t / dur, blur * s)
    soft = _fast_blur(m, 6.5 * s)
    _composite(out, X, Y, [(soft, INK_BLACK, 0.5 * alpha, "over"),
                           (soft, OFF_WHITE, 0.5 * glow * alpha if glow > 0.01 else 0.0, "add"),
                           (m, OFF_WHITE, alpha, "over")])
    last_base = base1 + (len(lines) - 1) * 1.45 * px
    if tail:
        a2 = alpha * _clamp((t - 0.25) / 0.6)
        rw = 34 * s * _ease_out_cubic((t - 0.15) / 0.6)
        ry = last_base + 26 * s
        _hline(out, cx - rw / 2, cx + rw / 2, ry, max(1.0, 1.4 * s), OFF_WHITE, 0.7 * alpha)
        sp = _line(tail, "plexmono", 22.0 * s, 400, 0.40, 2)
        m2, X2, Y2 = _warp(sp.a, sp.ox + sp.adv / 2, sp.base, cx, ry + 40 * s, 1.0, blur * s * 0.7)
        _over(out, m2, OFF_WHITE, X2, Y2, 0.8 * a2)
    return out


# ============================================================================= étalonnage
_PRESETS: dict[str, dict] = {
    # look principal : froid, désaturé (~45 %), verts tués vers le bleu-vert, noirs écrasés avec pied de
    # courbe, ombres bleu-vert, hautes lumières légèrement cyan, rouges vifs préservés
    "cold": dict(
        wb=(0.94, 1.0, 1.05), exposure=0.0, sat=0.55, red_sat=1.0, blue_sat=0.5, green_sat=0.28, green_teal=0.45,
        red_shift=0.12, red_lo=0.55,
        curve=[(0, 0), (0.045, 0.0), (0.11, 0.028), (0.24, 0.13), (0.5, 0.44), (0.75, 0.74), (0.92, 0.89), (1, 0.94)],
        black_tint=(0.0, 0.006, 0.011), shadow_tint=(-0.028, 0.012, 0.034), mid_tint=(-0.02, 0.006, 0.013),
        high_tint=(-0.026, 0.006, 0.016), glow=0.0, bleach=0.0),
    # habitacle rouge : rouge gardé (cramoisi), le reste désaturé, noirs très écrasés
    "night_red": dict(
        wb=(1.0, 0.98, 1.0), exposure=0.0, sat=0.8, red_sat=1.08, blue_sat=0.45, green_sat=0.35, green_teal=0.2,
        red_shift=0.32, red_lo=0.45,
        curve=[(0, 0), (0.07, 0.0), (0.15, 0.022), (0.3, 0.13), (0.55, 0.45), (0.8, 0.79), (1, 0.93)],
        black_tint=(0.002, 0.0, 0.003), shadow_tint=(0.006, -0.004, 0.004), mid_tint=(0.0, -0.006, -0.002),
        high_tint=(0.0, -0.012, -0.008), glow=0.0, bleach=0.0),
    # accueil bleu : bleu gardé mais désaturé, ombres bleu-vert profondes
    "blue": dict(
        wb=(0.91, 0.99, 1.09), exposure=0.0, sat=0.55, red_sat=0.95, blue_sat=0.95, green_sat=0.3, green_teal=0.5,
        red_shift=0.1, red_lo=0.55,
        curve=[(0, 0), (0.055, 0.0), (0.13, 0.03), (0.26, 0.135), (0.5, 0.44), (0.76, 0.75), (1, 0.94)],
        black_tint=(0.0, 0.005, 0.016), shadow_tint=(-0.03, 0.01, 0.046), mid_tint=(-0.022, 0.004, 0.024),
        high_tint=(-0.022, 0.004, 0.02), glow=0.0, bleach=0.0),
    # Burger King : chaud, saturé, lumineux, léger glow
    "warm": dict(
        wb=(1.07, 1.0, 0.88), exposure=0.12, sat=1.14, red_sat=1.18, blue_sat=1.0, green_sat=1.0, green_teal=0.0,
        red_shift=0.0, red_lo=0.55,
        curve=[(0, 0.01), (0.1, 0.092), (0.3, 0.30), (0.5, 0.525), (0.75, 0.78), (0.92, 0.925), (1, 0.97)],
        black_tint=(0.01, 0.004, -0.003), shadow_tint=(0.012, 0.004, -0.008), mid_tint=(0.008, 0.003, -0.006),
        high_tint=(0.012, 0.006, -0.012), glow=0.2, bleach=0.0),
    # arrêts sur image / screamer : bleach bypass contrasté, désaturé, hautes lumières argentées
    "bleach": dict(
        wb=(0.97, 1.0, 1.03), exposure=0.0, sat=0.36, red_sat=0.75, blue_sat=0.36, green_sat=0.25, green_teal=0.3,
        red_shift=0.1, red_lo=0.55,
        curve=[(0, 0), (0.05, 0.0), (0.18, 0.065), (0.35, 0.25), (0.5, 0.48), (0.7, 0.76), (0.88, 0.94), (1, 1.0)],
        black_tint=(0.0, 0.003, 0.006), shadow_tint=(-0.008, 0.002, 0.012), mid_tint=(-0.004, 0.0, 0.004),
        high_tint=(-0.004, 0.0, 0.006), glow=0.0, bleach=0.6),
}
_LUT_N = 4096


def _tone_lut(curve: tuple, black_tint, shadow_tint, mid_tint, high_tint) -> np.ndarray:
    from scipy.interpolate import PchipInterpolator
    xs, ys = zip(*curve)
    x = np.linspace(0.0, 1.0, _LUT_N)
    y = np.clip(PchipInterpolator(xs, ys)(x), 0.0, 1.0)
    w_black = 1.0 - _sstep(0.0, 0.06, y)
    w_sh = _sstep(0.0, 0.08, y) * (1.0 - _sstep(0.1, 0.5, y))
    w_mid = _sstep(0.15, 0.45, y) * (1.0 - _sstep(0.55, 0.85, y))
    w_hi = _sstep(0.55, 0.95, y)
    lut = np.empty((3, _LUT_N), np.float32)
    for c in range(3):
        lut[c] = y + black_tint[c] * w_black + shadow_tint[c] * w_sh + mid_tint[c] * w_mid + high_tint[c] * w_hi
    return np.clip(lut, 0.0, 1.0)


@functools.lru_cache(maxsize=64)
def _preset_resolved(preset: str, kw_items: tuple) -> dict:
    if preset not in _PRESETS:
        raise ValueError(f"preset d'étalonnage inconnu : {preset!r} ({sorted(_PRESETS)} + warm_to_cold)")
    p = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in _PRESETS[preset].items()}
    kw = dict(kw_items)
    ra = float(kw.get("red_accent") or 0.0)
    if ra:
        p["red_sat"] = p["red_sat"] * (1.0 + 0.5 * ra)
        p["red_shift"] = p["red_shift"] + 0.12 * ra
        p["red_gain"] = 1.0 + 0.10 * ra
        p["red_lo"] = p.get("red_lo", 0.55) - 0.08 * ra
    kb = kw.get("keep_blue")
    if kb:
        kbf = 1.0 if kb is True else float(kb)
        p["blue_sat"] = _lerp(p["blue_sat"], 1.0, kbf)
        p["keep_blue_hue"] = kbf
    wh = float(kw.get("warm_highlights") or 0.0)
    if wh:
        p["high_tint"] = [p["high_tint"][0] + 0.07 * wh, p["high_tint"][1] + 0.022 * wh, p["high_tint"][2] - 0.06 * wh]
    if "exposure" in kw and kw["exposure"] is not None:
        p["exposure"] = p["exposure"] + float(kw["exposure"])
    if "saturation" in kw and kw["saturation"] is not None:
        p["sat"] = p["sat"] * float(kw["saturation"])
    p["lut"] = _tone_lut(tuple(map(tuple, p["curve"])), p["black_tint"], p["shadow_tint"], p["mid_tint"],
                         p["high_tint"])
    p["lut"].flags.writeable = False
    return p


def _grade_params(preset: str, mix: float, kw: dict) -> dict:
    kw_items = tuple(sorted((k, v) for k, v in kw.items() if k not in ("strength",)))
    if preset != "warm_to_cold":
        return _preset_resolved(preset, kw_items)
    m = round(_clamp(float(mix)) * 128) / 128.0
    a, b = _preset_resolved("warm", kw_items), _preset_resolved("cold", kw_items)
    out = {}
    for k in set(a) | set(b):
        va, vb = a.get(k, b.get(k)), b.get(k, a.get(k))
        if k == "curve":
            continue
        if isinstance(va, np.ndarray):
            out[k] = (va + (vb - va) * m).astype(np.float32)
        elif isinstance(va, list):
            out[k] = [x + (y - x) * m for x, y in zip(va, vb)]
        elif isinstance(va, (int, float)):
            out[k] = va + (vb - va) * m
        else:
            out[k] = vb if m >= 0.5 else va
    return out


# ----------------------------------------------------------------------------- exposition automatique
# Plans de nuit iPhone (très sombres, bruités) : sans normalisation, la courbe « noirs écrasés » les rend
# illisibles sur téléphone. Relèvement AVANT la courbe, par une pente dans les ombres plafonnée :
#   f(v) = g·v / (1 + (g − 1)·v)    (v = max(R, V, B) ; f(0) = 0, f(1) = 1 : rien ne sature, teinte conservée)
# g est choisi pour amener la médiane vers AE_TARGET_MED et le 90e centile vers AE_TARGET_HI (le plus petit
# des deux gains), g ≤ AE_MAX_GAIN, g ≥ 1 (une image bien exposée n'est jamais modifiée). Une image noire
# (90e centile < 0,02) reste noire. Stable dans le temps : exposure_gain(images) se calcule UNE fois par plan.
AE_TARGET_MED = 0.15
AE_TARGET_HI = 0.50
AE_MAX_GAIN = 4.0
AE_BLACK_P90 = (0.02, 0.05)    # 90e centile en dessous duquel l'image est considérée noire (pas de relèvement)
AE_NR_FROM = 1.3               # gain à partir duquel la chrominance est débruitée


def _value_samples(img) -> np.ndarray:
    """max(R, V, B) d'une version réduite (~120 px de large) de l'image, aplati."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) * np.float32(1.0 / 255.0)
    else:
        a = a.astype(np.float32, copy=False)
    h, w = a.shape[:2]
    f = max(1, int(w // 120))
    if f > 1:
        a = cv2.resize(a, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    return np.clip(a[..., :3].max(axis=-1), 0.0, 1.0).ravel()


def _gain_for(level: float, target: float) -> float:
    level = max(float(level), 1e-4)
    if level >= target:
        return 1.0
    return target * (1.0 - level) / (level * (1.0 - target))


def exposure_gain(sample_frames, preset: str | None = None, max_gain: float | None = None,
                  target_med: float = AE_TARGET_MED, target_hi: float = AE_TARGET_HI) -> float:
    """Gain de normalisation d'exposition (≥ 1) d'un plan, à calculer UNE fois par plan sur quelques images
    (liste d'images float [0,1] ou uint8 de n'importe quelle taille, ou une seule image), puis à passer à
    grade(..., auto_exposure=gain) pour toutes les images du plan (aucun pompage). preset : réservé (même
    calcul pour tous les looks filmés)."""
    if isinstance(sample_frames, np.ndarray) and sample_frames.ndim == 3:
        sample_frames = [sample_frames]
    vals = [_value_samples(f) for f in sample_frames if f is not None]
    if not vals:
        return 1.0
    v = np.concatenate(vals)
    p50, p90 = (float(q) for q in np.percentile(v, [50, 90]))
    g = min(_gain_for(p50, target_med), _gain_for(p90, target_hi), float(max_gain or AE_MAX_GAIN))
    g = 1.0 + (max(1.0, g) - 1.0) * float(_sstep(AE_BLACK_P90[0], AE_BLACK_P90[1], p90))
    return round(max(1.0, g), 3)


def _resolve_gain(x: np.ndarray, auto_exposure) -> float:
    if auto_exposure is None or auto_exposure is False:
        return 1.0
    if auto_exposure is True:        # repli image par image (le rendu devrait passer un gain fixe par plan)
        return exposure_gain([x])
    return max(1.0, min(float(auto_exposure), 2.0 * AE_MAX_GAIN))


# ----------------------------------------------------------------------------- étalonnage par pixel (référence)
def _curve_apply(c: np.ndarray, lut_row: np.ndarray) -> np.ndarray:
    n = lut_row.shape[0] - 1
    return np.interp(np.clip(c, 0.0, 1.0) * n, np.arange(n + 1, dtype=np.float32), lut_row).astype(np.float32)


def _grade_pixels(x: np.ndarray, p: dict, gain: float = 1.0) -> np.ndarray:
    """Étalonnage d'un tableau (..., 3) pixel par pixel (aucune opération spatiale) : référence exacte qui sert
    à construire les LUT 3D. Ordre : normalisation d'exposition -> balance/exposition -> saturation sélective
    (rouges vifs gardés, bleus, verts tués vers le bleu-vert) -> rouges vers le cramoisi -> bleach -> courbe."""
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    if gain > 1.0 + 1e-4:
        v = x.max(axis=-1, keepdims=True)
        x = x * (np.float32(gain) / (1.0 + np.float32(gain - 1.0) * v))
    gains = (np.asarray(p["wb"], np.float32) * np.float32(2.0 ** p["exposure"]))
    r, g, b = (x[..., i] * gains[i] for i in range(3))
    L = r * _LUMA[0, 0] + g * _LUMA[0, 1] + b * _LUMA[0, 2]
    sat = float(p["sat"])
    rlo = float(p.get("red_lo", 0.55))
    rmask = _sstep(rlo, rlo + 0.3, (r - np.maximum(g, b)) / (r + np.float32(0.02))) * _sstep(0.025, 0.10, r)
    s_map = np.float32(sat) + np.float32(p["red_sat"] - sat) * rmask
    if abs(p["blue_sat"] - sat) > 1e-3:
        bmask = _sstep(0.15, 0.55, (b - np.maximum(r, g)) / (b + np.float32(0.02))) * _sstep(0.03, 0.12, b)
        s_map = s_map + np.float32(p["blue_sat"] - sat) * bmask
    gsat = float(p.get("green_sat", sat))
    gteal = float(p.get("green_teal", 0.0))
    gmask = None
    if abs(gsat - sat) > 1e-3 or gteal > 0:
        gmask = _sstep(0.04, 0.35, (g - np.maximum(r, b)) / (g + np.float32(0.02))) * _sstep(0.02, 0.08, g)
        s_map = s_map + np.float32(gsat - sat) * gmask
    r, g, b = L + (r - L) * s_map, L + (g - L) * s_map, L + (b - L) * s_map
    if gmask is not None and gteal:
        delta = np.maximum(g - r, 0) * (gmask * np.float32(gteal))
        r = r - delta * np.float32(0.35)
        b = b + delta * np.float32(0.6)
    rs, rg = float(p.get("red_shift", 0.0)), float(p.get("red_gain", 1.0))
    if rs:
        k = rmask * np.float32(rs)
        g = g * (1.0 - k)
        b = b * (1.0 - 0.4 * k)
    if rg != 1.0:
        r = r * (1.0 + np.float32(rg - 1.0) * rmask)
    if p.get("bleach", 0.0) > 0:
        bl = np.float32(p["bleach"])
        L2 = np.clip(r * _LUMA[0, 0] + g * _LUMA[0, 1] + b * _LUMA[0, 2], 0, 1)
        out_ = []
        for ch in (r, g, b):
            c = np.clip(ch, 0, 1)
            ov = np.where(c < 0.5, 2 * c * L2, 1 - 2 * (1 - c) * (1 - L2))
            out_.append(ch + (ov - ch) * bl)
        r, g, b = out_
    lut = p["lut"]
    res = np.stack([_curve_apply(r, lut[0]), _curve_apply(g, lut[1]), _curve_apply(b, lut[2])], -1)
    return np.clip(res, 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------- LUT 3D (rendu rapide)
# Tout l'étalonnage par pixel (exposition auto comprise) est précalculé dans une LUT 3D 49³ par preset, sur
# une grille en racine carrée (finesse dans les ombres, où sont la courbe et le relèvement), appliquée en
# trilinéaire (noyau numba si disponible, sinon cv2.remap). Écart à la référence < 1/255 en pratique.
LUT3_N = 49
_MIX_STEPS = 64               # warm_to_cold : LUT tous les 1/64, interpolées entre elles
_GAIN_STEPS = 16              # exposition : LUT tous les 1/16 de diaph, interpolées


@functools.lru_cache(maxsize=40)
def _lut3(preset: str, kw_items: tuple, mix_q: int, gain_q: int, strength: float | None) -> np.ndarray:
    """LUT (N, N, N, 3) indexée [b, g, r], nœuds en racine carrée (x = u²)."""
    N = LUT3_N
    p = _grade_params(preset, mix_q / _MIX_STEPS, dict(kw_items))
    u = (np.arange(N, dtype=np.float32) / (N - 1)) ** 2
    bb, gg, rr = np.meshgrid(u, u, u, indexing="ij")
    lattice = np.stack([rr, gg, bb], -1).reshape(N * N, N, 3)
    out = _grade_pixels(lattice, p, 2.0 ** (gain_q / _GAIN_STEPS))
    if strength is not None and strength < 1.0:
        out = lattice + (out - lattice) * np.float32(strength)
    out = np.ascontiguousarray(out.reshape(N, N, N, 3), dtype=np.float32)
    out.flags.writeable = False
    return out


_LUT_BLEND: dict = {}


def _lut3_for(preset: str, kw_items: tuple, mix: float, gain: float, strength) -> np.ndarray:
    key = (preset, kw_items, round(mix, 5), round(gain, 5), strength)
    hit = _LUT_BLEND.get(key)
    if hit is not None:
        return hit
    mf = _clamp(float(mix)) * _MIX_STEPS if preset == "warm_to_cold" else 0.0
    gf = math.log2(max(1.0, gain)) * _GAIN_STEPS
    m0, g0 = int(math.floor(mf)), int(math.floor(gf))
    fm, fg = mf - m0, gf - g0
    acc, wsum = None, 0.0
    for dm, wm in ((0, 1.0 - fm), (1, fm)):
        for dg, wg in ((0, 1.0 - fg), (1, fg)):
            w = wm * wg
            if w < 1e-4:
                continue
            lt = _lut3(preset, kw_items, min(m0 + dm, _MIX_STEPS), g0 + dg, strength)
            acc = lt * np.float32(w) if acc is None else acc + lt * np.float32(w)
            wsum += w
    out = acc if abs(wsum - 1.0) < 1e-6 else acc / np.float32(wsum)
    if len(_LUT_BLEND) > 6:
        _LUT_BLEND.clear()
    _LUT_BLEND[key] = out
    return out


try:   # noyaux compilés (optionnels) : même résultat que les replis numpy/cv2, 3 à 5x plus rapides
    import numba as _nb
except Exception:  # pragma: no cover - numba absent : replis cv2
    _nb = None

if _nb is not None:
    @_nb.njit(cache=True, nogil=True, fastmath=True)
    def _nb_lut3(x, lut, chroma, nr, out):
        H, W = x.shape[0], x.shape[1]
        N = lut.shape[0]
        s = np.float32(N - 1)
        hc, wc = chroma.shape[0], chroma.shape[1]
        use_nr = nr > 0.0
        for i in range(H):
            for j in range(W):
                r = x[i, j, 0]
                g = x[i, j, 1]
                b = x[i, j, 2]
                if use_nr:   # chrominance remplacée par sa version lissée (basse résolution, bilinéaire)
                    L = np.float32(0.2126) * r + np.float32(0.7152) * g + np.float32(0.0722) * b
                    cy = (i + np.float32(0.5)) * hc / H - np.float32(0.5)
                    cx = (j + np.float32(0.5)) * wc / W - np.float32(0.5)
                    y0 = int(np.floor(cy))
                    x0 = int(np.floor(cx))
                    fy = cy - y0
                    fx = cx - x0
                    ya, yb = min(max(y0, 0), hc - 1), min(max(y0 + 1, 0), hc - 1)
                    xa, xb = min(max(x0, 0), wc - 1), min(max(x0 + 1, 0), wc - 1)
                    w00 = (1 - fy) * (1 - fx)
                    w01 = (1 - fy) * fx
                    w10 = fy * (1 - fx)
                    w11 = fy * fx
                    cr = (chroma[ya, xa, 0] * w00 + chroma[ya, xb, 0] * w01 + chroma[yb, xa, 0] * w10
                          + chroma[yb, xb, 0] * w11)
                    cg = (chroma[ya, xa, 1] * w00 + chroma[ya, xb, 1] * w01 + chroma[yb, xa, 1] * w10
                          + chroma[yb, xb, 1] * w11)
                    cb = (chroma[ya, xa, 2] * w00 + chroma[ya, xb, 2] * w01 + chroma[yb, xa, 2] * w10
                          + chroma[yb, xb, 2] * w11)
                    r = r + (L + cr - r) * nr
                    g = g + (L + cg - g) * nr
                    b = b + (L + cb - b) * nr
                r = min(max(r, np.float32(0.0)), np.float32(1.0))
                g = min(max(g, np.float32(0.0)), np.float32(1.0))
                b = min(max(b, np.float32(0.0)), np.float32(1.0))
                ur = np.sqrt(r) * s
                ug = np.sqrt(g) * s
                ub = np.sqrt(b) * s
                r0 = min(int(ur), N - 2)
                g0 = min(int(ug), N - 2)
                b0 = min(int(ub), N - 2)
                fr = ur - r0
                fg = ug - g0
                fb = ub - b0
                for c in range(3):
                    c00 = lut[b0, g0, r0, c] + (lut[b0, g0, r0 + 1, c] - lut[b0, g0, r0, c]) * fr
                    c01 = lut[b0, g0 + 1, r0, c] + (lut[b0, g0 + 1, r0 + 1, c] - lut[b0, g0 + 1, r0, c]) * fr
                    c10 = lut[b0 + 1, g0, r0, c] + (lut[b0 + 1, g0, r0 + 1, c] - lut[b0 + 1, g0, r0, c]) * fr
                    c11 = (lut[b0 + 1, g0 + 1, r0, c]
                           + (lut[b0 + 1, g0 + 1, r0 + 1, c] - lut[b0 + 1, g0 + 1, r0, c]) * fr)
                    c0 = c00 + (c01 - c00) * fg
                    c1 = c10 + (c11 - c10) * fg
                    out[i, j, c] = c0 + (c1 - c0) * fb


def _chroma_small(x: np.ndarray) -> np.ndarray:
    """Chrominance (RVB − luminance) lissée au 1/4 de résolution (bruit chromatique des capteurs de nuit)."""
    h, w = x.shape[:2]
    small = cv2.resize(x, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    L = cv2.transform(small, _LUMA)[..., None]
    c = small - L
    return np.ascontiguousarray(cv2.GaussianBlur(c, (0, 0), 1.1), dtype=np.float32)


def _apply_lut3_cv(x: np.ndarray, lut: np.ndarray, chroma, nr: float) -> np.ndarray:
    """Repli sans numba : même interpolation trilinéaire via deux cv2.remap sur un atlas 2D des tranches b."""
    N = lut.shape[0]
    H, W = x.shape[:2]
    if nr > 0 and chroma is not None:
        L = cv2.transform(x, _LUMA)[..., None]
        cu = cv2.resize(chroma, (W, H), interpolation=cv2.INTER_LINEAR)
        x = x + ((L + cu) - x) * np.float32(nr)
    u = np.clip(x, 0.0, 1.0)
    cv2.sqrt(u, u)
    u *= np.float32(N - 1)
    ur, ug, ub = cv2.split(u)
    b0 = np.minimum(ub.astype(np.int32), N - 2).astype(np.float32)
    fb = ub - b0
    mx = ur + b0 * np.float32(N)
    atlas = lut.transpose(1, 0, 2, 3).reshape(N, N * N, 3)    # [g, b·N + r]
    o0 = cv2.remap(atlas, mx, ug, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mx += np.float32(N)
    o1 = cv2.remap(atlas, mx, ug, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    o1 -= o0
    o1 *= fb[..., None]
    o0 += o1
    return o0


def grade(img, preset: str, mix: float = 0.0, auto_exposure=True, exact: bool = False, **kw):
    """Étalonnage. preset : cold | night_red | blue | warm | warm_to_cold (mix 0 = chaud -> 1 = froid) | bleach.
    kw : red_accent (0..1, rouges renforcés), keep_blue (bool/0..1, bleus gardés : gyrophares),
    warm_highlights (0..1, hautes lumières chaudes : bougies), exposure (diaph. en plus),
    saturation (multiplicateur), strength (0..1, mélange avec l'original).
    auto_exposure : normalisation d'exposition avant la courbe — True (défaut : gain estimé sur CETTE image,
    repli), un gain fixe du plan (float, cf. exposure_gain : recommandé, stable), ou False/None (désactivée).
    exact : calcul de référence pixel par pixel (lent) au lieu de la LUT 3D."""
    x = np.ascontiguousarray(_f32(img)[..., :3])
    H, W = x.shape[:2]
    gain = _resolve_gain(x, auto_exposure)
    st = kw.pop("strength", None)
    st = None if st is None or float(st) >= 1.0 else _clamp(float(st))
    kw_items = tuple(sorted((k, v) for k, v in kw.items()))
    p = _grade_params(preset, mix, kw)            # (valide le preset ; glow)
    nr = _clamp((gain - AE_NR_FROM) / 1.2) * 0.85 if gain > AE_NR_FROM else 0.0
    glow = float(p.get("glow", 0.0))
    st_in_lut = st if glow <= 0 else None
    if exact:
        xin = x
        if nr > 0:
            L = cv2.transform(x, _LUMA)[..., None]
            xin = x + ((L + cv2.resize(_chroma_small(x), (W, H), interpolation=cv2.INTER_LINEAR)) - x) * np.float32(nr)
        res = _grade_pixels(xin, p, gain)
        if st_in_lut is not None:
            res = np.clip(x + (res - x) * np.float32(st_in_lut), 0, 1)
    else:
        lut = _lut3_for(preset, kw_items, float(mix), gain, st_in_lut)
        chroma = _chroma_small(x) if nr > 0 else None
        if _nb is not None:
            res = np.empty_like(x)
            _nb_lut3(x, lut, chroma if chroma is not None else np.zeros((1, 1, 3), np.float32), np.float32(nr), res)
        else:
            res = _apply_lut3_cv(x, lut, chroma, nr)
    if glow > 0:  # halo diffus des hautes lumières (chaud)
        gl = np.float32(glow)
        sw, sh_ = max(8, W // 4), max(8, H // 4)
        small = cv2.resize(res, (sw, sh_), interpolation=cv2.INTER_AREA)
        bright = np.clip(small - np.float32(0.5), 0, None) * np.float32(2.0)
        bright = cv2.GaussianBlur(bright, (0, 0), 10.0 * sw / 270.0)
        big = cv2.resize(bright, (W, H), interpolation=cv2.INTER_LINEAR)
        res = res + big * (gl * np.array([1.0, 0.82, 0.58], np.float32))
        np.clip(res, 0.0, 1.0, out=res)
        if st is not None:
            res = np.clip(x + (res - x) * np.float32(st), 0.0, 1.0)
    return res


# ============================================================================= finition argentique
_GRAIN_BANK: dict = {}
_FINISH_MAPS: dict = {}
_GRAIN_TILES, _GRAIN_PAD = 4, 96
_gl = np.linspace(0.0, 1.0, 256, dtype=np.float32)
# poids du grain selon la luminance : maximal dans les tons moyens-sombres, discret dans les hautes lumières,
# faible plancher dans les noirs (noirs denses, pas de « neige » colorée sur les aplats noirs)
_GRAIN_W_LUT = (0.16 + 3.9 * _gl * (1 - _gl) * np.sqrt(1 - _gl) + 0.6 * _sstep(0.0, 0.08, _gl) * (1 - _gl)
                ).astype(np.float32)[None, :]
_GRAIN_W_LUT.flags.writeable = False
del _gl
GRAIN_CHROMA = 0.12     # part de grain chromatique (rouge/bleu en opposition)


def _grain_bank(h: int, w: int) -> list:
    key = (h, w)
    if key not in _GRAIN_BANK:
        s = w / REF_W
        tiles = []
        for i in range(_GRAIN_TILES):
            rng = np.random.default_rng(90210 + i)
            n = rng.standard_normal((h + _GRAIN_PAD, w + _GRAIN_PAD), dtype=np.float32)
            n = cv2.GaussianBlur(n, (0, 0), max(0.35, 0.62 * s))   # grain d'environ 1,5 px à 1080
            n *= np.float32(1.0 / (n.std() + 1e-6))
            tiles.append(n)
        if len(_GRAIN_BANK) > 3:
            _GRAIN_BANK.clear()
        _GRAIN_BANK[key] = tiles
    return _GRAIN_BANK[key]


def _grain_field(h: int, w: int, frame_idx: int, salt: int) -> np.ndarray:
    tiles = _grain_bank(h, w)
    hh = _hash(0x6A, frame_idx, salt)
    t = tiles[hh % _GRAIN_TILES]
    oy, ox = (hh >> 8) % _GRAIN_PAD, (hh >> 20) % _GRAIN_PAD
    v = t[oy:oy + h, ox:ox + w]
    if (hh >> 32) & 1:
        v = v[::-1]
    if (hh >> 33) & 1:
        v = v[:, ::-1]
    return v


def _finish_maps(h: int, w: int, vignette: float) -> dict:
    key = (h, w, round(vignette, 3))
    if key not in _FINISH_MAPS:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u = (xx + 0.5 - w / 2) / (w / 2)
        v = (yy + 0.5 - h / 2) / (h / 2)
        d = np.sqrt(u * u * 1.0 + v * v * 0.92)                 # ovale qui épouse le 9:16
        vig = 1.0 - np.float32(vignette) * _sstep(0.42, 1.38, d) ** 1.25
        rc = math.hypot(w / 2, h / 2)
        rr = np.sqrt((xx + 0.5 - w / 2) ** 2 + (yy + 0.5 - h / 2) ** 2) / rc
        amt = 2.6 * (w / REF_W) / rc * rr ** 2                  # aberration : ~2,6 px dans les coins à 1080
        dx, dy = (xx + 0.5 - w / 2), (yy + 0.5 - h / 2)
        maps = dict(
            vig=vig.astype(np.float32), ca_k=float(2.6 * (w / REF_W) / rc ** 3),
            rx=(w / 2 - 0.5 + dx * (1 + amt)).astype(np.float32), ry=(h / 2 - 0.5 + dy * (1 + amt)).astype(np.float32),
            bx=(w / 2 - 0.5 + dx * (1 - amt)).astype(np.float32), by=(h / 2 - 0.5 + dy * (1 - amt)).astype(np.float32))
        if len(_FINISH_MAPS) > 4:
            _FINISH_MAPS.clear()
        _FINISH_MAPS[key] = maps
    return _FINISH_MAPS[key]


if _nb is not None:
    @_nb.njit(cache=True, nogil=True, fastmath=True, inline="always")
    def _nb_reflect(i, n):
        if i < 0:
            i = -i - 1
        if i >= n:
            i = 2 * n - i - 1
        return min(max(i, 0), n - 1)

    @_nb.njit(cache=True, nogil=True, fastmath=True, inline="always")
    def _nb_bilin(x, c, yy, xx):
        H, W = x.shape[0], x.shape[1]
        y0 = int(np.floor(yy))
        x0 = int(np.floor(xx))
        fy = yy - y0
        fx = xx - x0
        ya, yb = _nb_reflect(y0, H), _nb_reflect(y0 + 1, H)
        xa, xb = _nb_reflect(x0, W), _nb_reflect(x0 + 1, W)
        top = x[ya, xa, c] + (x[ya, xb, c] - x[ya, xa, c]) * fx
        bot = x[yb, xa, c] + (x[yb, xb, c] - x[yb, xa, c]) * fx
        return top + (bot - top) * fy

    @_nb.njit(cache=True, nogil=True, fastmath=True)
    def _nb_finish(x, halo, use_halo, ca_k, vig, n1, n2, wgt, use_grain, chroma, out):
        """Halation + aberration chromatique (rouge vers l'extérieur, bleu vers l'intérieur) + vignettage + grain,
        en une passe (repli cv2 équivalent ; seul le halo est pris au pixel et non au point décalé)."""
        H, W = x.shape[0], x.shape[1]
        cx, cy = W * 0.5, H * 0.5
        for i in range(H):
            dy = i + 0.5 - cy
            for j in range(W):
                dx = j + 0.5 - cx
                amt = ca_k * (dx * dx + dy * dy)
                r = _nb_bilin(x, 0, cy - 0.5 + dy * (1 + amt), cx - 0.5 + dx * (1 + amt))
                g = x[i, j, 1]
                b = _nb_bilin(x, 2, cy - 0.5 + dy * (1 - amt), cx - 0.5 + dx * (1 - amt))
                if use_halo:
                    h = halo[i, j]
                    r += h
                    g += h * np.float32(0.34)
                    b += h * np.float32(0.10)
                v = vig[i, j]
                r *= v
                g *= v
                b *= v
                if use_grain:
                    w = wgt[i, j]
                    a = n1[i, j] * w
                    c = n2[i, j] * w * chroma
                    g += a
                    r += a + c
                    b += a - c
                out[i, j, 0] = min(max(r, np.float32(0.0)), np.float32(1.0))
                out[i, j, 1] = min(max(g, np.float32(0.0)), np.float32(1.0))
                out[i, j, 2] = min(max(b, np.float32(0.0)), np.float32(1.0))


def film_finish(img, frame_idx: int, grain: float = 0.055, vignette: float = 0.38, halation: float = 0.22):
    """Finition argentique : halation (les hautes lumières bavent en rouge-orangé), légère aberration
    chromatique sur les bords, vignettage ovale 9:16, grain temporel (~1,5 px à 1080, pondéré par la
    luminance, déterministe par frame_idx). Noyau numba en une passe si disponible (sinon cv2)."""
    x = np.ascontiguousarray(_f32(img)[..., :3])
    H, W = x.shape[:2]
    mp_ = _finish_maps(H, W, float(vignette))
    sw, sh_ = max(8, W // 4), max(8, H // 4)
    Ls = cv2.transform(cv2.resize(x, (sw, sh_), interpolation=cv2.INTER_AREA), _LUMA)   # luminance au 1/4
    halo = None
    if halation > 0:
        hl = _sstep(0.55, 0.97, Ls)
        k = sw / 270.0   # 1 px à 1/4 de résolution ≈ 4 px à 1080
        # frange autour des hautes lumières (l'intérieur d'une grande zone claire n'est pas teinté)
        halo = 1.0 * cv2.GaussianBlur(hl, (0, 0), 1.6 * k) + 0.8 * cv2.GaussianBlur(hl, (0, 0), 5.5 * k) - 1.75 * hl
        halo = cv2.resize(np.maximum(halo, 0) * np.float32(halation * 1.3), (W, H), interpolation=cv2.INTER_LINEAR)
    wgt = n1 = n2 = None
    if grain > 0:
        # grain plus présent dans les ombres et tons moyens, discret dans les hautes lumières : poids calculé
        # au 1/4 de résolution (il varie lentement) puis agrandi ; le grain lui-même est pleine résolution
        Lv = np.clip(Ls * cv2.resize(mp_["vig"], (sw, sh_), interpolation=cv2.INTER_AREA), 0.0, 1.0)
        w_small = np.interp(Lv, np.linspace(0.0, 1.0, _GRAIN_W_LUT.shape[1]), _GRAIN_W_LUT[0]).astype(np.float32)
        wgt = cv2.resize(w_small * np.float32(0.8 * grain), (W, H), interpolation=cv2.INTER_LINEAR)  # σ max ≈ 0,8·grain
        n1 = _grain_field(H, W, int(frame_idx), 1)
        n2 = _grain_field(H, W, int(frame_idx), 2)
    if _nb is not None:
        out = np.empty_like(x)
        dummy = np.zeros((1, 1), np.float32)
        _nb_finish(x, halo if halo is not None else dummy, halo is not None, np.float32(mp_["ca_k"]), mp_["vig"],
                   n1 if n1 is not None else dummy, n2 if n2 is not None else dummy,
                   wgt if wgt is not None else dummy, wgt is not None, np.float32(GRAIN_CHROMA), out)
        return out
    r, g, b = cv2.split(x)
    if halo is not None:
        r += halo
        g += halo * np.float32(0.34)
        b += halo * np.float32(0.10)
    r = cv2.remap(r, mp_["rx"], mp_["ry"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    b = cv2.remap(b, mp_["bx"], mp_["by"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    vig = mp_["vig"]
    r *= vig
    g *= vig
    b *= vig
    if wgt is not None:
        a = n1 * wgt
        c = n2 * wgt * np.float32(GRAIN_CHROMA)
        g += a
        r += a + c
        b += a - c
    res = cv2.merge([r, g, b])
    np.clip(res, 0.0, 1.0, out=res)
    return res


# ============================================================================= effets d'image
def glitch(img, frame_idx: int, amount: float):
    """Glitch numérique déterministe : bandes horizontales décalées, déchirures RVB, blocs, postérisation."""
    amount = _clamp(float(amount))
    x = _f32(img)
    if amount <= 0:
        return x.copy()
    out = x.copy()
    H, W = x.shape[:2]
    s = W / REF_W
    rng = np.random.default_rng(_hash(0x61, int(frame_idx)) & 0xFFFFFFFF)
    # 1) une grande déchirure : un tiers de l'image glisse latéralement (signal qui décroche)
    if rng.random() < 0.35 + 0.5 * amount:
        h = int(rng.uniform(0.08, 0.35) * H * amount) + 2
        y0 = int(rng.integers(0, max(1, H - h)))
        out[y0:y0 + h] = np.roll(x[y0:y0 + h], int(rng.choice([-1, 1]) * rng.uniform(40, 160) * s * amount), axis=1)
    # 2) bandes fines décalées, déchirures RVB, postérisation
    for _ in range(3 + int(round(14 * amount * rng.uniform(0.6, 1.0)))):
        h = max(2, int(rng.uniform(2, 70) * s * (0.4 + amount)))
        y0 = int(rng.integers(0, max(1, H - h)))
        dx = int(rng.normal(0, 80 * s * amount))
        band = out[y0:y0 + h].copy()
        out[y0:y0 + h] = np.roll(band, dx, axis=1)
        if rng.random() < 0.6:
            out[y0:y0 + h, :, 0] = np.roll(band[:, :, 0], dx + int(rng.normal(0, 22 * s * amount)), axis=1)
            out[y0:y0 + h, :, 2] = np.roll(band[:, :, 2], dx - int(rng.normal(0, 14 * s * amount)), axis=1)
        if rng.random() < 0.25 * amount:
            out[y0:y0 + h] = np.round(out[y0:y0 + h] * 5.0) / 5.0
    # 3) bande pixelisée (macroblocs) et blocs recopiés (« datamosh »)
    if rng.random() < 0.7 * amount:
        h = max(8, int(rng.uniform(30, 160) * s))
        y0 = int(rng.integers(0, max(1, H - h)))
        bk = max(2, int(rng.choice([8, 12, 16]) * s))
        band = out[y0:y0 + h]
        small = cv2.resize(band, (max(1, W // bk), max(1, h // bk)), interpolation=cv2.INTER_AREA)
        out[y0:y0 + h] = cv2.resize(small, (W, h), interpolation=cv2.INTER_NEAREST)
    for _ in range(int(9 * amount * rng.random())):
        bs = max(4, int(rng.choice([16, 24, 32, 48]) * s))
        sy_, sx_ = int(rng.integers(0, max(1, H - bs))), int(rng.integers(0, max(1, W - bs)))
        dy_ = int(np.clip(sy_ + rng.normal(0, 40 * s), 0, H - bs))
        dx_ = int(np.clip(sx_ + rng.normal(0, 60 * s), 0, W - bs))
        out[dy_:dy_ + bs, dx_:dx_ + bs] = x[sy_:sy_ + bs, sx_:sx_ + bs]
    # 4) lignes de balayage parasites et léger décalage global du bleu
    for _ in range(int(5 * amount)):
        y = int(rng.integers(0, H))
        hh = max(1, int(2 * s))
        out[y:y + hh] = out[y:y + hh] * 0.4 + np.float32(rng.uniform(0.3, 0.9)) * 0.6
    if amount > 0.3:
        sh = int(round(5 * s * amount)) + 1
        out[:, :, 2] = np.roll(out[:, :, 2], sh, axis=1)
    return np.clip(out, 0.0, 1.0)


def rgb_split(img, amount: float, frame_idx: int):
    """Séparation RVB (rouge à gauche, bleu à droite), amplitude amount·≈14 px à 1080, légère gigue."""
    x = _f32(img)
    amount = float(amount)
    if amount <= 0:
        return x.copy()
    H, W = x.shape[:2]
    s = W / REF_W
    d = amount * 14.0 * s * (0.75 + 0.5 * _rand(0x52, int(frame_idx)))
    dy = d * 0.25 * (2 * _rand(0x53, int(frame_idx)) - 1)
    r, g, b = cv2.split(x)
    Mr = np.float32([[1, 0, -d], [0, 1, -dy]])
    Mb = np.float32([[1, 0, d], [0, 1, dy]])
    r = cv2.warpAffine(r, Mr, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    b = cv2.warpAffine(b, Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([r, g, b])


def negative(img):
    """Flash négatif."""
    return (1.0 - np.clip(_f32(img), 0.0, 1.0)).astype(np.float32)


def darken(img, amount: float):
    """Assombrit (0 = inchangé, 1 = noir) en écrasant un peu les noirs."""
    a = _clamp(float(amount))
    x = np.clip(_f32(img), 0.0, 1.0)
    if a <= 0:
        return x.copy()
    return (np.power(x, np.float32(1.0 + 0.5 * a)) * np.float32(1.0 - a)).astype(np.float32)
