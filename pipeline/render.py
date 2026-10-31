"""Rendu vidéo du film : render(edl, quality) -> RENDER_DIR/video_{quality}.mp4 (sans son).

Chaîne par image (ordre imposé) :
  décodage du proxy (pipe ffmpeg rawvideo rgb24, -ss avant -i, nombre d'images exact)
  → vitesse / rampes / freeze_after (mapping d'index d'images)
  → effets visages sur l'image proxy pleine (coordonnées FaceTrack)
  → incrustation de la silhouette (espace plaque)
  → caméra (recadrage/zoom/punch, rééchantillonnage Lanczos sous-pixel)
  → étalonnage → floutage des inconnus → glitch / négatif / subliminal
  → film_finish → overlays (timecode, cartons, prénoms, HUD, cercles, titre) → fondus → sous-titres.

Chaque shot est rendue dans un processus séparé (2–3 en parallèle) et mise en cache
(CACHE_DIR/shots/<quality>/<id>_<hash>.mp4, H.264 quasi sans perte), puis tout est concaténé
avec un nombre d'images exact (round(durée × 30)).

Les modules gfx / facefx / silhouette / parallax sont importés paresseusement. S'ils manquent,
un repli minimal local « STUB » est utilisé, signalé dans les avertissements et dans le rapport
RENDER_DIR/video_{quality}.json (le contrôle qualité refuse un rendu final fait avec des STUB).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # exécution directe : python pipeline/render.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

RENDER_VERSION = "render-1.0"
FPS = config.FPS
SIZES = {"preview": (540, 960), "final": (1080, 1920)}
SRC_W, SRC_H = config.W, config.H            # espace des proxies (coordonnées FaceTrack / decisions)
# Encodage des intermédiaires (quasi sans perte) : crf, preset
INTERMEDIATE = {"preview": ("12", "veryfast"), "final": ("10", "veryfast")}
FLASH_TYPES = ("negative", "subliminal")
MAX_FLASHES_PER_S = 3
FACE_FX = ("black_eyes", "stretch_smile", "melt", "widen_eyes", "demon_face", "head_tic", "portrait_eyes")
SUB_FADE_S = 0.12
# processus de rendu par défaut : un cœur laissé libre (4 CPU → 3 processus), au plus 4
DEFAULT_JOBS = max(1, min(4, (os.cpu_count() or 4) - 1))
OCC_STEP = {"preview": 3, "final": 3}  # masque d'occlusion des personnes recalculé toutes les 3 images (0,1 s),
#                                        même grille en preview et en final → cache disque partagé
LANCZOS_MIN_ZOOM = 1.12  # final : Lanczos pour les vrais agrandissements ; en dessous, bicubique (écart invisible
#                          sous le grain, warpAffine Lanczos 8x8 ≈ 4× plus coûteux)


# ---------------------------------------------------------------------------
# Outils généraux
# ---------------------------------------------------------------------------
def fidx(t: float) -> int:
    """Index d'image = round(t × 30) (arrondi au plus proche, demi vers le haut)."""
    return int(math.floor(t * FPS + 0.5 + 1e-9))


def _num(v, default=None):
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _ffmpeg() -> str:
    return os.environ.get("ESCAPE_FFMPEG", shutil.which("ffmpeg") or "ffmpeg")


def _ffprobe() -> str:
    return os.environ.get("ESCAPE_FFPROBE", shutil.which("ffprobe") or "ffprobe")


def ffprobe_json(path: Path, *args: str) -> dict:
    cmd = [_ffprobe(), "-v", "error", "-of", "json", *args, str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe a échoué sur {path} : {r.stderr.strip()[-500:]}")
    return json.loads(r.stdout or "{}")


def count_frames(path: Path) -> int:
    """Nombre exact d'images (paquets vidéo) d'un fichier."""
    d = ffprobe_json(path, "-select_streams", "v:0", "-count_packets",
                     "-show_entries", "stream=nb_read_packets")
    return int(d["streams"][0]["nb_read_packets"])


def video_size(path: Path) -> tuple[int, int]:
    d = ffprobe_json(path, "-select_streams", "v:0", "-show_entries", "stream=width,height")
    s = d["streams"][0]
    return int(s["width"]), int(s["height"])


def _file_sig(p: Path | None) -> str:
    if p is None:
        return "-"
    p = Path(p)
    if not p.exists():
        return f"{p.name}:absent"
    st = p.stat()
    return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def _code_sig() -> str:
    """Signature du code qui influence l'image (render + modules visuels)."""
    h = hashlib.sha1(RENDER_VERSION.encode())
    for name in ("render.py", "gfx.py", "facefx.py", "silhouette.py", "parallax.py"):
        p = Path(__file__).resolve().parent / name
        h.update(p.read_bytes() if p.exists() else b"absent")
    return h.hexdigest()[:10]


def load_edl(path: str | Path | None = None) -> dict:
    """Charge l'EDL : chemin explicite, sinon WORK_DIR/edl.json, sinon edit/edl.json."""
    cands = [Path(path)] if path else [config.WORK_DIR / "edl.json", config.EDIT_DIR / "edl.json"]
    for p in cands:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("edl.json introuvable (" + ", ".join(map(str, cands)) + ") : lance l'étape « edl ».")


def edl_duration(edl: dict) -> float:
    d = _num(edl.get("meta", {}).get("duration"))
    if d is None:
        d = max((s["start"] + s["dur"] for s in edl["shots"]), default=0.0)
    return float(d)


def frame_plan(edl: dict) -> list[tuple[int, int]]:
    """Partition exacte de la timeline en images : [(f0, f1)] par shot, contiguë, total = round(durée×30)."""
    shots = edl["shots"]
    f0s = [fidx(_num(s.get("start"), 0.0)) for s in shots]
    total = fidx(edl_duration(edl))
    out = []
    for i, f0 in enumerate(f0s):
        f1 = f0s[i + 1] if i + 1 < len(shots) else total
        out.append((f0, max(f0, f1)))
    return out


# ---------------------------------------------------------------------------
# Modules visuels (imports paresseux + STUB)
# ---------------------------------------------------------------------------
_MODS: dict[str, object] = {}
STUBS_USED: set[str] = set()


def _module(name: str):
    if name in _MODS:
        return _MODS[name]
    try:
        m = importlib.import_module(f"pipeline.{name}")
    except Exception as e:  # module absent ou en cours d'écriture : repli STUB signalé
        print(f"[render] ⚠ pipeline.{name} indisponible ({type(e).__name__}: {e}) -> STUB local", file=sys.stderr)
        m = _STUB_FACTORIES[name]()
        STUBS_USED.add(name)
    _MODS[name] = m
    return m


def gfx():
    return _module("gfx")


def facefx():
    return _module("facefx")


def silhouette_mod():
    return _module("silhouette")


def parallax():
    return _module("parallax")


# ----------------------------- STUBS --------------------------------------
# Replis minimaux pour tester la chaîne quand un module n'existe pas encore.
# STUB : à retirer à l'intégration (le contrôle qualité échoue en final si un STUB a servi).
class _StubGfx:  # STUB
    """STUB de pipeline.gfx (aspect volontairement basique)."""

    @staticmethod
    def font(name, size, weight=None):
        from PIL import ImageFont
        files = {"inter": "Interopsz_wght.ttf", "bodoni": "BodoniModaopsz_wght.ttf",
                 "plexmono": "IBMPlexMono-Regular.ttf", "cormorant": "CormorantGaramondwght.ttf"}
        p = config.FONTS_DIR / files.get(name, "Interopsz_wght.ttf")
        try:
            return ImageFont.truetype(str(p), size)
        except Exception:
            return ImageFont.load_default()

    @staticmethod
    def safe_zone():
        return (60, 160, config.W - 180, int(0.8 * config.H))

    @staticmethod
    def grade(img, preset, mix=0.0, **kw):
        y = img.mean(axis=2, keepdims=True)
        sat = {"cold": 0.55, "night_red": 0.8, "blue": 0.7, "warm": 1.25, "bleach": 0.3}.get(preset, 0.6)
        if preset == "warm_to_cold":
            sat = 1.25 + (0.55 - 1.25) * float(mix)
        out = y + (img - y) * sat
        tint = {"cold": (0.94, 1.0, 1.08), "warm": (1.08, 1.0, 0.9), "blue": (0.9, 1.0, 1.12)}.get(preset, (1, 1, 1))
        out = out * np.array(tint, np.float32)
        out = np.clip((out - 0.04) / 0.96, 0, 1) ** 1.1
        return out.astype(np.float32)

    @staticmethod
    def film_finish(img, frame_idx, grain=0.055, vignette=0.38, halation=0.22):
        h, w = img.shape[:2]
        rng = np.random.default_rng(frame_idx & 0xFFFFFFFF)
        n = rng.standard_normal((h, w, 1)).astype(np.float32) * grain * 0.6
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = ((xx / w - 0.5) ** 2 / 0.35 + (yy / h - 0.5) ** 2 / 0.6)
        v = 1 - vignette * np.clip(r, 0, 1)[..., None]
        return np.clip((img + n * np.sqrt(np.clip(img, 0, 1))) * v, 0, 1).astype(np.float32)

    @staticmethod
    def glitch(img, frame_idx, amount):
        out = img.copy()
        h = img.shape[0]
        rng = np.random.default_rng(frame_idx)
        for _ in range(int(6 * amount) + 1):
            y0 = int(rng.integers(0, h - 2)); hh = int(rng.integers(2, max(3, h // 12)))
            out[y0:y0 + hh] = np.roll(out[y0:y0 + hh], int(rng.integers(-40, 40) * amount), axis=1)
        return out

    @staticmethod
    def rgb_split(img, amount, frame_idx):
        d = max(1, int(round(amount * img.shape[1] * 0.006)))
        out = img.copy()
        out[..., 0] = np.roll(img[..., 0], d, axis=1)
        out[..., 2] = np.roll(img[..., 2], -d, axis=1)
        return out

    @staticmethod
    def negative(img):
        return (1.0 - img).astype(np.float32)

    @staticmethod
    def darken(img, amount):
        return (img * (1 - amount)).astype(np.float32)

    @staticmethod
    def _text(img, text, xy_rel, size_rel, color=(1, 1, 1), alpha=1.0, anchor="mm", fontname="inter"):
        from PIL import Image, ImageDraw
        h, w = img.shape[:2]
        layer = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(layer)
        f = _StubGfx.font(fontname, max(8, int(size_rel * w)))
        d.multiline_text((xy_rel[0] * w, xy_rel[1] * h), text, font=f, fill=255, anchor=anchor, align="center")
        a = (np.asarray(layer, np.float32) / 255.0 * alpha)[..., None]
        return (img * (1 - a) + np.array(color, np.float32) * a).astype(np.float32)

    @classmethod
    def draw_card(cls, img, text, t, dur, style="card"):
        if img is None:
            img = np.zeros((config.H, config.W, 3), np.float32)
        a = min(1.0, t / 0.4, max(0.0, (dur - t) / 0.4))
        return cls._text(img, text, (0.5, 0.45), 0.045, (0.92, 0.9, 0.86), max(0.0, a))

    @classmethod
    def draw_name(cls, img, name, t):
        return cls._text(img, name, (0.5, 0.66), 0.08, (0.95, 0.94, 0.9), min(1.0, t / 0.2), fontname="bodoni")

    @classmethod
    def draw_timecode(cls, img, hhmm, date, t):
        return cls._text(img, f"● {hhmm}  {date}", (0.07, 0.1), 0.03, (0.9, 0.9, 0.9), 0.85, anchor="lm",
                         fontname="plexmono")

    @classmethod
    def draw_subtitle(cls, img, text, kind="dialogue", alpha=1.0):
        col = {"entity": (0.75, 0.25, 0.25), "narrator": (0.92, 0.9, 0.85)}.get(kind, (1, 1, 1))
        return cls._text(img, text, (0.4444, 0.74), 0.038, col, alpha)

    @classmethod
    def draw_title_sub(cls, img, text, t, dur):
        return cls._text(img, text, (0.5, 0.62), 0.028, (0.9, 0.9, 0.9), min(1.0, t / 0.5))

    @classmethod
    def draw_hud_face(cls, img, bbox, t_since_lock, label, sublabel, frame_idx, mode="identify", landmarks=None):
        import cv2
        out = np.ascontiguousarray(img)
        col = (0.9, 0.15, 0.1) if mode == "victim" else (0.6, 0.95, 0.9)
        x0, y0, x1, y1 = [int(v) for v in bbox]
        cv2.rectangle(out, (x0, y0), (x1, y1), col, max(1, img.shape[1] // 400))
        n = int(len(label) * min(1.0, max(0.0, t_since_lock) / 0.6))
        return cls._text(out, label[:n], (x0 / img.shape[1], max(0.02, (y0 - 12) / img.shape[0])), 0.02, col,
                         1.0, anchor="ls", fontname="plexmono")

    @staticmethod
    def draw_red_circle(img, center_xy, axes_xy, progress, seed, frantic=False):
        import cv2
        out = np.ascontiguousarray(img)
        end = int(360 * (1.6 if frantic else 1.1) * max(0.0, min(1.0, progress)))
        if end > 0:
            cv2.ellipse(out, (int(center_xy[0]), int(center_xy[1])), (int(axes_xy[0]), int(axes_xy[1])),
                        -10 + seed % 20, 0, end, (0.85, 0.05, 0.05), max(2, img.shape[1] // 200), cv2.LINE_AA)
        return out


class _StubFace:  # boîte seule (analyse 5 i/s, boîte manuelle) : pas de points du maillage
    box_only = True

    def __init__(self, bbox, score=0.9):
        self.bbox = tuple(float(v) for v in bbox)
        self.score = score
        x0, y0, x1, y1 = self.bbox
        self.lm = np.array([[(x0 + x1) / 2, (y0 + y1) / 2]] * 478, np.float32)


class _StubFaceTrack:  # STUB : aucun visage (les effets sont neutres)
    def __init__(self, n=0):
        self.n = n

    @classmethod
    def compute(cls, video_path, t0, t1, fps=30):
        return cls(int(round((t1 - t0) * fps)))

    @classmethod
    def from_image(cls, img_uint8):
        return cls(1)

    def at(self, i):
        return []


def _stub_facefx():  # STUB
    import types
    m = types.SimpleNamespace(FaceTrack=_StubFaceTrack, STUB=True)
    ident = lambda img, *a, **k: img  # noqa: E731
    for n in FACE_FX:
        setattr(m, n, ident)
    return m


def _stub_silhouette():  # STUB : silhouette procédurale très simple
    import types
    from PIL import Image, ImageDraw

    def extract(video_path, t_range, out_dir, hint_center=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        w, h = 300, 900
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        c = (12, 12, 14, 255)
        d.ellipse([110, 20, 190, 120], fill=c)
        d.polygon([(95, 120), (205, 120), (240, 300), (260, 880), (40, 880), (60, 300)], fill=c)
        im.save(out_dir / "still.png")
        return {"still": str(out_dir / "still.png"), "bbox": [0, 0, w, h], "foot": [w / 2, h],
                "height_px": h, "method": "procedural"}

    def composite(img, sil_rgba, x, foot_y, height, opacity=0.9, blur=1.5, darkness=0.9, blend="normal",
                  frame_idx=0, occluder_mask=None):
        import cv2
        H, W = img.shape[:2]
        rgba = sil_rgba.astype(np.float32) / (255.0 if sil_rgba.dtype == np.uint8 else 1.0)
        hh = max(4, int(height * H))
        ww = max(2, int(rgba.shape[1] * hh / rgba.shape[0]))
        s = cv2.resize(rgba, (ww, hh), interpolation=cv2.INTER_AREA)
        x0, y0 = int(x * W - ww / 2), int(foot_y * H - hh)
        out = img.copy()
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + hh)
        if xb <= xa or yb <= ya:
            return out
        a = s[ya - y0:yb - y0, xa - x0:xb - x0, 3:4] * opacity
        if occluder_mask is not None:
            a = a * (1 - occluder_mask[ya:yb, xa:xb, None])
        dark = s[ya - y0:yb - y0, xa - x0:xb - x0, :3] * (1 - darkness)
        out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - a) + dark * a
        return out

    def person_mask(img):
        return np.zeros(img.shape[:2], np.float32)

    return types.SimpleNamespace(extract=extract, composite=composite, person_mask=person_mask, STUB=True)


def _stub_parallax():  # STUB : simple poussée 2D sans profondeur
    import types

    def depth(img):
        return np.zeros(img.shape[:2], np.float32)

    def render_parallax(img, depth, t, dur, zoom=(1.0, 1.16), center=(0.5, 0.45), amount=1.0, out_size=(1080, 1920)):
        import cv2
        ow, oh = out_size
        h, w = img.shape[:2]
        z = zoom[0] + (zoom[1] - zoom[0]) * (t / max(dur, 1e-6))
        s = max(ow / w, oh / h) * z
        cx, cy = center[0] * w, center[1] * h
        tx = min(0.0, max(ow - w * s, ow / 2 - cx * s))
        ty = min(0.0, max(oh - h * s, oh / 2 - cy * s))
        m = np.float32([[s, 0, tx], [0, s, ty]])
        return cv2.warpAffine(img, m, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return types.SimpleNamespace(depth=depth, render_parallax=render_parallax, STUB=True)


_STUB_FACTORIES = {"gfx": lambda: _StubGfx, "facefx": _stub_facefx, "silhouette": _stub_silhouette,
                   "parallax": _stub_parallax}


# ---------------------------------------------------------------------------
# Décodage / encodage
# ---------------------------------------------------------------------------
class FrameReader:
    """Lit `count` images à partir de l'index `first` (30 i/s) d'une vidéo, en flux, dans l'ordre.

    -ss AVANT -i (recherche précise : ffmpeg décode depuis l'image clé et jette ce qui précède),
    nombre d'images exact (-frames:v). Si la source est trop courte, la dernière image est répétée.
    """

    def __init__(self, path: Path, first: int, count: int, size: tuple[int, int] = (SRC_W, SRC_H)):
        self.path, self.first, self.count = Path(path), int(first), max(1, int(count))
        self.w, self.h = size
        self.nbytes = self.w * self.h * 3
        self.pos = self.first - 1  # index de l'image courante
        self.cur: np.ndarray | None = None
        self.hist: dict[int, np.ndarray] = {}  # dernières images (répétitions d'images des tics)
        self.short = False
        sw, sh = video_size(self.path)
        vf = []
        if (sw, sh) != (self.w, self.h):  # normalise (cover + recadrage centré)
            vf.append(f"scale={self.w}:{self.h}:force_original_aspect_ratio=increase:flags=lanczos,"
                      f"crop={self.w}:{self.h}")
        vf.append("format=rgb24")
        ss = max(0.0, (self.first - 0.25) / FPS)
        cmd = [_ffmpeg(), "-v", "error", "-nostdin", "-ss", f"{ss:.6f}", "-i", str(self.path), "-map", "0:v:0",
               "-frames:v", str(self.count), "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     bufsize=self.nbytes * 2)

    def get(self, j: int) -> np.ndarray:
        """Image d'index absolu j (lecture en flux ; un léger retour en arrière ≤ 4 images est possible)."""
        if j < self.pos and j in self.hist:
            return self.hist[j]
        while self.pos < j and not self.short:
            buf = self.proc.stdout.read(self.nbytes)
            if len(buf) < self.nbytes:
                self.short = True
                break
            self.cur = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
            self.pos += 1
            self.hist[self.pos] = self.cur
            self.hist.pop(self.pos - 5, None)
        if self.cur is None:
            raise RuntimeError(f"aucune image décodée dans {self.path.name} à partir de l'index {self.first}")
        return self.cur

    def close(self):
        try:
            if self.proc.stdout:
                self.proc.stdout.close()
        finally:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc.wait()


def read_single_frame(path: Path, j: int, size=(SRC_W, SRC_H)) -> np.ndarray:
    r = FrameReader(path, j, 1, size)
    try:
        return r.get(j).copy()
    finally:
        r.close()


class Encoder:
    """Encode des images float32 RGB [0,1] en H.264 quasi sans perte (BT.709, 30 i/s)."""

    def __init__(self, out: Path, w: int, h: int, quality: str):
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        crf, preset = INTERMEDIATE[quality]
        self.log = self.out.with_suffix(".log")
        self.n = 0
        cmd = [_ffmpeg(), "-y", "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-framerate", str(FPS), "-i", "-",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
               "-c:v", "libx264", "-preset", preset, "-crf", crf, "-tune", "grain", "-threads", "2",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
               "-video_track_timescale", "15360", "-an", str(self.out)]
        self._logf = open(self.log, "wb")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._logf)

    def write(self, img: np.ndarray):
        buf = np.multiply(img, 255.0, dtype=np.float32)
        buf += 0.5
        np.clip(buf, 0, 255, out=buf)
        self.proc.stdin.write(np.ascontiguousarray(buf.astype(np.uint8)).tobytes())
        self.n += 1

    def close(self) -> int:
        self.proc.stdin.close()
        rc = self.proc.wait()
        self._logf.close()
        if rc != 0:
            raise RuntimeError(f"encodage échoué ({self.out.name}) : {self.log.read_text(errors='ignore')[-800:]}")
        self.log.unlink(missing_ok=True)
        return self.n


# ---------------------------------------------------------------------------
# Mapping temporel (vitesse, rampes, gel)
# ---------------------------------------------------------------------------
def shot_local_times(start: float, f0: int, n: int) -> np.ndarray:
    """Temps relatif à la shot de chacune de ses n images (alignées sur la grille absolue de 30 i/s)."""
    return np.maximum(0.0, (f0 + np.arange(n)) / FPS - start)


def speed_curve(shot: dict, t: np.ndarray) -> np.ndarray:
    dur = max(1e-6, _num(shot.get("dur"), 1.0))
    sp = shot.get("speed", 1.0)
    if isinstance(sp, (list, tuple)) and len(sp) == 2:
        v = float(sp[0]) + (float(sp[1]) - float(sp[0])) * np.clip(t / dur, 0, 1)
    else:
        v = np.full_like(t, _num(sp, 1.0))
    for fx in shot.get("fx") or []:
        if isinstance(fx, dict) and fx.get("type") == "speed_ramp":
            t0 = _num(fx.get("t0"), 0.0)
            t1 = _num(fx.get("t1"), dur)
            to = _num(fx.get("to"), 1.0)
            k = np.clip((t - t0) / max(1e-6, t1 - t0), 0, 1)
            k = k * k * (3 - 2 * k)  # rampe douce
            v = v * (1 + (to - 1) * k)
    return np.maximum(v, 0.0)


def source_times(shot: dict, tl: np.ndarray) -> np.ndarray:
    """Temps source (s, dans le proxy) de chaque image : src_in + ∫ vitesse, gelé après freeze_after."""
    src_in = _num(shot.get("src_in"), 0.0)
    fa = _num(shot.get("freeze_after"))
    tq = np.minimum(tl, fa) if fa is not None else tl
    # intégration fine (pas de 1/240 s) de la courbe de vitesse
    tmax = float(tq.max()) if len(tq) else 0.0
    grid = np.linspace(0.0, tmax, max(2, int(tmax * 240) + 2))
    v = speed_curve(shot, grid)
    cum = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) * 0.5 * np.diff(grid))])
    return src_in + np.interp(tq, grid, cum)


def source_frames(shot: dict, tl: np.ndarray) -> np.ndarray:
    return np.array([fidx(s) for s in source_times(shot, tl)], dtype=np.int64)


# ---------------------------------------------------------------------------
# Préparation globale : silhouette, garde-fou flashs, résolution des gels
# ---------------------------------------------------------------------------
def _decisions() -> dict:
    p = config.EDIT_DIR / "decisions.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _plain(v, default):
    """Valeur concrète d'une décision ({auto, fallback} -> fallback)."""
    if isinstance(v, dict):
        return v.get("value", v.get("fallback", default))
    return default if v is None else v


def silhouette_source(edl: dict) -> dict:
    """Clip, plage et centre indicatif de la silhouette (EDL résolue si présent, sinon decisions.json)."""
    src = edl.get("silhouette_source") or edl.get("meta", {}).get("silhouette_source")
    dec = _decisions()
    if not isinstance(src, dict):
        d = dec.get("silhouette_source", {})
        src = {"clip": d.get("clip", "IMG_8346"), "t_range": d.get("t_range", [6.0, 9.5]),
               "best_t": _plain(d.get("best_t"), 7.5)}
    hint = src.get("hint_center")
    if hint is None:
        for s in edl.get("shots", []):
            c = (s.get("camera") or {}).get("center")
            if s.get("clip") == src.get("clip") and isinstance(c, (list, tuple)) and len(c) == 2:
                hint = [float(c[0]), float(c[1])]
                break
    if hint is None:
        h = _plain(dec.get("silhouette_center_8346"), None)
        hint = list(h) if isinstance(h, (list, tuple)) else None
    src["hint_center"] = hint
    return src


def needs_silhouette(edl: dict) -> bool:
    for s in edl.get("shots", []):
        if s.get("silhouette"):
            return True
        for fx in s.get("fx") or []:
            if isinstance(fx, dict) and fx.get("type") == "subliminal":
                return True
        for it in s.get("items") or []:
            if "silhouette" in (it.get("fx") or []):
                return True
        for ov in s.get("overlays") or []:
            if ov.get("type") == "red_circles" and ov.get("targets") == "silhouette":
                return True
    return False


def ensure_silhouette(edl: dict, log=print) -> dict | None:
    """Détoure (une fois, avec cache) la silhouette de 8346 ; renvoie le dict de silhouette.extract."""
    src = silhouette_source(edl)
    clip = src.get("clip", "IMG_8346")
    proxy = config.proxy_path(clip)
    sil_mod = silhouette_mod()
    key = _hash({"clip": clip, "t": src.get("t_range"), "hint": src.get("hint_center"), "proxy": _file_sig(proxy),
                 "stub": bool(getattr(sil_mod, "STUB", False))})
    out_dir = config.CACHE_DIR / "silhouette" / f"{clip}_{key}"
    meta_p = out_dir / "extract.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if Path(meta.get("still", "")).exists():
            return meta
    if not proxy.exists() and not getattr(sil_mod, "STUB", False):
        log(f"[render] ⚠ proxy {proxy.name} absent : silhouette procédurale (STUB)")
        sil_mod = _stub_silhouette()
        STUBS_USED.add("silhouette")
    t_range = tuple(float(v) for v in (src.get("t_range") or (6.0, 9.5)))
    log(f"[render] détourage de la silhouette ({clip} {t_range[0]:.1f}–{t_range[1]:.1f} s)…")
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    try:
        meta = sil_mod.extract(str(proxy), t_range, str(out_dir), hint_center=src.get("hint_center"))
    except Exception as e:
        failed = True  # le détourage ne doit jamais arrêter le film : silhouette procédurale, signalée
        log(f"[render] ⚠ silhouette.extract a échoué ({type(e).__name__}: {e}) : silhouette procédurale")
        proc = getattr(sil_mod, "_procedural_result", None)
        try:
            meta = proc(Path(out_dir), key, f"extract a échoué : {e}") if callable(proc) else None
        except Exception:
            meta = None
        if meta is None:
            meta = _stub_silhouette().extract(str(proxy), t_range, str(out_dir))
        meta = dict(meta)
        meta["method"] = f"procedural (extract a échoué : {type(e).__name__})"
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in dict(meta).items()}
    if not failed:  # un repli après échec n'est pas mis en cache (nouvel essai au prochain rendu)
        meta_p.write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[render] silhouette : méthode {meta.get('method')}, {meta.get('still')}")
    return meta


def plan_flashes(edl: dict) -> tuple[dict[str, list[int]], list[dict]]:
    """Garde-fou anti-flash : au plus 3 flashs (négatif, subliminal) par fenêtre glissante d'1 s.

    Renvoie ({shot_id: [index fx à sauter]}, rapport)."""
    events = []
    for s in edl["shots"]:
        start = _num(s.get("start"), 0.0)
        for i, fx in enumerate(s.get("fx") or []):
            if isinstance(fx, dict) and fx.get("type") in FLASH_TYPES:
                t0 = _num(fx.get("t0"))
                if t0 is None:
                    continue
                events.append((start + t0, s["id"], i, fx.get("type")))
    events.sort()
    kept, skips, report = [], {}, []
    for t, sid, i, typ in events:
        recent = [k for k in kept if k > t - 1.0 + 1e-6]
        if len(recent) >= MAX_FLASHES_PER_S:
            skips.setdefault(sid, []).append(i)
            report.append({"t": round(t, 3), "shot": sid, "fx": typ, "action": "sauté (> 3 flashs/s)"})
        else:
            kept.append(t)
            report.append({"t": round(t, 3), "shot": sid, "fx": typ, "action": "gardé"})
    return skips, report


def resolve_freeze_src(edl: dict, plan: list[tuple[int, int]], i: int, depth: int = 0) -> dict | None:
    """Source d'un gel : {"clip", "t"} ou {"photo"} ; résout 'prev_last' si le builder ne l'a pas fait."""
    s = edl["shots"][i]
    fs = s.get("freeze_src")
    if isinstance(fs, dict) and ("clip" in fs or "photo" in fs):
        return fs
    if isinstance(fs, (list, tuple)) and len(fs) >= 3 and fs[0] == "clip":
        return {"clip": fs[1], "t": _num(fs[2], 0.0)}
    if fs not in ("prev_last", None) or i == 0 or depth > 8:
        return None
    p = edl["shots"][i - 1]
    f0, f1 = plan[i - 1]
    if p.get("kind") == "clip":
        tl = shot_local_times(_num(p.get("start"), 0.0), f0, max(1, f1 - f0))
        return {"clip": p["clip"], "t": float(source_frames(p, tl)[-1]) / FPS}
    if p.get("kind") == "freeze":
        return resolve_freeze_src(edl, plan, i - 1, depth + 1)
    if p.get("kind") == "photo":
        return {"photo": p.get("photo")}
    if p.get("kind") == "montage" and p.get("items"):
        it = p["items"][-1]
        if "clip" in it:
            return {"clip": it["clip"], "t": _num(it.get("src_in"), 0.0) + _num(it.get("dur"), 0.1) - 1 / FPS}
        return {"photo": it.get("photo")}
    return None


# ---------------------------------------------------------------------------
# Rendu d'une shot (exécuté dans un processus de travail)
# ---------------------------------------------------------------------------
def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _faces_list(obj) -> list:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if hasattr(obj, "at"):
        try:
            return list(obj.at(0) or [])
        except Exception:
            return []
    return []


def _mesh_faces(faces: list) -> list:
    """Visages avec un vrai maillage (les boîtes seules ne servent qu'au floutage / au cadrage)."""
    return [f for f in faces or [] if not getattr(f, "box_only", False)]


def _dec_value(key: str, field: str):
    """decisions.json[key][field] si rempli (sinon None)."""
    v = (_decisions().get(key) or {})
    return v.get(field) if isinstance(v, dict) else None


def _clip_num(clip: str | None) -> str:
    return str(clip or "").replace("IMG_", "")


def main_faces(faces: list) -> list:
    """Sujets principaux : visages dont la hauteur ≥ 50 % du plus grand (4 au plus)."""
    if not faces:
        return []
    hmax = max(f.bbox[3] - f.bbox[1] for f in faces)
    m = [f for f in faces if (f.bbox[3] - f.bbox[1]) >= 0.5 * hmax]
    return sorted(m, key=lambda f: -_area(f.bbox))[:4]


class Camera:
    """Recadrage de l'espace proxy (sw×sh) vers la sortie (ow×oh) : p_out = s·(p − o) + shake."""

    def __init__(self, ow, oh, zoom, cx, cy, sw=SRC_W, sh=SRC_H, shake=(0.0, 0.0), lanczos=True):
        self.lanczos = lanczos  # Lanczos (final) ; bicubique en preview (warpAffine Lanczos ~3x plus lent)
        z = max(1.0, float(zoom))
        cw, ch = sw / z, sh / z
        self.s = ow / cw
        self.ox = min(max(cx - cw / 2, 0.0), sw - cw)
        self.oy = min(max(cy - ch / 2, 0.0), sh - ch)
        self.ow, self.oh = ow, oh
        self.dx, self.dy = shake

    def matrix(self) -> np.ndarray:
        return np.float32([[self.s, 0, -self.s * self.ox + self.dx], [0, self.s, -self.s * self.oy + self.dy]])

    def apply(self, img: np.ndarray, pre: np.ndarray | None = None) -> np.ndarray:
        """Recadrage + zoom Lanczos sous-pixel (uint8 ou float32 ; sortie du même type).

        En réduction (preview), l'image est d'abord ramenée à la taille de sortie avec un filtre de surface
        (anti-crénelage), puis agrandie par le zoom : `pre` = cette version réduite si déjà calculée."""
        import cv2
        h, w = img.shape[:2]
        M = self.matrix()
        r = self.ow / w
        if self.s < 0.999 and r < 0.999:
            if pre is None:
                pre = cv2.resize(img, (int(round(w * r)), int(round(h * r))), interpolation=cv2.INTER_AREA)
            ry = pre.shape[0] / h
            rx = pre.shape[1] / w
            # p = (p' + 0.5) / r − 0.5  (convention des centres de pixels de INTER_AREA)
            M = np.float32([[M[0, 0] / rx, 0, M[0, 2] + M[0, 0] * (0.5 / rx - 0.5)],
                            [0, M[1, 1] / ry, M[1, 2] + M[1, 1] * (0.5 / ry - 0.5)]])
            img = pre
        lanczos = self.lanczos and self.s * min(w / self.ow, h / self.oh) >= LANCZOS_MIN_ZOOM
        return cv2.warpAffine(img, M, (self.ow, self.oh),
                              flags=cv2.INTER_LANCZOS4 if lanczos else cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT101)

    def pt(self, x, y):
        return self.s * (x - self.ox) + self.dx, self.s * (y - self.oy) + self.dy

    def box(self, b):
        x0, y0 = self.pt(b[0], b[1])
        x1, y1 = self.pt(b[2], b[3])
        return [x0, y0, x1, y1]


def zoom_at(cam: dict, t: float, dur: float) -> float:
    z = cam.get("zoom", (1.0, 1.0))
    if not isinstance(z, (list, tuple)):
        z = (_num(z, 1.0), _num(z, 1.0))
    z0, z1 = float(z[0]), float(z[1])
    u = min(1.0, max(0.0, t / max(dur, 1e-6)))
    punch = _num(cam.get("punch"))
    if punch and punch > 0:  # punch-in rapide (ease-out) sur 85 % du trajet, puis glissement
        if t < punch:
            k = 1 - (1 - t / punch) ** 3
            return z0 + (z1 - z0) * 0.85 * k
        return z0 + (z1 - z0) * (0.85 + 0.15 * (t - punch) / max(1e-6, dur - punch))
    return z0 + (z1 - z0) * u


def _ramp_amount(fx: dict, t: float, dur: float) -> float:
    amount = _num(fx.get("amount"), 1.0)
    t0 = _num(fx.get("t0"), 0.0)
    ramp = _num(fx.get("ramp"), 0.0)
    if ramp and ramp > 0:
        return amount * min(1.0, max(0.0, (t - t0) / ramp))
    return amount


def _active(fx: dict, t: float, dur: float) -> bool:
    """Effet/overlay actif à l'instant t (relatif à la shot) : fidx(t0) <= fidx(t) < fidx(t1)."""
    t0 = _num(fx.get("t0"), 0.0)
    t1 = _num(fx.get("t1"), None)
    if t1 is None:
        return fidx(t) >= fidx(t0)
    return fidx(t0) <= fidx(t) < fidx(t1)


class ShotRenderer:
    def __init__(self, job: dict):
        self.job = job
        self.shot = job["shot"]
        self.q = job["quality"]
        self.ow, self.oh = SIZES[self.q]
        self.n = job["n"]
        self.f0 = job["f0"]
        self.start = _num(self.shot.get("start"), 0.0)
        self.dur = _num(self.shot.get("dur"), self.n / FPS)
        self.tl = shot_local_times(self.start, self.f0, self.n)
        self.seed = _seed(self.shot.get("id"))
        self.fbase = (self.seed % 50000) * 100  # base des index d'image (grain/glitch déterministes)
        self.warnings: list[str] = []
        self._warned: set[str] = set()
        skip = set(job.get("flash_skip") or [])
        self.fx = [fx if isinstance(fx, dict) else {"type": fx} for k, fx in enumerate(self.shot.get("fx") or [])
                   if k not in skip]
        self.sil_meta = job.get("silhouette")
        self._sil_rgba = None
        self._subl = {}
        self._proj = None  # projection photo -> sortie (shots photo)
        self.src_range = (0.0, 0.0)
        self.G = gfx()

    # --- utilitaires --------------------------------------------------------
    def warn(self, msg: str):
        if msg not in self._warned:
            self._warned.add(msg)
            self.warnings.append(f"{self.shot.get('id')}: {msg}")

    def safe(self, name: str, fn, img, *a, **k):
        """Appelle un effet ; en cas d'exception, image inchangée + avertissement (le rendu continue)."""
        try:
            out = fn(img, *a, **k)
            return img if out is None else out
        except Exception as e:  # pragma: no cover - dépend des modules des autres agents
            self.warn(f"{name} a échoué ({type(e).__name__}: {e})")
            return img

    def sil_rgba(self):
        if self._sil_rgba is None and self.sil_meta and self.sil_meta.get("still"):
            from PIL import Image
            self._sil_rgba = np.asarray(Image.open(self.sil_meta["still"]).convert("RGBA"))
        return self._sil_rgba

    # --- visages ------------------------------------------------------------
    def needs_faces(self) -> bool:
        cam = self.shot.get("camera") or {}
        if cam.get("center") in ("face", "faces"):
            return True
        if any(fx.get("type") in FACE_FX + ("privacy_blur",) for fx in self.fx):
            return True
        return any(ov.get("type") == "hud_faces" for ov in self.shot.get("overlays") or [])

    def full_res_needed(self, t: float) -> bool:
        """Un effet visage ou une incrustation de silhouette est-il actif à t (travail en pleine résolution) ?"""
        if any(fx.get("type") in FACE_FX and _active(fx, t, self.dur) for fx in self.fx):
            return True
        sl = self.shot.get("silhouette")
        if sl and sl.get("space") != "photo":
            t0, t1 = _num(sl.get("t0"), 0.0), _num(sl.get("t1"), self.dur)
            return t0 - 1e-6 <= t < t1
        return False

    def face_track(self, proxy: Path, j0: int, j1: int):
        FF = facefx()
        try:
            return FF.FaceTrack.compute(str(proxy), j0 / FPS, (j1 + 1) / FPS, fps=FPS)
        except Exception as e:
            self.warn(f"FaceTrack.compute a échoué ({type(e).__name__}: {e})")
            return None

    def faces_from_image(self, u8: np.ndarray) -> list:
        FF = facefx()
        try:
            return _faces_list(FF.FaceTrack.from_image(u8))
        except Exception as e:
            self.warn(f"FaceTrack.from_image a échoué ({type(e).__name__}: {e})")
            return []

    def analysis_faces(self, clip: str, t_src: float) -> list:
        """Visages de l'analyse (faces.json, 5 i/s) au plus près de t_src (complément pour le floutage)."""
        cache = self.__dict__.setdefault("_anaf", {})
        if clip not in cache:
            p = config.ANALYSIS_DIR / f"{clip}.faces.json"
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                cache[clip] = (np.asarray(d.get("t", []), float), d.get("faces", []))
            except Exception:
                cache[clip] = (np.zeros(0), [])
        ts, faces = cache[clip]
        if len(ts) == 0:
            return []
        k = int(np.argmin(np.abs(ts - t_src)))
        if abs(ts[k] - t_src) > 0.25:
            return []
        return [_StubFace(f["bbox"], f.get("score", 0.5)) for f in faces[k] if "bbox" in f]

    def analysis_persons(self, clip: str | None = None, t_src: float | None = None) -> list:
        """Boîtes des personnes de l'analyse (persons.json, 5 i/s, px proxy) au plus près de l'instant source."""
        clip = clip or self.cur_clip()
        if t_src is None:
            t_src = (self.__dict__.get("_cur_src") or (None, 0.0))[1]
        cache = self.__dict__.setdefault("_anap", {})
        if clip not in cache:
            try:
                d = json.loads((config.ANALYSIS_DIR / f"{clip}.persons.json").read_text(encoding="utf-8"))
                cache[clip] = (np.asarray(d.get("t", []), float), d.get("persons", []))
            except Exception:
                cache[clip] = (np.zeros(0), [])
        ts, pers = cache[clip]
        if len(ts) == 0:
            return []
        k = int(np.argmin(np.abs(ts - t_src)))
        if abs(ts[k] - t_src) > 0.25:
            return []
        return [q["bbox"] for q in pers[k] if q.get("score", 0) >= 0.3 and "bbox" in q]

    def apply_face_fx(self, img, faces, t, fi, box=None):
        """Effets visages. Sans maillage sur cette image : visage connu le plus proche dans la shot (déjà
        substitué par l'appelant), sinon `box` (analyse / boîte manuelle) avec facefx.<effet>_from_box s'il existe."""
        FF = facefx()
        main = main_faces(_mesh_faces(faces))
        for fx in self.fx:
            typ = fx.get("type")
            if typ not in FACE_FX or not _active(fx, t, self.dur):
                continue
            a = _ramp_amount(fx, t, self.dur)
            if a <= 0:
                continue
            fn = getattr(FF, typ, None)
            if fn is None:
                self.warn(f"effet {typ} absent de facefx")
                continue
            if typ == "head_tic":
                img = self.safe(typ, fn, img, main[0] if main else None, fi, a)
            elif typ == "portrait_eyes":
                img = self.portrait_eyes(img, fn, faces, t, a)
            elif main:
                for f in main:
                    if typ in ("melt", "demon_face"):
                        img = self.safe(typ, fn, img, f, a, t)
                    else:
                        img = self.safe(typ, fn, img, f, a)
            else:
                fb = getattr(FF, f"{typ}_from_box", None)
                if box is not None and callable(fb):
                    img = self.safe(f"{typ}_from_box", fb, img, box, a, t) if typ in ("melt", "demon_face") \
                        else self.safe(f"{typ}_from_box", fb, img, box, a)
                else:
                    self.warn(f"{typ} : aucun visage (ni maillage, ni boîte exploitable) sur une partie de la shot")
        return img

    def portrait_eyes(self, img, fn, faces, t, a):
        """Yeux des TABLEAUX seulement : visages proches des yeux indiqués dans decisions
        (portrait_eyes_<clip>.eyes), sinon visages sans corps détecté ; abstention en cas de doute."""
        faces = _mesh_faces(faces)
        eyes = _dec_value(f"portrait_eyes_{_clip_num(self.cur_clip())}", "eyes")
        if isinstance(eyes, list) and eyes:
            pts = [(float(e[0]) * SRC_W, float(e[1]) * SRC_H) for e in eyes]
            sel = [f for f in faces if any(f.bbox[0] - 0.1 * (f.bbox[2] - f.bbox[0]) <= x <= f.bbox[2] + 0.1 *
                                           (f.bbox[2] - f.bbox[0]) and f.bbox[1] <= y <= f.bbox[3] for x, y in pts)]
            at = getattr(facefx(), "portrait_eyes_at", None)
            if not sel and callable(at):
                return self.safe("portrait_eyes_at", at, img, eyes, t, a)
        else:
            persons = self.analysis_persons()
            sel = []
            for f in faces:
                fh = f.bbox[3] - f.bbox[1]
                cx, cy = (f.bbox[0] + f.bbox[2]) / 2, (f.bbox[1] + f.bbox[3]) / 2
                body = any(b[0] <= cx <= b[2] and b[1] - 0.2 * fh <= cy <= b[1] + 0.45 * (b[3] - b[1])
                           and (b[3] - b[1]) >= 2.6 * fh for b in persons)
                if not body:
                    sel.append(f)
        if not sel:
            self.warn("portrait_eyes : aucun visage de tableau identifié avec certitude → effet non appliqué")
            return img
        return self.safe("portrait_eyes", fn, img, sel, t, a)

    # --- silhouette ---------------------------------------------------------
    def apply_silhouette(self, img, t, fi, slot=None, opacity_mul=1.0):
        sl = slot or self.shot.get("silhouette")
        if not sl:
            return img
        rgba = self.sil_rgba()
        if rgba is None:
            self.warn("silhouette demandée mais aucune image détourée")
            return img
        t0 = _num(sl.get("t0"), 0.0)
        t1 = _num(sl.get("t1"), self.dur)
        if slot is None and not (t0 - 1e-6 <= t < t1):
            return img
        fade = _num(sl.get("fade_in"), 0.0) or 0.0
        op = _num(sl.get("opacity"), 0.9) * opacity_mul
        if fade > 0 and slot is None:
            op *= min(1.0, max(0.0, (t - t0) / fade))
        if op <= 0.003:
            return img
        S = silhouette_mod()
        occ = None
        if sl.get("occlude_persons"):
            # plaque fixe : masque des personnes calculé une seule fois ; plan animé : recalculé toutes les
            # OCC_STEP images (segmentation coûteuse, ~1 s/image ; les personnes bougent peu en 0,1 s)
            static = self.shot.get("kind") in ("photo", "freeze") or slot is not None
            key = (id(slot), img.shape) if static else ("dyn", (fi - self.fbase) // OCC_STEP[self.q], img.shape)
            cache = self.__dict__.setdefault("_occ", {})
            occ = cache.get(key)
            if occ is None:
                disk = self.occ_disk_path(img) if not static else None
                if disk is not None and disk.exists():
                    occ = self.occ_load(disk, img.shape)
                if occ is None:
                    try:
                        occ = S.person_mask(img)
                        if disk is not None:
                            self.occ_save(disk, occ)
                    except Exception as e:
                        self.warn(f"person_mask a échoué ({e})")
                if occ is not None:
                    if not static:
                        cache.clear()
                    cache[key] = occ
        return self.safe("silhouette.composite", S.composite, img, rgba,
                         _num(sl.get("x"), 0.5), _num(sl.get("foot_y"), 0.7), _num(sl.get("height"), 0.3),
                         opacity=op, blur=_num(sl.get("blur"), 1.5), darkness=_num(sl.get("darkness"), 0.9),
                         blend=sl.get("blend", "normal") or "normal", frame_idx=fi, occluder_mask=occ)

    def occ_disk_path(self, img) -> Path | None:
        """Cache disque du masque des personnes d'une image source (partagé preview/final : même image proxy)."""
        cur = self.__dict__.get("_cur_src")
        if not cur or not cur[0] or img.shape[:2] != (SRC_H, SRC_W):
            return None
        j = fidx(cur[1])
        b = (j // OCC_STEP["final"]) * OCC_STEP["final"]
        S = silhouette_mod()
        ver = _hash([getattr(S, "VERSION", "?"), _file_sig(config.proxy_path(cur[0]))])[:8]
        return config.CACHE_DIR / "person_mask" / f"{cur[0]}_{ver}" / f"{b:05d}.png"

    @staticmethod
    def occ_load(p: Path, shape) -> np.ndarray | None:
        import cv2
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None or m.shape[:2] != shape[:2]:
            return None
        return m.astype(np.float32) / 255.0

    @staticmethod
    def occ_save(p: Path, occ) -> None:
        import cv2
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.stem + f".{os.getpid()}.tmp.png")
            cv2.imwrite(str(tmp), np.clip(np.asarray(occ, np.float32) * 255 + 0.5, 0, 255).astype(np.uint8))
            os.replace(tmp, p)
        except Exception:
            pass

    def subliminal_image(self, what: str) -> np.ndarray:
        """Image subliminale plein cadre : silhouette.subliminal_frame (« lui » à ~0,8·H en contre-jour) si le
        module l'expose ; sinon repli local (silhouette noire sur mur éclairé, ou son négatif)."""
        if what in self._subl:
            return self._subl[what]
        fn = getattr(silhouette_mod(), "subliminal_frame", None)
        if callable(fn):
            try:
                img = fn((self.sil_meta or {}).get("still"), self.ow, self.oh,
                         negative=(what == "silhouette_negative"), frame_idx=self.fbase)
                self._subl[what] = np.asarray(img, np.float32)
                return self._subl[what]
            except Exception as e:
                self.warn(f"silhouette.subliminal_frame a échoué ({type(e).__name__}: {e}) -> repli local")
        import cv2
        W, H = self.ow, self.oh
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rng = np.random.default_rng(self.seed)
        # mur éclairé par des phares : halo ovale, bas plus sombre
        glow = np.exp(-(((xx / W - 0.5) / 0.55) ** 2 + ((yy / H - 0.42) / 0.5) ** 2))
        base = 0.18 + 0.62 * glow
        noise = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), W / 90) * 0.05
        plate = np.clip(base + noise, 0, 1)[..., None] * np.array([0.86, 0.9, 0.95], np.float32)
        plate = plate.astype(np.float32)
        rgba = self.sil_rgba()
        if rgba is not None:
            a = rgba[..., 3].astype(np.float32) / 255.0
            ys, xs = np.nonzero(a > 0.1)
            if len(ys):
                crop = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255.0
                # plein cadre : silhouette à ~0,8 × la hauteur de l'image
                hh = int(H * 0.8)
                ww = max(2, int(crop.shape[1] * hh / crop.shape[0]))
                if ww > W * 0.95:
                    ww = int(W * 0.95)
                    hh = max(2, int(crop.shape[0] * ww / crop.shape[1]))
                interp = cv2.INTER_AREA if hh < crop.shape[0] else cv2.INTER_CUBIC
                c = np.clip(cv2.resize(crop, (ww, hh), interpolation=interp), 0, 1)
                x0, y0 = (W - ww) // 2, H - hh - int(H * 0.04)
                al = cv2.GaussianBlur(c[..., 3], (0, 0), max(0.6, W / 900))[..., None]
                plate[y0:y0 + hh, x0:x0 + ww] = plate[y0:y0 + hh, x0:x0 + ww] * (1 - al) + 0.02 * al
        if what == "silhouette_negative":
            # négatif : fond presque noir, silhouette blanche qui irradie
            neg = np.clip(1.0 - plate, 0, 1) ** 2.4
            glow = cv2.GaussianBlur(neg, (0, 0), W / 60) * 0.55
            img = np.clip(neg + glow, 0, 1)
        else:
            img = plate
        self._subl[what] = img.astype(np.float32)
        return self._subl[what]

    # --- post-traitement commun -------------------------------------------------
    def grade(self, img, t):
        g = self.shot.get("grade")
        if not g:
            return img
        preset = g.get("preset", "cold")
        kw = {k: g[k] for k in ("red_accent", "keep_blue", "warm_highlights") if k in g}
        mix = g.get("mix")
        if isinstance(mix, (list, tuple)) and len(mix) == 2:
            u = min(1.0, max(0.0, t / max(self.dur, 1e-6)))
            kw["mix"] = float(mix[0]) + (float(mix[1]) - float(mix[0])) * u
        elif mix is not None:
            kw["mix"] = _num(mix, 0.0)
        if hasattr(self.G, "exposure_gain"):  # gain d'exposition FIXE par plan (aucun pompage)
            kw["auto_exposure"] = False if g.get("auto_exposure") is False else self.exposure_gain()
        return self.safe(f"grade {preset}", self.G.grade, img, preset, **kw)

    def exposure_gain(self):
        """Gain d'exposition du plan, calculé une fois (gfx.exposure_gain) sur ~6 images de la source ;
        True (= estimation image par image) en cas d'échec."""
        if "_ae_gain" in self.__dict__:
            return self._ae_gain
        gain = True
        try:
            kind = self.shot.get("kind")
            frames = []
            if kind == "clip":
                proxy = self._clip_proxy(self.shot["clip"])
                js = source_frames(self.shot, self.tl)
                for j in np.unique(np.linspace(js.min(), js.max(), 6).astype(int)):
                    frames.append(read_single_frame(proxy, int(j)))
            elif kind == "freeze":
                src = self.job.get("freeze") or {}
                if "clip" in src:
                    frames.append(read_single_frame(self._clip_proxy(src["clip"]), fidx(_num(src.get("t"), 0.0))))
            if frames:
                gain = float(self.G.exposure_gain(frames))
        except Exception as e:
            self.warn(f"exposure_gain a échoué ({type(e).__name__}: {e}) -> estimation image par image")
        self._ae_gain = gain
        return gain

    def privacy_regions(self, faces: list, cam) -> list:
        """Personnes secondaires (analyse efficientdet, hors sujet(s) principal(aux)) + zones manuelles
        decisions["privacy_zones"]["zones"][clip] = [[x0, y0, x1, y1, t0, t1]] (normalisées, temps source)."""
        clip, t_src = self.__dict__.get("_cur_src") or (self.cur_clip(), None)
        out = []
        if clip and t_src is not None:
            persons = self.analysis_persons(clip, t_src)
            mf = main_faces(faces)
            if persons:
                hmax = max(b[3] - b[1] for b in persons)
                for b in persons:
                    h = b[3] - b[1]
                    holds_main = any(b[0] <= (f.bbox[0] + f.bbox[2]) / 2 <= b[2] and b[1] - 0.1 * h <=
                                     (f.bbox[1] + f.bbox[3]) / 2 <= b[3] for f in mf)
                    if h >= 0.5 * hmax or holds_main or h < 0.03 * SRC_H:
                        continue  # sujet principal (jamais flouté) ou trop petit pour être reconnaissable
                    out.append(list(b))
            zones = ((_decisions().get("privacy_zones") or {}).get("zones") or {}).get(clip) or []
            for z in zones:
                if len(z) >= 4 and (len(z) < 6 or float(z[4]) <= t_src <= float(z[5])):
                    out.append([z[0] * SRC_W, z[1] * SRC_H, z[2] * SRC_W, z[3] * SRC_H])
        return [cam.box(b) if cam else list(b) for b in out]

    def blur_regions(self, img, boxes: list):
        """Assombrit + floute des zones (boîtes de sortie), bords adoucis."""
        import cv2
        H, W = img.shape[:2]
        for b in boxes:
            x0, y0 = int(max(0, b[0] - 0.08 * (b[2] - b[0]))), int(max(0, b[1] - 0.05 * (b[3] - b[1])))
            x1, y1 = int(min(W, b[2] + 0.08 * (b[2] - b[0]))), int(min(H, b[3] + 0.05 * (b[3] - b[1])))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            roi = img[y0:y1, x0:x1]
            sig = max(2.0, (x1 - x0) / 10)
            bl = cv2.GaussianBlur(roi, (0, 0), sig) * 0.55
            f = max(2.0, min(x1 - x0, y1 - y0) * 0.15)  # bords adoucis (rampe linéaire)
            ry = np.minimum(np.arange(y1 - y0), np.arange(y1 - y0)[::-1]) / f
            rx = np.minimum(np.arange(x1 - x0), np.arange(x1 - x0)[::-1]) / f
            m = np.clip(np.minimum.outer(ry, rx), 0.0, 1.0).astype(np.float32)[..., None]
            img[y0:y1, x0:x1] = roi * (1 - m) + bl * m
        return img

    def privacy_blur(self, img, faces_out: list):
        """Floute les visages hors sujets principaux (boîtes en coordonnées de sortie)."""
        import cv2
        if not faces_out:
            return img
        hmax = max(b[3] - b[1] for b in faces_out)
        H, W = img.shape[:2]
        for b in faces_out:
            h = b[3] - b[1]
            if h >= 0.5 * hmax and h > 0.06 * H:
                continue  # sujet principal
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            rx, ry = max(4.0, (b[2] - b[0]) * 0.8), max(5.0, h * 0.85)
            x0, x1 = int(max(0, cx - rx * 1.6)), int(min(W, cx + rx * 1.6))
            y0, y1 = int(max(0, cy - ry * 1.6)), int(min(H, cy + ry * 1.6))
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            roi = img[y0:y1, x0:x1]
            sig = max(2.0, rx / 2.5)
            bl = cv2.GaussianBlur(roi, (0, 0), sig) * 0.82
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            m = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
            m = np.clip((1.25 - m) / 0.5, 0, 1)[..., None]
            img[y0:y1, x0:x1] = roi * (1 - m) + bl * m
        return img

    def post(self, img, t, k, cam: Camera | None, faces: list, allow_finish=True, subl_ok=True):
        """Étalonnage → floutage → glitch/négatif/subliminal → film_finish → overlays → fondus → sous-titres."""
        fi = self.fbase + k
        G = self.G
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = self.grade(img, t)
        # floutage des inconnus
        if any(fx.get("type") == "privacy_blur" and _active(fx, t, self.dur) for fx in self.fx):
            img = self.blur_regions(img, self.privacy_regions(faces, cam))
            boxes = [cam.box(f.bbox) if cam else list(f.bbox) for f in faces]
            img = self.privacy_blur(img, boxes)
        # effets d'image
        subliminal = False
        for fx in self.fx:
            typ = fx.get("type")
            if typ == "subliminal" and subl_ok:
                t0 = _num(fx.get("t0"))
                nfr = int(_num(fx.get("frames"), 2))
                if t0 is not None and fidx(self.start + t0) <= self.f0 + k < fidx(self.start + t0) + nfr:
                    img = self.subliminal_image(fx.get("what", "silhouette")).copy()
                    subliminal = True
                continue
            if not _active(fx, t, self.dur):
                continue
            if typ == "glitch":
                img = self.safe("glitch", G.glitch, img, fi, _num(fx.get("amount"), 1.0))
            elif typ == "rgb_split":
                img = self.safe("rgb_split", G.rgb_split, img, _num(fx.get("amount"), 0.4), fi)
            elif typ == "negative":
                img = self.safe("negative", G.negative, img)
            elif typ == "darken":
                t0 = _num(fx.get("t0"), 0.0)
                a = _num(fx.get("amount"), 0.3) * min(1.0, max(0.0, (t - t0) / 0.5))
                img = self.safe("darken", G.darken, img, a)
        if allow_finish or subliminal:
            img = self.safe("film_finish", G.film_finish, img, fi)
        if not subliminal:
            img = self.overlays(img, t, k, cam, faces)
        # fondus (avant les sous-titres, qui restent lisibles)
        for fx in self.fx:
            typ = fx.get("type")
            if typ in ("fade_to_black", "fade_from_black"):
                t0 = _num(fx.get("t0"), 0.0)
                t1 = _num(fx.get("t1"), self.dur)
                u = min(1.0, max(0.0, (t - t0) / max(1e-6, t1 - t0)))
                if typ == "fade_to_black" and t >= t0:
                    img = img * (1.0 - u) ** 1.6
                elif typ == "fade_from_black" and t < t1:
                    img = img * u ** 1.6
        return self.subtitles(img, t)

    # --- overlays -------------------------------------------------------------
    def overlays(self, img, t, k, cam, faces):
        G = self.G
        fi = self.fbase + k
        for ov in self.shot.get("overlays") or []:
            typ = ov.get("type")
            t0 = _num(ov.get("t0"), 0.0)
            t1 = _num(ov.get("t1"), self.dur)
            if not _active(ov, t, self.dur):
                continue
            if typ == "timecode":
                img = self.safe("timecode", G.draw_timecode, img, str(ov.get("text", "")), str(ov.get("date", "")),
                                t - t0)
            elif typ == "card":
                img = self.safe("card", G.draw_card, img, str(ov.get("text", "")), t - t0, t1 - t0,
                                style=ov.get("style", "card"))
            elif typ == "name":
                if ov.get("show", False):
                    img = self.safe("name", G.draw_name, img, str(ov.get("name", "")), t - t0)
            elif typ == "title_sub":
                img = self.safe("title_sub", G.draw_title_sub, img, str(ov.get("text", "")), t - t0, t1 - t0)
            elif typ == "hud_faces":
                img = self.hud(img, ov, t, t0, fi, cam, faces)
            elif typ == "red_circles":
                img = self.red_circles(img, ov, t, t0, cam, faces, project=self._proj)
        return img

    def hud_faces(self, faces: list, want: int) -> list:
        """Visages du HUD : les 4 plus grands (≥ 30 % de la taille du plus grand), gauche → droite ; repli manuel
        decisions["hud_boxes_<clip>"]["boxes"] (boîtes normalisées) si la détection en trouve moins que voulu."""
        fs = []
        if faces:
            hmax = max(f.bbox[3] - f.bbox[1] for f in faces)
            fs = sorted([f for f in faces if f.bbox[3] - f.bbox[1] >= 0.3 * hmax], key=lambda f: -_area(f.bbox))[:4]
        boxes = _dec_value(f"hud_boxes_{_clip_num(self.cur_clip())}", "boxes")
        if len(fs) < want and isinstance(boxes, list) and boxes:
            fs = [_StubFace([b[0] * SRC_W, b[1] * SRC_H, b[2] * SRC_W, b[3] * SRC_H]) for b in boxes[:4]]
            self.warn(f"hud_faces : {len(fs)} boîte(s) manuelle(s) (decisions hud_boxes) au lieu de la détection")
        return sorted(fs, key=lambda f: (f.bbox[0] + f.bbox[2]) / 2)

    def cur_clip(self) -> str | None:
        if self.shot.get("clip"):
            return self.shot["clip"]
        return (self.job.get("freeze") or {}).get("clip")

    def hud(self, img, ov, t, t0, fi, cam, faces):
        G = self.G
        order = ov.get("order")
        want = len(order) if isinstance(order, (list, tuple)) and order else 4
        fs = self.hud_faces(faces, want)
        if not fs:
            self.warn("hud_faces : aucun visage détecté")
            return img
        names = None
        if isinstance(order, (list, tuple)) and order:
            if len(fs) >= len(order):
                if len(fs) > len(order):  # les plus grands, toujours de gauche à droite
                    fs = sorted(sorted(fs, key=lambda f: -_area(f.bbox))[:len(order)],
                                key=lambda f: (f.bbox[0] + f.bbox[2]) / 2)
                names = [str(n).upper() for n in order]
            else:
                self.warn(f"hud_faces : {len(fs)} visage(s) pour {len(order)} prénom(s) -> étiquettes sans prénom")
        mode = ov.get("mode", "identify")
        step = _num(ov.get("step"), 0.42 if mode == "identify" else 0.0)
        locks = ov.get("locks") if isinstance(ov.get("locks"), (list, tuple)) else None
        for i, f in enumerate(fs):
            lock = _num(locks[i], None) if locks and i < len(locks) else None
            tl = t - (lock if lock is not None else t0 + i * step)
            if tl < -0.3:  # gfx : entre −0,3 et 0 s = phase de recherche avant verrouillage
                continue
            n = f"{i + 1:02d}"
            if mode == "victim":
                label = f"VICTIME {n} — {names[i]}" if names else f"VICTIME {n}"
            else:
                label = f"SUJET {n} — {names[i]} — IDENTIFIÉ" if names else f"SUJET {n} — IDENTIFIÉ"
            sub = None  # sous-étiquette automatique (gfx)
            box = cam.box(f.bbox) if cam else list(f.bbox)
            lm = None
            if getattr(f, "lm", None) is not None and cam is not None and not getattr(f, "box_only", False):
                lm = (np.asarray(f.lm, np.float32) - [cam.ox, cam.oy]) * cam.s + [cam.dx, cam.dy]
            img = self.safe("hud", G.draw_hud_face, img, box, tl, label, sub, fi, mode=mode, landmarks=lm)
        return img

    def red_circles(self, img, ov, t, t0, cam, faces, project=None):
        G = self.G
        targets = ov.get("targets", "faces")
        step = _num(ov.get("step"), 0.38)
        draw = max(0.05, _num(ov.get("draw"), 0.32))
        frantic = bool(ov.get("frantic", False))
        proj = project or (lambda x, y: cam.pt(x, y) if cam else (x, y))
        circles = []
        if targets == "silhouette":
            sl = self.shot.get("silhouette") or {}
            sx, fy, sh = _num(sl.get("x"), 0.5), _num(sl.get("foot_y"), 0.6), _num(sl.get("height"), 0.3)
            pw, ph = self.job.get("plate_size") or (SRC_W, SRC_H)
            cx, cy = sx * pw, (fy - sh * 0.5) * ph
            circles.append(((cx, cy), (sh * ph * 0.32, sh * ph * 0.62)))
        else:
            want = len(ov.get("draws") or []) or 4
            centers = _dec_value("group_photo_faces", "centers")
            pw, ph = self.job.get("plate_size") or (SRC_W, SRC_H)
            if len(faces) != want and isinstance(centers, list) and centers:
                # repli manuel : [[x, y, rayon]] normalisés dans la photo (rayon en fraction de la largeur)
                for c in sorted(centers, key=lambda c: c[0])[:want]:
                    r = float(c[2]) if len(c) > 2 else 0.05
                    circles.append(((float(c[0]) * pw, float(c[1]) * ph), (r * pw, r * pw * 1.18)))
                self.warn(f"red_circles : {len(faces)} visage(s) détecté(s) pour {want} → centres manuels (decisions)")
            else:
                fs = sorted(sorted(faces, key=lambda f: -_area(f.bbox))[:want], key=lambda f: (f.bbox[0] + f.bbox[2]) / 2)
                for f in fs:
                    b = f.bbox
                    circles.append((((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), ((b[2] - b[0]) * 0.78, (b[3] - b[1]) * 0.8)))
                if len(fs) != want:
                    self.warn(f"red_circles : {len(fs)} visage(s) détecté(s) sur la photo pour {want} cercles "
                              "(renseigner decisions group_photo_faces.centers)")
        draws = ov.get("draws") if isinstance(ov.get("draws"), (list, tuple)) else None
        for i, ((cx, cy), (ax, ay)) in enumerate(circles):
            d0 = _num(draws[i], None) if draws and i < len(draws) else None
            p = (t - (d0 if d0 is not None else t0 + i * step)) / draw
            if p <= 0:
                continue
            (px, py) = proj(cx, cy)
            (qx, _) = proj(cx + 4.0, cy)  # échelle locale (même profondeur que le centre)
            sc = abs(qx - px) / 4.0
            img = self.safe("red_circle", G.draw_red_circle, img, (px, py), (ax * sc, ay * sc),
                            min(1.0, p), self.seed + 17 * i, frantic=frantic)
        return img

    def subtitles(self, img, t):
        T = self.start + t
        for sb in self.job.get("subs") or []:
            a, b = sb["start"], sb["end"]
            if a - 1e-6 <= T < b:
                alpha = min(1.0, (T - a) / SUB_FADE_S + 0.25, (b - T) / SUB_FADE_S)
                img = self.safe("subtitle", self.G.draw_subtitle, img, sb["text"], sb.get("kind", "dialogue"),
                                alpha=max(0.0, min(1.0, alpha)))
        return img

    # --- caméra -----------------------------------------------------------------
    def resolve_center(self, cam: dict, faces_seq: list[list]) -> tuple[float, float]:
        c = cam.get("center")
        if isinstance(c, (list, tuple)) and len(c) == 2 and all(_num(v) is not None for v in c):
            return float(c[0]) * SRC_W, float(c[1]) * SRC_H
        if c in ("face", "faces"):
            pts = []
            for faces in faces_seq:
                mf = main_faces(faces)
                if not mf:
                    continue
                if c == "face":
                    b = mf[0].bbox
                else:
                    b = (min(f.bbox[0] for f in mf), min(f.bbox[1] for f in mf),
                         max(f.bbox[2] for f in mf), max(f.bbox[3] for f in mf))
                pts.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
            if pts:
                cx, cy = np.median(np.asarray(pts), axis=0)
                z = cam.get("zoom", (1.0, 1.0))
                zmax = max(z) if isinstance(z, (list, tuple)) else _num(z, 1.0)
                if c == "face":  # visage légèrement au-dessus du centre (règle des tiers)
                    cy += 0.06 * SRC_H / max(1.0, zmax)
                return float(cx), float(cy)
            self.warn(f"centre caméra '{c}' : aucun visage, centre de l'image")
            return SRC_W * 0.5, SRC_H * 0.45
        if c == "person":
            clip = self.shot.get("clip")
            p = config.ANALYSIS_DIR / f"{clip}.persons.json"
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                ts = np.asarray(d.get("t", []), float)
                boxes = d.get("persons", d.get("boxes", d.get("faces", [])))
                a, b = self.src_range
                pts = []
                for ti, bl in zip(ts, boxes):
                    if a - 0.1 <= ti <= b + 0.1 and bl:
                        bb = max(bl, key=lambda o: _area(o["bbox"]))["bbox"]
                        pts.append(((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2))
                if pts:
                    cx, cy = np.median(np.asarray(pts), axis=0)
                    return float(cx), float(cy)
            except Exception:
                pass
            self.warn("centre caméra 'person' : pas de détection, centre de l'image")
            return SRC_W * 0.5, SRC_H * 0.45
        if c is not None:
            self.warn(f"centre caméra non résolu ({c!r}) -> centre de l'image")
        return SRC_W * 0.5, SRC_H * 0.5

    def camera_at(self, t, k, center) -> Camera:
        cam = self.shot.get("camera") or {}
        z = zoom_at(cam, t, self.dur)
        dx = dy = 0.0
        for fx in self.fx:
            if fx.get("type") == "shake" and _active(fx, t, self.dur):
                rng = np.random.default_rng(self.seed * 1000 + k)
                amp = _num(fx.get("amount"), 0.5) * 0.012 * self.ow
                dx, dy = (rng.standard_normal(2) * amp).tolist()
        return Camera(self.ow, self.oh, z, center[0], center[1], shake=(dx, dy), lanczos=self.q == "final")

    # --- types de shots -----------------------------------------------------------
    def render(self, enc: Encoder):
        kind = self.shot.get("kind", "clip")
        fn = {"clip": self.r_clip, "freeze": self.r_freeze, "black": self.r_black, "card": self.r_card,
              "photo": self.r_photo, "title": self.r_title, "montage": self.r_montage}.get(kind)
        if fn is None:
            self.warn(f"type de shot inconnu '{kind}' -> noir")
            fn = self.r_black
        fn(enc)

    def r_black(self, enc):
        for k in range(self.n):
            img = np.zeros((self.oh, self.ow, 3), np.float32)
            enc.write(self.post(img, float(self.tl[k]), k, None, [], allow_finish=False))

    def r_card(self, enc):
        for k in range(self.n):
            img = np.zeros((self.oh, self.ow, 3), np.float32)
            enc.write(self.post(img, float(self.tl[k]), k, None, []))

    def _clip_proxy(self, clip) -> Path:
        p = config.proxy_path(clip)
        if not p.exists():
            raise FileNotFoundError(f"proxy absent : {p} (lance l'étape « ingest »)")
        return p

    def r_clip(self, enc):
        clip = self.shot["clip"]
        proxy = self._clip_proxy(clip)
        js = source_frames(self.shot, self.tl)
        j0, j1 = int(js.min()), int(js.max())
        self.src_range = (j0 / FPS, j1 / FPS)
        track = self.face_track(proxy, j0, j1) if self.needs_faces() else None
        blur = any(fx.get("type") == "privacy_blur" for fx in self.fx)

        def faces_at(j):
            fs = []
            if track is not None:
                try:
                    fs = list(track.at(int(j - j0)) or [])
                except Exception as e:
                    self.warn(f"FaceTrack.at a échoué ({e})")
            if blur:  # complément : petits visages de l'analyse (5 i/s)
                for g in self.analysis_faces(clip, j / FPS):
                    if all(_iou(g.bbox, f.bbox) < 0.3 for f in fs):
                        fs.append(g)
            return fs

        cam_spec = self.shot.get("camera") or {}
        faces_seq = []
        if cam_spec.get("center") in ("face", "faces") and track is not None:
            sample = sorted(set(np.linspace(j0, j1, min(24, j1 - j0 + 1)).astype(int).tolist()))
            faces_seq = [faces_at(j) for j in sample]
        center = self.resolve_center(cam_spec, faces_seq)
        need_fx = any(fx.get("type") in FACE_FX for fx in self.fx)
        known = {}  # image source -> visages avec maillage (pour les images où le suivi décroche)
        if need_fx and track is not None:
            for jj in range(j0, j1 + 1):
                try:
                    mf = _mesh_faces(list(track.at(int(jj - j0)) or []))
                except Exception:
                    mf = []
                if mf:
                    known[jj] = mf
        manual = (_decisions().get("face_boxes") or {}).get(self.shot.get("id"))

        def fx_faces(j, faces):
            """(visages pour les effets, boîte de repli) : maillage de l'image, sinon visage connu le plus proche
            recalé sur la boîte de l'analyse, sinon boîte (analyse / manuelle decisions face_boxes)."""
            if main_faces(_mesh_faces(faces)) or not need_fx:
                return faces, None
            ab = sorted(self.analysis_faces(clip, j / FPS), key=lambda g: -_area(g.bbox))
            if known:
                jn = min(known, key=lambda x: abs(x - j))
                fs = known[jn]
                if ab and hasattr(fs[0], "transformed"):
                    f0 = max(fs, key=lambda f: _area(f.bbox))
                    b, g = f0.bbox, ab[0].bbox
                    sc = (g[2] - g[0]) / max(1.0, b[2] - b[0])
                    if 0.6 <= sc <= 1.6:
                        M = np.float32([[sc, 0, (g[0] + g[2]) / 2 - sc * (b[0] + b[2]) / 2],
                                        [0, sc, (g[1] + g[3]) / 2 - sc * (b[1] + b[3]) / 2]])
                        fs = [f.transformed(M) for f in fs]
                self.warn("effets visages : suivi perdu sur une partie de la shot → visage connu le plus proche")
                return fs, None
            if ab:
                return faces, ab[0].bbox
            if isinstance(manual, (list, tuple)) and len(manual) == 4:
                return faces, (manual[0] * SRC_W, manual[1] * SRC_H, manual[2] * SRC_W, manual[3] * SRC_H)
            return faces, None

        reader = FrameReader(proxy, j0, j1 - j0 + 1)
        try:
            hold_fn = getattr(facefx(), "head_tic_hold", None)
            tics = [fx for fx in self.fx if fx.get("type") == "head_tic"]
            for k in range(self.n):
                t = float(self.tl[k])
                fi = self.fbase + k
                j = int(js[k])
                for fx in tics:  # saut d'image pendant un tic (répétition d'images, extension facefx)
                    if hold_fn is not None and _active(fx, t, self.dur):
                        try:
                            j = max(j0, j - int(hold_fn(fi, _ramp_amount(fx, t, self.dur))))
                        except Exception:
                            pass
                u8 = reader.get(j)
                faces = faces_at(j)
                self._cur_src = (clip, j / FPS)
                cam = self.camera_at(t, k, center)
                if self.full_res_needed(t):
                    img = u8.astype(np.float32) * (1.0 / 255.0)
                    ff, fbox = fx_faces(j, faces)
                    img = self.apply_face_fx(img, ff, t, fi, box=fbox)
                    img = self.apply_silhouette(img, t, fi)
                    img = cam.apply(img)
                else:  # rien à faire en pleine résolution : recadrage direct en 8 bits (plus rapide)
                    img = cam.apply(u8).astype(np.float32) * (1.0 / 255.0)
                enc.write(self.post(img, t, k, cam, faces))
            if reader.short:
                self.warn(f"{clip} : source plus courte que demandé (dernière image répétée)")
        finally:
            reader.close()

    def r_freeze(self, enc):
        src = self.job.get("freeze") or {}
        if "photo" in src:
            p = config.proxy_path(src["photo"])
            u8 = _load_photo_u8(p, (SRC_W, SRC_H))
        elif "clip" in src:
            proxy = self._clip_proxy(src["clip"])
            u8 = read_single_frame(proxy, fidx(_num(src.get("t"), 0.0)))
        else:
            self.warn("gel sans source -> noir")
            return self.r_black(enc)
        self.src_range = (0, 0)
        if "clip" in src:
            self._cur_src = (src["clip"], _num(src.get("t"), 0.0))
        faces = self.faces_from_image(u8) if self.needs_faces() else []
        fbox = None
        if not main_faces(_mesh_faces(faces)) and "clip" in src:
            ab = sorted(self.analysis_faces(src["clip"], _num(src.get("t"), 0.0)), key=lambda g: -_area(g.bbox))
            manual = (_decisions().get("face_boxes") or {}).get(self.shot.get("id"))
            if ab:
                fbox = ab[0].bbox
            elif isinstance(manual, (list, tuple)) and len(manual) == 4:
                fbox = (manual[0] * SRC_W, manual[1] * SRC_H, manual[2] * SRC_W, manual[3] * SRC_H)
        center = self.resolve_center(self.shot.get("camera") or {}, [faces])
        base = u8.astype(np.float32) * (1.0 / 255.0)
        pre = None
        for k in range(self.n):
            t = float(self.tl[k])
            fi = self.fbase + k
            cam = self.camera_at(t, k, center)
            if self.full_res_needed(t):
                img = self.apply_face_fx(base.copy(), faces, t, fi, box=fbox)
                img = self.apply_silhouette(img, t, fi)
                img = cam.apply(img)
            else:
                if pre is None and cam.s < 0.999:
                    import cv2
                    pre = cv2.resize(base, (self.ow, int(round(base.shape[0] * self.ow / base.shape[1]))),
                                     interpolation=cv2.INTER_AREA)
                img = cam.apply(base, pre=pre)
            enc.write(self.post(img, t, k, cam, faces))

    def r_photo(self, enc):
        pid = self.shot.get("photo")
        p = config.proxy_path(pid)
        if not p.exists():
            raise FileNotFoundError(f"photo absente : {p} (lance l'étape « ingest »)")
        from PIL import Image
        u8 = np.asarray(Image.open(p).convert("RGB"))
        ph, pw = u8.shape[:2]
        self.job["plate_size"] = (pw, ph)
        plate = u8.astype(np.float32) / 255.0
        try:
            self._ae_gain = float(self.G.exposure_gain([plate])) if hasattr(self.G, "exposure_gain") else True
        except Exception as e:
            self.warn(f"exposure_gain a échoué ({e}) -> estimation image par image")
            self._ae_gain = True
        P = parallax()
        try:
            dmap = P.depth(plate)
        except Exception as e:
            self.warn(f"parallax.depth a échoué ({e}) -> profondeur plate")
            dmap = np.zeros(plate.shape[:2], np.float32)
        faces = self.faces_from_image(u8)
        cam = self.shot.get("camera") or {}
        z = cam.get("zoom", (1.0, 1.16))
        zoom = (float(z[0]), float(z[1])) if isinstance(z, (list, tuple)) else (1.0, 1.16)
        c = cam.get("center")
        center = (float(c[0]), float(c[1])) if isinstance(c, (list, tuple)) and len(c) == 2 else (0.5, 0.45)
        amount = _num(cam.get("parallax"), 1.0)
        sil = self.shot.get("silhouette")
        pl_key, pl = None, plate
        for k in range(self.n):
            t = float(self.tl[k])
            if sil:
                # plaque photo + silhouette mise à jour toutes les 3 images (la parallaxe réutilise alors ses
                # couches en cache ; la respiration de la silhouette est lente, 0,3 Hz)
                kq = (k // 3) * 3
                if kq != pl_key:
                    pl_key = kq
                    pl = self.apply_silhouette(plate, float(self.tl[kq]), self.fbase + kq)
            try:
                img = P.render_parallax(pl, dmap, t, self.dur, zoom=zoom, center=center, amount=amount,
                                        out_size=(self.ow, self.oh))
            except Exception as e:
                self.warn(f"render_parallax a échoué ({e}) -> poussée 2D")
                img = _stub_parallax().render_parallax(pl, dmap, t, self.dur, zoom, center, amount,
                                                       (self.ow, self.oh))
            self._proj = _photo_projector(P, pw, ph, self.ow, self.oh, zoom, center, t, self.dur, dmap, amount)
            enc.write(self.post(img, t, k, None, faces))

    def r_title(self, enc):
        seq = config.TITLE_DIR / "title_seq.mp4"
        if not seq.exists():
            self.warn("assets/title/title_seq.mp4 absent : carton titre de repli (gfx)")
            for k in range(self.n):
                t = float(self.tl[k])
                img = _fallback_title(self.ow, self.oh, t, self.G)
                enc.write(self.post(img, t, k, None, []))
            return
        reader = FrameReader(seq, 0, self.n, size=(self.ow, self.oh))
        try:
            for k in range(self.n):
                t = float(self.tl[k])
                img = reader.get(k).astype(np.float32) / 255.0
                enc.write(self.post(img, t, k, None, []))
            if reader.short:
                self.warn("title_seq.mp4 plus court que la shot (dernière image tenue)")
        finally:
            reader.close()

    def r_montage(self, enc):
        items = self.shot.get("items") or []
        if not items:
            return self.r_black(enc)
        # partition des images de la shot entre les items
        bounds, acc = [], 0.0
        if all(isinstance(it.get("frames"), int) for it in items):  # nombre d'images fourni par le builder
            for it in items:
                bounds.append(int(acc))
                acc += it["frames"]
        else:
            for it in items:
                st = _num(it.get("start"), acc)
                bounds.append(fidx(st))
                acc = st + _num(it.get("dur"), 0.2)
        bounds = [min(max(0, b), self.n) for b in bounds] + [self.n]
        bounds[0] = 0
        k = 0
        for i, it in enumerate(items):
            a, b = bounds[i], max(bounds[i], bounds[i + 1])
            if b <= a:
                continue
            self._montage_item(enc, it, i, a, b)
            k += b - a
        if k < self.n:  # sécurité : compléter
            for kk in range(k, self.n):
                enc.write(self.post(np.zeros((self.oh, self.ow, 3), np.float32), float(self.tl[kk]), kk, None, []))

    def _montage_item(self, enc, it, i, a, b):
        fxs = [f if isinstance(f, str) else f.get("type") for f in (it.get("fx") or [])]
        FF = facefx()
        n = b - a
        frames, faces_seq = [], []
        if not it.get("clip") and not it.get("photo"):  # micro-plan sans source (rush manquant) : noir
            for m in range(n):
                k = a + m
                enc.write(self.post(np.zeros((self.oh, self.ow, 3), np.float32), float(self.tl[k]), k, None, []))
            return
        if "clip" in it:
            proxy = self._clip_proxy(it["clip"])
            j0 = fidx(_num(it.get("src_in"), 0.0))
            r = FrameReader(proxy, j0, n)
            try:
                frames = [r.get(j0 + m).copy() for m in range(n)]
            finally:
                r.close()
            track = None
            if any(f in FACE_FX for f in fxs):
                track = self.face_track(proxy, j0, j0 + n - 1)
            for m in range(n):
                try:
                    faces_seq.append(list(track.at(m) or []) if track is not None else [])
                except Exception:
                    faces_seq.append([])
        else:
            p = config.proxy_path(it.get("photo"))
            if "silhouette" in fxs:  # incrustation dans l'espace de la photo entière, puis recadrage 9:16
                from PIL import Image
                ph = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
                slot = dict((_decisions().get("silhouette_slots") or {}).get("group_photo_bg") or {})
                for s_ in self.job.get("photo_slots") or []:
                    slot = s_
                ph = self.apply_silhouette(ph, 0.0, self.fbase + a, slot=slot) if slot else ph
                u8 = _cover_u8(np.clip(ph * 255 + 0.5, 0, 255).astype(np.uint8), (SRC_W, SRC_H))
                fxs = [f for f in fxs if f != "silhouette"]
            else:
                u8 = _load_photo_u8(p, (SRC_W, SRC_H))
            frames = [u8] * n
            faces_seq = [[]] * n
        center = (SRC_W * 0.5, SRC_H * 0.45)
        mf = [main_faces(f) for f in faces_seq if f]
        if mf and mf[0]:
            bb = mf[0][0].bbox
            center = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2 + 0.05 * SRC_H)
        for m in range(n):
            k = a + m
            t = float(self.tl[k])
            fi = self.fbase + k
            img = frames[m].astype(np.float32) / 255.0
            faces = faces_seq[m]
            for f in fxs:
                if f in ("black_eyes", "stretch_smile", "widen_eyes"):
                    for fc in main_faces(faces):
                        img = self.safe(f, getattr(FF, f), img, fc, 1.0)
                elif f in ("melt", "demon_face"):
                    for fc in main_faces(faces):
                        img = self.safe(f, getattr(FF, f), img, fc, 1.0, t)
                elif f == "head_tic":
                    mfc = main_faces(faces)
                    img = self.safe(f, FF.head_tic, img, mfc[0] if mfc else None, fi, 0.9)
            z = 1.06 + 0.06 * (m / max(1, n - 1))
            cam = Camera(self.ow, self.oh, z, center[0], center[1], lanczos=self.q == "final")
            img = cam.apply(img)
            if "rgb_split" in fxs:
                img = self.safe("rgb_split", self.G.rgb_split, img, 0.6, fi)
            if m == 0 and i > 0:  # micro-glitch sur chaque coupe
                img = self.safe("glitch", self.G.glitch, img, fi, 0.5)
            enc.write(self.post(img, t, k, cam, faces))


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = _area(a) + _area(b) - inter
    return inter / u if u > 0 else 0.0


def _load_photo_u8(p: Path, size) -> np.ndarray:
    """Photo recadrée « cover » au format 9:16 de l'espace proxy."""
    from PIL import Image
    if not Path(p).exists():
        raise FileNotFoundError(f"photo absente : {p}")
    return _cover_u8(np.asarray(Image.open(p).convert("RGB")), size)


def _cover_u8(u8: np.ndarray, size) -> np.ndarray:
    """Mise à l'échelle « cover » + recadrage centré à la taille (W, H)."""
    import cv2
    h, w = u8.shape[:2]
    W, H = size
    s = max(W / w, H / h)
    r = cv2.resize(u8, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (r.shape[0] - H) // 2, (r.shape[1] - W) // 2
    return np.ascontiguousarray(r[y0:y0 + H, x0:x0 + W])


def _photo_projector(P, pw, ph, ow, oh, zoom, center, t, dur, dmap=None, amount=1.0):
    """Projection photo -> sortie de la parallaxe (pour les cercles rouges).

    Utilise parallax.project_points(...) (même caméra que render_parallax, profondeur comprise) si le module
    l'expose, sinon le modèle « cover + zoom autour du centre »."""
    fn = getattr(P, "project_points", None)
    if callable(fn) and dmap is not None:
        def proj(x, y):
            try:
                r = fn([(x, y)], (ph, pw), dmap, t, dur, zoom=zoom, center=center, amount=amount,
                       out_size=(ow, oh))
                return float(r[0][0]), float(r[0][1])
            except Exception:
                return _cover_zoom(x, y, pw, ph, ow, oh, zoom, center, t, dur)
        return proj
    return lambda x, y: _cover_zoom(x, y, pw, ph, ow, oh, zoom, center, t, dur)


def _cover_zoom(x, y, pw, ph, ow, oh, zoom, center, t, dur):
    z = zoom[0] + (zoom[1] - zoom[0]) * min(1.0, max(0.0, t / max(dur, 1e-6)))
    s = max(ow / pw, oh / ph) * z
    cx, cy = center[0] * pw, center[1] * ph
    tx = min(0.0, max(ow - pw * s, ow / 2 - cx * s))
    ty = min(0.0, max(oh - ph * s, oh / 2 - cy * s))
    return x * s + tx, y * s + ty


def _fallback_title(w, h, t, G, y_center: float = 0.42) -> np.ndarray:
    """Carton titre de repli (si le titre 3D pré-rendu manque) : titre serif dans une brume froide."""
    from PIL import Image, ImageDraw
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sweep = 0.15 + 0.85 * min(1.0, t / 1.4)
    fog = 0.05 + 0.10 * np.exp(-(((xx / w - (0.2 + 0.5 * sweep)) / 0.5) ** 2 + ((yy / h - 0.48) / 0.25) ** 2))
    img = np.repeat(fog[..., None], 3, axis=2) * np.array([0.85, 0.92, 1.0], np.float32)
    layer = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(layer)
    try:
        f = G.font("bodoni", int(w * 0.16), 800)
    except Exception:
        f = _StubGfx.font("bodoni", int(w * 0.16))
    d.multiline_text((w / 2, h * y_center), "C'EST QUI\nLUI ?", font=f, fill=255, anchor="mm", align="center",
                     spacing=int(w * 0.02))
    a = (np.asarray(layer, np.float32) / 255.0)[..., None] * min(1.0, max(0.0, (t - 0.3) / 0.9))
    img = img * (1 - a) + np.array([0.86, 0.85, 0.82], np.float32) * a
    return img.astype(np.float32)


def _pool_init(log_dir: str):
    """Processus de travail : le bavardage C++ de MediaPipe/TFLite (stderr) part dans un journal."""
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        f = open(Path(log_dir) / f"render_worker_{os.getpid()}.log", "a", encoding="utf-8")
        os.dup2(f.fileno(), 2)
        sys.stderr = f
    except OSError:
        pass
    _worker_init()


def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    try:
        import cv2
        cv2.setNumThreads(2)
    except Exception:
        pass


def _close_models():
    """Ferme les modèles MediaPipe/ONNX chargés (sortie propre des processus de travail)."""
    for name in ("silhouette", "facefx"):
        m = _MODS.get(name)
        fn = getattr(m, "close_models", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def _render_shot_job(job: dict) -> dict:
    """Point d'entrée d'un processus de travail : rend une shot dans son fichier de cache."""
    t_start = time.time()
    out = Path(job["out"])
    tmp = out.with_name(out.stem + ".tmp.mp4")
    try:
        _worker_init()
        r = ShotRenderer(job)
        enc = Encoder(tmp, *SIZES[job["quality"]], job["quality"])
        try:
            r.render(enc)
        finally:
            n = enc.close()
        if n != job["n"]:
            raise RuntimeError(f"{n} images écrites au lieu de {job['n']}")
        os.replace(tmp, out)
        _close_models()
        return {"id": job["shot"]["id"], "ok": True, "out": str(out), "n": n, "warnings": r.warnings,
                "stubs": sorted(STUBS_USED), "seconds": round(time.time() - t_start, 1)}
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return {"id": job["shot"].get("id"), "ok": False, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(), "stubs": sorted(STUBS_USED),
                "seconds": round(time.time() - t_start, 1)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_jobs(edl: dict, quality: str, sil_meta: dict | None) -> tuple[list[dict], list[dict]]:
    plan = frame_plan(edl)
    skips, flash_report = plan_flashes(edl)
    subs = edl.get("subtitles") or []
    code = _code_sig()
    stub_sig = sorted(n for n in ("gfx", "facefx", "silhouette", "parallax") if _module(n) is not None and
                      n in STUBS_USED)  # un rendu STUB n'est jamais réutilisé pour un rendu réel
    # emplacement de la silhouette sur la photo de groupe (réutilisé par les micro-plans photo du montage)
    photo_slots = [dict(s["silhouette"]) for s in edl["shots"] if s.get("kind") == "photo" and s.get("silhouette")]
    # décisions lues au rendu (replis manuels) : elles font partie de la clé de cache
    dec = _decisions()
    dec_sig = _hash({k: v for k, v in dec.items() if k in ("group_photo_faces", "privacy_zones", "face_boxes",
                                                           "silhouette_slots")
                     or k.startswith(("hud_boxes_", "portrait_eyes_"))})
    jobs = []
    for i, s in enumerate(edl["shots"]):
        f0, f1 = plan[i]
        n = f1 - f0
        if n <= 0:
            continue
        start = _num(s.get("start"), 0.0)
        a, b = f0 / FPS, f1 / FPS
        my_subs = [{"start": _num(x["start"], 0.0), "end": _num(x["end"], 0.0),
                    "text": x.get("text", ""), "kind": x.get("kind", "dialogue")}
                   for x in subs if _num(x.get("end"), 0) > a and _num(x.get("start"), 0) < b]
        freeze = resolve_freeze_src(edl, plan, i) if s.get("kind") == "freeze" else None
        clips = [c for c in [s.get("clip"), (freeze or {}).get("clip")] + [it.get("clip") for it in s.get("items") or []]
                 if c]
        photos = [p for p in [s.get("photo"), (freeze or {}).get("photo")] + [it.get("photo") for it in
                                                                               s.get("items") or []] if p]
        key = {
            "v": code, "q": quality, "shot": s, "n": n, "off": round(f0 / FPS - start, 5),
            "subs": [[round(x["start"] - start, 3), round(x["end"] - start, 3), x["text"], x["kind"]] for x in my_subs],
            "skip": skips.get(s["id"], []), "freeze": freeze, "stubs": stub_sig, "dec": dec_sig,
            "analysis": [_file_sig(config.ANALYSIS_DIR / f"{c}.{k}.json") for c in clips for k in ("faces", "persons")],
            "pslot": photo_slots[:1] if s.get("kind") == "montage" else [],
            "files": [_file_sig(config.proxy_path(c)) for c in clips] + [_file_sig(config.proxy_path(p)) for p in photos]
            + ([_file_sig(config.TITLE_DIR / "title_seq.mp4")] if s.get("kind") == "title" else [])
            + ([_file_sig(Path(sil_meta["still"]))] if sil_meta and sil_meta.get("still") else []),
        }
        if s.get("kind") == "title":
            key["files"].append(_file_sig(config.TITLE_DIR / "title_meta.json"))
        h = _hash(key)
        out = config.CACHE_DIR / "shots" / quality / f"{s['id']}_{h}.mp4"
        jobs.append({"shot": s, "quality": quality, "n": n, "f0": f0, "subs": my_subs, "flash_skip": skips.get(s["id"]),
                     "freeze": freeze, "silhouette": sil_meta, "out": str(out),
                     "photo_slots": photo_slots[:1] if s.get("kind") == "montage" else []})
    return jobs, flash_report


def concat(files: list[Path], out: Path, n_expected: int, log=print) -> Path:
    """Concaténation sans réencodage (même codec/paramètres) ; vérifie le nombre d'images exact."""
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.with_suffix(".concat.txt")
    def q(f):  # échappement des apostrophes pour le démultiplexeur concat
        return Path(f).resolve().as_posix().replace("'", "'\\''")
    lst.write_text("".join(f"file '{q(f)}'\n" for f in files), encoding="utf-8")
    tmp = out.with_name(out.stem + ".tmp.mp4")
    cmd = [_ffmpeg(), "-y", "-v", "error", "-nostdin", "-f", "concat", "-safe", "0", "-i", str(lst),
           "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    n = count_frames(tmp) if r.returncode == 0 and tmp.exists() else -1
    if n != n_expected:
        log(f"[render] concat sans réencodage : {n} images au lieu de {n_expected} -> réencodage")
        cmd = [_ffmpeg(), "-y", "-v", "error", "-nostdin", "-f", "concat", "-safe", "0", "-i", str(lst),
               "-map", "0:v:0", "-frames:v", str(n_expected), "-c:v", "libx264", "-preset", "medium", "-crf", "10",
               "-pix_fmt", "yuv420p", "-r", str(FPS), "-colorspace", "bt709", "-color_primaries", "bt709",
               "-color_trc", "bt709", "-movflags", "+faststart", str(tmp)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError(f"concaténation échouée : {r.stderr[-800:]}")
        n = count_frames(tmp)
        if n != n_expected:
            raise RuntimeError(f"concaténation : {n} images au lieu de {n_expected}")
    os.replace(tmp, out)
    lst.unlink(missing_ok=True)
    return out


def render(edl: dict, quality: str = "preview", shots: list[str] | None = None, jobs: int | None = None,
           force: bool = False, log=print) -> Path:
    """Rend la vidéo du film (sans son) -> RENDER_DIR/video_{quality}.mp4.

    shots : sous-ensemble d'ids (test) -> RENDER_DIR/video_{quality}_subset.mp4.
    jobs  : processus parallèles (défaut ESCAPE_RENDER_JOBS ou 2).
    force : ignore le cache des shots."""
    if quality not in SIZES:
        raise ValueError(f"qualité inconnue : {quality}")
    t_all = time.time()
    config.ensure_dirs()
    STUBS_USED.clear()
    _MODS.clear()
    for name in ("gfx", "facefx", "silhouette", "parallax"):
        _module(name)  # détecte les STUB tôt (et les signale)
    sil = ensure_silhouette(edl, log) if needs_silhouette(edl) else None
    close = getattr(silhouette_mod(), "close_models", None)
    if callable(close):  # libère la mémoire des modèles de détourage dans le processus principal
        close()
    all_jobs, flash_report = build_jobs(edl, quality, sil)
    sel = [j for j in all_jobs if shots is None or j["shot"]["id"] in shots]
    if shots and len(sel) != len(set(shots)):
        missing = set(shots) - {j["shot"]["id"] for j in sel}
        raise ValueError(f"shots inconnues : {sorted(missing)}")
    todo = [j for j in sel if force or not Path(j["out"]).exists()]
    njobs = int(jobs or os.environ.get("ESCAPE_RENDER_JOBS", DEFAULT_JOBS))
    results = {j["shot"]["id"]: {"id": j["shot"]["id"], "ok": True, "cached": True, "n": j["n"], "warnings": []}
               for j in sel if j not in todo}
    log(f"[render] {quality} {SIZES[quality][0]}x{SIZES[quality][1]} : {len(sel)} shots, {len(todo)} à rendre, "
        f"{len(sel) - len(todo)} en cache, {njobs} processus" + (f" — STUB : {sorted(STUBS_USED)}" if STUBS_USED else ""))
    for fr in flash_report:
        if fr["action"] != "gardé":
            log(f"[render] ⚠ garde-fou flashs : {fr['fx']} de {fr['shot']} à {fr['t']} s {fr['action']}")
    todo.sort(key=lambda j: -j["n"])  # les plus longues d'abord (équilibrage)
    failures = []
    if todo:
        if njobs <= 1:
            for j in todo:
                res = _render_shot_job(j)
                results[res["id"]] = res
                _log_result(res, log)
                if not res["ok"]:
                    failures.append(res)
        else:
            ctx = mp.get_context("spawn")
            with cf.ProcessPoolExecutor(max_workers=njobs, mp_context=ctx, initializer=_pool_init,
                                        initargs=(str(config.WORK_DIR / "logs"),)) as ex:
                futs = {ex.submit(_render_shot_job, j): j for j in todo}
                for fut in cf.as_completed(futs):
                    res = fut.result()
                    results[res["id"]] = res
                    _log_result(res, log)
                    if not res["ok"]:
                        failures.append(res)
    if failures:
        msg = "; ".join(f"{f['id']}: {f['error']}" for f in failures)
        raise RuntimeError(f"rendu échoué pour {len(failures)} shot(s) : {msg}\n{failures[0].get('trace', '')}")
    n_exp = sum(j["n"] for j in sel)
    name = f"video_{quality}.mp4" if shots is None else f"video_{quality}_subset.mp4"
    out = concat([Path(j["out"]) for j in sel], config.RENDER_DIR / name, n_exp, log)
    stubs = sorted(set(STUBS_USED).union(*[set(r.get("stubs") or []) for r in results.values()]))
    report = {
        "file": str(out), "quality": quality, "size": SIZES[quality], "fps": FPS, "frames": n_exp,
        "duration": round(n_exp / FPS, 3), "subset": shots, "stubs": stubs, "flashes": flash_report,
        "silhouette": sil, "shots": [results[j["shot"]["id"]] for j in sel],
        "warnings": [w for j in sel for w in results[j["shot"]["id"]].get("warnings", [])],
        "render_s": round(time.time() - t_all, 1),
    }
    for r in report["shots"]:
        r.pop("trace", None)
    (config.RENDER_DIR / (Path(name).stem + ".json")).write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[render] {out.name} : {n_exp} images ({n_exp / FPS:.2f} s) en {report['render_s']} s"
        + (f", {len(report['warnings'])} avertissement(s)" if report["warnings"] else ""))
    for w in report["warnings"]:
        log(f"[render] ⚠ {w}")
    _prune_cache(quality, {Path(j["out"]).name for j in all_jobs})
    return out


def _log_result(res: dict, log):
    if res["ok"]:
        log(f"[render]   {res['id']:5s} {res['n']:4d} images  {res['seconds']:6.1f} s"
            + (f"  ⚠ {len(res['warnings'])}" if res.get("warnings") else ""))
    else:
        log(f"[render]   {res['id']:5s} ÉCHEC : {res['error']}")


def _prune_cache(quality: str, keep: set[str]):
    """Supprime les rendus de shots obsolètes (même id, autre hash)."""
    d = config.CACHE_DIR / "shots" / quality
    if not d.exists():
        return
    ids = {k.rsplit("_", 1)[0] for k in keep}
    for f in d.glob("*.mp4"):
        if f.name not in keep and f.name.rsplit("_", 1)[0] in ids and not f.name.endswith(".tmp.mp4"):
            try:
                f.unlink()
            except OSError:
                pass


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Rendu vidéo (sans son) du film")
    ap.add_argument("--edl", default=None)
    ap.add_argument("--quality", default="preview", choices=list(SIZES))
    ap.add_argument("--shots", default=None, help="ids séparés par des virgules (test)")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    edl = load_edl(a.edl)
    render(edl, a.quality, shots=a.shots.split(",") if a.shots else None, jobs=a.jobs, force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
