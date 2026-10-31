"""Aperçus avant/après des VFX (visages, silhouette, parallaxe) + mesures de temps.

Écrit des grilles JPG dans WORK_DIR/fx_previews (ou --out) à regarder avant de valider un réglage.
Utilise les proxies s'ils existent, sinon les rushs (décodés et remis en 1080x1920).

  python tools/preview_fx.py --what all|faces|track|silhouette|parallax [--out DOSSIER] [--testdata DOSSIER]

Tests sur faux rushs :
  ESCAPE_RUSH_DIR=.../fixtures/rushs ESCAPE_WORK_DIR=.../fixtures/work ESCAPE_MODELS_DIR=.../models \
  python tools/preview_fx.py --what all --out /home/user/edit_escape_tmp/fx_previews
"""
from __future__ import annotations

import os

# BLAS mono-thread : sous charge, les threads OpenBLAS en attente active ralentissent tout (cf. facefx)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from pipeline import facefx as fx  # noqa: E402
from pipeline import parallax as px  # noqa: E402
from pipeline import silhouette as sil  # noqa: E402

TIMINGS: dict[str, float] = {}
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def save_jpg(path: Path, img: np.ndarray, q: int = 88) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    path.write_bytes(buf.tobytes())
    print("  ->", path)
    return path


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)   # putText (OpenCV 5) : 8 bits uniquement
    s = max(0.45, img.shape[1] / 700)
    cv2.putText(out, text, (8, int(26 * s)), FONT, 0.7 * s, (0, 0, 0), max(2, int(4 * s)), cv2.LINE_AA)
    cv2.putText(out, text, (8, int(26 * s)), FONT, 0.7 * s, (255, 255, 150), max(1, int(1.5 * s)), cv2.LINE_AA)
    return out.astype(np.float32) / 255.0


def grid(rows: list[list[np.ndarray]], cell_w: int) -> np.ndarray:
    out_rows = []
    for r in rows:
        cells = [cv2.resize(c, (cell_w, int(round(c.shape[0] * cell_w / c.shape[1]))), interpolation=cv2.INTER_AREA)
                 for c in r]
        h = max(c.shape[0] for c in cells)
        cells = [np.pad(c, ((0, h - c.shape[0]), (0, 2), (0, 0))) for c in cells]
        out_rows.append(np.concatenate(cells, 1))
    w = max(r.shape[1] for r in out_rows)
    out_rows = [np.pad(r, ((0, 2), (0, w - r.shape[1]), (0, 0))) for r in out_rows]
    return np.concatenate(out_rows, 0)


def timed(name: str, fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    dt = (time.perf_counter() - t) * 1000
    TIMINGS.setdefault(name, dt)
    TIMINGS[name] = min(TIMINGS[name], dt)
    return r


def to_vertical(rgb_u8: np.ndarray, size=(1080, 1920)) -> np.ndarray:
    return cv2.resize(rgb_u8, size, interpolation=cv2.INTER_CUBIC if rgb_u8.shape[1] < size[0] else cv2.INTER_AREA
                      ).astype(np.float32) / 255.0


def clip_source(clip: str) -> Path:
    p = config.proxy_path(clip)
    return p if p.exists() else config.rush_path(clip)


def clip_frame(clip: str, t: float) -> np.ndarray:
    return to_vertical(next(fx.read_frames(clip_source(clip), t, t + 1 / 30)))


def photo(clip: str) -> np.ndarray:
    p = config.proxy_path(clip)
    if p.exists():
        rgb = cv2.cvtColor(cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    else:
        from PIL import Image, ImageOps
        rgb = np.asarray(ImageOps.exif_transpose(Image.open(config.rush_path(clip))).convert("RGB"))
        if rgb.shape[0] > 2560:
            rgb = cv2.resize(rgb, (rgb.shape[1] * 2560 // rgb.shape[0], 2560), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32) / 255.0


def testdata_canvas(td: Path, name: str, scale: float = 1.0, y: int = 300) -> np.ndarray:
    im = cv2.cvtColor(cv2.imread(str(td / name)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    c = np.full((1920, 1080, 3), 0.05, np.float32)
    s = 1080 * scale / im.shape[1]
    r = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h = min(r.shape[0], 1920 - y)
    x = (1080 - r.shape[1]) // 2
    c[y:y + h, x:x + r.shape[1]] = r[:h]
    return c


def face_crop(img: np.ndarray, f: fx.Face, pad: float = 0.7) -> np.ndarray:
    x0, y0, x1, y1 = f.bbox
    s = max(x1 - x0, y1 - y0) * (1 + pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + 0.08 * s
    a, b = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    return img[b:b + int(s), a:a + int(s)]


# ---------------------------------------------------------------------------
# Visages
# ---------------------------------------------------------------------------
def preview_faces(out: Path, td: Path) -> None:
    print("[faces] effets sur images de test (1080x1920)")
    tic_frames = [i for i in range(60) if fx._tic_event(i, 1.0) is not None]
    rows, ramp_rows = [], []
    for name in ("portrait.jpg", "lena.jpg", "business-person.png"):
        img = testdata_canvas(td, name)
        tr = timed("FaceTrack.from_image (1080x1920, tuiles)", fx.FaceTrack.from_image, img, use_cache=False)
        f = max(tr.at(0), key=lambda q: q.size)
        effects = [
            ("original", lambda: img),
            ("black_eyes 1", lambda: timed("black_eyes", fx.black_eyes, img, f, 1.0)),
            ("stretch_smile 1", lambda: timed("stretch_smile", fx.stretch_smile, img, f, 1.0)),
            ("melt 1 t=0.5", lambda: timed("melt", fx.melt, img, f, 1.0, 0.5)),
            ("melt 1 t=3", lambda: fx.melt(img, f, 1.0, 3.0)),
            ("widen_eyes 1", lambda: timed("widen_eyes", fx.widen_eyes, img, f, 1.0)),
            ("demon_face 1", lambda: timed("demon_face", fx.demon_face, img, f, 1.0, 0.3)),
            (f"head_tic #{tic_frames[0]}", lambda: timed("head_tic", fx.head_tic, img, f, tic_frames[0], 1.0)),
            (f"head_tic #{tic_frames[1]}", lambda: fx.head_tic(img, f, tic_frames[1], 1.0)),
        ]
        for _ in range(2):   # 2e passage pour des temps « à chaud »
            res = [(n, e()) for n, e in effects]
        rows.append([label(face_crop(o, f), n) for n, o in res])
        ramp = [label(face_crop(fx.black_eyes(img, f, a), f), f"black {a}") for a in (0.25, 0.5, 0.75)]
        ramp += [label(face_crop(fx.stretch_smile(img, f, a), f), f"smile {a}") for a in (0.4, 0.8)]
        ramp += [label(face_crop(fx.demon_face(img, f, a, 0.0), f), f"demon {a}") for a in (0.5, 0.8)]
        ramp += [label(face_crop(fx.portrait_eyes(img, [f], tt, 1.0), f, 0.2), f"portrait t={tt}") for tt in (0.5, 2.2)]
        ramp_rows.append(ramp)
    save_jpg(out / "faces_effects.jpg", grid(rows, 300))
    save_jpg(out / "faces_ramps.jpg", grid(ramp_rows, 300))

    print("[faces] petits visages (plan large, 4 visages ~70 px)")
    rng = np.random.default_rng(0)
    bg = cv2.GaussianBlur((rng.random((1920, 1080, 3)) * 0.05 + 0.08).astype(np.float32), (0, 0), 3)
    for (x, y), n in zip([(60, 820), (320, 760), (560, 840), (800, 790)],
                         ["portrait.jpg", "lena.jpg", "business-person.png", "portrait.jpg"]):
        im = cv2.cvtColor(cv2.imread(str(td / n)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        r = cv2.resize(im, None, fx=230 / im.shape[1], fy=230 / im.shape[1], interpolation=cv2.INTER_AREA)
        h = min(r.shape[0], 1920 - y)
        bg[y:y + h, x:x + r.shape[1]] = r[:h] * 0.75
    tr = timed("FaceTrack.from_image (petits visages)", fx.FaceTrack.from_image, bg, use_cache=False)
    fs = tr.at(0)
    print(f"  visages trouvés : {len(fs)} (tailles {[round(q.size) for q in fs]})")
    o1 = timed("black_eyes x4 petits visages", fx.black_eyes, bg, fs, 1.0)
    o2 = fx.demon_face(bg, fs, 1.0, 0.0)
    crop = lambda im: im[740:1120]  # noqa: E731
    z = lambda im: cv2.resize(im[800:960, 40:1060], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # noqa: E731
    save_jpg(out / "faces_small.jpg", grid([[label(crop(bg), f"{len(fs)} visages detectes")], [crop(o1)], [crop(o2)],
                                            [label(z(o1), "zoom x2 black_eyes")]], 1080))


def _relight(img: np.ndarray, kind: str) -> np.ndarray:
    """Éclairages difficiles : habitacle rouge sombre, accueil bleu, quasi-noir bruité (capteur de nuit)."""
    rng = np.random.default_rng(5)
    x = np.clip(img, 0, 1) ** 1.25
    if kind == "rouge":
        x = x * np.float32([0.95, 0.20, 0.17]) * 0.62 + np.float32([0.012, 0.0, 0.0])
    elif kind == "bleu":
        x = x * np.float32([0.30, 0.52, 1.0]) * 0.7
    else:   # nuit : −3,5 diaph + bruit
        x = x * 0.11
    x = x + rng.standard_normal(x.shape).astype(np.float32) * 0.012
    return np.clip(x, 0, 1).astype(np.float32)


def preview_lowlight(out: Path, td: Path) -> None:
    """VFX visages sur visages sombres / rouges / bleus, puis étalonnage du film (night_red, blue, cold+expo) et
    finition : contrôle à 100 % des bords, halos, effet « masque ». Plus le repli sans maillage (boîte seule)."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from pipeline import gfx
    print("[lowlight] visages sombres / rouges / bleus")
    rows = []
    for name in ("portrait.jpg", "business-person.png"):
        base = testdata_canvas(td, name)
        f = max(fx.FaceTrack.from_image(base, use_cache=False).at(0), key=lambda q: q.size)
        for kind, pre in (("rouge", "night_red"), ("bleu", "blue"), ("nuit", "cold")):
            img = _relight(base, kind)
            g = gfx.exposure_gain([img])
            fin = lambda im: gfx.film_finish(gfx.grade(im, pre, auto_exposure=g), 4)  # noqa: E731
            cells = [label(face_crop(fin(img), f), f"{kind} {pre} g={g:.1f}")]
            for nm, fn in (("black_eyes", lambda: fx.black_eyes(img, f, 1.0)),
                           ("smile", lambda: fx.stretch_smile(img, f, 1.0)),
                           ("demon", lambda: fx.demon_face(img, f, 1.0, 0.3)),
                           ("melt t=2", lambda: fx.melt(img, f, 1.0, 2.0))):
                cells.append(label(face_crop(fin(fn()), f), nm))
            rows.append(cells)
    save_jpg(out / "faces_lowlight.jpg", grid(rows, 300))
    # repli sans maillage : boîte « détecteur » seule, visage flou de bougé (screamer collé à l'objectif)
    print("[lowlight] repli depuis une boîte (maillage indisponible)")
    cells = []
    for name in ("portrait.jpg", "business-person.png", "lena.jpg"):
        im = cv2.cvtColor(cv2.imread(str(td / name)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        sc = 1920 / im.shape[0] * 1.15                                       # visage en gros plan
        big = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        ox, oy = max(0, (big.shape[1] - 1080) // 2), 0
        base = np.zeros((1920, 1080, 3), np.float32)
        crop_ = big[oy:oy + 1920, ox:ox + 1080]
        base[:crop_.shape[0], :crop_.shape[1]] = crop_
        f = max(fx.FaceTrack.from_image(base, use_cache=False).at(0), key=lambda q: q.size)
        x0, y0, x1, y1 = f.bbox
        bw = x1 - x0
        box = (x0 + 0.04 * bw, y0 + 0.13 * (y1 - y0), x1 - 0.04 * bw, y1)     # boîte type BlazeFace
        k = np.zeros((31, 31), np.float32)
        k[15, :] = 1.0 / 31                                                  # flou de bougé horizontal
        blurred = cv2.filter2D(base, -1, k)
        a = timed("demon_face_from_box", fx.demon_face_from_box, blurred, box, 1.0, 0.3)
        b = fx.demon_face(blurred, f, 1.0, 0.3)
        vis = blurred.copy()
        cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (1, 1, 0), 3)
        fin = lambda im: gfx.film_finish(gfx.grade(im, "bleach"), 9)  # noqa: E731
        cells += [label(face_crop(vis, f), f"{name} flou + boite"), label(face_crop(fin(a), f), "demon_from_box"),
                  label(face_crop(fin(b), f), "demon (maillage)")]
    save_jpg(out / "faces_from_box.jpg", grid([cells[:6], cells[6:]], 300))


def preview_track(out: Path) -> None:
    print("[track] FaceTrack.compute sur une vidéo + effets dans le temps")
    clip, t0, t1 = "IMG_8357", 4.0, 7.0
    src = clip_source(clip)
    tr = timed(f"FaceTrack.compute {clip} 3 s (par piste)", fx.FaceTrack.compute, src, t0, t1, use_cache=False)
    n = len(tr)
    TIMINGS["FaceTrack.compute par image"] = TIMINGS[f"FaceTrack.compute {clip} 3 s (par piste)"] / max(n, 1)
    frames = list(fx.read_frames(src, t0, t1))
    sx = 1080 / tr.frame_size[0]
    cells = []
    for i in np.linspace(0, n - 1, 5).astype(int):
        img = to_vertical(frames[i])
        fs = [f.scaled(sx) for f in tr.at(i)]
        vis = img.copy()
        for f in fs:
            for p in f.lm[::4]:
                cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0.2, 1, 0.3), -1, cv2.LINE_AA)
        o = fx.black_eyes(img, fs, 1.0)
        o = fx.stretch_smile(o, fs, 1.0)
        o = fx.head_tic(o, fs[0] if fs else None, int(i), 0.6)
        cells.append((label(vis, f"#{i} {len(fs)} visage(s)"), o))
    save_jpg(out / "track_8357.jpg", grid([[c[0] for c in cells], [c[1] for c in cells]], 360))
    # stabilité : 2e différence du centre de l'œil gauche (px)
    if tr.n_tracks:
        c = np.nanmean(tr.lm[:, 0, fx.LEFT_EYE], 1)
        TIMINGS["jitter oeil (px, 2e diff, source)"] = float(np.nanmean(np.abs(np.diff(c, 2, axis=0))))


# ---------------------------------------------------------------------------
# Silhouette
# ---------------------------------------------------------------------------
def preview_silhouette(out: Path, force: bool = False) -> dict:
    print("[silhouette] extraction IMG_8346 6,0–9,5 s")
    sdir = config.CACHE_DIR / "silhouette" / "IMG_8346"
    meta = timed("silhouette.extract (3,5 s @10 i/s)", sil.extract, clip_source("IMG_8346"), (6.0, 9.5), sdir,
                 hint_center=None, force=force)
    print("  ", {k: meta[k] for k in ("method", "t", "score", "height_px", "bbox")})
    still = meta["still"]
    tiles = []
    for p in [still] + sorted(Path(meta["frames"]).glob("*.png"))[::6]:
        a = sil.load_rgba(p).astype(np.float32) / 255
        for bgv in (0.85, 0.2):
            c = np.full(a.shape[:2] + (3,), bgv, np.float32)
            tiles.append(c * (1 - a[..., 3:]) + a[..., :3] * a[..., 3:])
    save_jpg(out / "silhouette_extract.jpg", grid([tiles], 110))
    proc = sil.procedural(1800)
    a = proc.astype(np.float32) / 255
    pv = np.full(a.shape[:2] + (3,), 0.8, np.float32) * (1 - a[..., 3:]) + a[..., :3] * a[..., 3:]
    save_jpg(out / "silhouette_procedural.jpg", pv)

    print("[silhouette] incrustations (slots de decisions.json)")
    slots = json.loads((config.EDIT_DIR / "decisions.json").read_text(encoding="utf-8"))["silhouette_slots"]
    plate57 = clip_frame("IMG_8357", 4.6)
    pm = timed("person_mask (1080x1920)", sil.person_mask, plate57)
    sil.person_mask(plate57)
    cells = [label(plate57, "8357 plaque"), label(np.dstack([pm] * 3), "person_mask")]
    for key in ("parking_8357_far", "parking_8357_mid", "parking_8357_near"):
        s = slots[key]
        o = timed("composite", sil.composite, plate57, still, s["x"], s["foot_y"], s["height"], 0.9, s["blur"],
                  s["darkness"], s.get("blend", "normal"), 12, pm if s.get("occlude_persons") else None)
        cells.append(label(o, key))
    s = slots["parking_8357_near"]
    cells.append(label(sil.composite(plate57, proc, s["x"], s["foot_y"], s["height"], 0.95, s["blur"], s["darkness"],
                                     "normal", 0, pm), "near (procedurale)"))
    plate49 = clip_frame("IMG_8349", 1.0)
    s = slots["room_corner_8349"]
    cells.append(label(sil.composite(plate49, still, s["x"], s["foot_y"], s["height"], 0.6, s["blur"], s["darkness"]),
                       "room_corner_8349"))
    plate56 = clip_frame("IMG_8356", 3.0)
    s = slots["bk_window_8356"]
    pm56 = sil.person_mask(plate56)
    cells.append(label(sil.composite(plate56, still, s["x"], s["foot_y"], s["height"], 0.35, s["blur"], s["darkness"],
                                     "multiply", 0, pm56), "bk_window multiply"))
    save_jpg(out / "silhouette_composites.jpg", grid([cells], 300))
    # respiration / balancement : 4 instants, zoom
    s = slots["parking_8357_mid"]
    seq = [sil.composite(plate57, still, s["x"], s["foot_y"], s["height"], 0.9, s["blur"], s["darkness"], "normal",
                         k, pm)[900:1400, 150:550] for k in (0, 25, 50, 75)]
    save_jpg(out / "silhouette_breath.jpg", grid([seq], 300))
    return meta


# ---------------------------------------------------------------------------
# Parallaxe
# ---------------------------------------------------------------------------
def preview_parallax(out: Path, td: Path, sil_meta: dict | None) -> None:
    print("[parallax] profondeur + poussée 2,5D (5,6 s)")
    sources = [("IMG_8350", photo("IMG_8350")), ("pose.jpg", None), ("messi5.jpg", None)]
    rows = []
    for name, img in sources:
        if img is None:
            im = cv2.cvtColor(cv2.imread(str(td / name)), cv2.COLOR_BGR2RGB)
            img = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        d = timed(f"depth {name}", px.depth, img, use_cache=False)
        px.render_parallax(img, d, 0.0, 5.6)    # préparation de la scène (hors mesure)
        frs = [timed("render_parallax 1080x1920", px.render_parallax, img, d, t, 5.6) for t in (0.0, 1.4, 2.8, 4.2, 5.6)]
        frs_prev = [timed("render_parallax 540x960", px.render_parallax, img, d, 2.8, 5.6, out_size=(540, 960))]
        dv = cv2.resize(np.dstack([d] * 3), (1080, int(1080 * d.shape[0] / d.shape[1])))
        rows.append([label(dv, f"depth {name}")] + [label(f, f"t={t}") for f, t in zip(frs, (0, 1.4, 2.8, 4.2, 5.6))])
        # contrôle des déchirures : écart image à image (doit rester faible et lisse)
        seq = [px.render_parallax(img, d, t, 5.6) for t in np.linspace(0, 5.6, 12)]
        jumps = [float(np.abs(a - b).mean()) for a, b in zip(seq[:-1], seq[1:])]
        TIMINGS[f"parallax écart moyen image/image {name}"] = float(np.mean(jumps))
        del frs_prev
    save_jpg(out / "parallax.jpg", grid(rows, 240))
    # photo de groupe : silhouette dans l'espace photo + suivi des points (project_points)
    if sil_meta:
        img = photo("IMG_8350")
        H, W = img.shape[:2]
        s = json.loads((config.EDIT_DIR / "decisions.json").read_text(encoding="utf-8"))["silhouette_slots"]["group_photo_bg"]
        pm = sil.person_mask(img)
        comp = sil.composite(img, sil_meta["still"], s["x"], s["foot_y"], s["height"], 0.9, s["blur"], s["darkness"],
                             "normal", 0, pm)
        d = px.depth(img)
        cells = []
        for t in (0.0, 2.8, 5.6):
            fr = px.render_parallax(comp, d, t, 5.6).copy()
            p = px.project_points([[s["x"] * W, s["foot_y"] * H]], img.shape, d, t, 5.6)[0]
            cv2.circle(fr, (int(p[0]), int(p[1])), 18, (1, 0.1, 0.1), 3, cv2.LINE_AA)
            cells.append(label(fr, f"t={t} (cercle = pieds projetés)"))
        save_jpg(out / "parallax_group_silhouette.jpg", grid([cells], 360))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", default="all", choices=["all", "faces", "lowlight", "track", "silhouette", "parallax"])
    ap.add_argument("--out", default=str(config.WORK_DIR / "fx_previews"))
    ap.add_argument("--testdata", default=str(config.WORK_DIR / "testdata"))
    ap.add_argument("--force", action="store_true", help="refaire l'extraction de la silhouette (sinon cache)")
    args = ap.parse_args()
    out = Path(args.out)
    td = Path(args.testdata)
    if not td.exists():
        td = Path("/home/user/edit_escape_tmp/testdata")
    sil_meta = None
    if args.what in ("all", "faces"):
        preview_faces(out, td)
    if args.what in ("all", "lowlight"):
        preview_lowlight(out, td)
    if args.what in ("all", "track"):
        preview_track(out)
    if args.what in ("all", "silhouette"):
        sil_meta = preview_silhouette(out, args.force)
    if args.what in ("all", "parallax"):
        if sil_meta is None:
            mp_ = config.CACHE_DIR / "silhouette" / "IMG_8346" / "meta.json"
            sil_meta = json.loads(mp_.read_text(encoding="utf-8")) if mp_.exists() else None
        preview_parallax(out, td, sil_meta)
    rep = out / f"timings_{args.what}.json"
    rep.write_text(json.dumps({k: round(v, 2) for k, v in TIMINGS.items()}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    for k, v in TIMINGS.items():
        print(f"  {k:55s} {v:9.1f}")
    fx.close_models()
    sil.close_models()


if __name__ == "__main__":
    main()
