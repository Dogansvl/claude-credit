"""Planches de contrôle visuel de pipeline/gfx.py -> WORK_DIR/gfx_previews/*.png (+ perf.json).

Usage :
  python tools/preview_gfx.py                      # toutes les planches, images 1080x1920
  python tools/preview_gfx.py --only cards,hud     # sélection (cards, names, timecode, subs, hud, circles,
                                                   #   grades, finish, composite, fx, check, perf)
  python tools/preview_gfx.py --res 540            # tout en 540x960 (résolution de la préversion)
Sources : images de test (--testdata, défaut WORK_DIR/testdata) et une image d'un rush ou proxy (--rush ;
défaut : proxy IMG_8346 s'il existe, sinon rush IMG_8346 de RUSH_DIR). Avec les faux rushs, ces planches ne
servent qu'au contrôle technique : aucune conclusion créative à en tirer.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import config  # noqa: E402
from pipeline import gfx  # noqa: E402

CAP_H = 34


# ----------------------------------------------------------------------------- utilitaires
def load_rgb(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def save_rgb(path: Path, img: np.ndarray) -> None:
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))


def cover(img: np.ndarray, w: int, h: int, center=(0.5, 0.5)) -> np.ndarray:
    """Recadrage « cover » vers w x h."""
    ih, iw = img.shape[:2]
    k = max(w / iw, h / ih)
    rw, rh = int(round(iw * k)), int(round(ih * k))
    r = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    x0 = int(np.clip(center[0] * rw - w / 2, 0, rw - w))
    y0 = int(np.clip(center[1] * rh - h / 2, 0, rh - h))
    return np.ascontiguousarray(np.clip(r[y0:y0 + h, x0:x0 + w], 0.0, 1.0))


def caption(tile: np.ndarray, text: str) -> np.ndarray:
    h, w = tile.shape[:2]
    bar = np.full((CAP_H, w, 3), 0.12, np.float32)
    spr = gfx._line(text, "plexmono", 17.0, 400, 0.02, 1)
    gfx._over(bar, spr.a, np.array([0.95, 0.85, 0.3], np.float32), 8 - int(spr.ox), int(CAP_H * 0.72 - spr.base))
    return np.concatenate([bar, tile], 0)


def sheet(tiles: list, cols: int, tile_w: int, gap: int = 6) -> np.ndarray:
    """tiles : [(image, légende)] ; chaque image est mise à la largeur tile_w (hauteur proportionnelle)."""
    rows = []
    for i in range(0, len(tiles), cols):
        row = []
        for img, cap in tiles[i:i + cols]:
            h = int(round(img.shape[0] * tile_w / img.shape[1]))
            t = cv2.resize(img, (tile_w, h), interpolation=cv2.INTER_AREA)
            row.append(caption(t, cap))
        hmax = max(r.shape[0] for r in row)
        row = [np.pad(r, ((0, hmax - r.shape[0]), (0, gap), (0, 0)), constant_values=0.05) for r in row]
        while len(row) < cols:
            row.append(np.full((hmax, tile_w + gap, 3), 0.05, np.float32))
        rows.append(np.concatenate(row, 1))
        rows.append(np.full((gap, rows[-1].shape[1], 3), 0.05, np.float32))
    return np.concatenate(rows, 0)


def crop(img: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    return img[max(0, y0):y0 + h, max(0, x0):x0 + w]


# ----------------------------------------------------------------------------- plaques de test
def rush_frame(src: Path | None, t: float, out_dir: Path, W: int, H: int) -> np.ndarray | None:
    if src is None or not src.exists():
        return None
    dst = out_dir / "_src" / f"{src.stem}_t{t:.2f}_{W}.png"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
               "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H}", str(dst)]
        if subprocess.run(cmd).returncode != 0:
            return None
    return load_rgb(dst)


def night_plate(td: Path, W: int, H: int) -> np.ndarray:
    """Scène de nuit synthétique : visage éclairé de côté, phares, feux rouges, gyrophare bleu, brume."""
    s = W / 1080
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = 0.035 + 0.05 * np.exp(-((xx - 0.3 * W) ** 2 + (yy - 0.75 * H) ** 2) / (2 * (0.35 * W) ** 2))
    img = np.dstack([base * 0.8, base * 0.9, base * 1.1]).astype(np.float32)
    face = cover(load_rgb(td / "lena.jpg"), int(560 * s), int(700 * s), (0.55, 0.55))
    light = np.clip(np.linspace(1.15, 0.12, face.shape[1])[None, :, None], 0, 1).astype(np.float32)
    face = face ** 1.3 * light
    fy, fx = int(0.20 * H), int(0.26 * W)
    yy2, xx2 = np.mgrid[0:face.shape[0], 0:face.shape[1]].astype(np.float32)
    m = np.clip(1.2 - np.hypot((xx2 - face.shape[1] / 2) / (face.shape[1] * 0.48),
                               (yy2 - face.shape[0] / 2) / (face.shape[0] * 0.5)) ** 4, 0, 1)[..., None]
    img[fy:fy + face.shape[0], fx:fx + face.shape[1]] = img[fy:fy + face.shape[0], fx:fx + face.shape[1]] * (1 - m) + face * m

    def blob(cx, cy, r, col, gain):
        d = np.hypot(xx - cx, yy - cy)
        img[...] += (np.exp(-(d / r) ** 2) * gain)[..., None] * np.asarray(col, np.float32)

    for cx in (0.30, 0.70):  # phares (hautes lumières -> halation)
        blob(cx * W, 0.80 * H, 30 * s, (1.0, 0.97, 0.9), 1.6)
        blob(cx * W, 0.80 * H, 140 * s, (0.9, 0.85, 0.75), 0.25)
    for cx in (0.18, 0.84):  # feux arrière rouges
        blob(cx * W, 0.62 * H, 22 * s, (0.95, 0.06, 0.04), 1.3)
        blob(cx * W, 0.62 * H, 80 * s, (0.6, 0.03, 0.02), 0.35)
    blob(0.85 * W, 0.12 * H, 60 * s, (0.1, 0.25, 1.0), 0.9)   # gyrophare
    # nuancier en bas : rouge sang, peau, bleu, gris
    sw = int(W / 5)
    for i, col in enumerate([(0.55, 0.03, 0.03), (0.85, 0.6, 0.48), (0.15, 0.3, 0.75), (0.5, 0.5, 0.5), (0.95, 0.95, 0.95)]):
        img[int(0.90 * H):int(0.95 * H), i * sw:(i + 1) * sw] = col
    ramp = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    img[int(0.95 * H):] = ramp
    return np.clip(img, 0, 1).astype(np.float32)


def group_plate(td: Path, W: int, H: int) -> np.ndarray:
    """« Accueil bleu » synthétique : 4 visages côte à côte (images de test) sur fond bleu nuit."""
    s = W / 1080
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = 0.05 + 0.12 * np.exp(-((xx - 0.5 * W) ** 2 / (0.6 * W) ** 2 + (yy - 0.35 * H) ** 2 / (0.4 * H) ** 2))
    img = np.dstack([g * 0.35, g * 0.6, g * 1.3]).astype(np.float32)
    srcs = [("portrait.jpg", (0.5, 0.25)), ("business-person.png", (0.5, 0.2)), ("lena.jpg", (0.55, 0.5)),
            ("portrait.jpg", (0.5, 0.22))]
    tw, th = int(250 * s), int(330 * s)
    for i, (f, c) in enumerate(srcs):
        t = cover(load_rgb(td / f), tw, th, c)
        if i == 3:
            t = t[:, ::-1]
        x0 = int((15 + i * 265) * s)
        y0 = int((640 + (i % 2) * 70) * s)
        img[y0:y0 + th, x0:x0 + tw] = t * np.array([0.75, 0.85, 1.1], np.float32)
    return np.clip(img, 0, 1)


def detect_faces(img: np.ndarray, n: int = 4) -> list:
    """[(bbox px, landmarks (478,2) px)] via MediaPipe FaceLandmarker (Tasks), triés de gauche à droite.
    Les petits visages d'une image 9:16 échappent au détecteur courte portée : analyse aussi par fenêtres
    glissantes, puis fusion des doublons."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError:
        return []
    model = config.model_path("face_landmarker.task")
    if not model.exists():
        return []
    H, W = img.shape[:2]
    u8 = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)), num_faces=n,
                                        running_mode=vision.RunningMode.IMAGE, min_face_detection_confidence=0.4)
    win = int(min(W, H) * 0.5)
    windows = [(0, 0, W, H)] + [(x, y, win, win) for y in range(0, H - win + 1, win // 2)
                                for x in range(0, W - win + 1, win // 2)]
    found = []
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for (x, y, w, h) in windows:
            crop_ = np.ascontiguousarray(u8[y:y + h, x:x + w])
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_))
            for f in res.face_landmarks:
                pts = np.array([[p.x * w + x, p.y * h + y] for p in f], np.float32)
                x0, y0 = pts.min(0)
                x1, y1 = pts.max(0)
                found.append(((float(x0), float(y0), float(x1), float(y1)), pts))
    out = []
    for bb, pts in sorted(found, key=lambda f: -(f[0][2] - f[0][0]) * (f[0][3] - f[0][1])):
        dup = False
        for ob, _ in out:
            ix = max(0.0, min(bb[2], ob[2]) - max(bb[0], ob[0]))
            iy = max(0.0, min(bb[3], ob[3]) - max(bb[1], ob[1]))
            if ix * iy > 0.3 * min((bb[2] - bb[0]) * (bb[3] - bb[1]), (ob[2] - ob[0]) * (ob[3] - ob[1])):
                dup = True
                break
        if not dup:
            out.append((bb, pts))
    return sorted(out[:n], key=lambda f: f[0][0])


# ----------------------------------------------------------------------------- planches
def sheet_cards(ctx) -> None:
    W, H = ctx["W"], ctx["H"]
    tiles = []
    for t in (0.03, 0.12, 0.25, 0.45, 0.9, 1.6, 1.82):
        tiles.append((gfx.draw_card(None, "INSPIRÉ DE FAITS RÉELS", t, 1.9, "card", size=(W, H)), f"card t={t:.2f}"))
    road = gfx.film_finish(gfx.grade(ctx["night"], "cold"), 3)
    for t in (0.05, 0.2, 0.5, 1.2, 2.0):
        tiles.append((gfx.draw_card(road, "30 OCTOBRE 2026", t, 2.1, "card_over"), f"card_over t={t:.2f}"))
    out = sheet(tiles, 6, 300)
    save_rgb(ctx["out"] / "01_cartons.png", out)
    full = gfx.draw_card(None, "INSPIRÉ DE FAITS RÉELS", 1.0, 1.9, "card", size=(W, H))
    save_rgb(ctx["out"] / "01b_carton_crop100.png", np.concatenate([
        crop(full, 0, int(0.40 * H), W, int(0.16 * H)),
        crop(gfx.draw_card(road, "30 OCTOBRE 2026", 1.2, 2.1, "card_over"), 0, int(0.40 * H), W, int(0.15 * H))], 0))


def sheet_names(ctx) -> None:
    W, H = ctx["W"], ctx["H"]
    face = gfx.film_finish(gfx.grade(ctx["night"], "cold"), 7)
    red = gfx.film_finish(gfx.grade(ctx["night"], "night_red"), 9)
    tiles = [(gfx.draw_name(face, "Gabriel", t), f"GABRIEL t={t:.2f}") for t in (0.0, 0.04, 0.1, 0.2, 0.35, 0.7)]
    tiles += [(gfx.draw_name(face, "Hugo", 0.7), "HUGO 0.70"), (gfx.draw_name(face, "Arsène", 0.7), "ARSÈNE 0.70"),
              (gfx.draw_name(red, "Dogan", 0.9), "DOGAN night_red 0.90")]
    save_rgb(ctx["out"] / "02_prenoms.png", sheet(tiles, 5, 330))
    save_rgb(ctx["out"] / "02b_prenom_crop100.png",
             crop(gfx.draw_name(face, "Arsène", 0.7), 0, int(0.56 * H), W, int(0.16 * H)))


def sheet_timecode(ctx) -> None:
    W = ctx["W"]
    a = gfx.film_finish(gfx.grade(ctx["rush"], "cold"), 2)
    b = gfx.film_finish(gfx.grade(ctx["night"], "cold"), 2)
    tiles = [(gfx.draw_timecode(a, "20:48", "30.10.2026", 0.2), "20:48 point allumé"),
             (gfx.draw_timecode(a, "20:48", "30.10.2026", 0.8), "20:48 point éteint"),
             (gfx.draw_timecode(b, "23:42", "30.10.2026", 0.1), "23:42 nuit")]
    save_rgb(ctx["out"] / "03_timecode.png", sheet(tiles, 3, 420))
    s = W / 1080
    save_rgb(ctx["out"] / "03b_timecode_crop100.png", np.concatenate([
        crop(tiles[0][0], 0, 0, int(600 * s), int(300 * s)), crop(tiles[2][0], 0, 0, int(600 * s), int(300 * s))], 1))


def sheet_subs(ctx) -> None:
    W, H = ctx["W"], ctx["H"]
    a = gfx.film_finish(gfx.grade(ctx["rush"], "cold"), 4)
    b = gfx.film_finish(gfx.grade(ctx["night"], "cold"), 4)
    c = gfx.film_finish(gfx.grade(ctx["night"], "night_red"), 4)
    items = [(a, "Je crois qu'il y a un gars là.", "dialogue"), (c, "Tu sais quand il vient ?", "dialogue"),
             (b, "Le 30 octobre 2026, quatre amis\nont pris la route pour un escape game.", "narrator"),
             (b, "Les organisateurs jurent que cette pièce ne fait pas partie du jeu.", "narrator"),
             (b, "Identification terminée.", "entity"), (a, "C'est qui lui ?", "entity")]
    tiles = [(gfx.draw_subtitle(img, txt, k), f"{k}") for img, txt, k in items]
    for img, cap in tiles:  # repères de la zone sûre (contrôle) : traits jaunes
        x0, y0, x1, y1 = gfx.safe_zone(W, H)
        img[y1:y1 + 2] = (1, 1, 0)
        img[:, x0:x0 + 1] = (1, 1, 0)
        img[:, x1:x1 + 1] = (1, 1, 0)
    save_rgb(ctx["out"] / "04_soustitres.png", sheet(tiles, 6, 300))
    rows = [crop(t[0], 0, int(0.68 * H), W, int(0.13 * H)) for t in tiles]
    save_rgb(ctx["out"] / "04b_soustitres_crop100.png", np.concatenate(rows, 0))


def _hud_frames(ctx, mode: str, times, lock_step=0.42):
    plate = ctx["group"]
    faces = ctx["group_faces"]
    graded = gfx.film_finish(gfx.grade(plate, "blue"), 1)
    names = ["DOGAN", "GABRIEL", "HUGO", "ARSÈNE"]
    tiles = []
    for t in times:
        img = graded
        for i, (bb, lm) in enumerate(faces):
            if mode == "identify":
                img = gfx.draw_hud_face(img, bb, t - i * lock_step, f"SUJET {i + 1:02d} — {names[i]} — IDENTIFIÉ",
                                        None, int(t * 30), "identify", lm)
            else:
                img = gfx.draw_hud_face(img, bb, t, f"VICTIME {i + 1:02d} — {names[i]}", None, int(t * 30) + i,
                                        "victim", lm)
        tiles.append((img, f"{mode} t={t:.2f}"))
    return tiles


def sheet_hud(ctx) -> None:
    W = ctx["W"]
    if not ctx["group_faces"]:
        print("  (pas de visage détecté : HUD sur boîtes fixes)")
        s = W / 1080
        ctx["group_faces"] = [((x * s, 690 * s, (x + 200) * s, 930 * s), None) for x in (40, 305, 570, 835)]
    tiles = _hud_frames(ctx, "identify", (-0.1, 0.05, 0.15, 0.3, 0.5, 0.8, 1.3, 2.2))
    save_rgb(ctx["out"] / "05_hud_identify.png", sheet(tiles, 4, 380))
    s = W / 1080
    bb = ctx["group_faces"][0][0]
    cx0, cy0 = int(max(0, bb[0] - 90 * s)), int(max(0, bb[1] - 120 * s))
    save_rgb(ctx["out"] / "05b_hud_crop100.png",
             np.concatenate([crop(tiles[i][0], cx0, cy0, int(560 * s), int(520 * s)) for i in (2, 4, 6)], 1))
    vt = _hud_frames(ctx, "victim", (0.0, 0.04, 0.1, 0.2, 0.4, 0.7, 1.0, 1.4))
    save_rgb(ctx["out"] / "06_hud_victim.png", sheet(vt, 4, 380))
    save_rgb(ctx["out"] / "06b_hud_victim_crop100.png",
             np.concatenate([crop(vt[i][0], cx0, cy0, int(560 * s), int(520 * s)) for i in (1, 4, 6)], 1))


def sheet_circles(ctx) -> None:
    W, H = ctx["W"], ctx["H"]
    s = W / 1080
    photo = gfx.film_finish(gfx.grade(ctx["photo"], "cold", red_accent=1.0), 5)
    faces = ctx["photo_faces"] or [((x * s, 760 * s, (x + 150) * s, 960 * s), None) for x in (120, 380, 640, 860)]
    tiles = []
    for p in (0.1, 0.3, 0.55, 0.8, 1.0):
        img = photo
        for i, (bb, _) in enumerate(faces):
            c = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
            ax = ((bb[2] - bb[0]) * 0.75, (bb[3] - bb[1]) * 0.78)
            img = gfx.draw_red_circle(img, c, ax, np.clip(p * 1.6 - i * 0.2, 0, 1), seed=11 + i)
        tiles.append((img, f"cercles p={p:.2f}"))
    sil_c, sil_ax = (0.5 * W, 0.33 * H), (110 * s, 230 * s)
    for p in (0.25, 0.6, 1.0):
        tiles.append((gfx.draw_red_circle(tiles[4][0], sil_c, sil_ax, p, seed=99, frantic=True),
                      f"frénétique p={p:.2f}"))
    save_rgb(ctx["out"] / "07_cercles.png", sheet(tiles, 4, 380))
    bb = faces[1][0]
    cx0, cy0 = int(max(0, bb[0] - 140 * s)), int(max(0, bb[1] - 150 * s))
    full = tiles[4][0]
    save_rgb(ctx["out"] / "07b_cercles_crop100.png", np.concatenate([
        crop(full, cx0, cy0, int(520 * s), int(560 * s)),
        crop(tiles[-1][0], int(sil_c[0] - 300 * s), int(sil_c[1] - 320 * s), int(600 * s), int(560 * s))], 1))
    # variété de tracés (graines)
    var = []
    blank = np.full((int(420 * s), int(420 * s), 3), 0.78, np.float32)
    for sd in range(8):
        var.append((gfx.draw_red_circle(blank, (210 * s, 210 * s), (150 * s, 120 * s), 1.0, sd, sd >= 6), f"seed {sd}"))
    save_rgb(ctx["out"] / "07c_cercles_graines.png", sheet(var, 4, 300))


def sheet_grades(ctx) -> None:
    looks = [("original", None, {}), ("cold", "cold", {}), ("cold red_accent", "cold", {"red_accent": 1.0}),
             ("cold keep_blue", "cold", {"keep_blue": True}), ("cold warm_hl", "cold", {"warm_highlights": 0.3}),
             ("night_red", "night_red", {}), ("blue", "blue", {}), ("warm", "warm", {}),
             ("warm>cold 0.5", "warm_to_cold", {"mix": 0.5}), ("warm>cold 1.0", "warm_to_cold", {"mix": 1.0}),
             ("bleach", "bleach", {})]
    for key, img in (("rush", ctx["rush"]), ("testdata", ctx["portrait"]), ("night", ctx["night"])):
        tiles = []
        for cap, pre, kw in looks:
            if pre is None:
                tiles.append((img, cap))
            else:
                kw = dict(kw)
                mix = kw.pop("mix", 0.0)
                tiles.append((gfx.grade(img, pre, mix, **kw), cap))
        save_rgb(ctx["out"] / f"08_etalonnages_{key}.png", sheet(tiles, 6, 260))


def sheet_finish(ctx) -> None:
    W, H = ctx["W"], ctx["H"]
    s = W / 1080
    g = gfx.grade(ctx["night"], "cold")
    f0 = gfx.film_finish(g, 0)
    f1 = gfx.film_finish(g, 1)
    tiles = [(g, "grade cold seul"), (f0, "film_finish frame 0"), (gfx.film_finish(g, 0, grain=0.0, halation=0.0), "vignette+CA seuls"),
             (np.clip(0.5 + (f1 - f0) * 4, 0, 1), "diff frame1-frame0 x4 (grain temporel)")]
    save_rgb(ctx["out"] / "09_film_finish.png", sheet(tiles, 4, 380))
    save_rgb(ctx["out"] / "09b_finish_crop100.png", np.concatenate([
        crop(g, int(0.15 * W), int(0.70 * H), int(500 * s), int(400 * s)),
        crop(f0, int(0.15 * W), int(0.70 * H), int(500 * s), int(400 * s)),
        crop(f0, 0, 0, int(400 * s), int(400 * s))], 1))
    # contrôle déterminisme
    assert np.array_equal(gfx.film_finish(g, 5), gfx.film_finish(g, 5)), "film_finish non déterministe"


def sheet_composite(ctx) -> None:
    """Images « comme au rendu » en 540x960 (préversion) : étalonnage + finition + overlays."""
    W, H = 540, 960
    night = cover(ctx["night"], W, H)
    group = cover(ctx["group"], W, H)
    k = W / ctx["W"]
    faces = [((b[0] * k, b[1] * k, b[2] * k, b[3] * k), None if lm is None else lm * k) for b, lm in ctx["group_faces"]]
    a = gfx.film_finish(gfx.grade(night, "cold"), 10)
    a = gfx.draw_timecode(a, "20:48", "30.10.2026", 0.2)
    a = gfx.draw_subtitle(a, "C'est qui lui ?", "dialogue")
    b = gfx.film_finish(gfx.grade(group, "blue"), 11)
    b = gfx.draw_timecode(b, "20:55", "30.10.2026", 0.2)
    for i, (bb, lm) in enumerate(faces):
        b = gfx.draw_hud_face(b, bb, 1.4 - i * 0.42, f"SUJET {i + 1:02d} — IDENTIFIÉ", None, 40, "identify", lm)
    b = gfx.draw_subtitle(b, "Identification terminée.", "dialogue")
    c = gfx.film_finish(gfx.grade(night, "night_red"), 12)
    c = gfx.draw_name(c, "Dogan", 0.8)
    d = gfx.draw_card(gfx.film_finish(gfx.grade(night, "cold"), 13), "30 OCTOBRE 2026", 1.0, 2.1, "card_over")
    d = gfx.draw_timecode(d, "20:40", "30.10.2026", 1.0)
    d = gfx.draw_subtitle(d, "Le 30 octobre 2026, quatre amis\nont pris la route pour un escape game.", "narrator")
    e = gfx.film_finish(gfx.grade(night, "bleach"), 14)
    e = gfx.draw_subtitle(e, "C'est qui lui ?", "entity")
    f = gfx.film_finish(np.zeros((H, W, 3), np.float32), 15)
    f = gfx.draw_title_sub(f, "INSPIRÉ DE FAITS RÉELS — 30.10.2026", 1.6, 2.4)
    tiles = [(a, "540 cold+TC+dialogue"), (b, "540 blue+HUD"), (c, "540 night_red+prénom"),
             (d, "540 card_over+narrateur"), (e, "540 bleach+entité"), (f, "540 title_sub")]
    save_rgb(ctx["out"] / "10_composite_540.png", sheet(tiles, 6, 270))


def sheet_fx(ctx) -> None:
    base = gfx.film_finish(gfx.grade(ctx["rush"], "cold"), 6)
    tiles = [(gfx.glitch(base, 3, 0.3), "glitch 0.3"), (gfx.glitch(base, 4, 1.0), "glitch 1.0"),
             (gfx.rgb_split(base, 0.4, 4), "rgb_split 0.4"), (gfx.rgb_split(base, 1.0, 5), "rgb_split 1.0"),
             (gfx.negative(base), "negative"), (gfx.darken(base, 0.35), "darken 0.35")]
    save_rgb(ctx["out"] / "12_effets.png", sheet(tiles, 6, 300))


def _ink_box(diff: np.ndarray, thr: float = 0.12):
    m = diff.max(axis=2) > thr
    ys, xs = np.nonzero(m)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else None


def check(ctx) -> None:
    """Contrôles automatiques du contrat : types, pureté, déterminisme, zones sûres (1080 et 540)."""
    errors = []
    for W, H in ((1080, 1920), (540, 960)):
        s = W / 1080
        img = np.full((H, W, 3), 0.25, np.float32)
        x0, y0, x1, y1 = gfx.safe_zone(W, H)
        texts = [("Je crois qu'il y a un gars là.", "dialogue"),
                 ("Les organisateurs jurent que cette pièce ne fait pas partie du jeu.", "narrator"),
                 ("Le 30 octobre 2026, quatre amis ont pris la route pour un escape game.", "narrator"),
                 ("Identification terminée.", "entity"), ("C'est qui lui ?", "entity")]
        for txt, kind in texts:
            o = gfx.draw_subtitle(img, txt, kind)
            b = _ink_box(np.abs(o - img))
            lines = gfx._fit(gfx._typo(txt, gfx._SUB_STYLES[kind]["family"]), gfx._SUB_STYLES[kind]["family"],
                             gfx._SUB_STYLES[kind]["size"] * s, gfx._SUB_STYLES[kind]["weight"],
                             gfx._SUB_STYLES[kind]["tracking"], (x1 - x0) - 28 * s, 2, 0.8)[0]
            ok = b is not None and b[0] >= x0 and b[2] <= x1 and b[3] <= 0.78 * H and len(lines) <= 2
            print(f"  {W}x{H} sous-titre {kind:8s} {len(lines)} l. encre={b} {'ok' if ok else 'ÉCHEC'}")
            if not ok:
                errors.append(f"sous-titre hors zone : {txt!r} {b}")
        b = _ink_box(np.abs(gfx.draw_timecode(img, "20:48", "30.10.2026", 0.1) - img), 0.2)
        ok = b is not None and b[0] >= x0 and b[1] >= y0 - 2 * s and b[3] <= 245 * s
        print(f"  {W}x{H} timecode encre={b} {'ok' if ok else 'ÉCHEC'}")
        if not ok:
            errors.append(f"timecode hors zone {b}")
        for nm in ("Gabriel", "Arsène", "Dogan", "Hugo"):
            for tt in (0.0, 1.0):  # début de l'animation (approche la plus large) et état final
                b = _ink_box(np.abs(gfx.draw_name(img, nm, tt) - img), 0.08)
                ok = b is not None and b[0] >= x0 and b[2] <= x1 and b[3] <= 0.8 * H
                print(f"  {W}x{H} prénom {nm:8s} t={tt} encre={b} {'ok' if ok else 'ÉCHEC'}")
                if not ok:
                    errors.append(f"prénom hors zone {nm} t={tt} {b}")
        for txt in ("INSPIRÉ DE FAITS RÉELS", "30 OCTOBRE 2026"):
            b = _ink_box(np.abs(gfx.draw_card(img, txt, 1.0, 2.0) - gfx.draw_card(img, "", 1.0, 2.0)), 0.2)
            ok = b is not None and b[0] >= x0 and b[2] <= x1
            print(f"  {W}x{H} carton {txt!r} encre={b} {'ok' if ok else 'ÉCHEC'}")
            if not ok:
                errors.append(f"carton hors zone {txt} {b}")
    # pureté, déterminisme, types
    img = ctx["night"]
    ref = img.copy()
    calls = {
        "grade": lambda: gfx.grade(img, "warm_to_cold", 0.4, red_accent=1.0),
        "film_finish": lambda: gfx.film_finish(img, 12),
        "glitch": lambda: gfx.glitch(img, 3, 0.8), "rgb_split": lambda: gfx.rgb_split(img, 0.5, 3),
        "negative": lambda: gfx.negative(img), "darken": lambda: gfx.darken(img, 0.4),
        "card": lambda: gfx.draw_card(img, "30 OCTOBRE 2026", 0.3, 2.0, "card_over"),
        "name": lambda: gfx.draw_name(img, "Arsène", 0.2),
        "timecode": lambda: gfx.draw_timecode(img, "22:12", "30.10.2026", 0.4),
        "subtitle": lambda: gfx.draw_subtitle(img, "C'est quoi ça ?", "dialogue", 0.6),
        "hud": lambda: gfx.draw_hud_face(img, (300, 500, 500, 760), 0.4, "SUJET 01 — DOGAN — IDENTIFIÉ", None, 9,
                                         "identify", np.random.default_rng(0).uniform(300, 500, (478, 2))),
        "hud_victim": lambda: gfx.draw_hud_face(img, (300, 500, 500, 760), 0.05, "VICTIME 01 — DOGAN", None, 9, "victim"),
        "circle": lambda: gfx.draw_red_circle(img, (400, 600), (100, 130), 0.7, 5),
        "circle_frantic": lambda: gfx.draw_red_circle(img, (400, 600), (100, 200), 0.7, 5, True),
        "title_sub": lambda: gfx.draw_title_sub(img, "INSPIRÉ DE FAITS RÉELS — 30.10.2026", 1.0, 2.4),
    }
    for name, fn in calls.items():
        a, b = fn(), fn()
        ok = (np.array_equal(a, b) and np.array_equal(img, ref) and a.dtype == np.float32 and a.shape == img.shape
              and np.isfinite(a).all() and a.min() >= 0.0 and a.max() <= 1.0)
        if not ok:
            errors.append(f"{name} : contrat non respecté")
    print(f"  pureté/déterminisme/types : {len(calls)} fonctions {'ok' if not errors else 'voir erreurs'}")
    (ctx["out"] / "check.json").write_text(json.dumps({"errors": errors}, indent=1, ensure_ascii=False))
    if errors:
        raise SystemExit("ÉCHECS : " + " | ".join(errors))


def perf(ctx) -> None:
    res = {}
    for W, H in ((1080, 1920), (540, 960)):
        img = cover(ctx["night"], W, H)
        s = W / 1080
        bb = (300 * s, 600 * s, 520 * s, 880 * s)
        lm = None
        cases = {
            "grade_cold": lambda: gfx.grade(img, "cold"),
            "grade_warm_to_cold": lambda: gfx.grade(img, "warm_to_cold", 0.4),
            "film_finish": lambda: gfx.film_finish(img, 3),
            "draw_card": lambda: gfx.draw_card(img, "INSPIRÉ DE FAITS RÉELS", 0.3, 1.9),
            "draw_name": lambda: gfx.draw_name(img, "Gabriel", 0.1),
            "draw_timecode": lambda: gfx.draw_timecode(img, "20:48", "30.10.2026", 0.2),
            "draw_subtitle": lambda: gfx.draw_subtitle(img, "Je crois qu'il y a un gars là.", "dialogue"),
            "draw_hud_face": lambda: gfx.draw_hud_face(img, bb, 0.6, "SUJET 01 — DOGAN — IDENTIFIÉ", None, 18,
                                                       "identify", lm),
            "draw_hud_face_victim": lambda: gfx.draw_hud_face(img, bb, 0.1, "VICTIME 01 — DOGAN", None, 18, "victim"),
            "draw_red_circle": lambda: gfx.draw_red_circle(img, (500 * s, 700 * s), (120 * s, 150 * s), 0.8, 3),
            "draw_red_circle_frantic": lambda: gfx.draw_red_circle(img, (500 * s, 700 * s), (110 * s, 230 * s), 0.8, 4, True),
            "draw_title_sub": lambda: gfx.draw_title_sub(img, "INSPIRÉ DE FAITS RÉELS — 30.10.2026", 1.0, 2.4),
            "glitch": lambda: gfx.glitch(img, 5, 0.8),
            "rgb_split": lambda: gfx.rgb_split(img, 0.4, 5),
            "copy (référence)": lambda: img.copy(),
        }
        for name, fn in cases.items():
            fn()
            ts = []
            for _ in range(7):
                t0 = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - t0) * 1000)
            res[f"{W}x{H} {name}"] = round(float(np.median(ts)), 2)
    (ctx["out"] / "perf.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for k, v in res.items():
        print(f"  {k:40s} {v:7.1f} ms")


SHEETS = {"cards": sheet_cards, "names": sheet_names, "timecode": sheet_timecode, "subs": sheet_subs,
          "hud": sheet_hud, "circles": sheet_circles, "grades": sheet_grades, "finish": sheet_finish,
          "composite": sheet_composite, "fx": sheet_fx, "check": check, "perf": perf}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=config.WORK_DIR / "gfx_previews")
    ap.add_argument("--testdata", type=Path, default=config.WORK_DIR / "testdata")
    ap.add_argument("--rush", type=Path, default=None, help="rush ou proxy vidéo pour l'image de test")
    ap.add_argument("--rush-t", type=float, default=7.5)
    ap.add_argument("--photo", type=Path, default=None, help="photo de groupe (défaut : proxy/rush IMG_8350)")
    ap.add_argument("--res", type=int, default=1080, choices=(540, 1080))
    ap.add_argument("--only", default="", help="liste séparée par des virgules : " + ",".join(SHEETS))
    args = ap.parse_args()
    W, H = (1080, 1920) if args.res == 1080 else (540, 960)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    td = args.testdata
    rush = args.rush
    if rush is None:
        for cand in (config.proxy_path("IMG_8346"),):
            if cand.exists():
                rush = cand
        if rush is None:
            try:
                rush = config.rush_path("IMG_8346")
            except FileNotFoundError:
                rush = None
    photo_path = args.photo
    if photo_path is None:
        for cand in (config.PROXY_DIR / "IMG_8350.png",):
            if cand.exists():
                photo_path = cand
        if photo_path is None:
            try:
                photo_path = config.rush_path("IMG_8350")
            except FileNotFoundError:
                photo_path = None
    ctx = dict(W=W, H=H, out=out)
    ctx["night"] = night_plate(td, W, H)
    ctx["portrait"] = cover(load_rgb(td / "portrait.jpg"), W, H, (0.5, 0.4))
    fr = rush_frame(rush, args.rush_t, out, W, H)
    ctx["rush"] = fr if fr is not None else ctx["portrait"]
    ctx["group"] = group_plate(td, W, H)
    ctx["group_faces"] = detect_faces(ctx["group"]) if any(k in args.only for k in ("hud", "composite")) or not args.only else []
    if photo_path is not None and photo_path.exists():
        ph = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)  # (cv2 applique l'orientation EXIF)
        ctx["photo"] = cover(cv2.cvtColor(ph, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, W, H)
    else:
        ctx["photo"] = ctx["group"]
    ctx["photo_faces"] = detect_faces(ctx["photo"]) if ("circles" in args.only or not args.only) else []
    print(f"sources : rush={rush} photo={photo_path} visages groupe={len(ctx['group_faces'])} "
          f"photo={len(ctx['photo_faces'])}")
    todo = [k.strip() for k in args.only.split(",") if k.strip()] or list(SHEETS)
    for k in todo:
        t0 = time.perf_counter()
        SHEETS[k](ctx)
        print(f"{k:10s} ok ({time.perf_counter() - t0:.1f} s)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
