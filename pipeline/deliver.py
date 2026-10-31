"""Livraison : deliver(video, mix) -> dict.

final   : OUT_DIR/CEST_QUI_LUI_bande-annonce.mp4 (H.264 High, yuv420p, 1080x1920, 30 i/s, CRF 16 preset slow,
          AAC 320 kb/s 48 kHz stéréo, -movflags +faststart), OUT_DIR/CEST_QUI_LUI_couverture.jpg (1080x1920),
          OUT_DIR/LISEZMOI.txt (≤ 5 lignes).
preview : RENDER_DIR/preview/CEST_QUI_LUI_preview.mp4 (540x960, mêmes réglages audio) + couverture d'aperçu ;
          rien n'est écrit dans OUT_DIR.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from pipeline import render as R  # noqa: E402

FINAL_SIZE = (config.W, config.H)


def _quality_of(video: Path) -> str:
    return "preview" if "preview" in Path(video).stem else "final"


def mux(video: Path, mix: Path, out: Path, quality: str, log=print) -> dict:
    """Mux vidéo + mix : nombre d'images conservé, audio complété/coupé à la durée exacte de la vidéo."""
    video, mix, out = Path(video), Path(mix), Path(out)
    if not video.exists():
        raise FileNotFoundError(f"vidéo absente : {video}")
    if not mix.exists():
        raise FileNotFoundError(f"mix absent : {mix}")
    n = R.count_frames(video)
    dur = n / config.FPS
    w, h = R.video_size(video)
    target = FINAL_SIZE if quality == "final" else R.SIZES["preview"]
    vf = []
    if (w, h) != target:
        log(f"[deliver] ⚠ vidéo {w}x{h} mise à l'échelle en {target[0]}x{target[1]}")
        vf.append(f"scale={target[0]}:{target[1]}:flags=lanczos")
    vf.append("format=yuv420p")
    crf, preset = ("16", "slow") if quality == "final" else ("20", "fast")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    cmd = [R._ffmpeg(), "-y", "-v", "error", "-nostdin", "-i", str(video), "-i", str(mix),
           "-map", "0:v:0", "-map", "1:a:0", "-frames:v", str(n),
           "-vf", ",".join(vf), "-r", str(config.FPS),
           "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-crf", crf, "-preset", preset,
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-af", f"apad=whole_dur={dur:.6f},atrim=0:{dur:.6f},aresample=48000",
           "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2",
           "-movflags", "+faststart", "-t", f"{dur:.6f}", str(tmp)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"mux échoué : {r.stderr[-1200:]}")
    n2 = R.count_frames(tmp)
    if n2 != n:
        raise RuntimeError(f"mux : {n2} images au lieu de {n}")
    os.replace(tmp, out)
    log(f"[deliver] {out.name} : {n} images, {dur:.2f} s ({time.time() - t0:.0f} s)")
    return {"file": str(out), "frames": n, "duration": round(dur, 3), "size": list(target)}


# ---------------------------------------------------------------------------
# Couverture
# ---------------------------------------------------------------------------
def _load_rgb(p: Path, size) -> np.ndarray:
    import cv2
    from PIL import Image
    im = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    h, w = im.shape[:2]
    W, H = size
    s = max(W / w, H / h)
    im = cv2.resize(im, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA if s < 1 else
                    cv2.INTER_LANCZOS4)
    y0, x0 = (im.shape[0] - H) // 2, (im.shape[1] - W) // 2
    return np.ascontiguousarray(im[y0:y0 + H, x0:x0 + W])


def _fallback_plate(size, G) -> np.ndarray:
    """Plaque de repli : brume froide, faisceau de phares, titre serif."""
    W, H = size
    img = R._fallback_title(W, H, 3.0, G, y_center=0.28)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    beam = np.exp(-(((xx / W - 0.5) / 0.35) ** 2 + ((yy / H - 0.72) / 0.18) ** 2)) * 0.22
    return np.clip(img + beam[..., None] * np.array([0.9, 0.95, 1.0], np.float32), 0, 1).astype(np.float32)


def make_cover(out: Path, edl: dict | None = None, size=FINAL_SIZE, log=print) -> Path:
    """Couverture : plaque du titre (assets/title/cover_plate.png) + silhouette détourée de 8346."""
    G = R.gfx()
    W, H = size
    plate_p = config.TITLE_DIR / "cover_plate.png"
    warnings = []
    if plate_p.exists():
        img = _load_rgb(plate_p, size)
    else:
        warnings.append("cover_plate.png absent : plaque de repli dessinée")
        img = _fallback_plate(size, G)
    # emplacement de la silhouette : celui prévu par le titre (title_meta.json > cover > silhouette_slot),
    # sinon sous la boîte du titre
    x, foot_y, height = 0.5, 0.83, 0.36
    meta_p = config.TITLE_DIR / "title_meta.json"
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        slot = (meta.get("cover") or {}).get("silhouette_slot")
        box = (meta.get("cover") or {}).get("text_box") or meta.get("text_box")
        if slot:
            x, foot_y, height = float(slot["x"]), float(slot["foot_y"]), float(slot["height"])
        elif box:
            bottom = float(box[3]) / config.H
            height = min(0.46, max(0.22, foot_y - bottom - 0.05))
    except Exception:
        pass
    sil = None
    try:
        sil = R.ensure_silhouette(edl or {"shots": []}, log)
    except Exception as e:
        warnings.append(f"silhouette indisponible ({e})")
    if sil and sil.get("still") and Path(sil["still"]).exists():
        from PIL import Image
        rgba = np.asarray(Image.open(sil["still"]).convert("RGBA"))
        S = R.silhouette_mod()
        try:
            img = S.composite(img, rgba, x, foot_y, height, opacity=0.96, blur=1.2, darkness=0.93,
                              blend="normal", frame_idx=0, occluder_mask=None)
        except Exception as e:
            warnings.append(f"incrustation de la silhouette échouée ({e})")
    try:
        img = G.film_finish(img, 4242, grain=0.04, vignette=0.42, halation=0.2)
    except Exception as e:
        warnings.append(f"film_finish échoué ({e})")
    from PIL import Image
    out.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(u8).resize((W, H), Image.LANCZOS).save(out, quality=94, subsampling=0, optimize=True)
    for w in warnings:
        log(f"[deliver] ⚠ couverture : {w}")
    log(f"[deliver] {out.name} ({W}x{H})")
    return out


def _voice_backend() -> str:
    try:
        b = json.loads((config.VOICE_DIR / "manifest.json").read_text(encoding="utf-8")).get("backend", "")
    except Exception:
        b = ""
    b = str(b).split(":")[0]
    return {"piper": "Piper", "edge-tts": "edge-tts", "edge": "edge-tts"}.get(b, b or "neuronale")


def write_lisezmoi(out: Path, duration: float) -> Path:
    lines = [
        f"C'EST QUI LUI ? : fausse bande-annonce d'horreur verticale (1080x1920, 30 i/s, {duration:.0f} s), "
        "montée automatiquement à partir des 18 vidéos et 4 photos du 30/10/2026.",
        "Image : étalonnage froid, VFX sur les visages (MediaPipe), silhouette détourée de IMG_8346 et "
        "incrustée, parallaxe 2,5D sur la photo de groupe, titre en 3D (Blender).",
        f"Son : voix off neuronale ({_voice_backend()}), vraies répliques des rushs, piano réel de IMG_8354, "
        "sound design 100 % synthétisé ; master à −14 LUFS et −1 dBTP.",
        "Sous-titres incrustés dans la zone sûre TikTok/Instagram ; aucun flash de plus de 3 par seconde.",
        "Vérifié automatiquement (durée, format, loudness, ASR, noirs, flashs) ; les originaux n'ont pas été touchés.",
    ]
    out.write_text("\n".join(lines[:5]) + "\n", encoding="utf-8")
    return out


def deliver(video, mix, quality: str | None = None, edl: dict | None = None, log=print) -> dict:
    """Mux final + couverture + LISEZMOI. quality déduite du nom de la vidéo si absente."""
    video, mix = Path(video), Path(mix)
    quality = quality or _quality_of(video)
    config.ensure_dirs()
    if edl is None:
        try:
            edl = R.load_edl()
        except Exception:
            edl = None
    if quality == "final":
        out_dir = config.OUT_DIR
        mp4 = out_dir / config.FINAL_MP4
        cover = out_dir / config.COVER_JPG
    else:
        out_dir = config.RENDER_DIR / "preview"
        mp4 = out_dir / "CEST_QUI_LUI_preview.mp4"
        cover = out_dir / "CEST_QUI_LUI_couverture_preview.jpg"
    res = {"quality": quality, **mux(video, mix, mp4, quality, log)}
    res["mp4"] = res.pop("file")
    try:
        res["cover"] = str(make_cover(cover, edl, FINAL_SIZE, log))
    except Exception as e:  # la couverture ne doit pas bloquer le film
        log(f"[deliver] ⚠ couverture non produite : {e}")
        res["cover"] = None
    if quality == "final":
        res["readme"] = str(write_lisezmoi(out_dir / config.README_TXT, res["duration"]))
    (config.RENDER_DIR / f"deliver_{quality}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                               encoding="utf-8")
    return res


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Mux final + couverture + LISEZMOI")
    ap.add_argument("--quality", default="preview", choices=["preview", "final"])
    ap.add_argument("--video", default=None)
    ap.add_argument("--mix", default=None)
    a = ap.parse_args(argv)
    video = a.video or config.RENDER_DIR / f"video_{a.quality}.mp4"
    mix = a.mix or config.RENDER_DIR / f"mix_{a.quality}.wav"
    print(json.dumps(deliver(video, mix, a.quality), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
