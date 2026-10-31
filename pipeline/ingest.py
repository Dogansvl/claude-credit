"""Ingestion des rushs : inventaire ffprobe, proxies vidéo 1080x1920 30 i/s CFR, photos PNG orientées.

Les originaux ne sont JAMAIS modifiés (lecture seule) ; tout est écrit dans config.PROXY_DIR
(fichiers temporaires puis renommage atomique) et l'inventaire dans config.ANALYSIS_DIR.

Proxy vidéo (contrat docs/ARCHITECTURE.md) :
  * rotation appliquée (autorotation ffmpeg via la matrice d'affichage), orientation vérifiée ;
  * `fps=30` (rééchantillonnage temporel correct pour 120 i/s et ~16 i/s VFR), nombre d'images exact ;
  * mise à l'échelle « cover » puis recadrage centré si le ratio n'est pas exactement 9:16
    (passage direct sans recadrage si la source est déjà en 1080x1920, ex. rushs « légers » préparés sous Windows) ;
  * HDR (HLG / PQ / Dolby Vision) → tonemapping zscale + tonemap vers BT.709 SDR ;
  * yuv420p, H.264 CRF 14 preset medium, GOP 1 s (recherche précise rapide au rendu) ;
  * premier flux audio décodable (pistes de métadonnées ignorées), AAC 48 kHz stéréo 256k,
    silence si pas d'audio ; durée audio = durée vidéo (au nombre d'échantillons près).

CLI : python -m pipeline.ingest [--force] [--jobs 2] [--only IMG_8346 IMG_8350]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config  # noqa: E402

FFMPEG = os.environ.get("ESCAPE_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("ESCAPE_FFPROBE", "ffprobe")

CHAIN_VERSION = "ingest-v3"          # change → proxies reconstruits
PROXY_JOBS = 2                        # processus ffmpeg en parallèle (4 CPU partagés)
PHOTO_MAX_H = 2560
HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}
# codecs audio que ffmpeg sait décoder sur un iPhone ; tout le reste (mebx, tmcd, apac Apple...) est ignoré
AUDIO_OK = {"aac", "alac", "pcm_s16le", "pcm_s16be", "pcm_s24le", "pcm_s24be", "pcm_f32le", "pcm_s32le",
            "mp3", "ac3", "eac3", "opus", "flac", "vorbis", "pcm_u8", "mp2"}


def log(msg: str) -> None:
    """print robuste : une console Windows en cp1252 ne doit jamais faire planter le pipeline (→, ≠, …)."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(msg.encode(enc, "replace").decode(enc, "replace"), flush=True)


# ---------------------------------------------------------------------------
# ffprobe
# ---------------------------------------------------------------------------
def _run(cmd: list, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def _ratio(s: str | None) -> float | None:
    try:
        f = Fraction(s)
        return float(f) if f > 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _float(x) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _rotation(stream: dict) -> int:
    """Rotation (degrés, sens ffmpeg) depuis la matrice d'affichage ou l'ancien tag `rotate`."""
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"])))
            except (TypeError, ValueError):
                pass
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        try:
            return -int(tag)  # le tag est en sens horaire, la matrice en sens trigonométrique
        except ValueError:
            pass
    return 0


def probe(path) -> dict:
    """Infos ffprobe utiles : durée, fps moyen / r_frame_rate, codec, rotation, dimensions, flux audio, HDR."""
    path = Path(path)
    r = _run([FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", path], timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe a échoué sur {path}: {r.stderr.strip()[-500:]}")
    data = json.loads(r.stdout or "{}")
    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    vids = [s for s in streams if s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")]
    auds = [s for s in streams if s.get("codec_type") == "audio"]
    info: dict = {
        "path": str(path),
        "size_bytes": int(fmt.get("size") or path.stat().st_size),
        "format": fmt.get("format_name"),
        "duration": _float(fmt.get("duration")),
        "start_time": _float(fmt.get("start_time")) or 0.0,
        "creation_time": (fmt.get("tags") or {}).get("creation_time"),
        "n_streams": len(streams),
        "other_streams": [f"{s.get('index')}:{s.get('codec_type')}:{s.get('codec_tag_string') or s.get('codec_name')}"
                          for s in streams if s.get("codec_type") not in ("video", "audio")],
        "video": None,
        "audio": [],
        "audio_index": None,
    }
    if vids:
        v = vids[0]
        rot = _rotation(v)
        w, h = int(v.get("width") or 0), int(v.get("height") or 0)
        dw, dh = (h, w) if abs(rot) % 180 == 90 else (w, h)
        sd_types = [sd.get("side_data_type", "") for sd in v.get("side_data_list") or []]
        transfer = v.get("color_transfer") or "unknown"
        dovi = any("DOVI" in t or "Dolby" in t for t in sd_types)
        pix = v.get("pix_fmt") or ""
        info["video"] = {
            "index": v.get("index"),
            "codec": v.get("codec_name"),
            "profile": v.get("profile"),
            "pix_fmt": pix,
            "bit_depth": 10 if ("10" in pix or "p010" in pix) else 8,
            "width": w, "height": h, "rotation": rot, "display_w": dw, "display_h": dh,
            "r_frame_rate": v.get("r_frame_rate"), "avg_frame_rate": v.get("avg_frame_rate"),
            "fps_avg": _ratio(v.get("avg_frame_rate")), "fps_r": _ratio(v.get("r_frame_rate")),
            "nb_frames": int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None,
            "duration": _float(v.get("duration")),
            "start_time": _float(v.get("start_time")) or 0.0,
            "color_transfer": transfer, "color_primaries": v.get("color_primaries") or "unknown",
            "color_space": v.get("color_space") or "unknown", "color_range": v.get("color_range") or "unknown",
            "side_data": sd_types,
            "dolby_vision": dovi,
            "hdr": transfer in HDR_TRANSFERS or dovi,
        }
        fa, fr = info["video"]["fps_avg"], info["video"]["fps_r"]
        info["video"]["vfr_suspect"] = bool(fa and fr and abs(fa - fr) / fr > 0.02)
    for a in auds:
        info["audio"].append({
            "index": a.get("index"), "codec": a.get("codec_name"), "tag": a.get("codec_tag_string"),
            "channels": int(a.get("channels") or 0), "sample_rate": int(a.get("sample_rate") or 0),
            "duration": _float(a.get("duration")), "start_time": _float(a.get("start_time")) or 0.0,
        })
    info["audio_index"] = _pick_audio(info["audio"])
    return info


def _pick_audio(audios: list[dict]) -> int | None:
    """Premier flux audio décodable (ignore les pistes de métadonnées / codecs inconnus)."""
    for a in audios:
        if a["codec"] in AUDIO_OK and a["channels"] > 0:
            return a["index"]
    for a in audios:  # codec inattendu mais déclaré audio : on tente quand même
        if a["codec"] and a["codec"] != "none" and a["channels"] > 0:
            return a["index"]
    return None


# ---------------------------------------------------------------------------
# Proxies vidéo
# ---------------------------------------------------------------------------
def _sidecar(clip: str) -> Path:
    return config.PROXY_DIR / f"{clip}.proxy.json"


def _signature(src: Path) -> dict:
    st = src.stat()
    return {"source": str(src), "size": st.st_size, "mtime": int(st.st_mtime), "chain": CHAIN_VERSION,
            "w": config.W, "h": config.H, "fps": config.FPS}


def _shift(start: float) -> str:
    """Expression setpts ramenant le début commun A/V à 0 (gère un début négatif, ex. amorce AAC)."""
    return f"PTS-{start:.6f}/TB" if start >= 0 else f"PTS+{-start:.6f}/TB"


def _video_filters(info: dict, start: float, n_frames: int) -> tuple[str, list[str]]:
    """Chaîne de filtres vidéo (après l'autorotation automatique de ffmpeg) + notes."""
    v = info["video"]
    W, H, FPS = config.W, config.H, config.FPS
    notes = []
    f = [f"setpts={_shift(start)}", f"fps={FPS}:start_time=0"]
    if v["hdr"]:
        tin = v["color_transfer"] if v["color_transfer"] in HDR_TRANSFERS else "arib-std-b67"
        f += [f"zscale=tin={tin}:min=bt2020nc:pin=bt2020:rin=tv:t=linear:npl=100",
              "format=gbrpf32le", "zscale=p=bt709", "tonemap=tonemap=hable:desat=0",
              "zscale=t=bt709:m=bt709:r=tv"]
        notes.append(f"HDR ({tin}{', Dolby Vision' if v['dolby_vision'] else ''}) → tonemap hable BT.709")
    dw, dh = v["display_w"], v["display_h"]
    if (dw, dh) == (W, H):
        notes.append("source déjà en 1080x1920 : pas de mise à l'échelle ni de recadrage")
    else:
        f += [f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos+accurate_rnd+full_chroma_int",
              f"crop={W}:{H}"]
        if dw * H != dh * W:
            notes.append(f"ratio {dw}x{dh} ≠ 9:16 : recadrage centré")
        if dw > dh:
            notes.append("ATTENTION : source paysage (rotation absente ?) → recadrage centré important")
    f += ["setsar=1", "tpad=stop_mode=clone:stop_duration=2", f"trim=end_frame={n_frames}",
          "format=yuv420p"]
    return ",".join(f), notes


def make_proxy(clip: str, force: bool = False, threads: int | None = None) -> Path:
    """Crée (si besoin) le proxy 1080x1920 30 i/s CFR du clip ; renvoie son chemin."""
    config.PROXY_DIR.mkdir(parents=True, exist_ok=True)
    src = config.rush_path(clip)
    out = config.PROXY_DIR / f"{clip}.mp4"
    sig = _signature(src)
    if out.exists() and not force and _sidecar(clip).exists():
        try:
            old = json.loads(_sidecar(clip).read_text(encoding="utf-8"))
            if all(old.get(k) == sig[k] for k in sig):
                return out
        except (OSError, ValueError):
            pass
    info = probe(src)
    if not info["video"]:
        raise RuntimeError(f"{clip}: aucun flux vidéo dans {src}")
    v = info["video"]
    # Début commun (A/V synchrones) et durée = fin de la vidéo sur la timeline source
    start = min([v["start_time"]] + [a["start_time"] for a in info["audio"]] + [info["start_time"]])
    vdur = v["duration"] or info["duration"]
    if not vdur and v["nb_frames"] and v["fps_avg"]:
        vdur = v["nb_frames"] / v["fps_avg"]
    if not vdur:
        raise RuntimeError(f"{clip}: durée vidéo inconnue")
    end = v["start_time"] + vdur - start
    n_frames = max(1, int(round(end * config.FPS)))
    n_samples = int(round(n_frames / config.FPS * config.SR))
    vf, notes = _video_filters(info, start, n_frames)

    tmp = out.with_name(f"{clip}.part.mp4")
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", src]
    ai = info["audio_index"]
    if ai is None:
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r={config.SR}:cl=stereo"]
        af = f"apad=whole_len={n_samples},atrim=end_sample={n_samples}"
        amap = "1:a:0"
        notes.append("pas de piste audio : silence ajouté")
    else:
        af = (f"asetpts={_shift(start)},aresample={config.SR}:async=1:first_pts=0,"
              f"apad=whole_len={n_samples},atrim=end_sample={n_samples}")
        amap = f"0:{ai}"
        if len(info["audio"]) > 1:
            notes.append(f"{len(info['audio'])} pistes audio : flux #{ai} retenu")
    if threads is None:
        threads = max(1, (os.cpu_count() or 2) // PROXY_JOBS)
    cmd += ["-map", f"0:{v['index']}", "-map", amap, "-vf", vf, "-af", af,
            "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-g", str(config.FPS), "-threads", str(threads),
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-r", str(config.FPS), "-fps_mode", "cfr",
            "-c:a", "aac", "-b:a", "256k", "-ar", str(config.SR), "-ac", "2",
            "-map_metadata", "-1", "-map_chapters", "-1", "-movflags", "+faststart", tmp]
    t0 = time.time()
    r = _run(cmd)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{clip}: ffmpeg a échoué\n{r.stderr.strip()[-2000:]}")
    os.replace(tmp, out)
    check = verify_proxy(out, n_frames)
    sig.update({"frames": n_frames, "duration": n_frames / config.FPS, "notes": notes,
                "encode_s": round(time.time() - t0, 1), "check": check})
    _sidecar(clip).write_text(json.dumps(sig, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[ingest] {clip}: {n_frames} images ({n_frames / config.FPS:.2f} s) en {time.time() - t0:.1f} s"
        + (f" — {'; '.join(notes)}" if notes else "") + ("" if check["ok"] else f" — PROBLÈME {check['problems']}"))
    return out


def verify_proxy(path: Path, n_expected: int | None = None) -> dict:
    """Contrôle du proxy : 1080x1920, 30 i/s CFR, nombre d'images, sans rotation, audio 48 kHz stéréo, durées A/V."""
    r = _run([FFPROBE, "-v", "error", "-count_packets", "-show_streams", "-show_format", "-of", "json", path],
             timeout=120)
    data = json.loads(r.stdout or "{}")
    probs = []
    vs = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    as_ = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    out = {"ok": False}
    if not vs:
        return {"ok": False, "problems": ["pas de vidéo"]}
    v = vs[0]
    n = int(v.get("nb_read_packets") or v.get("nb_frames") or 0)
    rot = _rotation(v)
    vdur = _float(v.get("duration"))
    out.update({"w": v.get("width"), "h": v.get("height"), "fps": v.get("r_frame_rate"),
                "avg_fps": v.get("avg_frame_rate"), "frames": n, "rotation": rot, "pix_fmt": v.get("pix_fmt"),
                "v_duration": vdur})
    if (v.get("width"), v.get("height")) != (config.W, config.H):
        probs.append(f"taille {v.get('width')}x{v.get('height')}")
    if _ratio(v.get("avg_frame_rate")) is None or abs(_ratio(v.get("avg_frame_rate")) - config.FPS) > 1e-3:
        probs.append(f"cadence {v.get('avg_frame_rate')}")
    if rot:
        probs.append(f"rotation résiduelle {rot}")
    if v.get("pix_fmt") != "yuv420p":
        probs.append(f"pix_fmt {v.get('pix_fmt')}")
    if n_expected is not None and n != n_expected:
        probs.append(f"{n} images au lieu de {n_expected}")
    if as_:
        a = as_[0]
        adur = _float(a.get("duration"))
        out.update({"a_rate": a.get("sample_rate"), "a_channels": a.get("channels"), "a_duration": adur})
        if int(a.get("sample_rate") or 0) != config.SR or int(a.get("channels") or 0) != 2:
            probs.append("audio pas en 48 kHz stéréo")
        if vdur and adur and abs(vdur - adur) > 1.5 / config.FPS:
            probs.append(f"durées A/V {vdur:.3f}/{adur:.3f}")
    else:
        probs.append("pas d'audio")
    out["ok"] = not probs
    out["problems"] = probs
    return out


# ---------------------------------------------------------------------------
# Photos
# ---------------------------------------------------------------------------
def _open_image(path: Path):
    from PIL import Image
    if path.suffix.lower() in (".heic", ".heif"):
        try:
            from pillow_heif import register_heif_opener
        except ImportError as e:  # pragma: no cover - dépend de la machine
            raise RuntimeError("Photo HEIC : installer pillow-heif (pip install pillow-heif)") from e
        register_heif_opener()
    return Image.open(path)


def _to_srgb(im):
    """Convertit vers sRGB si un profil ICC est embarqué (iPhone = Display P3)."""
    from PIL import ImageCms
    icc = im.info.get("icc_profile")
    if not icc:
        return im.convert("RGB"), "sans profil (supposé sRGB)"
    try:
        import io
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        name = ImageCms.getProfileDescription(src).strip()
        if "srgb" in name.lower():
            return im.convert("RGB"), name
        dst = ImageCms.createProfile("sRGB")
        rgb = im.convert("RGB")
        out = ImageCms.profileToProfile(rgb, src, dst, renderingIntent=0, outputMode="RGB")
        return out, f"{name} → sRGB"
    except Exception as e:  # profil illisible : on garde les valeurs telles quelles
        return im.convert("RGB"), f"profil ICC ignoré ({e.__class__.__name__})"


def make_photo_proxy(photo: str, force: bool = False) -> Path:
    """PNG orienté selon l'EXIF, hauteur ≤ 2560 px, sRGB. Renvoie le chemin du PNG."""
    from PIL import Image, ImageOps
    config.PROXY_DIR.mkdir(parents=True, exist_ok=True)
    src = config.rush_path(photo)
    out = config.PROXY_DIR / f"{photo}.png"
    sig = _signature(src)
    side = config.PROXY_DIR / f"{photo}.proxy.json"
    if out.exists() and side.exists() and not force:
        try:
            old = json.loads(side.read_text(encoding="utf-8"))
            if all(old.get(k) == sig[k] for k in sig):
                return out
        except (OSError, ValueError):
            pass
    im = _open_image(src)
    exif = im.getexif()
    orient = exif.get(0x0112, 1)
    dt = exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132)
    raw_size = im.size
    im = ImageOps.exif_transpose(im)
    im, icc_note = _to_srgb(im)
    if im.height > PHOTO_MAX_H:
        w = int(round(im.width * PHOTO_MAX_H / im.height))
        im = im.resize((w, PHOTO_MAX_H), Image.LANCZOS)
    tmp = out.with_name(f"{photo}.part.png")
    im.save(tmp, format="PNG", compress_level=6)
    os.replace(tmp, out)
    sig.update({"exif_orientation": orient, "datetime": dt, "raw_size": list(raw_size),
                "size": list(im.size), "color": icc_note})
    side.write_text(json.dumps(sig, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[ingest] {photo}: {raw_size[0]}x{raw_size[1]} (EXIF {orient}) → {im.width}x{im.height} PNG ({icc_note})")
    return out


# ---------------------------------------------------------------------------
# Inventaire
# ---------------------------------------------------------------------------
def _find_rush(name: str) -> Path | None:
    try:
        return config.rush_path(name)
    except FileNotFoundError:
        return None


def run_ingest(force: bool = False, jobs: int = PROXY_JOBS, only: list[str] | None = None) -> dict:
    """Proxies de tous les rushs + inventaire `analysis/inventory.json`. Renvoie l'inventaire."""
    config.ensure_dirs()
    if shutil.which(FFMPEG) is None or shutil.which(FFPROBE) is None:
        raise RuntimeError("ffmpeg/ffprobe introuvables (Windows : winget install Gyan.FFmpeg)")
    clips = [c for c in config.CLIPS if not only or c in only]
    photos = [p for p in config.PHOTOS if not only or p in only]
    inv_path = config.ANALYSIS_DIR / "inventory.json"
    inv = {"rush_dir": str(config.RUSH_DIR), "proxy_dir": str(config.PROXY_DIR), "chain": CHAIN_VERSION,
           "clips": {}, "photos": {}, "missing": [], "unexpected": [], "problems": []}
    if inv_path.exists() and only:
        try:  # mise à jour partielle : on garde les autres entrées
            old = json.loads(inv_path.read_text(encoding="utf-8"))
            inv["clips"].update(old.get("clips", {}))
            inv["photos"].update(old.get("photos", {}))
        except (OSError, ValueError):
            pass
    t0 = time.time()
    todo = []
    for c in clips:
        src = _find_rush(c)
        if src is None:
            inv["missing"].append(c)
            continue
        todo.append(c)
    threads = max(1, (os.cpu_count() or 2) // max(1, jobs))
    results: dict[str, Exception | Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = {ex.submit(make_proxy, c, force, threads): c for c in todo}
        for fu in as_completed(futs):
            c = futs[fu]
            try:
                results[c] = fu.result()
            except Exception as e:  # on continue avec les autres clips
                results[c] = e
                log(f"[ingest] ÉCHEC {c}: {e}")
    for c in todo:
        src = config.rush_path(c)
        entry = {"source": str(src), "probe": probe(src)}
        res = results.get(c)
        if isinstance(res, Exception):
            entry["error"] = str(res)
            inv["problems"].append(f"{c}: {res}")
        else:
            side = json.loads(_sidecar(c).read_text(encoding="utf-8"))
            entry.update({"proxy": str(res), "frames": side.get("frames"), "duration": side.get("duration"),
                          "notes": side.get("notes", []), "check": side.get("check")})
            if not (side.get("check") or {}).get("ok", False):
                inv["problems"].append(f"{c}: {(side.get('check') or {}).get('problems')}")
        inv["clips"][c] = entry
    for p in photos:
        src = _find_rush(p)
        if src is None:
            inv["missing"].append(p)
            continue
        try:
            out = make_photo_proxy(p, force)
            side = json.loads((config.PROXY_DIR / f"{p}.proxy.json").read_text(encoding="utf-8"))
            inv["photos"][p] = {"source": str(src), "proxy": str(out), **{k: side.get(k) for k in (
                "exif_orientation", "datetime", "raw_size", "size", "color")}}
        except Exception as e:
            inv["photos"][p] = {"source": str(src), "error": str(e)}
            inv["problems"].append(f"{p}: {e}")
            log(f"[ingest] ÉCHEC {p}: {e}")
    known = set(config.CLIPS) | set(config.PHOTOS)
    if config.RUSH_DIR.exists():
        for f in sorted(config.RUSH_DIR.iterdir()):
            if f.is_file() and f.suffix.lower() in (".mov", ".mp4", ".jpeg", ".jpg", ".heic", ".png") \
                    and f.stem not in known:
                inv["unexpected"].append(f.name)
    inv["total_duration"] = round(sum((e.get("duration") or 0) for e in inv["clips"].values()), 2)
    inv["elapsed_s"] = round(time.time() - t0, 1)
    tmp = inv_path.with_suffix(".part.json")
    tmp.write_text(json.dumps(inv, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, inv_path)
    log(f"[ingest] {len(inv['clips'])} clips, {len(inv['photos'])} photos, {inv['total_duration']} s de rushs, "
        f"{len(inv['missing'])} manquants, {len(inv['problems'])} problèmes → {inv_path}")
    return inv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ingestion des rushs (proxies + inventaire)")
    ap.add_argument("--force", action="store_true", help="reconstruire les proxies existants")
    ap.add_argument("--jobs", type=int, default=PROXY_JOBS)
    ap.add_argument("--only", nargs="*", help="limiter à ces rushs (IMG_xxxx)")
    a = ap.parse_args(argv)
    inv = run_ingest(force=a.force, jobs=a.jobs, only=a.only)
    return 1 if inv["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
