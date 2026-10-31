"""Pilote du titre 3D « C'EST QUI LUI ? » (Blender headless + recomposition Python).

Produit :
  assets/title/title_seq.mp4    1080x1920, 30 i/s, 150 images (5,0 s), H.264 CRF 12, yuv420p, plaque propre (sans grain)
  assets/title/title_meta.json  {"dur", "reveal", "text_box", "fps", ...}
  assets/title/cover_plate.png  image fixe 1080x1920 pour la couverture (titre en haut, contre-jour au centre/bas)
  assets/title/BodoniModa-Black-opsz28.ttf  instance statique (Blender ne gère pas les axes variables)

Passes Blender (tools/blender_title.py, Cycles CPU), en EXR linéaire dans WORK_DIR/title_frames/exr_<passe>/ :
  bg   : surfaces sans volume, 34 % (le fond est flou de toute façon : profondeur de champ + brouillard)
  text : surfaces sans volume, 1080x1920 mais uniquement dans un cadre autour du titre (bords nets)
  fog  : brouillard seul, 16 % : transmittance T + diffusion S par groupe de lumière (+ brouillard reflété)
Post (ce script) : débruitage OIDN (albédo + normales), suréchantillonnage bilatéral joint guidé par la
profondeur, recomposition image = Σ surf·T + S + brume, halo, traînée anamorphique, courbe filmique,
étalonnage froid (rouges préservés), PNG 8 bits tramé dans WORK_DIR/title_frames/png/.
WORK_DIR = config.WORK_DIR (défaut /home/user/edit_escape_tmp), surchargeable par --work.

Usage (chaque commande est reprenable : ce qui existe déjà est sauté ; --time-limit borne la durée d'un appel,
code de sortie 3 = interrompu, relancer la même commande) :
  python tools/render_title.py font                         # instance statique de la police
  python tools/render_title.py render [--frames 1-150] [--time-limit 300] [--pass fog|bg|text]
  python tools/render_title.py post   [--frames 1-150] [--time-limit 300]
  python tools/render_title.py encode | meta | cover | contact
  python tools/render_title.py all                          # tout, d'un bloc (long : ~1 h sur 3 threads chargés)
Dépendances (uniquement pour re-rendre le titre, déjà fourni dans assets/title) : Blender >= 4.0, fonttools,
OpenEXR, pyoidn, opencv-python, numpy.
Essais rapides : render --scale 13 --samples 16 --work DIR, puis post --lookdev --work DIR, contact --work DIR.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import config  # noqa: E402

SCRIPT = REPO / "tools" / "blender_title.py"
TITLE_DIR = config.TITLE_DIR
FONT_SRC = config.FONTS_DIR / "BodoniModaopsz_wght.ttf"
FONT_STATIC = TITLE_DIR / "BodoniModa-Black-opsz28.ttf"
FONT_AXES = {"wght": 900, "opsz": 28}
FPS, N_FRAMES, W, H = 30, 150, 1080, 1920
SUB_T0 = 2.6     # apparition de « INSPIRÉ DE FAITS RÉELS — 30.10.2026 » (edit/edl_spec.py, S7a)

# Réglages de rendu (mesurés : ~25 s/frame au total, 3 threads, CPU partagé avec d'autres rendus)
PASSES = {
    "fog": dict(scale=16, samples=16, extra=["--step-rate", "2.0"]),
    "bg": dict(scale=34, samples=7, extra=[]),
    "text": dict(scale=100, samples=6, extra=[]),
}
OIDN_THREADS = 2
# Fin du plan (voiture arrêtée, seule la caméra avance lentement) : une image sur deux est rendue, les autres sont
# synthétisées en post (moyenne des deux voisines recalées sur la boîte du titre : zoom + translation < 1 px).
HALF_RATE_FROM = 106


CHUNK = 15  # frames par appel Blender (chaque appel reste court : ~2-4 min)


def is_interpolated(f: int) -> bool:
    return HALF_RATE_FROM <= f < N_FRAMES and (f - HALF_RATE_FROM) % 2 == 1


# Étalonnage (linéaire scène -> affichage)
GRADE = dict(
    exposure=2.2,           # multiplicateur linéaire avant la courbe
    k_head=1.0, k_red=1.0, k_amb=2.0,          # surfaces par groupe de lumière
    k_fog_head=0.5, k_fog_red=1.15, k_fog_amb=1.1,   # diffusion du brouillard par groupe
    amb_ramp=(0.40, 0.30, 1.70),               # ambiance : facteur initial, début et fin de la montée (s)
    amb_ramp2=(4.0, 1.60, 2.50),               # puis lumière froide de face x4 quand les phares sont passés
    haze=(0.0030, 0.0042, 0.0066),             # brume ambiante (clair de lune diffus) * (1 - T)
    haze_dyn=0.03,          # brume qui s'éclaire avec les phares (diffusion multiple approximée)
    bloom=0.10, bloom_thr=0.9,
    streak=0.02, streak_thr=2.5,               # traînée anamorphique bleutée des sources très vives (phares)
    black=0.0025,           # point noir (noirs écrasés, mais texture du brouillard conservée)
    shadow_tint=(0.90, 0.98, 1.10),
    sat=0.80,
)


# Lisibilité du titre après l'éblouissement des phares (relecture phase 2) : de 1,8 s à la fin, le métal gris se
# confondait avec le brouillard gris (lettres plus sombres que le fond). Recomposition sans Blender :
#   * gain des surfaces des lettres (toutes lumières) ;
#   * liseré froid synthétique sur les biseaux, calculé depuis les normales de la passe « text » (contre-jour
#     haut-arrière = lampadaire, + clé latérale gauche = lune), atténué par la transmittance du brouillard ;
#   * brouillard légèrement assombri derrière les lettres (ombre portée diffuse), sources très vives épargnées.
# Montée progressive entre t_on[0] et t_on[1] (s) : avant, image strictement inchangée (éblouissement).
TEXT_LOOK = dict(
    t_on=(1.45, 2.15),
    gain=2.5,                       # surfaces des lettres
    rim=0.075, rim_pow=1.6,         # liseré (linéaire scène, avant exposition) et dureté
    rim_dirs=((0.25, 0.92, 0.30, 1.0), (-0.85, 0.35, 0.15, 0.45)),   # (x, y, z, poids) ; +y haut, +z arrière
    rim_color=(0.82, 0.92, 1.0),
    spec=0.06, spec_pow=18.0,       # reflet franc du métal (lumière froide de face-haut)
    spec_dir=(0.15, 0.55, -0.82),
    backdrop=0.55, backdrop_sigma=44.0, backdrop_grow=9,
)


def text_look_amount(t: float | None) -> float:
    if t is None:
        return 0.0
    a, b = TEXT_LOOK["t_on"]
    u = min(1.0, max(0.0, (t - a) / (b - a)))
    return u * u * (3 - 2 * u)


def text_rim(normal: np.ndarray, T: np.ndarray, k: float) -> np.ndarray:
    """Liseré + reflet synthétiques (linéaire, (h,w,3)) depuis les normales de la passe text."""
    tl = TEXT_LOOK
    n = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    shade = np.zeros(n.shape[:2], np.float32)
    for x, y, z, w in tl["rim_dirs"]:
        L = np.array([x, y, z], np.float32)
        L /= np.linalg.norm(L)
        shade += w * np.clip(n @ L, 0, 1) ** tl["rim_pow"]
    S = np.array(tl["spec_dir"], np.float32)
    S /= np.linalg.norm(S)
    spec = np.clip(n @ S, 0, 1) ** tl["spec_pow"]
    col = np.array(tl["rim_color"], np.float32)
    return (k * (tl["rim"] * shade + tl["spec"] * spec))[..., None] * col * T


# Couverture : brouillard plus sombre, titre plus éclairé (doit rester lisible en vignette)
GRADE_COVER = dict(GRADE, exposure=1.8, k_fog_head=0.45, k_head=1.4, k_amb=1.0, k_fog_amb=1.0, haze_dyn=0.05)
COVER_PASSES = (("fog", "cover_fog", 33, 32), ("bg", "cover_surf", 50, 16), ("text", "cover_text", 100, 24))


def work_dir(args) -> Path:
    base = Path(args.work) if getattr(args, "work", None) else config.WORK_DIR
    d = base / "title_frames"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(msg: str) -> None:
    print(f"[render_title] {msg}", flush=True)


# ----------------------------------------------------------------------------- outils externes
def find_blender() -> str:
    cand = [os.environ.get("BLENDER"), shutil.which("blender")]
    if config.IS_WINDOWS:
        pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation"
        if pf.exists():
            cand += [str(p) for p in sorted(pf.glob("Blender*/blender.exe"), reverse=True)]
    for c in cand:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("Blender introuvable (définir la variable BLENDER)")


def find_ffmpeg() -> str:
    c = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not c:
        raise FileNotFoundError("ffmpeg introuvable")
    return c


def parse_frames(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def frames_spec(frames: list[int]) -> str:
    return ",".join(str(f) for f in frames)


# ----------------------------------------------------------------------------- police
def make_font(force: bool = False) -> Path:
    """Instance statique Bodoni Moda Black (wght 900, opsz 28) : capitales à fort contraste, empattements nets."""
    if FONT_STATIC.exists() and not force:
        return FONT_STATIC
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer
    TITLE_DIR.mkdir(parents=True, exist_ok=True)
    vf = TTFont(str(FONT_SRC))
    try:
        st = instancer.instantiateVariableFont(vf, FONT_AXES, updateFontNames=True)
    except Exception:  # noqa: BLE001 - table STAT absente : noms conservés
        st = instancer.instantiateVariableFont(TTFont(str(FONT_SRC)), FONT_AXES)
    st.save(str(FONT_STATIC))
    log(f"police statique -> {FONT_STATIC.name}")
    return FONT_STATIC


# ----------------------------------------------------------------------------- Blender
def run_blender(pass_: str, frames: list[int], out_dir: Path, mode: str = "seq", samples: int | None = None,
                scale: int | None = None, extra: list[str] | None = None, threads: int = 0) -> None:
    cfg = PASSES[pass_]
    cmd = [find_blender(), "-b", "--factory-startup", "-noaudio", "-P", str(SCRIPT), "--",
           "--font", str(make_font()), "--pass", "fog" if pass_ == "fog" else ("text" if pass_ == "text" else "surf"),
           "--mode", mode, "--frames", frames_spec(frames), "--out", str(out_dir), "--skip-existing",
           "--scale", str(scale or cfg["scale"]), "--samples", str(samples or cfg["samples"])]
    cmd += cfg["extra"] + (extra or [])
    if threads:
        cmd += ["--threads", str(threads)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace")
    for line in proc.stdout.splitlines():
        if line.startswith("[title]") or "Error" in line or "Traceback" in line:
            print("   ", line, flush=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-25:])
        raise RuntimeError(f"Blender a échoué ({pass_}, frames {frames[0]}..{frames[-1]}) :\n{tail}")


def render_passes(args, passes: list[str], frames: list[int], t_start: float) -> bool:
    """Rend les frames manquantes par paquets de CHUNK, toutes les passes d'un paquet à la suite (le post peut
    suivre au fil de l'eau). Renvoie False si la limite de temps (--time-limit) a interrompu : relancer."""
    wd = work_dir(args)
    todo = [f for f in frames if not is_interpolated(f)
            and any(not (wd / f"exr_{p}" / f"f_{f:04d}.exr").exists() for p in passes)]
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        for p in passes:
            if args.time_limit and time.time() - t_start > args.time_limit:
                log(f"limite de temps atteinte : reste {len(todo) - i} frames (relancer la même commande)")
                return False
            missing = [f for f in chunk if not (wd / f"exr_{p}" / f"f_{f:04d}.exr").exists()]
            if missing:
                run_blender(p, missing, wd / f"exr_{p}", threads=args.threads, scale=args.scale,
                            samples=args.samples)
    return True


# ----------------------------------------------------------------------------- EXR / débruitage
def read_exr(path: Path) -> dict[str, np.ndarray]:
    """EXR multicouche Blender -> {couche: (H, W, C) float32}."""
    import OpenEXR
    f = OpenEXR.File(str(path), separate_channels=True)
    layers: dict[str, dict[str, np.ndarray]] = {}
    for name, ch in f.parts[0].channels.items():
        parts = name.split(".")
        layer, comp = ".".join(parts[1:-1]) or parts[0], parts[-1]
        layers.setdefault(layer, {})[comp] = ch.pixels
    out = {}
    for layer, comps in layers.items():
        order = [k for k in ("R", "G", "B", "A", "X", "Y", "Z") if k in comps]
        out[layer] = np.stack([comps[k].astype(np.float32) for k in order], -1)
    return out


_OIDN = None


def oidn(color: np.ndarray, albedo: np.ndarray | None = None, normal: np.ndarray | None = None,
         hdr: bool = True) -> np.ndarray:
    """Débruitage Intel Open Image Denoise (pyoidn). Repli : flou bilatéral léger si indisponible."""
    global _OIDN
    color = np.ascontiguousarray(color[..., :3], dtype=np.float32)
    try:
        import pyoidn
    except ImportError:
        import cv2
        return cv2.bilateralFilter(color, 5, 0.05, 2.0)
    if _OIDN is None:
        _OIDN = pyoidn.Device(pyoidn.OIDN_DEVICE_TYPE_CPU)
        _OIDN.set_int("numThreads", OIDN_THREADS)      # CPU partagé : ne pas étouffer le rendu Blender
        _OIDN.commit()
    out = np.zeros_like(color)
    flt = pyoidn.Filter(_OIDN, pyoidn.OIDN_FILTER_TYPE_RT)
    flt.set_image(pyoidn.OIDN_IMAGE_COLOR, color, pyoidn.OIDN_FORMAT_FLOAT3)
    if albedo is not None:
        alb = np.ascontiguousarray(np.clip(albedo[..., :3], 0, 1), dtype=np.float32)
        flt.set_image(pyoidn.OIDN_IMAGE_ALBEDO, alb, pyoidn.OIDN_FORMAT_FLOAT3)
        if normal is not None:
            nrm = np.ascontiguousarray(normal[..., :3], dtype=np.float32)
            flt.set_image(pyoidn.OIDN_IMAGE_NORMAL, nrm, pyoidn.OIDN_FORMAT_FLOAT3)
    flt.set_image(pyoidn.OIDN_IMAGE_OUTPUT, out, pyoidn.OIDN_FORMAT_FLOAT3)
    flt.set_bool("hdr", hdr)
    flt.set_quality(pyoidn.OIDN_QUALITY_HIGH)
    flt.commit()
    flt.execute()
    err = _OIDN.get_error()
    flt.release()
    if err:
        raise RuntimeError(f"OIDN : {err}")
    return np.maximum(out, 0.0)


def denoise_groups(ex: dict, crop=None) -> dict[str, np.ndarray]:
    """Débruite chaque groupe de lumière avec albédo/normales ; crop=(y0,y1,x0,x1) limite le calcul."""
    sl = (slice(None), slice(None)) if crop is None else (slice(crop[0], crop[1]), slice(crop[2], crop[3]))
    alb = ex["Denoising Albedo"][sl]
    nrm = ex["Denoising Normal"][sl]
    out = {}
    for g in ("head", "red", "amb"):
        key = f"Combined_{g}"
        img = np.zeros(ex[key].shape[:2] + (3,), np.float32)
        img[sl] = oidn(ex[key][sl], alb, nrm)
        out[g] = img
    return out


# ----------------------------------------------------------------------------- suréchantillonnage guidé
def jbu(lo: np.ndarray, lo_depth: np.ndarray, hi_depth: np.ndarray, sigma: float = 0.25,
        radius: int = 3, offset=(0.0, 0.0), scale=None) -> np.ndarray:
    """Suréchantillonnage bilatéral joint : les valeurs basse définition (h,w,C) sont interpolées vers la
    grille de hi_depth (H',W') en ne mélangeant que des échantillons de profondeur voisine.
    offset/scale : position (y, x) du coin de la grille fine et rapport lo/hi (pour une fenêtre recadrée)."""
    h, w = lo_depth.shape
    Hh, Wh = hi_depth.shape
    sy, sx = scale if scale is not None else (h / Hh, w / Wh)
    yy = (np.arange(Hh) + offset[0] + 0.5) * sy - 0.5
    xx = (np.arange(Wh) + offset[1] + 0.5) * sx - 0.5
    y0, x0 = np.floor(yy).astype(np.int64), np.floor(xx).astype(np.int64)
    ldh = np.log(np.maximum(hi_depth, 1e-3)).astype(np.float32)
    ldl = np.log(np.maximum(lo_depth, 1e-3)).astype(np.float32)
    acc = np.zeros((Hh, Wh, lo.shape[2]), np.float32)
    wsum = np.zeros((Hh, Wh), np.float32)
    for dy in range(-radius + 1, radius + 1):
        yi = np.clip(y0 + dy, 0, h - 1)
        wy = np.maximum(0.0, 1.0 - np.abs(yy - (y0 + dy)) / radius).astype(np.float32)
        for dx in range(-radius + 1, radius + 1):
            xi = np.clip(x0 + dx, 0, w - 1)
            wx = np.maximum(0.0, 1.0 - np.abs(xx - (x0 + dx)) / radius).astype(np.float32)
            ws = wy[:, None] * wx[None, :]
            d = ldh - ldl[yi[:, None], xi[None, :]]
            wgt = ws * (np.exp(-(d / sigma) ** 2) + 1e-4)
            acc += wgt[..., None] * lo[yi[:, None], xi[None, :]]
            wsum += wgt
    return acc / np.maximum(wsum, 1e-8)[..., None]


# ----------------------------------------------------------------------------- étalonnage
def bloom(img: np.ndarray, thr: float, amount: float) -> np.ndarray:
    import cv2
    hi = np.maximum(img - thr, 0.0)
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    for div, wgt in ((4, 0.5), (8, 0.3), (16, 0.2)):
        small = cv2.resize(hi, (w // div, h // div), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 2.5)
        out += wgt * cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return img + amount * out


def filmic(x: np.ndarray) -> np.ndarray:
    """Courbe ACES (approximation Narkowicz) appliquée à la luminance, chroma préservée puis désaturée
    dans les hautes lumières (pas de virage orange des rouges)."""
    lum = np.maximum(0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2], 1e-6)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    lt = np.clip((lum * (a * lum + b)) / (lum * (c * lum + d) + e), 0, 1)
    y = x * (lt / lum)[..., None]
    over = np.clip((y.max(-1) - 1.0) / 1.0, 0, 1)[..., None]       # hors gamut -> vers le blanc
    y = y * (1 - over) + lt[..., None] * over
    return np.clip(y, 0, 1)


def streak(img: np.ndarray, thr: float, amount: float) -> np.ndarray:
    """Traînée horizontale (optique anamorphique) autour des sources très vives : couleur de la source,
    légèrement décalée vers le bleu acier."""
    import cv2
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    peak = img.max(-1, keepdims=True)
    hi = np.ascontiguousarray(img * (np.maximum(peak - thr, 0.0) / np.maximum(peak, 1e-6)), dtype=np.float32)
    small = cv2.resize(hi, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(small)
    for sig, wgt in ((12, 0.5), (45, 0.35), (110, 0.15)):
        acc += wgt * cv2.GaussianBlur(small, (0, 0), sigmaX=sig, sigmaY=0.6)
    out = cv2.resize(acc, (w, h), interpolation=cv2.INTER_LINEAR)
    lum = out.mean(-1, keepdims=True)
    out = 0.6 * out + 0.4 * lum * np.array([0.55, 0.78, 1.0], np.float32)
    return img + amount * out


def grade(lin: np.ndarray, g: dict = GRADE) -> np.ndarray:
    """Linéaire scène -> sRGB affichage [0,1] : exposition, halo, courbe, noirs, teinte froide, saturation."""
    x = bloom(lin * g["exposure"], g["bloom_thr"], g["bloom"])
    x = streak(x, g["streak_thr"], g["streak"])
    x = np.maximum(x - g["black"], 0.0)
    y = filmic(x)
    lum = (0.2126 * y[..., 0] + 0.7152 * y[..., 1] + 0.0722 * y[..., 2])[..., None]
    sh = np.clip(1.0 - lum * 3.0, 0, 1)                    # teinte froide dans les ombres
    y = y * (1 - sh * (1 - np.array(g["shadow_tint"], np.float32)))
    red = np.clip((y[..., :1] - np.maximum(y[..., 1:2], y[..., 2:3])) * 4, 0, 1)  # protège les rouges
    s = g["sat"] + (1 - g["sat"]) * red
    y = lum + (y - lum) * s
    y = np.clip(y, 0, 1)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)


def to_u8(img: np.ndarray, seed: int) -> np.ndarray:
    """Quantification 8 bits avec tramage triangulaire (pas de bandes dans les dégradés du brouillard)."""
    rng = np.random.default_rng(seed)
    n = rng.random(img.shape, dtype=np.float32) - rng.random(img.shape, dtype=np.float32)
    return np.clip(np.round(img * 255.0 + n), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- recomposition
def neutral_chroma(img: np.ndarray) -> np.ndarray:
    """Le brouillard est quasi neutre et chaque groupe a une couleur de source fixe : on remplace la chrominance
    par pixel (bruitée, artefacts colorés du débruitage en basse définition) par la chrominance moyenne du groupe."""
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    tot = float(lum.sum())
    if tot <= 0:
        return img
    mean = img.reshape(-1, 3).sum(0) / tot                                  # couleur moyenne / luminance
    return lum[..., None] * mean[None, None, :].astype(np.float32)


def fog_terms(fx: dict) -> dict[str, np.ndarray]:
    t = oidn(np.repeat(fx["Emit"][..., :1], 3, -1), hdr=False)[..., :1]
    return dict(T=np.clip(t, 0, 1), S_head=neutral_chroma(oidn(fx["Combined_head"])),
                S_red=neutral_chroma(oidn(fx["Combined_red"])), S_amb=neutral_chroma(oidn(fx["Combined_amb"])),
                depth=fx["Depth"][..., 0])


def gains(g: dict, t: float | None) -> dict:
    """Gains dépendant du temps : l'ambiance froide monte un peu après le passage des phares (l'œil s'habitue)."""
    g = dict(g)
    if t is not None:
        a0, t0, t1 = g["amb_ramp"]
        u = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
        k = a0 + (1 - a0) * u * u * (3 - 2 * u)
        g["k_amb"] *= k
        g["k_fog_amb"] *= k
        # le titre « émerge » dans la lumière froide (lampadaire + lune) quand les phares ne l'éclairent plus,
        # juste avant l'apparition de la ligne « INSPIRÉ DE FAITS RÉELS » (2,6 s)
        a2, t2, t3 = g["amb_ramp2"]
        u = min(1.0, max(0.0, (t - t2) / (t3 - t2)))
        g["k_amb"] *= 1 + (a2 - 1) * u * u * (3 - 2 * u)
    return g


def despeckle(img: np.ndarray, ratio: float = 4.0, floor: float = 0.05) -> np.ndarray:
    """Supprime les lucioles résiduelles (reflets spéculaires isolés sur l'asphalte à faible échantillonnage) :
    un pixel plus de `ratio` fois plus lumineux que la médiane 5x5 de son voisinage est ramené à ce plafond."""
    import cv2
    lum = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)
    med = cv2.medianBlur(np.ascontiguousarray(np.clip(lum, 0, 1e4)), 5)
    cap = np.maximum(med * ratio, floor)
    k = np.where(lum > cap, cap / np.maximum(lum, 1e-6), 1.0)[..., None]
    return img * k


def assemble(surf: dict[str, np.ndarray], fog_up: np.ndarray, g: dict, haze_dyn: np.ndarray,
             surf_gain=1.0) -> np.ndarray:
    """fog_up : (H,W,10) = [T, S_head(3), S_red(3), S_amb(3)] déjà suréchantillonné sur la grille des surfaces.
    surf_gain : gain des surfaces seules (scalaire ou (H,W,1)) — le brouillard devant elles n'est pas touché."""
    T = fog_up[..., :1]
    S = g["k_fog_head"] * fog_up[..., 1:4] + g["k_fog_red"] * fog_up[..., 4:7] + g["k_fog_amb"] * fog_up[..., 7:10]
    L = (g["k_head"] * surf["head"] + g["k_red"] * surf["red"] + g["k_amb"] * surf["amb"]) * T * surf_gain
    haze = np.array(g["haze"], np.float32) * g["k_fog_amb"] + haze_dyn
    return L + S + (1.0 - T) * haze


def backdrop(lin: np.ndarray, mask: np.ndarray, amount: float, exposure: float) -> np.ndarray:
    """Assombrit le brouillard autour des lettres (ombre diffuse, comme si le titre masquait la lueur derrière
    lui) ; les lettres elles-mêmes et les sources très vives (feux, phares) sont épargnées."""
    import cv2
    tl = TEXT_LOOK
    m8 = mask.astype(np.uint8)
    k = 2 * tl["backdrop_grow"] + 1
    grown = cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    q = 4
    small = cv2.resize(grown, (W // q, H // q), interpolation=cv2.INTER_AREA)
    halo = cv2.GaussianBlur(small, (0, 0), tl["backdrop_sigma"] / q)
    halo = cv2.resize(halo / max(float(halo.max()), 1e-6), (W, H), interpolation=cv2.INTER_LINEAR)
    halo = np.clip(halo * 1.6, 0, 1)
    lum = (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]) * exposure
    keep_src = np.clip((lum - 0.6) / 1.4, 0, 1)
    soft_letters = cv2.GaussianBlur(m8.astype(np.float32), (0, 0), 1.0)
    d = amount * halo * (1.0 - soft_letters) * (1.0 - keep_src)
    return lin * (1.0 - d)[..., None]


def composite_frame(wd: Path, f: int, g: dict = GRADE, cover: bool = False) -> tuple[np.ndarray, dict]:
    """Recompose une frame -> (sRGB float (1920,1080,3), stats du titre)."""
    import cv2
    name = "cover.exr" if cover else f"f_{f:04d}.exr"
    fx = read_exr(wd / ("cover_fog" if cover else "exr_fog") / name)
    fog = fog_terms(fx)
    g = gains(g, None if cover else (f - 1) / FPS)
    fog_stack = np.concatenate([fog["T"], fog["S_head"], fog["S_red"], fog["S_amb"]], -1)
    s_mean = fog["S_head"].mean(axis=(0, 1))
    haze_dyn = g["haze_dyn"] * s_mean / max(float(s_mean.mean()), 1e-6) * float(s_mean.mean())

    bx = read_exr(wd / ("cover_surf" if cover else "exr_bg") / name)
    bg = denoise_groups(bx)
    up = jbu(fog_stack, fog["depth"], bx["Depth"][..., 0])
    lin_bg = despeckle(assemble(bg, up, g, haze_dyn))
    if lin_bg.shape[:2] != (H, W):
        lin = cv2.resize(lin_bg, (W, H), interpolation=cv2.INTER_LANCZOS4)
        lin = np.maximum(lin, 0.0)
    else:
        lin = lin_bg
    mask = (bx["IndexOB"][..., 0] > 0.5) if "IndexOB" in bx and bx["IndexOB"].shape[:2] == (H, W) else None

    text_exr = wd / ("cover_text" if cover else "exr_text") / name
    k_look = 0.0 if cover else text_look_amount((f - 1) / FPS)
    if text_exr.exists():
        tx = read_exr(text_exr)
        alpha = tx["Combined"][..., 3]
        ys, xs = np.nonzero(alpha > 0)
        if len(ys):
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = (y0, y1, x0, x1)
            surf = {k: v[y0:y1, x0:x1] for k, v in denoise_groups(tx, crop).items()}
            h_lo = fog["depth"].shape[0]
            upc = jbu(fog_stack, fog["depth"], tx["Depth"][y0:y1, x0:x1, 0], offset=(y0, x0),
                      scale=(h_lo / H, h_lo / H))
            obj = (tx["IndexOB"][y0:y1, x0:x1, :1] > 0.5).astype(np.float32)
            if k_look > 0:
                # lettres : gain des surfaces + liseré/reflet synthétiques (bords adoucis à 1 px)
                obj_s = cv2.GaussianBlur(obj[..., 0], (0, 0), 0.7)[..., None]
                gain = 1.0 + (TEXT_LOOK["gain"] - 1.0) * k_look * obj_s
                lin_hi = assemble(surf, upc, g, haze_dyn, surf_gain=gain)
                lin_hi = lin_hi + text_rim(tx["Denoising Normal"][y0:y1, x0:x1], upc[..., :1], k_look) * obj_s
            else:
                lin_hi = assemble(surf, upc, g, haze_dyn)
            # fondu de raccord sur 14 px au bord du cadre pleine définition
            m = np.ones((y1 - y0, x1 - x0), np.float32)
            m = cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
            m = cv2.GaussianBlur(cv2.erode(m, np.ones((15, 15), np.uint8), borderType=cv2.BORDER_CONSTANT,
                                           borderValue=0), (0, 0), 4.0)[..., None]
            lin[y0:y1, x0:x1] = lin[y0:y1, x0:x1] * (1 - m) + lin_hi * m
            mask = np.zeros((H, W), bool)
            mask[y0:y1, x0:x1] = tx["IndexOB"][y0:y1, x0:x1, 0] > 0.5
            if k_look > 0:
                lin = backdrop(lin, mask, k_look * TEXT_LOOK["backdrop"], g["exposure"])
    out = grade(lin, g)
    stats = text_stats(out, mask)
    return out, stats


def text_stats(img: np.ndarray, mask: np.ndarray | None) -> dict:
    """Lisibilité du titre (valeurs d'affichage sRGB) : contraste entre l'intérieur des lettres et un anneau de fond
    autour d'elles, plus le contraste des contours (biseaux éclairés). Vaut pour des lettres claires sur fond sombre
    comme pour une silhouette sombre devant la brume éclairée."""
    if mask is None or not mask.any():
        return {}
    import cv2
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    ys, xs = np.nonzero(mask)
    m8 = mask.astype(np.uint8)
    inner = cv2.erode(m8, np.ones((5, 5), np.uint8)) > 0
    ring = (cv2.dilate(m8, np.ones((31, 31), np.uint8)) > 0) & ~(cv2.dilate(m8, np.ones((9, 9), np.uint8)) > 0)
    edge = (cv2.dilate(m8, np.ones((5, 5), np.uint8)) > 0) & ~inner
    t, b = lum[inner if inner.any() else mask], lum[ring]
    gy, gx = np.gradient(cv2.GaussianBlur(lum, (0, 0), 1.0))
    grad = np.hypot(gx, gy)
    legib = abs(float(np.median(t)) - float(np.median(b))) + 2.0 * float(np.mean(grad[edge]))
    return dict(box=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                text_med=round(float(np.median(t)), 4), ring_med=round(float(np.median(b)), 4),
                edge_grad=round(float(np.mean(grad[edge])), 4), legib=round(legib, 4))


def layout_boxes(args) -> dict[int, list[float]]:
    """Boîtes projetées du titre (px) pour les 150 frames, calculées par Blender (--layout), en cache."""
    path = work_dir(args) / "layout.json"
    if not path.exists():
        cmd = [find_blender(), "-b", "--factory-startup", "-noaudio", "-P", str(SCRIPT), "--",
               "--font", str(make_font()), "--frames", f"1-{N_FRAMES}", "--layout", str(path)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return {int(k): v[:4] for k, v in json.loads(path.read_text())["boxes"].items()}


def interpolate_frame(args, f: int) -> tuple[np.ndarray, dict] | None:
    """Image f synthétisée à partir de f-1 et f+1 : chaque voisine est recalée (échelle + translation) sur la
    boîte projetée du titre à l'instant f, puis les deux sont moyennées."""
    import cv2
    wd = work_dir(args)
    boxes = layout_boxes(args)
    stats = json.loads((wd / "stats.json").read_text()) if (wd / "stats.json").exists() else {}
    acc, sts = np.zeros((H, W, 3), np.float32), []
    for n in (f - 1, f + 1):
        p = wd / "png" / f"f_{n:04d}.png"
        if not p.exists() or str(n) not in stats:
            return None
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        bx, bt = boxes[n], boxes[f]
        sx, sy = (bt[2] - bt[0]) / (bx[2] - bx[0]), (bt[3] - bt[1]) / (bx[3] - bx[1])
        M = np.float32([[sx, 0, bt[0] - sx * bx[0]], [0, sy, bt[1] - sy * bx[1]]])
        acc += 0.5 * cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        sts.append(stats[str(n)])
    st = {k: round((sts[0][k] + sts[1][k]) / 2, 4) for k in sts[0] if k != "box"}
    st["box"] = [int(round((a + b) / 2)) for a, b in zip(sts[0]["box"], sts[1]["box"])]
    st["interpolated"] = True
    return np.clip(acc, 0, 1), st


def post(args, frames: list[int], t_start: float) -> bool:
    import cv2
    wd = work_dir(args)
    png = wd / "png"
    png.mkdir(exist_ok=True)
    stats_path = wd / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    # les images rendues d'abord, les images synthétisées ensuite (elles dépendent de leurs voisines)
    order = [f for f in frames if not is_interpolated(f)] + [f for f in frames if is_interpolated(f)]
    for f in order:
        dst = png / f"f_{f:04d}.png"
        if dst.exists() and str(f) in stats and not args.force:
            continue
        if args.time_limit and time.time() - t_start > args.time_limit:
            log("limite de temps atteinte (post)")
            stats_path.write_text(json.dumps(stats, indent=0))
            return False
        t0 = time.time()
        if is_interpolated(f):
            stats_path.write_text(json.dumps(stats, indent=0))
            res = interpolate_frame(args, f)
            if res is None:
                log(f"frame {f} : voisines pas encore prêtes, ignorée")
                continue
            img, st = res
            dither = False
        else:
            need = ("fog", "bg") if args.lookdev else ("fog", "bg", "text")
            if not all((wd / f"exr_{p}" / f"f_{f:04d}.exr").exists() for p in need):
                log(f"frame {f} : passes manquantes, ignorée")
                continue
            img, st = composite_frame(wd, f)
            dither = True
        out = to_u8(img, seed=1000 + f) if dither else np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
        cv2.imwrite(str(dst), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        stats[str(f)] = st
        log(f"post frame {f} en {time.time() - t0:.1f} s  {st}")
        stats_path.write_text(json.dumps(stats, indent=0))
    return True


# ----------------------------------------------------------------------------- encodage / méta / couverture
def encode(args) -> Path:
    wd = work_dir(args)
    missing = [f for f in range(1, N_FRAMES + 1) if not (wd / "png" / f"f_{f:04d}.png").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} PNG manquants (ex. {missing[:5]})")
    TITLE_DIR.mkdir(parents=True, exist_ok=True)
    dst = TITLE_DIR / "title_seq.mp4"
    cmd = [find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(FPS), "-start_number", "1",
           "-i", str(wd / "png" / "f_%04d.png"), "-frames:v", str(N_FRAMES),
           "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
           "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
           "-r", str(FPS), "-movflags", "+faststart", "-an", str(dst)]
    subprocess.run(cmd, check=True)
    ffm = Path(find_ffmpeg())
    probe = subprocess.run([str(ffm.with_name(ffm.name.replace("ffmpeg", "ffprobe"))), "-v", "error", "-count_frames",
                            "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,nb_read_frames,r_frame_rate,pix_fmt", "-of", "json", str(dst)],
                           capture_output=True, text=True)
    log(f"encodé -> {dst} {probe.stdout.strip()}")
    return dst


def meta(args) -> dict:
    """title_meta.json : instant de lisibilité mesuré sur les images, boîte du titre (px) sous laquelle placer
    la ligne « INSPIRÉ DE FAITS RÉELS — 30.10.2026 »."""
    wd = work_dir(args)
    stats = {int(k): v for k, v in json.loads((wd / "stats.json").read_text()).items() if v}
    frames = sorted(stats)
    end = stats[frames[-1]]
    leg = {f: stats[f]["legib"] for f in frames}
    plateau = float(np.median([leg[f] for f in frames if (f - 1) / FPS >= SUB_T0]))
    thr = max(0.05, 0.5 * plateau)
    reveal = None
    for f in frames:   # lisible = lisibilité ≥ seuil et qui le reste (6 images suivantes ≥ 80 % du seuil)
        nxt = [leg[k] for k in frames if f <= k < f + 6]
        if leg[f] >= thr and min(nxt) >= 0.8 * thr:
            reveal = (f - 1) / FPS
            break
    sub_frames = [f for f in frames if (f - 1) / FPS >= SUB_T0]
    boxes = np.array([stats[f]["box"] for f in sub_frames])
    box = [int(boxes[:, 0].min()), int(boxes[:, 1].min()), int(boxes[:, 2].max()), int(boxes[:, 3].max())]
    out = {
        "dur": N_FRAMES / FPS,
        "reveal": round(reveal if reveal is not None else 1.2, 3),
        "text_box": box,
        "fps": FPS,
        "frames": N_FRAMES,
        "size": [W, H],
        "text_box_note": "union des boîtes du titre (px, x0,y0,x1,y1) de t=2,6 s à la fin (travelling avant)",
        "reveal_note": f"premier instant où la lisibilité mesurée dépasse {thr:.3f} (50 % du plateau final)",
        "text_box_start": stats[frames[0]]["box"],
        "text_box_end": end["box"],
        # centre optique du titre (≈ 480 px : il est décalé à gauche pour la zone sûre TikTok) ; gfx.draw_title_sub
        # utilise l'ancre telle quelle
        "sub_anchor": [int(round((box[0] + box[2]) / 2)), int(box[3] + 70)],
        "font": FONT_STATIC.name,
        "plate": "propre (sans grain ni vignettage) : à finir avec gfx.film_finish",
    }
    cover_json = wd / "cover_meta.json"
    if cover_json.exists():
        out["cover"] = json.loads(cover_json.read_text())
    TITLE_DIR.mkdir(parents=True, exist_ok=True)
    (TITLE_DIR / "title_meta.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"title_meta.json : reveal={out['reveal']} s text_box={box}")
    return out


def cover(args) -> Path:
    """Plaque de couverture 1080x1920 : rendu fixe plus échantillonné (brouillard 33 %, fond 50 %, titre en pleine
    définition), étalonnage GRADE_COVER (titre plus lumineux, brouillard plus sombre)."""
    import cv2
    wd = work_dir(args)
    for pass_, d, scale, spp in COVER_PASSES:
        if not (wd / d / "cover.exr").exists():
            run_blender(pass_, [1], wd / d, mode="cover", samples=spp, scale=scale, threads=args.threads)
    img, st = composite_frame(wd, 1, GRADE_COVER, cover=True)
    dst = TITLE_DIR / "cover_plate.png"
    cv2.imwrite(str(dst), cv2.cvtColor(to_u8(img, seed=77), cv2.COLOR_RGB2BGR))
    (wd / "cover_meta.json").write_text(json.dumps({
        "text_box": st.get("box"),
        # phares au centre (x ≈ 0,49, y ≈ 0,45) : la silhouette (tête vers 0,42·H, auréolée par les phares)
        # s'y découpe en contre-jour, pieds sur les reflets de l'asphalte mouillé
        "silhouette_slot": {"x": 0.49, "foot_y": 0.78, "height": 0.36,
                            "note": "contre-jour des phares au centre : silhouette sombre devant la lueur"}}))
    log(f"couverture -> {dst} {st}")
    return dst


def contact(args, frames: list[int], src: str = "png") -> Path:
    import cv2
    wd = work_dir(args)
    tiles = []
    for f in frames:
        p = wd / src / f"f_{f:04d}.png"
        if p.exists():
            im = cv2.imread(str(p))
            im = cv2.resize(im, (270, 480), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"{(f - 1) / FPS:.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            tiles.append(im)
    if not tiles:
        raise RuntimeError("aucune image")
    cols = min(len(tiles), 6)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    dst = wd / "contact.jpg"
    cv2.imwrite(str(dst), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return dst


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["font", "render", "post", "encode", "meta", "cover", "contact", "all"])
    ap.add_argument("--pass", dest="pass_", choices=list(PASSES), default=None)
    ap.add_argument("--frames", default=f"1-{N_FRAMES}")
    ap.add_argument("--time-limit", type=float, default=0,
                    help="s ; arrête proprement après ce délai (reprendre ensuite)")
    ap.add_argument("--threads", type=int, default=0, help="threads Blender (0 = tous ; 3 sur une machine partagée)")
    ap.add_argument("--scale", type=int, default=None, help="render : surcharge du %% de résolution (essais)")
    ap.add_argument("--samples", type=int, default=None, help="render : surcharge des échantillons (essais)")
    ap.add_argument("--work", default=None, help="WORK_DIR (défaut : config.WORK_DIR)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lookdev", action="store_true", help="post sans la passe text (aperçu rapide)")
    args = ap.parse_args(argv)
    frames = parse_frames(args.frames)
    t0 = time.time()

    if args.cmd == "font":
        make_font(force=args.force)
    elif args.cmd == "render":
        if not render_passes(args, [args.pass_] if args.pass_ else ["fog", "bg", "text"], frames, t0):
            return 3
    elif args.cmd == "post":
        return 0 if post(args, frames, t0) else 3
    elif args.cmd == "encode":
        encode(args)
    elif args.cmd == "meta":
        meta(args)
    elif args.cmd == "cover":
        cover(args)
    elif args.cmd == "contact":
        log(f"planche -> {contact(args, frames)}")
    elif args.cmd == "all":
        make_font()
        if not render_passes(args, ["fog", "bg", "text"], frames, t0):
            return 3
        if not post(args, frames, t0):
            return 3
        encode(args)
        if not (TITLE_DIR / "cover_plate.png").exists():
            cover(args)
        meta(args)
    log(f"terminé en {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
