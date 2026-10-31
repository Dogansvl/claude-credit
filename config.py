"""Configuration centrale du projet « C'EST QUI LUI ? ».

Tous les chemins peuvent être surchargés par variables d'environnement :
  ESCAPE_RUSH_DIR   dossier des rushs originaux (lecture seule, jamais modifié)
  ESCAPE_WORK_DIR   dossier de travail (proxies, caches, rendus intermédiaires) HORS OneDrive
  ESCAPE_OUT_DIR    dossier des livrables finaux
Par défaut : chemins Windows du PC de Dogan ; sous Linux : /home/user/...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

IS_WINDOWS = sys.platform.startswith("win")

# Console Windows redirigée (cp1252) : jamais de plantage sur « → », « ≠ », « è »… (print des modules)
for _st in (sys.stdout, sys.stderr):
    if _st is not None and (getattr(_st, "encoding", "") or "").lower().replace("-", "") != "utf8":
        try:
            _st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

_DEFAULT_RUSH = (
    r"C:\Users\dogan\OneDrive\Bureau\Documents\Photos, Souvenirs et Voyages\Montages\Escape game horreur"
    if IS_WINDOWS
    else "/home/user/rushs"
)
_DEFAULT_WORK = r"C:\Users\dogan\edit_escape_tmp" if IS_WINDOWS else "/home/user/edit_escape_tmp"

REPO = Path(__file__).resolve().parent

RUSH_DIR = Path(os.environ.get("ESCAPE_RUSH_DIR", _DEFAULT_RUSH))
WORK_DIR = Path(os.environ.get("ESCAPE_WORK_DIR", _DEFAULT_WORK))


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


# Livrables : à côté des rushs (brief) ; mais si les rushs sont DANS le dépôt (flux cloud : <dépôt>/rushs),
# jamais dans le dépôt (vidéo > 15 Mo) → dossier de travail.
_DEFAULT_OUT = (WORK_DIR if _inside(RUSH_DIR, REPO) else RUSH_DIR) / "MONTAGE_CEST_QUI_LUI"
OUT_DIR = Path(os.environ.get("ESCAPE_OUT_DIR", str(_DEFAULT_OUT)))

ASSETS = REPO / "assets"
FONTS_DIR = ASSETS / "fonts"
VOICE_DIR = ASSETS / "audio" / "voice"
SFX_DIR = ASSETS / "audio" / "sfx"
TITLE_DIR = ASSETS / "title"
EDIT_DIR = REPO / "edit"

# Models are shared with the work dir (never committed). Override with ESCAPE_MODELS_DIR.
MODELS_DIR = Path(os.environ.get("ESCAPE_MODELS_DIR", str(WORK_DIR / "models")))
PROXY_DIR = WORK_DIR / "proxies"        # IMG_xxxx.mp4 (1080x1920, 30 fps CFR) + IMG_xxxx.png (photos)
ANALYSIS_DIR = WORK_DIR / "analysis"    # transcripts, faces, persons, contact sheets, report
CACHE_DIR = WORK_DIR / "cache"          # face tracks, silhouettes, depth maps, per-shot renders
RENDER_DIR = WORK_DIR / "render"        # preview + final intermediates

# Output format
W, H, FPS = 1080, 1920, 30
SR = 48000  # audio sample rate

# Rushs
CLIPS = [f"IMG_{n}" for n in list(range(8336, 8350)) + list(range(8354, 8358))]
PHOTOS = [f"IMG_{n}" for n in range(8350, 8354)]

# Characters (exact spelling — never change)
NAMES = ["Dogan", "Gabriel", "Hugo", "Arsène"]

# Deliverables
FINAL_MP4 = "CEST_QUI_LUI_bande-annonce.mp4"
COVER_JPG = "CEST_QUI_LUI_couverture.jpg"
README_TXT = "LISEZMOI.txt"

# Target duration window (seconds)
DURATION_MIN, DURATION_MAX = 85.0, 95.0

# Loudness targets
TARGET_LUFS = -14.0
TARGET_TP = -1.0

# Words/phrases that must NEVER be audible (shooting instructions). Normalised, accent-free, lowercase.
# Recherche par MOTS ENTIERS (pipeline.analyze.match_blacklist) : « souris » ne touche pas « sourire ».
INSTRUCTION_BLACKLIST = [
    "fais genre", "fait genre", "fais comme si", "fais semblant", "fait semblant",
    "vas y filme", "filme moi", "t'es possede", "tes possede", "regarde la camera",
    "action", "coupe", "on refait", "recommence",
    "vas y", "fais le", "fais la", "souris", "crie", "bouge pas", "reste comme ca", "on la refait",
    "regarde moi", "c est bon", "t as filme", "attends",
]
# Mots courants de la liste : on coupe le MOT seul (± 0,12 s), pas tout l'énoncé qui le contient
# (les vraies consignes « fais genre… » emportent tout l'énoncé).
INSTRUCTION_WORDS_ONLY = ["vas y", "fais le", "fais la", "souris", "crie", "bouge pas", "regarde moi", "c est bon",
                          "attends"]


def ensure_dirs() -> None:
    for d in (WORK_DIR, PROXY_DIR, ANALYSIS_DIR, CACHE_DIR, RENDER_DIR, OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def model_path(name: str) -> Path:
    return MODELS_DIR / name


def rush_path(clip: str) -> Path:
    """Return the original rush path (tries .MOV/.mov/.jpeg/.jpg/.JPG/.HEIC)."""
    for ext in (".MOV", ".mov", ".MP4", ".mp4", ".jpeg", ".JPEG", ".jpg", ".JPG", ".HEIC", ".heic", ".png"):
        p = RUSH_DIR / f"{clip}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Rush introuvable : {clip} dans {RUSH_DIR}")


def proxy_path(clip: str) -> Path:
    if clip in PHOTOS:
        return PROXY_DIR / f"{clip}.png"
    return PROXY_DIR / f"{clip}.mp4"
