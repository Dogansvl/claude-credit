"""Construction de l'EDL : edit/edl_spec.py (temps relatifs) → edl.json (temps absolus, tout résolu).

Sources :
  * edit/edl_spec.py          spécification créative (SHOTS, MUSIC, TARGET_DURATION)
  * edit/decisions.json       choix validés visuellement (identités, instants, positions d'incrustation)
  * analysis/ (facultatif)    transcriptions mot à mot, visages, personnes, audio, luminance (pipeline/analyze.py)
  * assets/audio/voice/manifest.json (facultatif) durées réelles des voix off
  * assets/title/title_meta.json     (facultatif) instant de révélation du titre

Sans analyse ni manifeste, l'EDL est quand même complète (timecodes approximatifs du brief, durées de
parole estimées à ~13 caractères/s, révélation du titre à 1,2 s) : chaque repli est signalé dans `warnings`.

Sorties : WORK_DIR/edl.json + WORK_DIR/edl_report.md + WORK_DIR/script.md (copie) ;
avec --commit : aussi edit/edl.json et edit/edl_report.md (plan par défaut versionné).

Conventions de temps (contrat docs/ARCHITECTURE.md, « Détails de edl.json ») :
  * start/dur des shots : absolus, alignés sur les images (multiples de 1/30 s, sans dérive) ;
  * fx / overlays / silhouette : t0/t1 relatifs à la shot ; voices / subtitles / cues / music / silence / src_audio : absolus ;
  * vitesse de lecture d'une shot `clip` à l'instant τ (relatif) : speed (float ou rampe linéaire [v0, v1])
    × rampe des fx `speed_ramp` (1 → `to` en courbe douce « smoothstep » sur [t0, t1], comme le rendu, puis `to`), 0 après `freeze_after` ;
    l'image affichée à l'image k est celle du proxy à src_in + ∫₀^{k/30} v(τ) dτ.

CLI : python -m edit.build_edl [--commit] [--no-analysis]
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config  # noqa: E402

FPS = config.FPS
CHARS_PER_S = 13.0          # débit de parole estimé sans manifeste
REVEAL_FALLBACK = 1.2       # révélation du titre sans title_meta.json
HUD_LOCK_DB = -11.0         # blips de verrouillage HUD auto
MONTAGE_HIT_DB = -14.0      # impacts légers aux coupes du montage accéléré
CUT_FADE = 0.01             # fondu du son source aux coupes (s)
MAX_CHARS = 34              # sous-titres : ≤ 2 lignes de ≤ 34 caractères
SUB_HOLD = 0.3              # maintien d'un sous-titre après la fin de la parole
SUB_GAP = 0.04              # écart minimal entre deux sous-titres
AMBIENCE_DB = -4.0          # son source plus bas que ça = ambiance : tous ses mots transcrits sont coupés
WORD_PAD = 0.1              # marge autour d'un mot coupé (s)
NAMES_UP = {n.upper() for n in config.NAMES}
_DEC = re.compile(r"^decision:([A-Za-z0-9_]+)(?:\.([A-Za-z_]+))?([+-]\d+(?:\.\d+)?)?$")


def log(msg: str) -> None:
    """print robuste : une console Windows en cp1252 ne doit jamais faire planter le pipeline (→, ≠, …)."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(msg.encode(enc, "replace").decode(enc, "replace"), flush=True)


def fr(t: float) -> int:
    return int(round(t * FPS))


def r4(x):
    return None if x is None else round(float(x), 4)


def _tuples_to_lists(o):
    if isinstance(o, (tuple, list)):
        return [_tuples_to_lists(v) for v in o]
    if isinstance(o, dict):
        return {k: _tuples_to_lists(v) for k, v in o.items()}
    return o


# ---------------------------------------------------------------------------
# Contexte de résolution
# ---------------------------------------------------------------------------
class Ctx:
    def __init__(self, spec, decisions, adir: Path | None, manifest: dict | None, title_meta: dict | None):
        self.spec = spec
        self.dec = decisions or {}
        self.adir = adir
        self.manifest = (manifest or {}).get("items") if manifest else None
        self.title_meta = title_meta
        self.warnings: list[str] = []
        self.resolution: list[dict] = []  # journal pour le rapport
        self.anchors_found = 0
        self.anchors_total = 0
        self.script_texts = _script_texts()
        self._warned = set()
        self.voices: list[dict] = []
        self._an = None

    def warn(self, msg: str, once_key: str | None = None):
        if once_key:
            if once_key in self._warned:
                return
            self._warned.add(once_key)
        if msg not in self.warnings:
            self.warnings.append(msg)

    def note(self, shot, field, source, value, detail=""):
        self.resolution.append({"shot": shot, "field": field, "source": source, "value": value, "detail": detail})

    @property
    def an(self):
        """Module pipeline.analyze (fonctions de lecture uniquement), ou None sans analyse."""
        if self.adir is None:
            return None
        if self._an is None:
            from pipeline import analyze
            self._an = analyze
        return self._an


def _script_texts() -> dict:
    """Textes des voix depuis les tableaux de edit/script.md (repli sans manifeste)."""
    out = {}
    p = config.EDIT_DIR / "script.md"
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*((?:N\d+[a-z]?)|(?:E_[A-Z]+))\s*\|\s*([^|]+?)\s*\|", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


# ---------------------------------------------------------------------------
# Résolution des valeurs
# ---------------------------------------------------------------------------
def _pick_candidate(entry) -> tuple[dict, bool]:
    """Entrée de décision avec `candidates` : 1er candidat confirmé, sinon le 1er (non confirmé)."""
    if isinstance(entry, dict) and isinstance(entry.get("candidates"), list) and entry["candidates"]:
        conf = [c for c in entry["candidates"] if c.get("confirmed")]
        return (conf[0], True) if conf else (entry["candidates"][0], False)
    if isinstance(entry, dict):
        return entry, bool(entry.get("confirmed"))
    return {}, False


def resolve_value(v, ctx: Ctx, shot: str, field: str):
    """{"auto": "...", "fallback": x} → valeur d'analyse ou repli ; listes/dicts résolus récursivement."""
    if isinstance(v, dict) and "auto" in v:
        val, why = (None, "pas d'analyse")
        if ctx.an is not None:
            val, why = ctx.an.resolve_auto(v["auto"], ctx.adir)
        if val is None:
            ctx.note(shot, field, "repli", v.get("fallback"), f"{v['auto']} : {why}")
            ctx.warn(f"auto {v['auto']} indisponible ({why}) : repli {v.get('fallback')}", once_key="auto:" + v["auto"])
            return copy.deepcopy(v.get("fallback"))
        ctx.note(shot, field, "auto", val, f"{v['auto']} : {why}")
        return val
    if isinstance(v, dict):
        return {k: resolve_value(x, ctx, shot, f"{field}.{k}") for k, x in v.items()}
    if isinstance(v, list):
        return [resolve_value(x, ctx, shot, field) for x in v]
    return v


def resolve_decision(ref: str, ctx: Ctx, shot: str, field: str):
    m = _DEC.match(ref.strip())
    if not m:
        ctx.warn(f"{shot}.{field} : référence de décision mal formée « {ref} »")
        return None
    key, sub, off = m.group(1), m.group(2), m.group(3)
    if key not in ctx.dec:
        ctx.warn(f"{shot}.{field} : clé absente de decisions.json « {key} »")
        return None
    entry = ctx.dec[key]
    if sub:
        cand, _ = _pick_candidate(entry)
        if sub not in cand:
            ctx.warn(f"{shot}.{field} : champ « {sub} » absent de decisions.json[{key}]")
            return None
        val = resolve_value(cand[sub], ctx, shot, field)
    else:
        val = resolve_value(entry, ctx, shot, field)
        if isinstance(val, dict) and "order_left_to_right" in val:
            val = val["order_left_to_right"]
    if off:
        try:
            val = float(val) + float(off)
        except (TypeError, ValueError):
            ctx.warn(f"{shot}.{field} : décalage {off} sur une valeur non numérique ({val})")
    ctx.note(shot, field, "décision", val if not isinstance(val, dict) else "…", ref)
    return val


def _guard_cut(t: float, m: dict, a: dict, clip: str, ctx: Ctx, shot: str, field: str) -> float:
    """Point de coupe : un décalage ne doit jamais entamer le mot voisin (mot suivant pour une fin, précédent
    pour un début) ; on s'arrête 30 ms avant lui."""
    tr = ctx.an.load_transcript(clip, ctx.adir) or {}
    words = tr.get("words") or []
    off = float(a.get("offset", 0.0))
    if a["edge"] == "end" and off > 0 and m["i1"] + 1 < len(words):
        lim = words[m["i1"] + 1]["start"] - 0.03
        if t > lim:
            new = max(m["end"], lim)
            ctx.note(shot, field, "coupe protégée", round(new, 3),
                     f"{t:.2f} → {new:.2f} s : n'entame pas « {words[m['i1'] + 1]['w']} »")
            return new
    if a["edge"] == "start" and off < 0 and m["i0"] > 0:
        lim = words[m["i0"] - 1]["end"] + 0.03
        if t < lim:
            new = min(m["start"], lim)
            ctx.note(shot, field, "coupe protégée", round(new, 3),
                     f"{t:.2f} → {new:.2f} s : n'entame pas « {words[m['i0'] - 1]['w']} »")
            return new
    return t


def resolve_anchor(a: dict, clip: str | None, ctx: Ctx, shot: str, field: str, cut: bool = False) -> float:
    ctx.anchors_total += 1
    m = None
    if ctx.an is not None and clip:
        m = ctx.an.find_phrase_match(clip, a["anchor"], a.get("occ", 0), a.get("window"), ctx.adir)
    if m is None:
        ctx.note(shot, field, "repli", a["fallback"], f"« {a['anchor']} » ({a['edge']}) introuvable dans {clip}")
        why = "pas de transcription" if ctx.an is None else f"introuvable dans la transcription de {clip}"
        ctx.warn(f"ancre « {a['anchor']} » ({clip}) {why} : timecodes approximatifs du brief", once_key=f"A:{clip}:{a['anchor']}:{a['edge']}")
        return float(a["fallback"])
    ctx.anchors_found += 1
    t = (m["start"] if a["edge"] == "start" else m["end"]) + float(a.get("offset", 0.0))
    ctx.note(shot, field, "transcription", round(t, 3),
             f"« {a['anchor']} » → « {m['text']} » {m['start']:.2f}–{m['end']:.2f} s (score {m['score']:.2f})")
    if cut:
        t = _guard_cut(t, m, a, clip, ctx, shot, field)
    if abs(t - float(a["fallback"])) > 4.0:
        ctx.warn(f"{shot}.{field} : ancre « {a['anchor']} » trouvée à {t:.2f} s, loin du repli {a['fallback']} s "
                 f"(mauvaise occurrence ?)")
    return t


def resolve_scalar(v, ctx: Ctx, shot: str, field: str, clip: str | None = None, cut: bool = False):
    """Nombre, ancre A(...), « decision:... » ou valeur auto (`cut` : point de coupe protégé)."""
    if isinstance(v, dict) and "anchor" in v:
        return resolve_anchor(v, clip, ctx, shot, field, cut)
    if isinstance(v, str) and v.startswith("decision:"):
        return resolve_decision(v, ctx, shot, field)
    if isinstance(v, dict) and "auto" in v:
        return resolve_value(v, ctx, shot, field)
    return v


# ---------------------------------------------------------------------------
# Voix
# ---------------------------------------------------------------------------
def voice_info(vid: str, ctx: Ctx) -> dict:
    it = (ctx.manifest or {}).get(vid) if ctx.manifest else None
    if it:
        return {"file": f"assets/audio/voice/{it.get('file', vid + '.wav')}", "onset_s": float(it.get("onset_s", 0.0)),
                "speech_dur": float(it["speech_dur"]), "dur": float(it["dur"]), "text": it.get("text", ""),
                "tts_text": it.get("tts_text") or it.get("text", ""),
                "kind": it.get("kind") or ("narrator" if vid.startswith("N") else "entity"), "estimated": False}
    text = ctx.script_texts.get(vid, vid)
    sd = max(0.4, len(text) / CHARS_PER_S)
    ctx.warn(f"voix {vid} absente du manifeste : durée de parole estimée ({sd:.2f} s à {CHARS_PER_S:.0f} car./s)",
             once_key="voice:" + vid)
    return {"file": f"assets/audio/voice/{vid}.wav", "onset_s": 0.05, "speech_dur": round(sd, 3),
            "dur": round(sd + 0.65, 3), "text": text, "kind": "narrator" if vid.startswith("N") else "entity",
            "estimated": True}


def _voice_pauses(file_rel: str, onset: float, speech_dur: float) -> list[float]:
    """Milieux des pauses (≥ 60 ms) dans la parole d'un fichier voix (temps relatifs au début de parole)."""
    p = config.REPO / file_rel
    if not p.exists():
        return []
    try:
        import soundfile as sf
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    except Exception:
        return []
    x = x.mean(axis=1)
    hop = int(0.01 * sr)
    n = len(x) // hop
    if n < 5:
        return []
    e = 20 * np.log10(np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-9)
    a, b = int(onset / 0.01), min(n, int((onset + speech_dur) / 0.01))
    if b - a < 5:
        return []
    seg = e[a:b]
    thr = float(np.percentile(seg, 95)) - 30.0
    quiet = seg < thr
    out, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if j - i >= 6 and i > 0 and j < len(quiet):
                out.append((i + j) / 2 * 0.01)
            i = j
        else:
            i += 1
    return out


def resolve_voice_time(v, shot_voices: list[dict], ctx: Ctx, sid: str) -> float:
    """t relatif d'une voix : nombre ou « after:<id>:<écart> » (fin de parole de <id> + écart)."""
    if isinstance(v, (int, float)):
        return float(v)
    m = re.match(r"^after:([A-Za-z0-9_]+):(-?\d+(?:\.\d+)?)$", str(v))
    if m:
        ref = next((x for x in reversed(shot_voices) if x["id"] == m.group(1)), None)
        if ref is None:
            ctx.warn(f"{sid} : « {v} » référence une voix absente de la shot")
            return 0.0
        return ref["_t_rel"] + ref["speech_dur"] + float(m.group(2))
    ctx.warn(f"{sid} : temps de voix non reconnu « {v} »")
    return 0.0


def resolve_rel_time(v, ctx: Ctx, sid: str, shot_voices: list[dict]):
    """t relatif d'un fx/cue : nombre, « voice_end:<id>:<décalage> », « after:<id>:<écart> », « title:reveal »."""
    if v is None or isinstance(v, (int, float)):
        return v
    s = str(v)
    m = re.match(r"^voice_end:([A-Za-z0-9_]+):?(-?\d+(?:\.\d+)?)?$", s)
    if m:
        ref = next((x for x in reversed(shot_voices) if x["id"] == m.group(1)), None)
        if ref is None:
            ctx.warn(f"{sid} : « {s} » référence une voix absente de la shot")
            return 0.0
        return ref["_t_rel"] + ref["speech_dur"] + float(m.group(2) or 0.0)
    if s.startswith("after:"):
        return resolve_voice_time(s, shot_voices, ctx, sid)
    if s == "title:reveal":
        rv = (ctx.title_meta or {}).get("reveal")
        if rv is None:
            ctx.warn(f"assets/title/title_meta.json absent : révélation du titre supposée à {REVEAL_FALLBACK} s",
                     once_key="title_meta")
            return REVEAL_FALLBACK
        return float(rv)
    ctx.warn(f"{sid} : temps non reconnu « {s} »")
    return 0.0


# ---------------------------------------------------------------------------
# Vitesse / correspondance temps de shot ↔ temps source
# ---------------------------------------------------------------------------
class TimeMap:
    """τ (relatif à la shot) ↔ temps source du proxy, selon speed, fx speed_ramp et freeze_after."""

    def __init__(self, src_in: float, dur: float, speed=1.0, fx=None, freeze_after=None):
        self.src_in = float(src_in)
        self.dur = max(float(dur), 1e-3)
        n = int(math.ceil(self.dur / 0.002)) + 1
        tau = np.linspace(0.0, self.dur, n)
        if isinstance(speed, (list, tuple)):
            v0, v1 = float(speed[0]), float(speed[-1])
        else:
            v0 = v1 = float(speed if speed is not None else 1.0)
        v = v0 + (v1 - v0) * tau / self.dur
        def num(x, default):
            try:
                return float(x)
            except (TypeError, ValueError):
                return default
        for f in fx or []:
            if f.get("type") == "speed_ramp":
                t0 = num(f.get("t0"), 0.0)
                t1 = num(f.get("t1"), self.dur)
                ramp = np.clip((tau - t0) / max(t1 - t0, 1e-3), 0, 1)
                ramp = ramp * ramp * (3 - 2 * ramp)  # rampe douce, identique à pipeline/render.py
                v = v * (1 + (float(f.get("to", 0.6)) - 1) * ramp)
        if freeze_after is not None:
            v = np.where(tau >= float(freeze_after), 0.0, v)
        pos = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(tau))])
        self.tau, self.pos = tau, pos

    def src(self, t: float) -> float:
        return self.src_in + float(np.interp(t, self.tau, self.pos))

    def rel(self, s: float) -> float:
        """Premier τ où la source atteint s (borné à [0, dur])."""
        x = s - self.src_in
        if x <= 0:
            return 0.0 if x > -1e-6 else float(x)  # négatif : avant le début de la shot
        k = int(np.searchsorted(self.pos, x))
        if k >= len(self.pos):
            return self.dur + (x - self.pos[-1])  # au-delà de la fin (vitesse ~1)
        if k == 0:
            return 0.0
        p0, p1 = self.pos[k - 1], self.pos[k]
        return float(self.tau[k - 1] + (x - p0) / max(p1 - p0, 1e-9) * (self.tau[k] - self.tau[k - 1]))


def _mean_speed(speed) -> float:
    if isinstance(speed, (list, tuple)):
        return (float(speed[0]) + float(speed[-1])) / 2
    return float(speed if speed is not None else 1.0)


# ---------------------------------------------------------------------------
# Sous-titres
# ---------------------------------------------------------------------------
def _sub_tokens(text: str) -> list[str]:
    """Mots ; la ponctuation française détachée (« ? », « ! », « : », « … ») reste collée au mot précédent."""
    toks: list[str] = []
    for t in text.split():
        if toks and re.fullmatch(r"[?!:;…»]+", t):
            toks[-1] += " " + t
        elif toks and toks[-1] == "«":
            toks[-1] += " " + t
        else:
            toks.append(t)
    return toks


_FUNCTION_WORDS = {"le", "la", "les", "l'", "un", "une", "des", "de", "du", "d'", "à", "au", "aux", "et", "ou", "en",
                   "pour", "par", "sur", "dans", "que", "qui", "ce", "cette", "ces", "son", "sa", "ses", "ne", "se",
                   "il", "ils", "je", "tu", "on", "nous", "vous", "mais", "pas"}


_BREAK_BEFORE = {"pour", "par", "sur", "dans", "avec", "sans", "et", "ou", "mais", "que", "qui", "quand", "comme",
                 "à", "au", "aux", "en", "parce", "puis"}


def _two_lines(toks: list[str]) -> tuple[str, float] | None:
    """Meilleure coupure en ≤ 2 lignes de ≤ MAX_CHARS : (texte, coût). Coût : déséquilibre des lignes, bonus après
    une ponctuation ou avant une préposition/conjonction, pénalité si la ligne 1 finit sur un mot-outil."""
    s = " ".join(toks)
    if len(s) <= MAX_CHARS:
        return s, 0.0
    best = None
    for k in range(1, len(toks)):
        a, b = " ".join(toks[:k]), " ".join(toks[k:])
        if len(a) > MAX_CHARS or len(b) > MAX_CHARS:
            continue
        cost = abs(len(a) - len(b)) - (8 if re.search(r"[,;:.?!…]$", a) else 0) + (6 if len(toks[k:]) == 1 else 0) \
            + (12 if toks[k - 1].lower() in _FUNCTION_WORDS else 0) - (4 if toks[k].lower() in _BREAK_BEFORE else 0)
        if best is None or cost < best[1]:
            best = (a + "\n" + b, cost)
    return best


def _best_split(toks: list[str]) -> tuple[list[str], float]:
    """Blocs (≤ 2 lignes chacun) et coût total ; un bloc supplémentaire coûte 14."""
    one = _two_lines(toks)
    cands = [([one[0]], one[1])] if one else []
    if len(toks) > 1 and (one is None or one[1] >= 12):
        for k in range(1, len(toks)):
            la, lb = len(" ".join(toks[:k])), len(" ".join(toks[k:]))
            if one is not None and (la > 2 * MAX_CHARS or lb > 2 * MAX_CHARS):
                continue
            if one is None and max(la, lb) > 2 * MAX_CHARS and min(la, lb) < 12:
                continue
            A, B = _best_split(toks[:k]), _best_split(toks[k:])
            punct = re.search(r"[,;:.?!…]$", toks[k - 1]) is not None
            cost = A[1] + B[1] + 14 + 0.3 * abs(la - lb) - (20 if punct else 0) \
                + (12 if toks[k - 1].lower() in _FUNCTION_WORDS else 0)
            cands.append((A[0] + B[0], cost))
    if not cands:  # mot isolé trop long : on le laisse tel quel
        return [" ".join(toks)], 1000.0
    return min(cands, key=lambda c: c[1])


def split_blocks(text: str) -> list[str]:
    """Découpe un texte en blocs de sous-titres (≤ 2 lignes de ≤ 34 caractères, '\\n' inséré)."""
    toks = _sub_tokens(text.strip())
    return _best_split(toks)[0] if toks else []


_PUNCT_SPLIT = re.compile(r"(?<=[,.;:?!…])\s+")


def _spoken_fraction(text: str, tts_text: str, pos: int) -> float:
    """Fraction du temps de parole écoulée à la position `pos` du texte affiché. Le texte prononcé peut différer
    (« 30 octobre 2026 » → « trente octobre deux mille vingt-six ») : si les deux textes ont le même découpage en
    ponctuation, on mesure la proportion sur le texte prononcé, segment par segment."""
    total = max(1, len(text))
    a, b = _PUNCT_SPLIT.split(text.strip()), _PUNCT_SPLIT.split(tts_text.strip())
    if len(a) != len(b) or len(a) < 2:
        return min(1.0, pos / total)
    tts_total = sum(len(x) + 1 for x in b)
    acc_d = acc_t = 0
    for da, tb in zip(a, b):
        la, lb = len(da) + 1, len(tb) + 1
        if pos <= acc_d + la:
            return min(1.0, (acc_t + (pos - acc_d) / la * lb) / tts_total)
        acc_d += la
        acc_t += lb
    return 1.0


def voice_subtitles(v: dict, ctx: Ctx) -> list[dict]:
    """Sous-titres d'une voix off : blocs au prorata du texte prononcé, frontières recalées sur les pauses du fichier."""
    blocks = split_blocks(v["text"])
    if not blocks:
        return []
    sd = v["speech_dur"]
    lens = np.cumsum([len(b.replace("\n", " ")) + 1 for b in blocks])[:-1]
    cuts = [_spoken_fraction(v["text"], v.get("tts_text") or v["text"], int(c)) * sd for c in lens]
    pauses = [] if v.get("estimated") else _voice_pauses(v["file"], v["t_speech"] - v["t_file"], sd)
    for i, c in enumerate(cuts):
        near = [p for p in pauses if abs(p - c) <= 0.6]
        if near:
            cuts[i] = min(near, key=lambda p: abs(p - c))
    edges = [0.0] + cuts + [sd]
    out = []
    for i, b in enumerate(blocks):
        st = v["t_speech"] + edges[i]
        en = v["t_speech"] + edges[i + 1] + (SUB_HOLD if i == len(blocks) - 1 else 0.0)
        out.append({"start": r4(st), "end": r4(en), "text": b, "kind": v["kind"], "voice": v["id"]})
    return out


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


def _resolve_center(c, ctx, sid):
    if isinstance(c, str) and c.startswith("decision:"):
        v = resolve_decision(c, ctx, sid, "camera.center")
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return [float(v[0]), float(v[1])]
        ctx.warn(f"{sid} : centre caméra « {c} » non résolu → [0.5, 0.5]")
        return [0.5, 0.5]
    return _tuples_to_lists(c)


def _faces_in_range(clip, tm, dur, ctx: Ctx) -> int | None:
    """Nombre de visages typique (75e centile) dans la plage source d'une shot, d'après l'analyse."""
    if ctx.an is None or not clip or tm is None:
        return None
    d = ctx.an._read_json(ctx.adir / f"{clip}.faces.json")
    if not d or not d.get("t"):
        return None
    a, b = tm.src(0.0), tm.src(dur)
    counts = [len(f) for t, f in zip(d["t"], d["faces"]) if a - 0.1 <= t <= b + 0.1]
    if not counts:
        return None
    n = int(np.percentile(counts, 75))
    return min(4, n) if n > 0 else None


def _merge_ranges(rs: list) -> list:
    out = []
    for a, b in sorted([float(x), float(y)] for x, y in rs if float(y) > float(x)):
        if out and a <= out[-1][1] + 0.02:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [[r4(a), r4(b)] for a, b in out]


def _available(name: str | None) -> bool | None:
    """Rush disponible ? True (proxy ou original trouvé), False (rushs présents mais pas celui-ci, ou listé
    manquant par l'ingestion), None (inconnu : aucun rush ni inventaire, ex. plan par défaut)."""
    if not name or not isinstance(name, str):
        return None
    if config.proxy_path(name).exists():
        return True
    try:
        config.rush_path(name)
        return True
    except FileNotFoundError:
        pass
    inv = _load_json(config.ANALYSIS_DIR / "inventory.json") or {}
    if name in (inv.get("missing") or []):
        return False
    try:
        if config.RUSH_DIR.is_dir() and any(config.RUSH_DIR.iterdir()):
            return False
    except OSError:
        pass
    return None


def _photo_or_fallback(pid, ctx: Ctx, sid: str):
    """Photo demandée, ou une autre photo du groupe si elle manque (message clair, pas de plantage au rendu)."""
    if _available(pid) is not False:
        return pid
    alt = next((p for p in config.PHOTOS if p != pid and _available(p)), None)
    ctx.warn(f"{sid} : PHOTO MANQUANTE {pid} → " + (f"repli sur {alt}" if alt else "aucune autre photo disponible"))
    return alt or pid


def _clamp_src(t: float, clip: str | None, ctx: Ctx, sid: str, field: str) -> float:
    """Instant source borné à [0, durée du clip − 1 image] (avertissement si corrigé)."""
    L = _clip_len(clip, ctx)
    hi = None if L is None else max(0.0, L - 1 / FPS)
    t2 = max(0.0, t if hi is None else min(t, hi))
    if abs(t2 - t) > 1e-3:
        ctx.warn(f"{sid}.{field} : {t:.2f} s hors de {clip} ({'?' if L is None else f'{L:.2f}'} s) → {t2:.2f} s")
    return t2


def _clip_len(clip: str, ctx: Ctx) -> float | None:
    if ctx.an is None or not clip:
        return None
    return ctx.an.clip_duration(clip, ctx.adir)


def build(spec_module=None, decisions: dict | None = None, analysis_dir=None, voice_manifest: dict | None = None,
          *, write: bool = True, commit: bool = False, title_meta: dict | None = None) -> dict:
    """Résout la spec en EDL absolue (dict). Écrit WORK_DIR/edl.json (+ rapport, copie du script) si `write`,
    et edit/edl.json + edit/edl_report.md si `commit`.
    analysis_dir : None = config.ANALYSIS_DIR s'il existe ; False/"" = sans analyse."""
    spec = spec_module or importlib.import_module("edit.edl_spec")
    if decisions is None:
        decisions = _load_json(config.EDIT_DIR / "decisions.json") or {}
    if analysis_dir is None:
        analysis_dir = config.ANALYSIS_DIR
    adir = None
    if analysis_dir not in ("", False) and Path(analysis_dir).is_dir() and any(Path(analysis_dir).glob("*.json")):
        adir = Path(analysis_dir)
    if voice_manifest is None:
        voice_manifest = _load_json(config.VOICE_DIR / "manifest.json")
    if title_meta is None:
        title_meta = _load_json(config.TITLE_DIR / "title_meta.json")
    overrides: dict[str, dict] = {}
    for _ in range(3):  # passes : corrections de rejeu (S2k/S2l…) appliquées puis reconstruction
        edl, ctx, conflicts = _build_pass(spec, decisions, adir, voice_manifest, title_meta, overrides)
        new = {k: v for k, v in conflicts.items() if overrides.get(k) != v}
        if not new:
            break
        overrides.update(new)
    if write:
        write_outputs(edl, ctx, commit=commit)
    return edl


def _build_pass(spec, decisions: dict, adir, voice_manifest, title_meta, overrides: dict):
    """Une passe de résolution : renvoie (edl, ctx, corrections de rejeu à appliquer à la passe suivante)."""
    ctx = Ctx(spec, decisions, adir, voice_manifest, title_meta)
    ctx.overrides = overrides
    if adir is None:
        ctx.warn("aucune analyse (transcriptions, visages...) : EDL construite sur les replis du brief", once_key="noanalysis")
    if not voice_manifest:
        ctx.warn("assets/audio/voice/manifest.json absent : durées de parole estimées", once_key="nomanifest")
    if not decisions.get("_reviewed"):
        ctx.warn("edit/decisions.json pas encore revu (_reviewed=false) : identités/instants à vérifier sur les planches contact")
    fixtures = adir is not None and (config.RUSH_DIR / "FIXTURES_NOT_REAL.txt").exists()

    shots, voices, subs, cues, silence, src_audio = [], [], [], [], [], []
    t_frames = 0  # position en images (entier) : aucune dérive
    prev = None
    dropped: list[str] = []
    order = [sp["id"] for sp in spec.SHOTS]
    for sp in spec.SHOTS:
        sid = sp["id"]
        kind = sp["kind"]
        drop_key = sp.get("drop_if_unconfirmed")
        if drop_key and not _pick_candidate(ctx.dec.get(drop_key))[1]:
            ctx.warn(f"{sid} supprimée de la timeline : aucune identité confirmée pour « {drop_key} » "
                     f"(decisions.json) — pas de gel sans prénom")
            dropped.append(sid)
            continue
        sh = {"id": sid, "section": sp.get("section"), "label": sp.get("label", ""), "kind": kind}
        clip = resolve_scalar(sp.get("clip"), ctx, sid, "clip") if sp.get("clip") else None
        if kind == "clip" and _available(clip) is False:
            ctx.warn(f"{sid} : RUSH MANQUANT {clip} (ni original dans {config.RUSH_DIR} ni proxy) → plan remplacé "
                     f"par du noir ; dépose le fichier puis relance ingest")
            kind = sh["kind"] = "black"
            sh["label"] += f" [RUSH MANQUANT {clip}]"
            est = sp.get("dur")
            if est is None and isinstance(sp.get("src_out"), dict) and isinstance(sp.get("src_in"), dict):
                est = float(sp["src_out"].get("fallback", 1.0)) - float(sp["src_in"].get("fallback", 0.0))
            sp = {k: v for k, v in sp.items() if k not in ("fx", "silhouette", "camera", "subs", "cues")}
            sp["dur"] = max(0.5, float(est or 1.0))
            clip = None
        tm = None
        if kind == "clip":
            speed = _tuples_to_lists(sp.get("speed", 1.0))
            src_in = float(resolve_scalar(sp.get("src_in", 0.0), ctx, sid, "src_in", clip, cut=True) or 0.0)
            if "src_out" in sp:
                src_out = float(resolve_scalar(sp["src_out"], ctx, sid, "src_out", clip, cut=True))
                dur = (src_out - src_in) / max(_mean_speed(speed), 1e-3)
                if dur < 0.2:
                    ctx.warn(f"{sid} : src_out ({src_out:.2f}) ≤ src_in ({src_in:.2f}) → durée minimale 0,5 s")
                    dur = 0.5
            else:
                dur = float(sp["dur"])
            ovr = ctx.overrides.get(sid) or {}
            if "src_in" in ovr:
                src_in = float(ovr["src_in"])
                ctx.warn(ovr["why"])
            if "src_out" in ovr:
                dur = max(1 / FPS, (float(ovr["src_out"]) - src_in) / max(_mean_speed(speed), 1e-3))
                ctx.warn(ovr["why"])
            if src_in < 0:
                if src_in < -0.1:
                    ctx.warn(f"{sid} : src_in négatif ({src_in:.2f} s) ramené à 0")
                src_in = 0.0
            n = max(1, fr(dur))
            dur = n / FPS
            fx_raw = sp.get("fx") or []
            freeze_after = sp.get("freeze_after")
            tm = TimeMap(src_in, dur, speed, fx_raw, freeze_after)
            L = _clip_len(clip, ctx)
            if L is not None:
                end_src = tm.src(dur - 1 / FPS)
                if end_src > L - 1 / FPS + 1e-6:
                    excess = end_src - (L - 1 / FPS)
                    new_in = max(0.0, src_in - excess)
                    ctx.warn(f"{sid} : dépasse la fin de {clip} ({end_src:.2f} > {L:.2f} s) → src_in {src_in:.2f} → {new_in:.2f}")
                    src_in = new_in
                    tm = TimeMap(src_in, dur, speed, fx_raw, freeze_after)
                    if tm.src(dur - 1 / FPS) > L:
                        ctx.warn(f"{sid} : {clip} trop court pour {dur:.2f} s : la dernière image sera répétée")
            sh.update({"clip": clip, "src_in": r4(src_in), "src_out": r4(tm.src(dur)), "speed": speed,
                       "freeze_after": freeze_after})
        elif kind == "freeze":
            dur = fr(float(sp["dur"])) / FPS
            fs = sp.get("freeze_src")
            if fs == "prev_last":
                if prev is None or "_last" not in prev:
                    ctx.warn(f"{sid} : freeze prev_last sans shot précédente exploitable")
                    sh["freeze_src"] = None
                else:
                    sh["freeze_src"] = dict(prev["_last"])
            elif isinstance(fs, (list, tuple)) and len(fs) == 3 and fs[0] == "clip":
                fclip = resolve_scalar(fs[1], ctx, sid, "freeze_src.clip")
                ft = resolve_scalar(fs[2], ctx, sid, "freeze_src.t", fclip)
                try:
                    ft = float(ft)
                except (TypeError, ValueError):
                    ctx.warn(f"{sid} : instant du gel non résolu ({fs[2]!r}) → 0 s")
                    ft = 0.0
                ft = _clamp_src(ft, fclip, ctx, sid, "freeze_src.t")
                if _available(fclip) is False:
                    ctx.warn(f"{sid} : RUSH MANQUANT {fclip} pour le gel → noir")
                    sh["freeze_src"] = None
                else:
                    sh["freeze_src"] = {"clip": fclip, "t": r4(ft)}
            else:
                ctx.warn(f"{sid} : freeze_src non reconnu {fs}")
                sh["freeze_src"] = None
        else:
            dur = fr(float(sp.get("dur", 1.0))) / FPS
        if kind == "photo":
            sh["photo"] = _photo_or_fallback(resolve_scalar(sp.get("photo"), ctx, sid, "photo"), ctx, sid)
        start = t_frames / FPS
        sh["start"] = r4(start)
        sh["dur"] = r4(dur)
        sh["frames"] = fr(dur)

        # --- voix (avant fx/cues : voice_end / after en dépendent)
        shot_voices = []
        for v in sp.get("voice") or []:
            vid = resolve_scalar(v["id"], ctx, sid, "voice.id")
            if not vid:
                continue
            with_sub = True
            mk = _DEC.match(v["id"]) if isinstance(v["id"], str) else None
            if mk and not _pick_candidate(ctx.dec.get(mk.group(1)))[1]:
                with_sub = False
                ctx.warn(f"{sid} : {mk.group(1)} non confirmé → chuchotement {vid} gardé, SANS sous-titre de prénom")
            info = voice_info(str(vid), ctx)
            t_rel = resolve_voice_time(v.get("t", 0.0), shot_voices, ctx, sid)
            vo = {"id": str(vid), "file": info["file"], "t_speech": r4(start + t_rel),
                  "t_file": r4(start + t_rel - info["onset_s"]), "dur": r4(info["dur"]),
                  "speech_dur": r4(info["speech_dur"]), "gain_db": float(v.get("gain_db", 0.0)), "kind": info["kind"],
                  "shot": sid, "allow_in_silence": bool(sp.get("silence")), "text": info["text"],
                  "tts_text": info.get("tts_text", info["text"]), "_t_rel": t_rel, "estimated": info["estimated"],
                  "subtitle": with_sub}
            shot_voices.append(vo)
        voices += shot_voices
        ctx.voices = voices

        # --- caméra / étalonnage
        if sp.get("camera"):
            cam = _tuples_to_lists(copy.deepcopy(sp["camera"]))
            if "center" in cam:
                cam["center"] = _resolve_center(cam["center"], ctx, sid)
            sh["camera"] = cam
        if sp.get("grade"):
            sh["grade"] = _tuples_to_lists(copy.deepcopy(sp["grade"]))

        # --- fx
        fx_out = []
        for f in sp.get("fx") or []:
            f2 = _tuples_to_lists(copy.deepcopy(f))
            t0 = resolve_rel_time(f2.get("t0", 0.0), ctx, sid, shot_voices)
            t0 = 0.0 if t0 is None else float(t0)
            if f2.get("type") == "subliminal" and f2.get("frames"):
                t1 = t0 + int(f2["frames"]) / FPS
            else:
                t1 = resolve_rel_time(f2.get("t1"), ctx, sid, shot_voices)
                t1 = dur if t1 is None else float(t1)
            if t0 > dur + 1e-6 or t1 < -1e-6:
                ctx.warn(f"{sid} : fx {f2.get('type')} hors de la shot ({t0:.2f}–{t1:.2f} / {dur:.2f} s)")
            f2["t0"], f2["t1"] = r4(min(max(t0, 0.0), dur)), r4(min(max(t1, 0.0), dur))
            fx_out.append(f2)
        sh["fx"] = fx_out

        # --- overlays
        ov_out = []
        for o in sp.get("overlays") or []:
            o2 = _tuples_to_lists(copy.deepcopy(o))
            t0 = resolve_rel_time(o2.get("t0", 0.0), ctx, sid, shot_voices) or 0.0
            t1 = resolve_rel_time(o2.get("t1"), ctx, sid, shot_voices)
            o2["t0"], o2["t1"] = r4(t0), r4(dur if t1 is None else min(float(t1), dur))
            typ = o2.get("type")
            if typ == "name":
                key = o2.get("confirm")
                _, ok = _pick_candidate(ctx.dec.get(key)) if key else ({}, False)
                o2["show"] = bool(ok)
                if o2.get("name", "").upper() not in NAMES_UP:
                    ctx.warn(f"{sid} : prénom inconnu « {o2.get('name')} » (orthographe : {', '.join(config.NAMES)})")
                    o2["show"] = False
                if not ok:
                    ctx.warn(f"{sid} : identité « {o2.get('name')} » non confirmée → gel sans prénom")
            elif typ == "hud_faces":
                order = o2.get("order")
                if isinstance(order, str) and order.startswith("decision:"):
                    order = resolve_decision(order, ctx, sid, "hud.order")
                if order is not None:
                    if not (isinstance(order, list) and all(isinstance(x, str) and x.upper() in NAMES_UP for x in order)):
                        ctx.warn(f"{sid} : ordre HUD invalide {order} → étiquettes sans prénom")
                        order = None
                o2["order"] = order
                n_faces = len(order) if order else _faces_in_range(clip, tm, dur, ctx) or 4
                if o2.get("mode", "identify") == "identify":
                    step = float(o2.get("step", 0.4))
                    o2["locks"] = [r4(o2["t0"] + i * step) for i in range(n_faces)]
                    for lt in o2["locks"]:
                        if lt < dur:
                            cues.append({"t": r4(start + lt), "sfx": "hud_lock", "gain_db": HUD_LOCK_DB, "dur": None,
                                         "params": {}, "shot": sid, "generated": True})
            elif typ == "red_circles":
                if o2.get("targets") == "faces":
                    o2["draws"] = [r4(o2["t0"] + i * float(o2.get("step", 0.38))) for i in range(4)]
            ov_out.append(o2)
        sh["overlays"] = ov_out

        # --- silhouette
        if sp.get("silhouette"):
            s_sp = sp["silhouette"]
            slot = s_sp.get("slot")
            base = {"slot": slot, "x": 0.5, "foot_y": 0.7, "height": 0.35, "blur": 1.5, "darkness": 0.9,
                    "blend": "normal", "occlude_persons": False, "opacity": 0.9, "t0": 0.0, "t1": None,
                    "fade_in": 0.0, "space": "photo" if kind == "photo" else "plate"}
            slots = ctx.dec.get("silhouette_slots") or {}
            if slot not in slots:
                ctx.warn(f"{sid} : emplacement de silhouette « {slot} » absent de decisions.json (valeurs par défaut)")
            base.update(resolve_value(copy.deepcopy(slots.get(slot, {})), ctx, sid, "silhouette"))
            base.update({k: v for k, v in s_sp.items() if k != "slot"})
            t1 = base.get("t1")
            base["t0"] = r4(float(base.get("t0") or 0.0))
            base["t1"] = r4(dur if t1 is None else min(float(t1), dur))
            sh["silhouette"] = base
        else:
            sh["silhouette"] = None

        # --- son source
        au = copy.deepcopy(sp.get("audio") or {})
        audio = {"src_gain_db": float(au.get("src_gain_db", 0.0)), "mute_src": bool(au.get("mute_src", False)),
                 "src_fx": _tuples_to_lists(au.get("src_fx", [])), "mute_ranges": _tuples_to_lists(au.get("mute_ranges", [])),
                 "fade_in": float(au.get("fade_in", CUT_FADE)), "fade_out": float(au.get("fade_out", CUT_FADE))}
        if kind == "clip" and not audio["mute_src"] and ctx.an is not None:
            play_src = (tm.src_in, tm.src(dur))
            for h in ctx.an.detect_instructions(clip, ctx.adir):
                if h["end"] > play_src[0] and h["start"] < play_src[1]:
                    a0, a1 = max(0.0, tm.rel(h["start"])), min(dur, tm.rel(h["end"]))
                    if a1 > a0:
                        audio["mute_ranges"].append([r4(a0), r4(a1)])
                        ctx.warn(f"{sid} : consigne « {h['text']} » ({clip} {h['start']:.2f}–{h['end']:.2f} s) "
                                 f"dans la plage utilisée → son source coupé {a0:.2f}–{a1:.2f} s")
            if audio["src_gain_db"] < AMBIENCE_DB and not sp.get("keep_words"):
                # son source d'ambiance : aucun mot parasite audible (bavardages hors sujet)
                tr = ctx.an.load_transcript(clip, ctx.adir) or {}
                cut, heard = [], []
                for w in tr.get("words") or []:
                    if w["end"] > play_src[0] and w["start"] < play_src[1]:
                        a0 = max(0.0, tm.rel(w["start"] - WORD_PAD))
                        a1 = min(dur, tm.rel(w["end"] + WORD_PAD))
                        if a1 > a0:
                            cut.append([r4(a0), r4(a1)])
                            heard.append(w["w"])
                if cut:
                    audio["mute_ranges"] += cut
                    ctx.warn(f"{sid} : son d'ambiance ({audio['src_gain_db']:+.0f} dB) → {len(cut)} mot(s) parasite(s) "
                             f"coupé(s) : « {' '.join(heard)[:60]} »")
            audio["mute_ranges"] = _merge_ranges(audio["mute_ranges"])
        sh["audio"] = audio
        if kind == "clip" and not audio["mute_src"]:
            play = dur if sh["freeze_after"] is None else min(dur, float(sh["freeze_after"]))
            src_audio.append({"clip": clip, "src_in": sh["src_in"], "src_out": r4(tm.src(play)), "t": r4(start),
                              "dur": r4(play), "speed": sh["speed"], "gain_db": audio["src_gain_db"],
                              "fade_in": audio["fade_in"], "fade_out": audio["fade_out"], "src_fx": audio["src_fx"],
                              "mute_ranges": [[r4(start + a), r4(start + b)] for a, b in audio["mute_ranges"]],
                              "shot": sid})

        # --- montage
        if kind == "montage":
            items = []
            raw = sp.get("items") or []
            want = [float(it.get("dur", 0.2)) for it in raw]
            total = sh["frames"]
            scale = total / max(1e-6, sum(want) * FPS)
            exact = [w * FPS * scale for w in want]
            frames = [int(math.floor(e)) for e in exact]
            for k in sorted(range(len(raw)), key=lambda k: -(exact[k] - frames[k]))[: total - sum(frames)]:
                frames[k] += 1
            if abs(scale - 1) > 0.02:
                ctx.warn(f"{sid} : durées des micro-plans ({sum(want):.2f} s) ≠ durée de la shot ({dur:.2f} s), mises à l'échelle")
            acc = 0
            for k, it in enumerate(raw):
                it2 = {"start": r4(acc / FPS), "dur": r4(frames[k] / FPS), "frames": frames[k],
                       "fx": _tuples_to_lists(it.get("fx", []))}
                if it.get("clip"):
                    c = resolve_scalar(it["clip"], ctx, sid, f"items[{k}].clip")
                    s_in = float(resolve_scalar(it.get("src_in", 0.0), ctx, sid, f"items[{k}].src_in", c) or 0.0)
                    if _available(c) is False:
                        prev_ok = next((x for x in reversed(items) if x.get("clip") or x.get("photo")), None)
                        ctx.warn(f"{sid} : micro-plan {k} : RUSH MANQUANT {c} → "
                                 + ("répétition du micro-plan précédent" if prev_ok else "noir"))
                        if prev_ok:
                            it2.update({x: prev_ok[x] for x in ("clip", "src_in", "photo") if x in prev_ok})
                    else:
                        Lc = _clip_len(c, ctx)
                        if Lc is not None and s_in + frames[k] / FPS > Lc:
                            new_in = max(0.0, Lc - frames[k] / FPS - 1 / FPS)
                            ctx.warn(f"{sid} : micro-plan {k} ({c}) {s_in:.2f} s hors du clip ({Lc:.2f} s) → {new_in:.2f} s")
                            s_in = new_in
                        if s_in < 0:
                            ctx.warn(f"{sid} : micro-plan {k} ({c}) instant négatif {s_in:.2f} s → 0")
                        it2.update({"clip": c, "src_in": r4(max(0.0, s_in))})
                elif it.get("photo"):
                    it2["photo"] = _photo_or_fallback(resolve_scalar(it["photo"], ctx, sid, f"items[{k}].photo"),
                                                      ctx, sid)
                items.append(it2)
                if k > 0:
                    cues.append({"t": r4(start + acc / FPS), "sfx": "impact_hit", "gain_db": MONTAGE_HIT_DB, "dur": None,
                                 "params": {}, "shot": sid, "generated": True})
                acc += frames[k]
            sh["items"] = items

        # --- cues
        for c in sp.get("cues") or []:
            c2 = _tuples_to_lists(copy.deepcopy(c))
            if "at" in c2:
                s_t = resolve_scalar(c2.pop("at"), ctx, sid, f"cue {c2.get('sfx')}", clip)
                rel = tm.rel(float(s_t)) if tm is not None else float(s_t)
            else:
                rel = resolve_rel_time(c2.get("t", 0.0), ctx, sid, shot_voices) or 0.0
            if rel < -1e-3 or rel > dur + 1e-3:
                ctx.warn(f"{sid} : cue {c2.get('sfx')} hors de la shot ({rel:.2f} s / {dur:.2f} s)")
            c2["t"] = r4(start + rel)
            c2.setdefault("gain_db", 0.0)
            c2.setdefault("dur", None)
            c2.setdefault("params", {})
            c2["shot"] = sid
            cues.append(c2)

        # --- sous-titres des vraies répliques
        for s in sp.get("subs") or []:
            a = resolve_scalar(s["at"], ctx, sid, "sub.at", clip)
            b = resolve_scalar(s["until"], ctx, sid, "sub.until", clip)
            ra = tm.rel(float(a)) if tm is not None else float(a)
            rb = tm.rel(float(b)) if tm is not None else float(b)
            if ra < -0.3 or rb > dur + 0.3:
                ctx.warn(f"{sid} : sous-titre « {s['text']} » déborde de la shot ({ra:.2f}–{rb:.2f} / {dur:.2f} s), recadré")
            ra, rb = max(0.0, ra), min(dur, rb)
            if rb - ra < 0.5:
                rb = min(dur, ra + 0.6)
            for blk in split_blocks(s["text"]):
                subs.append({"start": r4(start + ra), "end": r4(start + rb), "text": blk, "kind": s.get("kind", "dialogue"),
                             "shot": sid})

        if sp.get("silence"):
            silence.append([r4(start), r4(start + dur)])
        # dernière image (pour freeze prev_last)
        if kind == "clip":
            sh["_last"] = {"clip": clip, "t": r4(tm.src(max(0.0, dur - 1 / FPS)))}
        elif kind == "freeze" and sh.get("freeze_src"):
            sh["_last"] = dict(sh["freeze_src"])
        elif kind == "photo" and sh.get("photo"):
            sh["_last"] = {"photo": sh["photo"]}
        elif kind == "montage" and sh.get("items"):
            it = sh["items"][-1]
            sh["_last"] = ({"clip": it["clip"], "t": r4(it["src_in"] + max(0, it["frames"] - 1) / FPS)}
                           if it.get("clip") else {"photo": it.get("photo")})
        shots.append(sh)
        prev = sh
        t_frames += sh["frames"]

    total = t_frames / FPS
    # --- sous-titres des voix off
    for v in voices:
        if v.get("subtitle", True):
            subs += voice_subtitles(v, ctx)
    subs.sort(key=lambda s: (s["start"], s["end"]))
    for i in range(len(subs) - 1):
        a, b = subs[i], subs[i + 1]
        if a["end"] > b["start"] - SUB_GAP:
            new_end = max(a["start"] + 0.3, b["start"] - SUB_GAP)
            if a["end"] - new_end > 0.35:
                ctx.warn(f"sous-titres qui se chevauchent : « {a['text'][:30]} » raccourci de {a['end'] - new_end:.2f} s "
                         f"(« {b['text'][:30]} » à {b['start']:.2f} s)")
            a["end"] = r4(new_end)
    for s in subs:
        s["end"] = r4(min(s["end"], total))

    # --- musique
    by_id = {s["id"]: s for s in shots}

    def ref_t(rf) -> float:
        s = by_id.get(rf["ref"])
        if s is None and rf["ref"] in dropped:  # shot supprimée : point de la shot gardée suivante
            nxt = next((by_id[x] for x in order[order.index(rf["ref"]) + 1:] if x in by_id), None)
            if nxt is not None:
                return nxt["start"] + float(rf.get("offset", 0.0))
        if s is None:
            ctx.warn(f"musique : shot inconnue {rf['ref']}")
            return 0.0
        base = s["start"] + (s["dur"] if rf.get("edge") == "end" else 0.0)
        return base + float(rf.get("offset", 0.0))
    music = []
    for m in getattr(spec, "MUSIC", []):
        t0, t1 = max(0.0, ref_t(m["start"])), min(total, ref_t(m["end"]))
        if t1 <= t0:
            ctx.warn(f"lit {m['kind']} vide ({t0:.2f}–{t1:.2f} s)")
            continue
        auto = []
        for p in m.get("automation") or []:
            tt, dbv = p[0], float(p[1])
            if isinstance(tt, str):
                mm = re.match(r"^rel_end([+-]\d+(?:\.\d+)?)?$", tt)
                tt = t1 + float(mm.group(1) or 0.0) if mm else t0
            else:
                tt = t0 + float(tt)
            auto.append([r4(tt), dbv])
        music.append({"kind": m["kind"], "t0": r4(t0), "t1": r4(t1), "gain_db": float(m.get("gain_db", 0.0)),
                      "fade_in": float(m.get("fade_in", 0.0)), "fade_out": float(m.get("fade_out", 0.0)),
                      "duck": float(m.get("duck", 0.0)), "automation": auto,
                      "params": _tuples_to_lists(m.get("params", {}))})

    cues.sort(key=lambda c: c["t"])
    for v in voices:
        v.pop("_t_rel", None)
        v.pop("tts_text", None)
    for s in shots:
        s.pop("_last", None)

    # --- contrôles
    checks = run_checks(shots, voices, subs, cues, silence, total, getattr(spec, "TARGET_DURATION", 92.0), ctx)

    edl = {
        "meta": {"fps": FPS, "w": config.W, "h": config.H, "duration": r4(total),
                 "target": getattr(spec, "TARGET_DURATION", None),
                 "built_with": {"transcripts": ctx.anchors_found > 0, "voice_manifest": bool(voice_manifest),
                                "title_meta": bool(title_meta), "analysis": adir is not None,
                                "decisions_reviewed": bool(decisions.get("_reviewed"))},
                 "anchors": {"found": ctx.anchors_found, "total": ctx.anchors_total},
                 "dropped_shots": dropped,
                 "fixtures": fixtures,
                 "conventions": "start/dur absolus alignés sur les images ; fx/overlays/silhouette relatifs à la shot ; "
                                "voices/subtitles/cues/music/silence/src_audio absolus ; vitesse = speed × speed_ramp (smoothstep), "
                                "0 après freeze_after"},
        "shots": shots, "voices": voices, "subtitles": subs, "cues": cues, "music": music, "silence": silence,
        "src_audio": src_audio, "checks": checks, "warnings": ctx.warnings,
    }
    return edl, ctx, _replay_conflicts(shots, ctx)


def _cuts_word(clip: str, t: float, ctx: Ctx) -> bool:
    """Une coupe à l'instant source t tombe-t-elle au milieu d'un mot transcrit ?"""
    if ctx.an is None:
        return False
    tr = ctx.an.load_transcript(clip, ctx.adir) or {}
    return any(w["start"] + 0.02 < t < w["end"] - 0.02 for w in tr.get("words") or [])


def _replay_conflicts(shots: list[dict], ctx: Ctx) -> dict:
    """Pas de rejeu : deux shots consécutives du même clip ne doivent pas se chevaucher dans la source
    (ex. le cri S2l ne rejoue pas la fin de S2k). On raccourcit la première (src_out = src_in suivant − 0,05 s)
    si elle garde ≥ 0,4 s sans couper un mot ; sinon on décale le début de la seconde."""
    out = {}
    for a, b in zip(shots, shots[1:]):
        if not (a["kind"] == b["kind"] == "clip" and a.get("clip") and a["clip"] == b.get("clip")):
            continue
        if a.get("freeze_after") is not None:
            continue
        a_in, a_out, b_in = float(a["src_in"]), float(a["src_out"]), float(b["src_in"])
        if not (a_in - 0.01 <= b_in < a_out - 0.01):  # la seconde ne reprend pas dans la plage de la première
            continue
        ov = a_out - b_in
        new_out = float(b["src_in"]) - 0.05
        if new_out - float(a["src_in"]) >= 0.4 and not _cuts_word(a["clip"], new_out, ctx):
            out[a["id"]] = {"src_out": round(new_out, 4),
                            "why": f"{a['id']} rejouait {ov:.2f} s reprises par {b['id']} : fin ramenée de "
                                   f"{a['src_out']:.2f} à {new_out:.2f} s"}
        else:
            new_in = float(a["src_out"])
            out[b["id"]] = {"src_in": round(new_in, 4),
                            "why": f"{b['id']} rejouait {ov:.2f} s de {a['id']} : début décalé de "
                                   f"{b['src_in']:.2f} à {new_in:.2f} s (raccourcir {a['id']} couperait un mot)"}
    return out


# ---------------------------------------------------------------------------
# Contrôles
# ---------------------------------------------------------------------------
def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def run_checks(shots, voices, subs, cues, silence, total, target, ctx: Ctx) -> dict:
    out = {"duration_ok": config.DURATION_MIN <= total <= config.DURATION_MAX, "duration": r4(total),
           "voice_voice": [], "voice_dialogue": [], "voice_silence": [], "voice_tail_beyond_end": [], "subtitles_format": []}
    if not out["duration_ok"]:
        ctx.warn(f"DURÉE HORS FENÊTRE : {total:.2f} s (attendu {config.DURATION_MIN:.0f}–{config.DURATION_MAX:.0f} s)")
    elif abs(total - target) > 2.5:
        ctx.warn(f"durée {total:.2f} s (cible {target} s)")
    sp = [(v["id"], v["t_speech"], v["t_speech"] + v["speech_dur"]) for v in voices]
    for i in range(len(sp)):
        for j in range(i + 1, len(sp)):
            ov = _overlap(sp[i][1], sp[i][2], sp[j][1], sp[j][2])
            if ov > 0.0:
                out["voice_voice"].append([sp[i][0], sp[j][0], r4(ov)])
                if ov > 0.05:
                    ctx.warn(f"chevauchement voix/voix : {sp[i][0]} et {sp[j][0]} ({ov:.2f} s)")
    dlg = [s for s in subs if s["kind"] == "dialogue"]
    for vid, a, b in sp:
        for s in dlg:
            ov = _overlap(a, b, s["start"], s["end"])
            if ov > 0.05:
                out["voice_dialogue"].append([vid, s["text"].replace("\n", " "), r4(ov)])
                ctx.warn(f"chevauchement voix/réplique : {vid} et « {s['text']} » ({ov:.2f} s)")
    for v in voices:
        if v["allow_in_silence"]:
            continue
        for a, b in silence:
            ov = _overlap(v["t_speech"], v["t_speech"] + v["speech_dur"], a, b)
            if ov > 0:
                out["voice_silence"].append([v["id"], r4(ov)])
                ctx.warn(f"la voix {v['id']} tombe dans une plage de silence ({ov:.2f} s) : elle sera coupée")
        tail = v["t_file"] + v["dur"] - total
        if tail > 0.05:
            out["voice_tail_beyond_end"].append([v["id"], r4(tail)])
            ctx.warn(f"le fichier {v['id']} dépasse la fin du film de {tail:.2f} s (traîne coupée)")
        if v["t_speech"] + v["speech_dur"] > total:
            ctx.warn(f"la parole de {v['id']} dépasse la fin du film")
    for s in subs:
        lines = s["text"].split("\n")
        if len(lines) > 2 or any(len(l) > MAX_CHARS for l in lines):
            out["subtitles_format"].append(s["text"])
            ctx.warn(f"sous-titre trop long : « {s['text']} »")
        if s["end"] - s["start"] < 0.25:
            ctx.warn(f"sous-titre trop bref ({s['end'] - s['start']:.2f} s) : « {s['text']} »")
    prev = None
    for s in shots:
        if s["kind"] == "clip":
            if prev is not None and prev["clip"] == s["clip"]:
                ov = _overlap(prev["src_in"], prev["src_out"], s["src_in"], s["src_out"])
                if ov > 0.1:
                    ctx.warn(f"{s['id']} rejoue {ov:.2f} s de {s['clip']} déjà montrées dans {prev['id']} "
                             f"({prev['src_in']:.2f}–{prev['src_out']:.2f} / {s['src_in']:.2f}–{s['src_out']:.2f})")
            prev = s
        elif s["kind"] != "freeze":
            prev = None
    for c in cues:
        for a, b in silence:
            if a <= c["t"] < b:
                ctx.warn(f"cue {c['sfx']} à {c['t']:.2f} s dans une plage de silence (coupée)")
    return out


# ---------------------------------------------------------------------------
# Sorties
# ---------------------------------------------------------------------------
def _fmt(t) -> str:
    return "—" if t is None else f"{t:6.2f}"


def write_report(edl: dict, ctx: Ctx | None = None) -> str:
    m = edl["meta"]
    bw = m["built_with"]
    L = ["# Liste de montage — C'EST QUI LUI ?", "",
         f"Durée totale : **{m['duration']:.2f} s** ({m['duration'] * FPS:.0f} images, cible {m.get('target')} s, "
         f"fenêtre {config.DURATION_MIN:.0f}–{config.DURATION_MAX:.0f} s) — "
         f"{'OK' if edl['checks']['duration_ok'] else 'HORS FENÊTRE'}", "",
         f"Construite avec : transcriptions {'oui' if bw['transcripts'] else 'non'} "
         f"({m['anchors']['found']}/{m['anchors']['total']} ancres trouvées), manifeste des voix "
         f"{'oui' if bw['voice_manifest'] else 'non'}, title_meta {'oui' if bw['title_meta'] else 'non'}, "
         f"analyse {'oui' if bw['analysis'] else 'non'}, decisions.json revu {'oui' if bw['decisions_reviewed'] else 'NON'}.", ""]
    if m.get("fixtures"):
        L += ["> **EDL construite sur les FAUX RUSHS de test : ne pas utiliser pour un livrable.**", ""]
    L += ["## Shots", "", "| # | id | § | type | début | fin | durée | source | voix off | remarques |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    vby = {}
    for v in edl["voices"]:
        vby.setdefault(v["shot"], []).append(v["id"])
    for i, s in enumerate(edl["shots"], 1):
        src = ""
        if s["kind"] == "clip":
            src = f"{s['clip']} {s['src_in']:.2f}→{s['src_out']:.2f}"
            if s.get("freeze_after") is not None:
                src += f" (gel à +{s['freeze_after']})"
        elif s["kind"] == "freeze" and s.get("freeze_src"):
            fs = s["freeze_src"]
            src = f"gel {fs['clip']} @{fs['t']:.2f}" if fs.get("clip") else f"gel photo {fs.get('photo')}"
        elif s["kind"] == "photo":
            src = f"photo {s.get('photo')}"
        elif s["kind"] == "montage":
            src = " · ".join(f"{it.get('clip') or it.get('photo')}" for it in s.get("items", []))
        rem = []
        for o in s.get("overlays", []):
            if o["type"] == "card":
                rem.append(f"carton « {o['text']} »")
            elif o["type"] == "name":
                rem.append(f"prénom {o['name']} {'affiché' if o['show'] else 'MASQUÉ (non confirmé)'}")
            elif o["type"] == "timecode":
                rem.append(f"TC {o['text']}")
            elif o["type"] == "hud_faces":
                rem.append(f"HUD {o.get('mode')} {o.get('order') or 'sans prénoms'}")
            elif o["type"] == "red_circles":
                rem.append(f"cercles ({o.get('targets')})")
        if s.get("silhouette"):
            rem.append(f"silhouette {s['silhouette']['slot']}")
        if s["audio"].get("mute_src"):
            rem.append("son source coupé")
        if s["audio"].get("mute_ranges"):
            rem.append(f"consignes coupées {s['audio']['mute_ranges']}")
        L.append(f"| {i} | {s['id']} | {s['section']} | {s['kind']} | {s['start']:.2f} | {s['start'] + s['dur']:.2f} | "
                 f"{s['dur']:.2f} | {src} | {', '.join(vby.get(s['id'], []))} | {'; '.join(rem)} |")
    L += ["", "## Voix off", "", "| id | type | parole | fichier | shot | texte |", "|---|---|---|---|---|---|"]
    for v in edl["voices"]:
        L.append(f"| {v['id']} | {v['kind']} | {v['t_speech']:.2f}→{v['t_speech'] + v['speech_dur']:.2f} | "
                 f"{v['t_file']:.2f}→{v['t_file'] + v['dur']:.2f}{' (estimé)' if v.get('estimated') else ''} | "
                 f"{v['shot']} | {v['text']} |")
    L += ["", "## Sous-titres", "", "| début | fin | type | texte |", "|---|---|---|---|"]
    for s in edl["subtitles"]:
        L.append(f"| {s['start']:.2f} | {s['end']:.2f} | {s['kind']} | {s['text'].replace(chr(10), ' ⏎ ')} |")
    L += ["", f"## Cues SFX ({len(edl['cues'])})", "", "| t | sfx | gain | durée | shot |", "|---|---|---|---|---|"]
    for c in edl["cues"]:
        L.append(f"| {c['t']:.2f} | {c['sfx']}{' (auto)' if c.get('generated') else ''} | {c['gain_db']} | "
                 f"{'' if c.get('dur') is None else c['dur']} | {c['shot']} |")
    L += ["", "## Lits musicaux", "", "| type | début | fin | gain | duck | automation |", "|---|---|---|---|---|---|"]
    for mu in edl["music"]:
        L.append(f"| {mu['kind']} | {mu['t0']:.2f} | {mu['t1']:.2f} | {mu['gain_db']} | {mu['duck']} | {mu['automation'] or ''} |")
    L += ["", f"## Silences francs : {edl['silence']}", "",
          "## Son source (segments pour le mixage)", "", "| t | clip | src | durée | gain | coupures |", "|---|---|---|---|---|---|"]
    for sa in edl["src_audio"]:
        L.append(f"| {sa['t']:.2f} | {sa['clip']} | {sa['src_in']:.2f}→{sa['src_out']:.2f} | {sa['dur']:.2f} | "
                 f"{sa['gain_db']} | {sa['mute_ranges'] or ''} |")
    ck = edl["checks"]
    L += ["", "## Contrôles", "",
          f"- durée : {ck['duration']:.2f} s → {'OK' if ck['duration_ok'] else 'HORS FENÊTRE'}",
          f"- chevauchements voix/voix : {ck['voice_voice'] or 'aucun'}",
          f"- chevauchements voix/répliques : {ck['voice_dialogue'] or 'aucun'}",
          f"- voix dans un silence : {ck['voice_silence'] or 'aucune'}",
          f"- fichiers voix au-delà de la fin : {ck['voice_tail_beyond_end'] or 'aucun'}",
          f"- sous-titres hors format : {ck['subtitles_format'] or 'aucun'}"]
    if ctx is not None:
        L += ["", "## Résolution des valeurs", "", "| shot | champ | source | valeur | détail |", "|---|---|---|---|---|"]
        for r in ctx.resolution:
            val = r["value"]
            if isinstance(val, float):
                val = round(val, 3)
            L.append(f"| {r['shot']} | {r['field']} | {r['source']} | {val} | {r['detail']} |")
    L += ["", f"## Avertissements ({len(edl['warnings'])})", ""] + [f"- {w}" for w in edl["warnings"]] + [""]
    return "\n".join(L)


def write_outputs(edl: dict, ctx: Ctx | None, commit: bool = False) -> dict:
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    js = json.dumps(edl, ensure_ascii=False, indent=1)
    rep = write_report(edl, ctx)
    out = {"edl": config.WORK_DIR / "edl.json", "report": config.WORK_DIR / "edl_report.md",
           "script": config.WORK_DIR / "script.md"}
    out["edl"].write_text(js, encoding="utf-8")
    out["report"].write_text(rep, encoding="utf-8")
    if (config.EDIT_DIR / "script.md").exists():
        shutil.copyfile(config.EDIT_DIR / "script.md", out["script"])
    if commit:
        if edl["meta"].get("fixtures"):
            raise RuntimeError("refus de --commit : l'EDL contient des valeurs issues des FAUX rushs de test")
        (config.EDIT_DIR / "edl.json").write_text(js, encoding="utf-8")
        (config.EDIT_DIR / "edl_report.md").write_text(rep, encoding="utf-8")
        out["commit"] = [config.EDIT_DIR / "edl.json", config.EDIT_DIR / "edl_report.md"]
    return out


def edl_path() -> Path:
    """EDL courante : WORK_DIR/edl.json si elle existe, sinon le plan par défaut edit/edl.json."""
    p = config.WORK_DIR / "edl.json"
    return p if p.exists() else config.EDIT_DIR / "edl.json"


def load_edl(path=None) -> dict:
    return json.loads(Path(path or edl_path()).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Construit l'EDL absolue depuis edit/edl_spec.py")
    ap.add_argument("--commit", action="store_true", help="copier aussi dans edit/edl.json et edit/edl_report.md")
    ap.add_argument("--no-analysis", action="store_true", help="ignorer l'analyse (replis du brief)")
    a = ap.parse_args(argv)
    try:
        edl = build(analysis_dir="" if a.no_analysis else None, commit=a.commit)
    except RuntimeError as e:
        log(f"[edl] ERREUR : {e}")
        return 3
    m = edl["meta"]
    log(f"[edl] {len(edl['shots'])} shots, {m['duration']:.2f} s, {len(edl['voices'])} voix, "
          f"{len(edl['subtitles'])} sous-titres, {len(edl['cues'])} cues, {len(edl['music'])} lits, "
          f"ancres {m['anchors']['found']}/{m['anchors']['total']}, {len(edl['warnings'])} avertissements")
    log(f"[edl] → {config.WORK_DIR / 'edl.json'}" + ("  (+ edit/edl.json)" if a.commit else ""))
    for w in edl["warnings"]:
        log("  ! " + w)
    return 0 if edl["checks"]["duration_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
