"""Enchaîne toute la chaîne « C'EST QUI LUI ? ».

    python run_all.py --quality preview            # ingest → analyze → edl → render → mix → deliver → qc
    python run_all.py --quality final              # après la revue de edit/decisions.json
    python run_all.py --stage render,mix --quality preview
    python run_all.py --from render --quality final
    python run_all.py --stage voices               # (option) régénère les voix off
    python run_all.py --stage title                # (option) re-rend le titre 3D (Blender, ~1 h)

Étapes : models, ingest, analyze, edl, voices (option), title (option), render, mix, deliver, qc.
(`models` télécharge les modèles manquants dans MODELS_DIR ; instantané s'ils sont déjà là.)
Journal complet : WORK_DIR/logs/run_<date>.log. Arrêt à la première erreur bloquante.
"""
from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

STAGES = ["models", "ingest", "analyze", "edl", "voices", "title", "render", "mix", "deliver", "qc"]
DEFAULT = ["models", "ingest", "analyze", "edl", "render", "mix", "deliver", "qc"]

# Modèles (fichier, URL, taille exacte en octets). Les modèles de transcription sherpa-onnx sont
# téléchargés par pipeline/asr.py lui-même.
_MP = "https://storage.googleapis.com/mediapipe-models/"
_REMBG = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
MODELS = [
    ("face_landmarker.task", _MP + "face_landmarker/face_landmarker/float16/1/face_landmarker.task", 3758596),
    ("blaze_face_short_range.tflite",
     _MP + "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite", 229746),
    ("efficientdet_lite2.tflite", _MP + "object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite",
     23096891),
    ("pose_landmarker_full.task", _MP + "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
     9398198),
    ("selfie_multiclass.tflite",
     _MP + "image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite", 16371837),
    ("u2net_human_seg.onnx", _REMBG + "u2net_human_seg.onnx", 175997641),
    ("isnet-general-use.onnx", _REMBG + "isnet-general-use.onnx", 178648008),
    ("u2net.onnx", _REMBG + "u2net.onnx", 175997641),
    ("midas_small.onnx", "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx", 66764249),
]
PIPER_VOICES = ["tom-medium", "upmc-medium", "gilles-low", "siwis-medium"]  # étape « voices » seulement


class Tee:
    """Duplique la sortie console dans le journal."""

    def __init__(self, stream, logf):
        self.stream, self.logf = stream, logf

    def write(self, s):
        try:
            self.stream.write(s)
        except UnicodeEncodeError:  # console Windows non UTF-8
            self.stream.write(s.encode("ascii", "replace").decode("ascii"))
        self.logf.write(s)
        self.logf.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.logf.flush()


def _call(fn, **kw):
    """Appelle fn avec les seuls arguments qu'elle accepte (interfaces des autres modules)."""
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return fn(**kw)
    return fn(**{k: v for k, v in kw.items() if k in params})


class Pipeline:
    def __init__(self, args):
        self.a = args
        self.q = args.quality
        self.edl: dict | None = None
        self.outputs: dict = {}

    # --- étapes ------------------------------------------------------------
    def models(self, piper: bool = False):
        import urllib.request
        d = config.MODELS_DIR
        d.mkdir(parents=True, exist_ok=True)
        got = []
        for name, url, size in MODELS:
            p = d / name
            if p.exists() and p.stat().st_size == size:
                continue
            print(f"téléchargement de {name} ({size / 1e6:.0f} Mo)…")
            tmp = p.with_suffix(p.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            if tmp.stat().st_size != size:
                raise RuntimeError(f"{name} : taille inattendue ({tmp.stat().st_size} au lieu de {size})")
            tmp.replace(p)
            got.append(name)
        if piper:
            import tarfile
            for v in PIPER_VOICES:
                dd = d / f"vits-piper-fr_FR-{v}"
                if (dd / f"fr_FR-{v}.onnx").exists():
                    continue
                url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fr_FR-{v}.tar.bz2"
                print(f"téléchargement de la voix {v}…")
                arch = d / f"vits-piper-fr_FR-{v}.tar.bz2"
                urllib.request.urlretrieve(url, arch)
                with tarfile.open(arch) as tf:
                    try:
                        tf.extractall(d, filter="data")
                    except TypeError:  # Python < 3.12
                        tf.extractall(d)
                arch.unlink()
                got.append(v)
        return f"{config.MODELS_DIR} : " + (f"téléchargés {got}" if got else "tous présents")

    def ingest(self):
        from pipeline import ingest
        inv = _call(ingest.run_ingest, force=self.a.force)
        n = len(inv.get("clips", inv)) if isinstance(inv, dict) else "?"
        return f"proxies dans {config.PROXY_DIR} ({n} entrées)"

    def analyze(self):
        from pipeline import analyze
        rep = _call(analyze.run_analysis, force=self.a.force)
        return f"rapport : {rep or config.ANALYSIS_DIR / 'report.md'}"

    def edl_stage(self):
        from edit import build_edl
        out = config.WORK_DIR / "edl.json"
        if hasattr(build_edl, "main"):
            rc = build_edl.main([])
            if rc not in (0, None):
                raise RuntimeError(f"build_edl a renvoyé {rc}")
        else:
            import importlib
            spec = importlib.import_module("edit.edl_spec")
            dec = json.loads((config.EDIT_DIR / "decisions.json").read_text(encoding="utf-8"))
            edl = build_edl.build(spec, dec, config.ANALYSIS_DIR, config.VOICE_DIR / "manifest.json")
            out.write_text(json.dumps(edl, indent=1, ensure_ascii=False), encoding="utf-8")
        self.edl = None
        edl = self.get_edl()
        warn = edl.get("warnings") or []
        d = edl.get("meta", {}).get("duration")
        if d is not None and not (config.DURATION_MIN <= float(d) <= config.DURATION_MAX):
            raise StageError(f"durée de l'EDL {d:.2f} s hors fenêtre {config.DURATION_MIN}–{config.DURATION_MAX} s")
        return f"{len(edl['shots'])} shots, {d} s, {len(warn)} avertissement(s) (voir {config.WORK_DIR / 'edl_report.md'})"

    def voices(self):
        self.models(piper=True)
        self._tool([sys.executable, str(ROOT / "tools" / "make_voices.py")])
        return f"voix dans {config.VOICE_DIR}"

    def title(self):
        self._tool([sys.executable, str(ROOT / "tools" / "render_title.py"), "all"])
        return f"titre dans {config.TITLE_DIR}"

    def render(self):
        self._review_gate()
        from pipeline import render
        out = render.render(self.get_edl(), self.q, jobs=self.a.jobs, force=self.a.force)
        self.outputs["video"] = out
        rep = json.loads((config.RENDER_DIR / f"video_{self.q}.json").read_text(encoding="utf-8"))
        extra = f", STUB : {rep['stubs']}" if rep.get("stubs") else ""
        return f"{out} ({rep['frames']} images, {len(rep.get('warnings', []))} avertissement(s){extra})"

    def mix(self):
        from pipeline.audio import mixer
        out = mixer.mix(self.get_edl(), self.q)
        self.outputs["mix"] = out
        return str(out)

    def deliver(self):
        from pipeline import deliver
        video = self.outputs.get("video") or config.RENDER_DIR / f"video_{self.q}.mp4"
        mix = self.outputs.get("mix") or config.RENDER_DIR / f"mix_{self.q}.wav"
        res = deliver.deliver(video, mix, self.q, edl=self.get_edl())
        self.outputs["mp4"] = res["mp4"]
        return f"{res['mp4']} ; couverture {res.get('cover')}"

    def qc(self):
        from pipeline import qc
        mp4 = self.outputs.get("mp4")
        if mp4 is None:
            mp4 = (config.OUT_DIR / config.FINAL_MP4 if self.q == "final"
                   else config.RENDER_DIR / "preview" / "CEST_QUI_LUI_preview.mp4")
        res = qc.qc(mp4, self.get_edl(), self.q, run_asr=not self.a.no_asr)
        self.outputs["qc"] = res
        fails = [c["name"] for c in res["checks"] if c["status"] == "fail"]
        if fails:
            raise QCFailed(f"contrôle qualité : {len(fails)} échec(s) ({', '.join(fails)}) — voir "
                           f"{config.RENDER_DIR / 'qc' / 'qc_report.md'}")
        return f"VALIDÉ — {config.RENDER_DIR / 'qc' / 'qc_report.md'}"

    # --- utilitaires -------------------------------------------------------------
    def get_edl(self) -> dict:
        if self.edl is None:
            from pipeline.render import load_edl
            self.edl = load_edl(self.a.edl)
        return self.edl

    def _tool(self, cmd):
        print("$ " + " ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            raise RuntimeError(f"{Path(cmd[1]).name} a échoué (code {rc})")

    def _review_gate(self):
        if self.q != "final" or self.a.ignore_review:
            return
        try:
            dec = json.loads((config.EDIT_DIR / "decisions.json").read_text(encoding="utf-8"))
        except Exception:
            dec = {}
        if not dec.get("_reviewed", False):
            raise StageError(
                "revue obligatoire non faite : regarde analysis/contact/*.jpg et analysis/report.md, complète "
                "edit/decisions.json puis mets \"_reviewed\": true (ou relance avec --ignore-review en "
                "connaissance de cause).")


class QCFailed(RuntimeError):
    pass


class StageError(RuntimeError):
    """Erreur bloquante attendue (message clair, sans trace Python)."""


def parse_stages(a) -> list[str]:
    if a.stage and a.stage != "all":
        st = [s.strip() for s in a.stage.split(",") if s.strip()]
        bad = [s for s in st if s not in STAGES]
        if bad:
            raise SystemExit(f"étape(s) inconnue(s) : {bad} ; choix : {', '.join(STAGES)}")
        return [s for s in STAGES if s in st]
    st = list(DEFAULT)
    if a.from_:
        if a.from_ not in st:
            raise SystemExit(f"--from : choisir parmi {', '.join(DEFAULT)}")
        st = st[st.index(a.from_):]
    if a.to:
        if a.to not in st:
            raise SystemExit(f"--to : choisir parmi {', '.join(st)}")
        st = st[: st.index(a.to) + 1]
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C'EST QUI LUI ? — chaîne complète")
    ap.add_argument("--quality", choices=["preview", "final"], default="preview")
    ap.add_argument("--stage", default="all", help=f"'all' ou liste séparée par des virgules parmi : {', '.join(STAGES)}")
    ap.add_argument("--from", dest="from_", default=None, help="reprendre la chaîne par défaut à partir de cette étape")
    ap.add_argument("--to", default=None, help="arrêter la chaîne par défaut après cette étape")
    ap.add_argument("--force", action="store_true", help="recalculer même ce qui est en cache (proxies, shots)")
    ap.add_argument("--jobs", type=int, default=None, help="processus de rendu en parallèle (défaut 2)")
    ap.add_argument("--edl", default=None, help="EDL à utiliser (défaut : WORK_DIR/edl.json)")
    ap.add_argument("--no-asr", action="store_true", help="contrôle qualité sans transcription")
    ap.add_argument("--ignore-review", action="store_true", help="autorise le final sans revue de decisions.json")
    a = ap.parse_args(argv)
    stages = parse_stages(a)

    config.ensure_dirs()
    logdir = config.WORK_DIR / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logp = logdir / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{a.quality}.log"
    logf = open(logp, "a", encoding="utf-8")
    out0, err0 = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(out0, logf), Tee(err0, logf)
    print(f"=== C'EST QUI LUI ? — {a.quality} — étapes : {', '.join(stages)}")
    print(f"    rushs : {config.RUSH_DIR}\n    travail : {config.WORK_DIR}\n    livrables : {config.OUT_DIR}")
    pipe = Pipeline(a)
    summary, rc = [], 0
    try:
        for st in stages:
            t0 = time.time()
            print(f"\n--- [{st}] ---")
            try:
                fn = pipe.edl_stage if st == "edl" else getattr(pipe, st)
                msg = fn()
                summary.append((st, "OK", time.time() - t0, msg))
            except QCFailed as e:
                summary.append((st, "À CORRIGER", time.time() - t0, str(e)))
                rc = 2
                break
            except StageError as e:
                print(f"ERREUR : {e}")
                summary.append((st, "ÉCHEC", time.time() - t0, str(e)))
                rc = 1
                break
            except Exception as e:
                traceback.print_exc()
                summary.append((st, "ÉCHEC", time.time() - t0, f"{type(e).__name__}: {e}"))
                rc = 1
                break
        done = {s for s, *_ in summary}
        for st in stages:
            if st not in done:
                summary.append((st, "non lancée", 0.0, ""))
        print("\n=== Résumé ===")
        for st, status, dt, msg in summary:
            print(f"  {st:8s} {status:11s} {dt:7.1f} s  {msg}")
        if rc == 0 and "qc" in stages and a.quality == "preview":
            print("\nÉTAPE DE REVUE OBLIGATOIRE avant le final : regarde "
                  f"{config.ANALYSIS_DIR / 'contact'}/*.jpg, {config.ANALYSIS_DIR / 'report.md'} et "
                  f"{config.RENDER_DIR / 'qc' / 'contact_preview.jpg'}, complète edit/decisions.json "
                  "(\"_reviewed\": true), puis : python run_all.py --quality final")
        if rc == 0 and a.quality == "final" and "deliver" in stages:
            print(f"\nLivrables : {config.OUT_DIR}")
        print(f"Journal : {logp}")
    finally:
        sys.stdout, sys.stderr = out0, err0
        logf.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
