"""Tests de robustesse (sans rushs) : HDR→SDR, liste noire par mots entiers, noirs vs plans sombres,
garde-fous du montage (rejeu, shots supprimées, chuchotements non confirmés, rush manquant), replis du rendu.

    python -m unittest tests.test_robustness -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from pipeline import analyze, ingest, qc  # noqa: E402
from pipeline import render as R  # noqa: E402
from edit import build_edl as B  # noqa: E402


class TestHDR(unittest.TestCase):
    def test_reference_levels(self):
        f = lambda v: ingest.hdr_to_sdr(np.array([v, v, v]), "arib-std-b67")[0] * 255
        self.assertGreater(f(0.75), 232)          # blanc de référence HLG (203 nits) → quasi blanc SDR
        self.assertLess(f(0.75), 248)
        self.assertAlmostEqual(f(1.0), 255, delta=1)
        self.assertAlmostEqual(f(0.0), 0, delta=0.5)
        self.assertTrue(95 < f(0.38) < 125)       # gris 18 % (BT.2408)

    def test_monotonic_and_hue(self):
        ramp = np.linspace(0, 1, 200)
        g = ingest.hdr_to_sdr(np.stack([ramp] * 3, 1), "arib-std-b67")[:, 0]
        self.assertTrue(np.all(np.diff(g) >= -1e-9))
        red = ingest.hdr_to_sdr(np.array([0.9, 0.3, 0.3]), "arib-std-b67")
        self.assertGreater(red[0], red[1] + 0.3)  # un rouge reste rouge (pas de virage)

    def test_lut_file(self):
        with tempfile.TemporaryDirectory() as td:
            old = config.CACHE_DIR
            config.CACHE_DIR = Path(td)
            try:
                p = ingest.hdr_lut("arib-std-b67")
                head = p.read_text(encoding="ascii").splitlines()[:5]
            finally:
                config.CACHE_DIR = old
        self.assertIn(f"LUT_3D_SIZE {ingest.HDR_LUT_SIZE}", head)
        self.assertNotIn(":", p.name)              # nom seul dans le graphe de filtres (Windows)


class TestBlacklist(unittest.TestCase):
    def hits(self, text):
        return [ph for _, _, ph in analyze.match_blacklist(analyze.norm_text(text).split())]

    def test_whole_words(self):
        self.assertIn("fais genre", self.hits("Fais genre t'es possédé"))
        self.assertIn("souris", self.hits("allez souris"))
        self.assertEqual(self.hits("un grand sourire"), [])
        self.assertEqual(self.hits("c'est qui lui"), [])
        self.assertEqual(self.hits("c'est quoi ça"), [])
        self.assertEqual(self.hits("oh putain"), [])
        self.assertIn("coupe", self.hits("coupez"))

    def test_word_only_scope(self):
        words = [{"w": w, "start": i * 0.3, "end": i * 0.3 + 0.25} for i, w in
                 enumerate("et là il me dit attends c'est pas fini".split())]
        h = analyze.detect_instructions_in_words(words)
        self.assertEqual(len(h), 1)
        self.assertLess(h[0]["end"] - h[0]["start"], 0.6)   # le mot seul, pas tout l'énoncé
        words2 = [{"w": w, "start": i * 0.3, "end": i * 0.3 + 0.25} for i, w in
                  enumerate("fais genre t'es possédé regarde".split())]
        h2 = analyze.detect_instructions_in_words(words2)
        self.assertGreater(h2[0]["end"] - h2[0]["start"], 1.2)  # vraie consigne : tout l'énoncé


class TestDarkVsHole(unittest.TestCase):
    def setUp(self):
        n = 120
        rng = np.random.default_rng(0)
        Y = np.zeros((n, 96, 54), np.float32)
        Y[:60] = 0.004 + 0.04 * (rng.random((60, 96, 54)) > 0.95)   # plan de nuit très sombre (contenu épars)
        Y[60:62] = 0.0                                                # trou de 2 images
        Y[62:] = 0.004 + 0.04 * (rng.random((58, 96, 54)) > 0.95)
        self.stats = {"n": n, "Y": Y, "Ylin": Y.reshape(n, -1).mean(1)}
        self.edl = {"shots": [{"id": "A", "kind": "clip", "clip": "X", "start": 0.0, "dur": 4.0, "src_in": 0.0}],
                    "subtitles": []}

    def test_classification(self):
        cl = qc.classify_dark(self.edl, self.stats, [(0.0, 4.0)], [])
        self.assertEqual(len(cl["holes"]), 1)
        self.assertEqual(cl["holes"][0]["t"], [2.0, 2.07])
        self.assertTrue(cl["dark"])

    def test_planned_black_is_ok(self):
        edl = {"shots": [{"id": "A", "kind": "clip", "start": 0.0, "dur": 2.0},
                         {"id": "B", "kind": "black", "start": 2.0, "dur": 0.1},
                         {"id": "C", "kind": "clip", "start": 2.1, "dur": 1.9}], "subtitles": []}
        cl = qc.classify_dark(edl, self.stats, [], qc.planned_dark(edl))
        self.assertEqual(cl["holes"], [])

    def test_text_on_black_is_still_a_hole(self):
        self.stats["Y"][60:66] = 0.0
        self.stats["Y"][60:66, 62:70, 10:44] = 0.9                    # un sous-titre sur du noir
        edl = dict(self.edl, subtitles=[{"start": 1.9, "end": 2.4, "text": "x", "kind": "narrator"}])
        cl = qc.classify_dark(edl, self.stats, [], [])
        self.assertTrue(cl["holes"])


class _Spec(types.SimpleNamespace):
    pass


def _spec(shots, music=()):
    return _Spec(SHOTS=shots, MUSIC=list(music), TARGET_DURATION=90.0)


class TestBuilder(unittest.TestCase):
    def build(self, shots, dec=None, music=()):
        return B.build(_spec(shots, music), dec or {}, analysis_dir="", voice_manifest={"items": {}}, write=False,
                       title_meta={"reveal": 1.0})

    def test_drop_unconfirmed(self):
        shots = [dict(id="A", kind="black", dur=1.0),
                 dict(id="B", kind="black", dur=1.0, drop_if_unconfirmed="name_x"),
                 dict(id="C", kind="black", dur=1.0)]
        edl = self.build(shots, {"name_x": {"candidates": [{"clip": "IMG_1", "confirmed": False}]}},
                         music=[dict(kind="drone_low", start=B.importlib.import_module("edit.edl_spec").ref("B"),
                                     end=B.importlib.import_module("edit.edl_spec").ref("C", "end"))])
        self.assertEqual([s["id"] for s in edl["shots"]], ["A", "C"])
        self.assertAlmostEqual(edl["meta"]["duration"], 2.0)
        self.assertAlmostEqual(edl["music"][0]["t0"], 1.0)     # référence à la shot supprimée → shot suivante
        edl2 = self.build(shots, {"name_x": {"candidates": [{"clip": "IMG_1", "confirmed": True}]}})
        self.assertEqual(len(edl2["shots"]), 3)

    def test_unconfirmed_whisper_has_no_subtitle(self):
        shots = [dict(id="A", kind="black", dur=2.0, voice=[dict(id="decision:possess_1.whisper", t=0.1)]),
                 dict(id="B", kind="black", dur=2.0, voice=[dict(id="decision:possess_2.whisper", t=0.1)])]
        dec = {"possess_1": {"whisper": "E_DOGAN", "confirmed": False},
               "possess_2": {"whisper": "E_GABRIEL", "confirmed": True}}
        edl = self.build(shots, dec)
        self.assertEqual([v["id"] for v in edl["voices"]], ["E_DOGAN", "E_GABRIEL"])   # l'audio reste
        self.assertEqual({s.get("voice") for s in edl["subtitles"]}, {"E_GABRIEL"})

    def test_replay_guard(self):
        shots = [dict(id="K", kind="clip", clip="IMG_X", src_in=2.0, src_out=4.0),
                 dict(id="L", kind="clip", clip="IMG_X", src_in=3.6, dur=0.9)]
        edl = self.build(shots)
        k, l_ = edl["shots"]
        self.assertGreaterEqual(l_["src_in"], k["src_out"] - 1e-6)   # le cri ne rejoue pas la fin de K
        self.assertTrue(any("rejouait" in w for w in edl["warnings"]))

    def test_missing_photo_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td) / "rushs"
            rd.mkdir()
            (rd / "IMG_8351.jpeg").write_bytes(b"x")
            old_r, old_w = config.RUSH_DIR, config.WORK_DIR
            config.RUSH_DIR, config.PROXY_DIR = rd, Path(td) / "proxies"
            config.ANALYSIS_DIR = Path(td) / "analysis"
            try:
                edl = self.build([dict(id="P", kind="photo", photo="IMG_8350", dur=1.0),
                                  dict(id="C", kind="clip", clip="IMG_8346", src_in=1.0, dur=1.0)])
            finally:
                config.RUSH_DIR = old_r
                config.PROXY_DIR = old_w / "proxies"
                config.ANALYSIS_DIR = old_w / "analysis"
        self.assertEqual(edl["shots"][0]["photo"], "IMG_8351")
        self.assertEqual(edl["shots"][1]["kind"], "black")
        self.assertTrue(any("RUSH MANQUANT" in w for w in edl["warnings"]))


class TestRenderFallbacks(unittest.TestCase):
    def renderer(self, shot, freeze=None):
        job = {"shot": shot, "quality": "preview", "n": 3, "f0": 0, "subs": [], "freeze": freeze, "out": "x"}
        return R.ShotRenderer(job)

    def test_privacy_secondary_persons_and_zones(self):
        sh = {"id": "S", "kind": "clip", "clip": "IMG_T", "start": 0.0, "dur": 1.0, "fx": [{"type": "privacy_blur"}]}
        r = self.renderer(sh)
        r._cur_src = ("IMG_T", 1.0)
        r.analysis_persons = lambda clip=None, t_src=None: [[300, 400, 800, 1800], [900, 900, 1000, 1150]]
        main = R._StubFace([450, 450, 650, 700])
        old = R._decisions
        R._decisions = lambda: {"privacy_zones": {"zones": {"IMG_T": [[0.0, 0.0, 0.1, 0.1, 0.5, 2.0],
                                                                         [0.5, 0.5, 0.6, 0.6, 5.0, 6.0]]}}}
        try:
            boxes = r.privacy_regions([main], None)
        finally:
            R._decisions = old
        self.assertIn([900, 900, 1000, 1150], boxes)               # personne secondaire floutée
        self.assertNotIn([300, 400, 800, 1800], boxes)              # le sujet principal jamais
        self.assertEqual(len(boxes), 2)                             # + zone manuelle active (pas l'inactive)
        img = np.full((960, 540, 3), 0.5, np.float32)
        out = r.blur_regions(img.copy(), [[100, 100, 300, 400]])
        self.assertLess(out[250, 200].mean(), 0.4)                  # assombri au centre
        self.assertAlmostEqual(float(out[10, 10].mean()), 0.5, places=4)

    def test_hud_manual_boxes(self):
        sh = {"id": "H", "kind": "clip", "clip": "IMG_8347", "start": 0.0, "dur": 1.0}
        r = self.renderer(sh)
        old = R._decisions
        R._decisions = lambda: {"hud_boxes_8347": {"boxes": [[0.6, 0.2, 0.7, 0.3], [0.1, 0.2, 0.2, 0.3]]}}
        try:
            fs = r.hud_faces([R._StubFace([100, 100, 200, 220])], 4)
        finally:
            R._decisions = old
        self.assertEqual(len(fs), 2)
        self.assertLess(fs[0].bbox[0], fs[1].bbox[0])               # gauche → droite

    def test_camera_interp_choice(self):
        cam = R.Camera(1080, 1920, 1.05, 540, 960, lanczos=True)
        img = np.zeros((1920, 1080, 3), np.uint8)
        self.assertEqual(cam.apply(img).shape, (1920, 1080, 3))


class TestConfig(unittest.TestCase):
    def test_out_dir_not_in_repo_for_cloud_rushes(self):
        import subprocess
        env = dict(os.environ, ESCAPE_RUSH_DIR=str(ROOT / "rushs"), ESCAPE_WORK_DIR="/tmp/escape_w")
        env.pop("ESCAPE_OUT_DIR", None)
        out = subprocess.run([sys.executable, "-c", "import config; print(config.OUT_DIR)"], cwd=str(ROOT),
                             env=env, capture_output=True, text=True, encoding="utf-8").stdout.strip()
        self.assertFalse(Path(out).resolve().is_relative_to(ROOT.resolve()), out)

    def test_cloud_rush_extensions(self):
        with tempfile.TemporaryDirectory() as td:
            old = config.RUSH_DIR
            config.RUSH_DIR = Path(td)
            (Path(td) / "IMG_8346.mp4").write_bytes(b"x")
            try:
                self.assertEqual(config.rush_path("IMG_8346").suffix, ".mp4")
            finally:
                config.RUSH_DIR = old


if __name__ == "__main__":
    unittest.main()
