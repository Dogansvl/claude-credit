"""Tests du rendu, de la livraison et du contrôle qualité (sans les vrais rushs).

    python -m unittest tests.test_render -v        (ou : python tests/test_render.py)

Les tests « bout en bout » fabriquent un petit proxy synthétique avec ffmpeg dans un dossier temporaire :
ils ne dépendent ni des rushs, ni des faux rushs de fixtures.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from pipeline import qc as Q  # noqa: E402
from pipeline import render as R  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _edl(shots, duration=None):
    t = 0.0
    for s in shots:
        s.setdefault("start", round(t, 6))
        t = s["start"] + s["dur"]
    return {"meta": {"fps": 30, "duration": duration if duration is not None else t}, "shots": shots,
            "subtitles": [], "cues": [], "music": [], "silence": [], "voices": [], "warnings": []}


class TestTiming(unittest.TestCase):
    def test_frame_plan_partition(self):
        edl = _edl([{"id": "a", "kind": "black", "dur": 1.234}, {"id": "b", "kind": "black", "dur": 0.95},
                    {"id": "c", "kind": "black", "dur": 2.017}])
        plan = R.frame_plan(edl)
        self.assertEqual(plan[0][0], 0)
        for (_, f1), (g0, _) in zip(plan[:-1], plan[1:]):
            self.assertEqual(f1, g0)  # contigu
        self.assertEqual(plan[-1][1], R.fidx(edl["meta"]["duration"]))
        self.assertEqual(sum(b - a for a, b in plan), R.fidx(1.234 + 0.95 + 2.017))

    def test_source_frames_constant_and_freeze(self):
        s = {"kind": "clip", "src_in": 2.0, "dur": 1.0, "freeze_after": 0.5}
        js = R.source_frames(s, R.shot_local_times(0.0, 0, 30))
        self.assertEqual(js[0], 60)
        self.assertEqual(js[10], 70)
        self.assertTrue(np.all(js[15:] == js[15]))  # gelé après 0,5 s
        self.assertEqual(js[15], 75)

    def test_speed_ramp_slows_down(self):
        s = {"kind": "clip", "src_in": 0.0, "dur": 2.0, "fx": [{"type": "speed_ramp", "t0": 1.0, "t1": 2.0, "to": 0.5}]}
        js = R.source_frames(s, R.shot_local_times(0.0, 0, 60))
        self.assertTrue(np.all(np.diff(js) >= 0))
        self.assertEqual(js[29], 29)          # vitesse 1 avant la rampe
        self.assertLess(js[-1], 59)            # ralenti ensuite
        self.assertGreater(js[-1], 59 * 0.7)

    def test_speed_pair(self):
        s = {"kind": "clip", "src_in": 0.0, "dur": 1.0, "speed": [1.0, 0.0]}
        t = R.source_times(s, np.array([1.0]))
        self.assertAlmostEqual(float(t[0]), 0.5, places=2)

    def test_zoom_punch(self):
        cam = {"zoom": [1.0, 1.4], "punch": 0.2}
        z = [R.zoom_at(cam, t, 1.0) for t in np.linspace(0, 1, 31)]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(z, z[1:])))
        self.assertAlmostEqual(z[-1], 1.4, places=6)
        self.assertGreater(R.zoom_at(cam, 0.2, 1.0), 1.0 + 0.8 * 0.4)  # punch rapide

    def test_active_window_frames(self):
        fx = {"t0": 0.0, "t1": 0.067}
        frames = [k for k in range(10) if R._active(fx, k / 30, 1.0)]
        self.assertEqual(frames, [0, 1])


class TestPlanning(unittest.TestCase):
    def test_flash_guard(self):
        shots = [{"id": "a", "kind": "black", "dur": 1.0,
                  "fx": [{"type": "negative", "t0": 0.0, "t1": 0.05}, {"type": "subliminal", "t0": 0.2, "frames": 2},
                         {"type": "negative", "t0": 0.4, "t1": 0.45}, {"type": "subliminal", "t0": 0.6, "frames": 2},
                         {"type": "glitch", "t0": 0.7, "t1": 0.8}]},
                 {"id": "b", "kind": "black", "dur": 1.0, "fx": [{"type": "negative", "t0": 0.5, "t1": 0.55}]}]
        skips, rep = R.plan_flashes(_edl(shots))
        self.assertEqual(skips, {"a": [3]})
        self.assertEqual(sum(r["action"] != "gardé" for r in rep), 1)

    def test_prev_last(self):
        shots = [{"id": "a", "kind": "clip", "clip": "IMG_X", "src_in": 1.0, "dur": 1.0},
                 {"id": "b", "kind": "freeze", "freeze_src": "prev_last", "dur": 0.5},
                 {"id": "c", "kind": "freeze", "freeze_src": "prev_last", "dur": 0.5}]
        edl = _edl(shots)
        plan = R.frame_plan(edl)
        fb = R.resolve_freeze_src(edl, plan, 1)
        self.assertEqual(fb["clip"], "IMG_X")
        self.assertAlmostEqual(fb["t"], 1.0 + 29 / 30, places=6)
        self.assertEqual(R.resolve_freeze_src(edl, plan, 2), fb)  # gel de gel

    def test_camera_mapping_matches_warp(self):
        img = np.zeros((R.SRC_H, R.SRC_W, 3), np.float32)
        x, y = 700.0, 500.0
        img[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3] = 1.0
        cam = R.Camera(540, 960, 1.3, 600, 700)
        out = cam.apply(img)
        ys, xs = np.nonzero(out[..., 0] > 0.5)
        px, py = cam.pt(x, y)
        self.assertLess(abs(xs.mean() - px), 1.5)
        self.assertLess(abs(ys.mean() - py), 1.5)

    def test_main_faces(self):
        F = R._StubFace
        faces = [F([0, 0, 100, 120]), F([300, 300, 340, 345]), F([500, 0, 580, 100])]
        m = R.main_faces(faces)
        self.assertEqual(len(m), 2)


class TestQC(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(Q.norm_text("T'es possédé, regarde la caméra !"), "t es possede regarde la camera")

    def test_flash_count(self):
        y = np.full(90, 0.05, np.float32)
        for k in (5, 11, 17, 23, 29):  # 5 flashs en 1 s
            y[k] = 0.6
        best, _, _ = Q.count_flashes(y)
        self.assertGreaterEqual(best, 4)
        y2 = np.full(90, 0.05, np.float32)
        y2[10] = y2[40] = 0.6
        self.assertLessEqual(Q.count_flashes(y2)[0], 1)

    def test_names(self):
        edl = _edl([{"id": "n", "kind": "black", "dur": 1.0,
                     "overlays": [{"type": "name", "name": "ARSÈNE", "show": True}]}])
        edl["subtitles"] = [{"start": 0, "end": 1, "text": "Arsene est là", "kind": "narrator"}]
        checks, _ = Q.check_names(edl)
        self.assertEqual(checks[0]["status"], "fail")
        edl["subtitles"] = [{"start": 0, "end": 1, "text": "À 22h12, Dogan s'est mis au piano.", "kind": "narrator"}]
        self.assertEqual(Q.check_names(edl)[0][0]["status"], "pass")

    def test_planned_dark(self):
        edl = _edl([{"id": "a", "kind": "clip", "dur": 2.0, "fx": [{"type": "fade_to_black", "t0": 1.0, "t1": 2.0}]},
                    {"id": "b", "kind": "black", "dur": 0.5}, {"id": "c", "kind": "card", "dur": 1.0}])
        pd = Q.planned_dark(edl)
        self.assertEqual(len(pd), 1)
        self.assertAlmostEqual(pd[0][0], 1.4, places=6)
        self.assertAlmostEqual(pd[0][1], 3.5, places=6)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg absent")
class TestEndToEnd(unittest.TestCase):
    """Petit rendu preview complet sur un proxy synthétique + mux + contrôles de format."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cql_test_"))
        cls.saved = {k: getattr(config, k) for k in ("WORK_DIR", "PROXY_DIR", "ANALYSIS_DIR", "CACHE_DIR",
                                                     "RENDER_DIR", "OUT_DIR")}
        config.WORK_DIR = cls.tmp
        config.PROXY_DIR = cls.tmp / "proxies"
        config.ANALYSIS_DIR = cls.tmp / "analysis"
        config.CACHE_DIR = cls.tmp / "cache"
        config.RENDER_DIR = cls.tmp / "render"
        config.OUT_DIR = cls.tmp / "out"
        config.ensure_dirs()
        # proxy synthétique 1080x1920 30 i/s, 3 s, avec un compteur d'images
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30:duration=3",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=48000",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                        str(config.PROXY_DIR / "IMG_9001.mp4")], check=True)

    @classmethod
    def tearDownClass(cls):
        for k, v in cls.saved.items():
            setattr(config, k, v)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_render_mux(self):
        shots = [{"id": "A", "kind": "clip", "clip": "IMG_9001", "src_in": 0.5, "dur": 0.73,
                  "camera": {"zoom": [1.0, 1.1], "center": [0.5, 0.5]}, "grade": {"preset": "cold"},
                  "overlays": [{"type": "timecode", "text": "20:48", "date": "30.10.2026"}]},
                 {"id": "B", "kind": "freeze", "freeze_src": "prev_last", "dur": 0.41,
                  "fx": [{"type": "negative", "t0": 0.0, "t1": 0.067}]},
                 {"id": "C", "kind": "black", "dur": 0.3},
                 {"id": "D", "kind": "card", "dur": 0.6, "overlays": [{"type": "card", "text": "TEST", "t0": 0.0}]}]
        edl = _edl(shots)
        edl["subtitles"] = [{"start": 0.1, "end": 0.6, "text": "C'est qui lui ?", "kind": "dialogue"}]
        out = R.render(edl, "preview", jobs=1, log=lambda *a: None)
        n = R.fidx(edl["meta"]["duration"])
        self.assertEqual(R.count_frames(out), n)
        self.assertEqual(R.video_size(out), R.SIZES["preview"])
        # gel = dernière image de A (le compteur de testsrc2 ne bouge plus)
        f_last_a = R.read_single_frame(out, R.frame_plan(edl)[0][1] - 1, R.SIZES["preview"]).astype(float)
        f_freeze = R.read_single_frame(out, R.frame_plan(edl)[1][0] + 5, R.SIZES["preview"]).astype(float)
        self.assertLess(np.abs(f_last_a.mean() - f_freeze.mean()), 25)
        # le noir est noir
        f_black = R.read_single_frame(out, R.frame_plan(edl)[2][0] + 2, R.SIZES["preview"])
        self.assertLess(f_black.mean(), 2.0)
        # cache : second rendu instantané, même fichier
        out2 = R.render(edl, "preview", jobs=1, log=lambda *a: None)
        self.assertEqual(R.count_frames(out2), n)
        # mux
        from pipeline import deliver as D
        sr = 48000
        t = np.arange(int(sr * (n / 30 + 0.3))) / sr
        wav = config.RENDER_DIR / "mix_preview.wav"
        import soundfile as sf
        sf.write(str(wav), np.stack([0.1 * np.sin(2 * math.pi * 220 * t)] * 2, 1).astype(np.float32), sr)
        mp4 = config.RENDER_DIR / "preview" / "t.mp4"
        res = D.mux(out, wav, mp4, "preview", log=lambda *a: None)
        self.assertEqual(res["frames"], n)
        checks, _ = Q.check_format(mp4, "preview", n)
        bad = {c["name"]: c for c in checks if c["status"] == "fail" and c["name"] != "durée"}
        self.assertEqual(bad, {})
        boxes = [b[0] for b in Q.mp4_top_boxes(mp4)]
        self.assertLess(boxes.index("moov"), boxes.index("mdat"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
