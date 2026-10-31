"""Génère de FAUX rushs de test (synthétiques) pour valider la chaîne de bout en bout
sans les vraies vidéos. Ces fichiers ne servent qu'aux tests techniques : ils ne doivent
JAMAIS être utilisés pour un livrable.

Reproduit les pièges des vrais rushs iPhone : HEVC dans un .MOV, rotation −90° en métadonnée,
cadences 30 / 120 / ~16 fps, photos JPEG avec orientation EXIF 6.

Usage : python tools/make_fixtures.py [--out /home/user/edit_escape_tmp/fixtures/rushs]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import config  # noqa: E402

TESTDATA = config.WORK_DIR / "testdata"
SR = 48000

# clip: (durée, fps, image de visage, lignes [(t, texte, voix)], extras)
CLIPS = {
    "IMG_8336": (6, 30, None, [], {}),
    "IMG_8337": (6, 30, "portrait.jpg", [], {}),
    "IMG_8338": (9, 30, "lena.jpg", [(0.4, "Tu sais quand il vient ? Il m'a dit, il vient directement, lui, en voiture.", 1)], {"scream": 4.7}),
    "IMG_8339": (6, 30, "business-person.png", [], {}),
    "IMG_8340": (12, 120, "portrait.jpg", [(1.5, "Fais genre t'es possédé, regarde la caméra.", 0)], {}),
    "IMG_8341": (6, 120, "messi5.jpg", [], {"flashes": [1.8, 3.4]}),
    "IMG_8342": (6, 30, None, [], {}),
    "IMG_8343": (6, 30, None, [], {}),
    "IMG_8344": (9, 30, "lena.jpg", [(0.8, "Fais genre tu souris, vas-y.", 0)], {}),
    "IMG_8345": (6, 30, None, [], {}),
    "IMG_8346": (14, 30, None, [(0.3, "T'as entendu le bruit ?", 1), (3.2, "Je crois qu'il y a un gars là.", 0),
                                 (6.4, "C'est qui lui ?", 1), (10.6, "Oh putain !", 0)], {"person": (6.0, 12.5)}),
    "IMG_8347": (13, 16, "portrait.jpg", [(4.3, "Identification terminée.", 2)], {"scream": 11.0}),
    "IMG_8348": (12, 30, None, [(7.6, "C'est quoi ça ?", 1)], {}),
    "IMG_8349": (17, 30, "lena.jpg", [], {"scream": 15.2}),
    "IMG_8354": (10, 30, "portrait.jpg", [], {"piano": True}),
    "IMG_8355": (10, 30, "messi5.jpg", [(1.0, "Et là il me dit, attends, c'est pas fini !", 0), (4.0, "Ha ha ha ha ha !", 1)], {}),
    "IMG_8356": (10, 30, "lena.jpg", [(0.5, "Ha ha ha ha, non mais arrête !", 1), (4.0, "Ha ha ha ha ha ha !", 0)], {}),
    "IMG_8357": (10, 30, "business-person.png", [], {"person": (0.0, 10.0)}),
}
PHOTOS = ["IMG_8350", "IMG_8351", "IMG_8352", "IMG_8353"]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(" ".join(map(str, cmd)) + "\n" + r.stderr[-3000:])
    return r


def tts(text: str, voice_idx: int, out_wav: Path) -> np.ndarray:
    from piper import PiperVoice, SynthesisConfig
    voices = [
        ("vits-piper-fr_FR-upmc-medium/fr_FR-upmc-medium.onnx", 1),
        ("vits-piper-fr_FR-tom-medium/fr_FR-tom-medium.onnx", None),
        ("vits-piper-fr_FR-siwis-medium/fr_FR-siwis-medium.onnx", None),
    ]
    model, spk = voices[voice_idx]
    v = PiperVoice.load(str(config.MODELS_DIR / model))
    with wave.open(str(out_wav), "wb") as w:
        v.synthesize_wav(text, w, syn_config=SynthesisConfig(speaker_id=spk))
    with wave.open(str(out_wav)) as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    n = int(len(a) * SR / sr)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def make_audio(clip, dur, lines, extras, tmp: Path) -> Path:
    rng = np.random.default_rng(abs(hash(clip)) % 2**32)
    n = int(dur * SR)
    a = (rng.standard_normal(n) * 0.004).astype(np.float32)
    for i, (t, text, vi) in enumerate(lines):
        s = tts(text, vi, tmp / f"{clip}_{i}.wav") * 0.5
        k = int(t * SR)
        a[k:k + len(s)] += s[: max(0, n - k)]
    if "scream" in extras:
        k = int(extras["scream"] * SR)
        m = int(0.9 * SR)
        tt = np.arange(m) / SR
        burst = rng.standard_normal(m) * np.exp(-tt * 2) * 0.6 + 0.3 * np.sin(2 * np.pi * (520 + 60 * np.sin(30 * tt)) * tt)
        a[k:k + m] += burst[: max(0, n - k)].astype(np.float32)
    if extras.get("piano"):
        for j, (t, f) in enumerate([(0.5, 220), (1.2, 261.6), (1.9, 329.6), (2.6, 311.1), (3.6, 220), (4.4, 196), (5.2, 207.7)]):
            k = int(t * SR)
            m = int(1.6 * SR)
            tt = np.arange(m) / SR
            note = sum(np.sin(2 * np.pi * f * h * tt) / h for h in (1, 2, 3, 4)) * np.exp(-tt * 3) * 0.25
            a[k:k + m] += note[: max(0, n - k)].astype(np.float32)
    a = np.clip(a, -1, 1)
    st = np.stack([a, a], 1)
    p = tmp / f"{clip}.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())
    return p


def make_clip(clip, out_dir: Path, tmp: Path):
    dur, fps, face, lines, extras = CLIPS[clip]
    wav = make_audio(clip, dur, lines, extras, tmp)
    # Portrait 540x960 content, then transposed to landscape + rotate metadata (like iPhone).
    inputs = ["-f", "lavfi", "-i", f"color=c=0x0d1117:s=540x960:r={fps}:d={dur}"]
    filt = []
    idx = 1
    base = "[0:v]"
    if extras.get("person"):
        inputs += ["-stream_loop", "-1", "-i", str(TESTDATA / "vtest.avi")]
        filt.append(f"[{idx}:v]scale=-2:960,crop=540:960,eq=brightness=-0.15,setpts=PTS-STARTPTS[bg]")
        filt.append(f"{base}[bg]overlay=shortest=1[b0]")
        base = "[b0]"
        idx += 1
    if face:
        inputs += ["-loop", "1", "-i", str(TESTDATA / face)]
        filt.append(f"[{idx}:v]scale=360:-2[f]")
        filt.append(f"{base}[f]overlay=x='90+40*sin(t*0.9)':y='300+30*cos(t*0.7)':shortest=1[b1]")
        base = "[b1]"
        idx += 1
    if extras.get("flashes"):
        en = "+".join(f"between(t\\,{f - 0.05:.2f}\\,{f + 0.12:.2f})" for f in extras["flashes"])
        filt.append(f"{base}eq=brightness=-0.35[b2a];[b2a]eq=brightness=0.6:enable='{en}'[b2]")
        base = "[b2]"
    filt.append(f"{base}drawtext=text='{clip}  t=%{{pts\\:flt}}':x=20:y=40:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.5,"
                f"drawtext=text='HAUT':x=(w-tw)/2:y=8:fontsize=24:fontcolor=yellow,noise=alls=12:allf=t,"
                f"format=yuv420p,transpose=2[v]")
    out = out_dir / f"{clip}.MOV"
    cmd = ["ffmpeg", "-y", "-v", "error", *inputs, "-i", str(wav), "-filter_complex", ";".join(filt),
           "-map", "[v]", "-map", f"{idx}:a", "-t", str(dur), "-c:v", "libx265", "-preset", "ultrafast", "-crf", "28",
           "-tag:v", "hvc1", "-r", str(fps), "-c:a", "aac", "-b:a", "128k", "-metadata:s:v:0", "rotate=90", str(out)]
    if fps == 16:  # variable frame rate-ish : drop random frames
        cmd[cmd.index("-r") + 1] = "16"
    run(cmd)
    from mov_rotate import patch  # iPhone-style display matrix (rotation -90)
    patch(str(out))
    return out


def make_photo(photo, out_dir: Path, i: int):
    from PIL import Image, ImageDraw
    W, H = 3024, 4032  # displayed portrait
    img = Image.new("RGB", (W, H), (22, 18, 16))
    d = ImageDraw.Draw(img)
    d.rectangle([150, 400, 900, 2600], fill=(40, 28, 20))       # armoire
    d.ellipse([2000, 500, 2700, 1200], fill=(60, 50, 30))         # horloge
    faces = ["portrait.jpg", "lena.jpg", "business-person.png", "messi5.jpg"]
    for k, f in enumerate(faces):
        fi = Image.open(TESTDATA / f).convert("RGB")
        fi.thumbnail((700, 900))
        img.paste(fi, (60 + k * 740, 1900 + (k % 2) * 120))
    d.text((80, 80), f"{photo} (photo test) HAUT", fill=(255, 255, 0))
    stored = img.rotate(90, expand=True)  # stored landscape; EXIF 6 => rotate 270 (CW 90) for display
    exif = Image.Exif()
    exif[0x0112] = 6
    stored.save(out_dir / f"{photo}.jpeg", quality=88, exif=exif)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.WORK_DIR / "fixtures" / "rushs"))
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for clip in CLIPS:
            if args.only and clip not in args.only:
                continue
            p = make_clip(clip, out, Path(td))
            print("ok", p)
    for i, ph in enumerate(PHOTOS):
        if args.only and ph not in args.only:
            continue
        make_photo(ph, out, i)
        print("ok", ph)
    (out / "FIXTURES_NOT_REAL.txt").write_text("Faux rushs synthétiques de test. Ne jamais utiliser pour un livrable.\n")
    print(json.dumps({"out": str(out)}))


if __name__ == "__main__":
    main()
