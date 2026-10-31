"""Rend la bibliothèque SFX (aperçus versionnés) + planche de spectrogrammes de contrôle.

    python tools/make_sfx_library.py [--only impact_big braam] [--real-piano] [--out DIR]

Sortie : assets/audio/sfx/<nom>.wav (48 kHz, 16 bits, ≤ 6 s), _spectrograms.png, _stats.json.
Le mixer re-synthétise chaque son à la bonne durée : ces fichiers sont des aperçus.
Par défaut, piano_note et music_box_leitmotif utilisent le motif de SECOURS (pas le vrai piano) :
les aperçus versionnés ne dépendent ainsi d'aucun rush. --real-piano : utilise le vrai piano (8354).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pipeline.audio import dsp, synth  # noqa: E402

MAX_DUR = 6.0
PREVIEW_DUR = {  # lits longs : aperçu de 6 s
    "drone_low": 6.0, "drone_tension": 6.0, "wind": 6.0, "warm_pad": 6.0, "clock_tick": 6.0,
    "music_box_leitmotif": 6.0, "heartbeat": 6.0, "piano_note": 6.0, "impact_big": 5.0,
}
PREVIEW_PARAMS = {"heartbeat": {"bpm_start": 62, "bpm_end": 118}, "tinnitus": {"ramp_to_db": 0.0}}


def analyse(x: np.ndarray) -> dict:
    m = dsp.to_mono(x).astype(np.float64)
    d = np.abs(np.diff(m))
    loc = np.sqrt(np.convolve(m ** 2, np.ones(480) / 480, mode="same"))[1:] + 1e-4
    click = float(np.max(d / loc)) if len(d) else 0.0
    _, M = dsp.momentary(x)
    spec = np.abs(np.fft.rfft(m * np.hanning(len(m)))) if len(m) > 16 else np.zeros(2)
    fr = np.fft.rfftfreq(len(m), 1 / dsp.SR)
    cen = float(np.sum(fr * spec) / (np.sum(spec) + 1e-12))
    hf = float(np.sum(spec[fr > 16000] ** 2) / (np.sum(spec ** 2) + 1e-20))
    return {"dur": round(len(x) / dsp.SR, 3), "peak_db": round(dsp.peak_db(x), 2),
            "true_peak_db": round(dsp.true_peak_db(x), 2), "rms_db": round(dsp.rms_db(x), 2),
            "max_momentary_lufs": round(float(M.max()), 2), "dc": round(float(np.abs(x.mean(axis=0)).max()), 6),
            "max_jump_over_local_rms": round(click, 1), "centroid_hz": round(cen, 0), "hf_ratio_16k": round(hf, 5),
            "end_abs": round(float(np.abs(x[-1]).max()), 5) if len(x) else 0.0}


def spectrogram_sheet(items: list[tuple[str, np.ndarray, dict]], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal as sig

    cols = 5
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.9), facecolor="#111")
    for ax in np.ravel(axes):
        ax.set_facecolor("#111")
        ax.axis("off")
    for ax, (name, x, st) in zip(np.ravel(axes), items):
        ax.axis("on")
        m = dsp.to_mono(x).astype(np.float64)
        f, t, S = sig.spectrogram(m, fs=dsp.SR, nperseg=2048, noverlap=1536, scaling="spectrum")
        S = 10 * np.log10(S + 1e-14)
        S -= S.max()
        f[0] = 10
        ax.pcolormesh(t, f, S, vmin=-90, vmax=0, cmap="magma", shading="auto")
        ax.set_yscale("log")
        ax.set_ylim(20, 24000)
        ax.set_yticks([30, 100, 300, 1000, 3000, 10000])
        ax.set_yticklabels(["30", "100", "300", "1k", "3k", "10k"], fontsize=6, color="#bbb")
        ax.tick_params(axis="x", labelsize=6, colors="#bbb")
        ax2 = ax.twinx()
        tt = np.arange(len(m)) / dsp.SR
        step = max(1, len(m) // 1500)
        ax2.plot(tt[::step], m[::step], color="#4fd1ff", lw=0.35, alpha=0.55)
        ax2.set_ylim(-1.05, 1.05)
        ax2.axis("off")
        ax.set_title(f"{name}  {st['dur']}s  pk {st['peak_db']}  Mmax {st['max_momentary_lufs']}",
                     fontsize=7.5, color="w")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=85, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="noms à rendre (défaut : tous les sons exigés)")
    ap.add_argument("--extras", action="store_true", help="inclure les variantes piano_* (vrai piano)")
    ap.add_argument("--real-piano", action="store_true", help="aperçus piano depuis le vrai 8354")
    ap.add_argument("--out", default=str(config.SFX_DIR))
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.only or (synth.names() if a.extras else list(synth.REQUIRED))
    items, stats = [], {}
    for name in names:
        t0 = time.time()
        params = dict(PREVIEW_PARAMS.get(name, {}))
        if name in ("piano_note", "music_box_leitmotif") or name.startswith("piano_"):
            params["source"] = "real" if a.real_piano else "fallback"
        x = synth.render(name, PREVIEW_DUR.get(name), seed=0, **params)
        x = x[: int(MAX_DUR * dsp.SR)]
        dsp.save(out / f"{name}.wav", x, subtype="PCM_16")
        st = analyse(x)
        stats[name] = st
        items.append((name, x, st))
        print(f"{name:22s} {st}  ({time.time() - t0:.1f} s)")
    if a.only:  # fusion avec les stats existantes
        old = out / "_stats.json"
        if old.exists():
            stats = {**json.loads(old.read_text(encoding="utf-8")), **stats}
    (out / "_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    if not a.no_png:
        if a.only:  # planche complète à partir des fichiers présents
            items = []
            for name in (synth.REQUIRED + [n for n in synth.names() if n not in synth.REQUIRED]):
                p = out / f"{name}.wav"
                if p.exists():
                    x = dsp.load(p)
                    items.append((name, x, analyse(x)))
        spectrogram_sheet(items, out / "_spectrograms.png")
        print(f"planche : {out / '_spectrograms.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
