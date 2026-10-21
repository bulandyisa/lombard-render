"""Финиш 3D-версии «Ломбарда» (10 с): кадры Blender → дождь/пыль, свечение (bloom), цвет фильма, виньетка, зерно,
звук из библиотеки SFX проекта → lombard/render/Ломбард_3D_10с.mp4.
Запуск: .venv/bin/python lombard/3d/post3d.py"""
import json
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "lombard/audio"))
import motion as M  # noqa: E402
import mix as AU  # noqa: E402

FR = ROOT / "lombard/3d/frames"
OUT = ROOT / "lombard/render/Ломбард_3D_10с.mp4"
W, H, N, CUT = 1920, 1080, 240, 108          # кадры 0..107 — улица, 108..239 — ломбард
SR, SPF = AU.SR, AU.SPF


def bloom(rgb, tint=(1.0, 0.72, 0.42)):
    """Ореол ярких мест; белые (пересвеченные) неон и лампы светятся янтарным ореолом, как настоящий неон."""
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    hi = rgb * np.clip((lum - 0.62) / 0.3, 0, 1)[..., None] * np.array(tint, np.float32)
    return rgb + M.blur(hi, 7) * 0.45 + M.blur(hi, 28) * 0.4 + M.blur(hi, 70) * 0.18


def grade(rgb, sat):
    lum = (rgb @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    warm = (rgb[..., 0] - rgb[..., 2])[..., None]
    glow = np.clip((lum - 0.45) / 0.25, 0, 1) * np.clip((warm - 0.08) / 0.15, 0, 1)    # неон и банки остаются тёплыми
    k = sat + (1 - sat) * glow
    out = lum + (rgb - lum) * k
    # ночь: тени чуть в синий, света в янтарь
    out = out + (1 - lum) * np.array([-0.008, 0.0, 0.018], np.float32) * 0.9
    return np.clip(out, 0, 1)


def audio():
    track = np.zeros((N * SPF + SR, 2), np.float32)

    def bed(name, f0, f1, gain, fin=6, fout=10):
        x = AU.bed(name, (f1 - f0) * SPF)
        x = AU.fade(x, fin * SPF, fout * SPF)
        AU.place(track, x, f0 * SPF, gain)

    def hit(name, f, gain, length=None):
        x = AU.load(name)
        if length:
            x = AU.fade(x[:length * SPF], 0, 2 * SPF)
        AU.place(track, x, f * SPF, gain)

    bed("rain", 0, CUT + 10, -15, fin=12, fout=12)          # дождь глохнет за дверью (L-cut)
    hit("neon_buzz", 4, -13)
    hit("neon_buzz", 56, -17, length=20)                    # дребезг неона
    hit("door_creak", 83, -12)
    hit("door_bell", 91, -7)
    bed("ticking_many", CUT - 8, N, -18, fin=8, fout=14)     # J-cut: часы ломбарда слышны до склейки
    bed("jar_hum", CUT, N, -21, fin=10, fout=14)
    hit("glass_chime", 150, -19)
    track = track[:N * SPF]
    track[-14 * SPF:] *= np.linspace(1, 0, 14 * SPF)[:, None]
    raw = ROOT / "lombard/audio/mix3d_raw.wav"
    with wave.open(str(raw), "wb") as w_:
        w_.setnchannels(2)
        w_.setsampwidth(2)
        w_.setframerate(SR)
        w_.writeframes((np.clip(track, -1, 1) * 32767).astype("<i2").tobytes())
    p1 = subprocess.run(["ffmpeg", "-v", "info", "-i", str(raw), "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
                         "-f", "null", "-"], capture_output=True, text=True).stderr
    m = json.loads(p1[p1.rindex("{"):p1.rindex("}") + 1])
    af = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
          f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    out = ROOT / "lombard/audio/mix3d.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af", af, "-ar", str(SR), str(out)], check=True)
    return out


def main():
    snd = audio()
    rain = M.Rain(W, H, density=1.0, angle=7, seed=71, ripples=(0, 760, W, H))
    dust = M.Dust(W, H, n=45, seed=8, region=(640, 60, 1280, 700), size=(0.6, 1.5))
    vig = M.vignette(W, H, 0.32)
    wr = M.Writer(OUT, w=W, h=H, audio=snd)
    for i in range(N):
        im = cv2.imread(str(FR / f"f_{i + 1:04d}.png"), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255
        if i < CUT:
            im = 1 - (1 - np.clip(im * 1.35, 0, 1)) ** 1.15          # ночь светлее: экспозиция и подъём теней
            rgb = bloom(im) + rain.render(i) * 0.7
            sat = 0.78
        else:
            rgb = bloom(im) + dust.render(i) * 0.25
            sat = 0.8
        rgb = grade(np.clip(rgb, 0, 1), sat) * vig + M.grain(W, H, i, 0.012)
        if i < 10:
            rgb = rgb * M.smooth(i / 10)
        wr.write(np.clip(rgb, 0, 1))
    wr.close()
    print(OUT)


if __name__ == "__main__":
    main()
