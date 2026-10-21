"""Черновой звук аниматика «Ломбарда»: атмосферы и SFX из shots.json с точностью до кадра.
Без музыки. Мастер −14 LUFS / −1 dBTP (loudnorm, 2 прохода), 48 кГц.
Запуск: .venv/bin/python lombard/audio/mix.py"""
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "lombard/audio"
SR, FPS = 48000, 24
SPF = SR // FPS  # сэмплов на кадр

# id атмосферы → (файл, громкость дБ, зациклить)
AMB = {
    "rain": ("rain", -18, True),
    "street_night": ("street_night", -22, True),
    "ticking_many": ("ticking_many", -21, True),
    "ticking_loud": ("ticking_many", -14, True),
    "ticking_slow": ("ticking_slow", -13, False),
    "ticking_stop": ("ticking_stop", -12, False),
    "jar_hum": ("jar_hum", -24, True),
    "office_hum": ("office_hum", -24, True),
    "car_interior": ("car_interior", -20, True),
    "room_tone": ("room_tone", -22, True),
    "hall_ambience": ("hall_ambience", -20, True),
}
SFX_GAIN = {
    "voice_daughter": -3, "neon_buzz": -14, "step_puddle": -9, "typing": -16, "msg_send": -12,
    "door_bell": -8, "tick_single": -10, "glass_chime": -10, "subway_muffled": -16, "cash_thud": -11,
    "clasp_click": -8, "cloth_swish": -12, "notif": -12, "paper_rustle": -12, "waves_laugh": -9,
    "jar_close": -8, "keys_jingle": -8, "girl_laugh": -10, "swipe": -12, "jar_thud": -6,
    "heartbeat": -7, "glass_resonance": -8, "watch_clink": -8, "glass_slide": -10, "run_puddles": -8,
    "door_creak": -12, "audience_whisper": -16, "star_flash": -7, "jars_fade": -10, "rain_tail": -18,
}
JCUT = 8      # атмосфера следующей сцены начинается на 8 к раньше склейки
FADE = 6      # кадров на подъём/спад атмосферы
_cache = {}


def load(name):
    if name not in _cache:
        f = A / ("voice/daughter_phone.wav" if name == "voice_daughter" else f"sfx/{name}.mp3")
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(f), "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        _cache[name] = np.frombuffer(raw, np.float32).reshape(-1, 2).copy()
    return _cache[name]


def phone_voice():
    """Голосовое дочери: чуть выше тон (ребёнок), полоса телефона 300–3400 Гц."""
    src, dst = A / "voice/daughter_lunya.mp3", A / "voice/daughter_phone.wav"
    up = 2 ** (2.5 / 12)
    af = (f"asetrate=44100*{up:.5f},aresample=48000,atempo={1 / up:.5f},"
          "silenceremove=start_periods=1:start_threshold=-45dB,"
          "highpass=f=300,lowpass=f=3400,acompressor=threshold=-20dB:ratio=3,volume=2dB")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-af", af, "-ar", str(SR), str(dst)], check=True)


def fade(x, n_in, n_out):
    x = x.copy()
    if n_in:
        x[:n_in] *= np.linspace(0, 1, n_in)[:, None]
    if n_out:
        x[-n_out:] *= np.linspace(1, 0, n_out)[:, None]
    return x


def place(track, x, at, gain_db):
    at = max(0, at)
    end = min(len(track), at + len(x))
    if end > at:
        track[at:end] += x[:end - at] * 10 ** (gain_db / 20)


def bed(name, n):
    x = load(name)
    if len(x) >= n:
        return x[:n]
    # петля со склейкой внахлёст 0,25 с, чтобы не щёлкало
    ov = SR // 4
    out = x.copy()
    while len(out) < n:
        k = np.linspace(0, 1, ov)[:, None]
        out[-ov:] = out[-ov:] * (1 - k) + x[:ov] * k
        out = np.concatenate([out, x[ov:]])
    return out[:n]


def main():
    import sys
    shots = json.load(open(ROOT / "lombard/shots.json"))["shots"]
    upto = sys.argv[sys.argv.index("--upto") + 1] if "--upto" in sys.argv else None
    if upto:                                   # короткая версия: только планы до upto включительно
        shots = shots[:[s["id"] for s in shots].index(upto) + 1]
    starts, f = [], 0
    for s in shots:
        starts.append(f)
        f += s["frames"]
    total = f
    track = np.zeros((total * SPF + SR, 2), np.float32)
    phone_voice()

    # атмосферы: соседние планы с одинаковой атмосферой сливаются в один непрерывный слой
    runs = []
    for i, s in enumerate(shots):
        for amb in s["amb"]:
            if runs and runs[-1][0] == amb and runs[-1][2] == starts[i]:
                runs[-1][2] = starts[i] + s["frames"]
            else:
                runs.append([amb, starts[i], starts[i] + s["frames"]])
    for amb, f0, f1 in runs:
        name, gain, loop = AMB[amb]
        a0 = max(0, f0 - (JCUT if f0 else 0)) * SPF
        n = f1 * SPF - a0
        x = bed(name, n) if loop else load(name)[:n]
        x = fade(x, FADE * SPF if f0 else SPF * 10, min(len(x), FADE * SPF))
        place(track, x, a0, gain)

    # эффекты — точно в кадр
    for i, s in enumerate(shots):
        for e in s["sfx"]:
            name = e["id"]
            x = load("rain" if name == "rain_tail" else name)
            if name == "rain_tail":
                x = fade(x[:s["frames"] * SPF], 0, s["frames"] * SPF)
            if "len" in e:
                x = fade(x[:e["len"] * SPF], 0, SPF)
            place(track, x, (starts[i] + e["f"]) * SPF, SFX_GAIN.get(name, -10))

    track = track[:total * SPF]
    if upto:                                   # мягкий уход в тишину на последних 18 к
        n = 18 * SPF
        track[-n:] *= np.linspace(1, 0, n)[:, None] ** 1.5
    raw = A / "mix_raw.wav"
    import wave
    pcm = (np.clip(track, -1, 1) * 32767).astype("<i2")
    with wave.open(str(raw), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())

    # громкость: loudnorm в два прохода
    p1 = subprocess.run(["ffmpeg", "-v", "info", "-i", str(raw), "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
                         "-f", "null", "-"], capture_output=True, text=True).stderr
    m = json.loads(p1[p1.rindex("{"):p1.rindex("}") + 1])
    af = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    out = A / (f"mix_upto{upto}.wav" if upto else "mix_v1.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af", af, "-ar", str(SR), str(out)], check=True)
    print(out, f"{total / FPS:.2f} с", "вход", m["input_i"], "LUFS")


if __name__ == "__main__":
    main()
