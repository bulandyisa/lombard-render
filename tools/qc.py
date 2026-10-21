"""Автоматический контроль качества рендера — обязателен ПЕРЕД показом пользователю (CLAUDE.md, гейт QC).

Проверяет видео или папку кадров: число кадров, чёрные/замёрзшие/дублированные кадры, мерцание,
скачки (случайные склейки), ореолы альфы; для звука — громкость и пик. Делает контактный лист и
кинопленку ключевых кадров для глазного ревью по чек-листу §10. Выход: отчёт в консоль + JSON + JPG.

Запуск:
  .venv/bin/python tools/qc.py видео.mp4 [--frames N] [--cuts 60,228,...] [--out папка]
  .venv/bin/python tools/qc.py папка_кадров/ [--frames N]
Из кода: qc.check_video(...), qc.bg_stability(...), qc.tears(...)."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# Пороги (эмпирические, подобраны на «Ломбарде»; менять осознанно)
TH = dict(
    black_mean=0.02,        # средняя яркость ниже — чёрный кадр (кроме намеренных затемнений на склейках/титре)
    frozen_mad=0.0015,      # средняя абс. разница соседних кадров ниже — кадр «замёрз» (на статике с живой паузой ≥ 0.002)
    flicker_lum=0.06,       # скачок средней яркости между соседними кадрами (не на склейке) — мерцание
    jump_mad=0.25,          # средняя разница соседних кадров выше — скачок изображения вне объявленной склейки
    bg_drift=0.01,          # фон вне персонажа между далёкими кадрами должен отличаться не больше этого
    tear_px=400,            # замкнутая дыра в силуэте больше N пикселей (на 1672×941) — разрыв деталей
)


def read_video(path, step=1, scale=0.5):
    cap = cv2.VideoCapture(str(path))
    frames = []
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i % step == 0:
            if scale != 1:
                im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(im[..., ::-1].astype(np.float32) / 255)
        i += 1
    cap.release()
    return frames, i


def read_frames_dir(path, scale=0.5):
    files = sorted(Path(path).glob("*.png"))
    frames = []
    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if scale != 1:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(im[..., ::-1].astype(np.float32) / 255)
    return frames, len(files)


def luminance(f):
    return float((f @ np.array([0.299, 0.587, 0.114], np.float32)).mean())


def check_frames(frames, cuts=(), fades=()):
    """cuts — кадры, где склейка законна; fades — диапазоны (a, b), где чёрный кадр законен."""
    issues = []
    cuts = set(cuts)
    lums = [luminance(f) for f in frames]
    for i, f in enumerate(frames):
        if lums[i] < TH["black_mean"] and not any(a <= i <= b for a, b in fades):
            issues.append(dict(frame=i, kind="black", value=round(lums[i], 4)))
    for i in range(1, len(frames)):
        mad = float(np.abs(frames[i] - frames[i - 1]).mean())
        if mad < TH["frozen_mad"] and lums[i] > TH["black_mean"]:
            issues.append(dict(frame=i, kind="frozen", value=round(mad, 5)))
        if i not in cuts:
            if abs(lums[i] - lums[i - 1]) > TH["flicker_lum"]:
                issues.append(dict(frame=i, kind="flicker", value=round(lums[i] - lums[i - 1], 3)))
            if mad > TH["jump_mad"]:
                issues.append(dict(frame=i, kind="jump", value=round(mad, 3)))
    # серии замёрзших кадров — одна запись
    merged, run = [], None
    for it in sorted(issues, key=lambda x: (x["kind"], x["frame"])):
        if it["kind"] == "frozen" and run and run["kind"] == "frozen" and it["frame"] == run["to"] + 1:
            run["to"] = it["frame"]
        else:
            if run:
                merged.append(run)
            run = dict(it, to=it["frame"])
    if run:
        merged.append(run)
    return merged, lums


def loudness(path):
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-af", "loudnorm=I=-14:TP=-1:LRA=11:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True).stderr
    try:
        m = json.loads(p[p.rindex("{"):p.rindex("}") + 1])
        return dict(lufs=float(m["input_i"]), tp=float(m["input_tp"]), lra=float(m["input_lra"]))
    except ValueError:
        return None


def contact_sheet(frames, out, n=12, cols=4, label=True):
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    tiles = []
    for i in idx:
        t = (frames[i] * 255).astype(np.uint8).copy()
        if label:
            cv2.putText(t, f"{i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(t, f"{i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    cv2.imwrite(str(out), cv2.resize(sheet, (1920, int(1920 * sheet.shape[0] / sheet.shape[1])))[..., ::-1])
    return out


def filmstrip(frames, out, a, b, step=1, max_tiles=16):
    """Подряд идущие кадры a..b — для проверки движения (overlap, дуги, тайминг)."""
    idx = list(range(a, min(b, len(frames) - 1) + 1, step))[:max_tiles]
    tiles = [(frames[i] * 255).astype(np.uint8) for i in idx]
    h, w = tiles[0].shape[:2]
    cols = min(8, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    cv2.imwrite(str(out), cv2.resize(sheet, (1920, int(1920 * sheet.shape[0] / sheet.shape[1])))[..., ::-1])
    return out


def bg_stability(frame_a, frame_b, char_mask, grow=41):
    """Фон вне персонажа (маска 0..1 в координатах кадра) между двумя кадрами одного плана без движения камеры."""
    far = cv2.dilate((char_mask > 0.02).astype(np.uint8), np.ones((grow, grow), np.uint8)) == 0
    d = np.abs(frame_a - frame_b).max(-1)
    return float(d[far].max()), float(d[far].mean())


def tears(alpha_union, min_px=None):
    """Замкнутые дыры в объединённом силуэте слоёв персонажа (разрывы между деталями). alpha_union — 0..1."""
    min_px = min_px or TH["tear_px"]
    b = (alpha_union > 0.5).astype(np.uint8)
    h, w = b.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((1 - b).astype(np.uint8))
    out = []
    for i in range(1, n):
        x, y, ww, hh, area = st[i]
        if area >= min_px and x > 0 and y > 0 and x + ww < w and y + hh < h:
            out.append(dict(x=int(x), y=int(y), w=int(ww), h=int(hh), px=int(area)))
    return out


def check_video(path, expect_frames=None, cuts=(), fades=(), out_dir=None, strips=()):
    path = Path(path)
    out_dir = Path(out_dir) if out_dir else path.parent / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        frames, total = read_frames_dir(path)
        loud = None
    else:
        frames, total = read_video(path)
        loud = loudness(path)
    issues, lums = check_frames(frames, cuts, fades)
    report = dict(file=str(path), frames=total, expect_frames=expect_frames, loudness=loud, issues=issues)
    verdict = []
    if expect_frames and total != expect_frames:
        verdict.append(f"кадров {total}, ожидалось {expect_frames}")
    if loud:
        if not (-15.0 <= loud["lufs"] <= -13.0):
            verdict.append(f"громкость {loud['lufs']} LUFS (норма −14 ±1)")
        if loud["tp"] > -0.9:
            verdict.append(f"пик {loud['tp']} dBTP (норма ≤ −1)")
    kinds = {}
    for it in issues:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    for k, v in kinds.items():
        verdict.append(f"{k}: {v}")
    report["verdict"] = "OK" if not verdict else "; ".join(verdict)
    stem = path.stem if path.is_file() else path.name
    contact_sheet(frames, out_dir / f"{stem}_contact.jpg")
    for a, b in strips:
        filmstrip(frames, out_dir / f"{stem}_strip_{a}_{b}.jpg", a, b)
    json.dump(report, open(out_dir / f"{stem}_qc.json", "w"), ensure_ascii=False, indent=1)
    print(f"QC {path.name}: кадров {total}" + (f" / {expect_frames}" if expect_frames else "") +
          (f", {loud['lufs']} LUFS, {loud['tp']} dBTP" if loud else ""))
    for it in issues[:30]:
        rng = f"{it['frame']}–{it['to']}" if it.get("to", it["frame"]) != it["frame"] else str(it["frame"])
        print(f"  {it['kind']:8s} кадр {rng:>9s}  {it['value']}")
    if len(issues) > 30:
        print(f"  … ещё {len(issues) - 30}")
    print("  ВЕРДИКТ:", report["verdict"])
    print("  лист:", out_dir / f"{stem}_contact.jpg")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--cuts", default="", help="кадры законных склеек через запятую")
    ap.add_argument("--fades", default="", help="диапазоны законного чёрного: a-b,c-d")
    ap.add_argument("--strips", default="", help="диапазоны кинопленки: a-b,c-d")
    ap.add_argument("--out")
    a = ap.parse_args()
    cuts = [int(x) for x in a.cuts.split(",") if x]
    fades = [tuple(map(int, x.split("-"))) for x in a.fades.split(",") if x]
    strips = [tuple(map(int, x.split("-"))) for x in a.strips.split(",") if x]
    r = check_video(a.path, a.frames, cuts, fades, a.out, strips)
    sys.exit(0 if r["verdict"] == "OK" else 1)
