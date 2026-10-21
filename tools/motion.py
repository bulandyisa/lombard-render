"""Движок «живой иллюстрации» (2.5D, как Puppet Warp / Live2D) для «Ломбарда».

Кадр = слои в премультиплицированной альфе (float32, 0..1). Персонажи «гнутся» деформацией
Moving Least Squares (rigid) по точкам-шарнирам: точки на голове поворачиваются вокруг шеи,
точки на теле стоят — картинка между ними деформируется плавно. Сверху — свечение, частицы,
цвет и камера. Пресеты тайминга — из docs/ПРАВИЛА_АНИМАЦИИ.md (моргание 2+1+3, overlap 2–4 к).
"""
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

FPS = 24


# ---------- картинки ----------
def load(path, size=None):
    """RGBA → float32 premultiplied; size=(w, h) — уменьшение ступенями по 2× + INTER_AREA (без «звона»)."""
    im = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255
    if size is not None:
        while im.shape[1] >= size[0] * 2 and im.shape[0] >= size[1] * 2:
            im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA if im.shape[1] > size[0] else cv2.INTER_LANCZOS4)
    im[..., :3] *= im[..., 3:4]
    return np.clip(im, 0, 1)


def rgb_layer(rgb):
    """Непрозрачный слой из RGB-массива."""
    return np.dstack([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)])


def with_alpha(layer, mask):
    """Накладывает маску (0..1) на слой."""
    out = layer * mask[..., None]
    return out


def over(dst, src, k=1.0):
    """src поверх dst (оба premultiplied)."""
    s = src * k
    return s + dst * (1 - s[..., 3:4])


def add_light(dst, rgb, k=1.0):
    """Свет: аддитивно к цвету, альфа не меняется."""
    out = dst.copy()
    out[..., :3] = out[..., :3] + rgb * k
    return out


def screen(dst, rgb, k=1.0):
    out = dst.copy()
    out[..., :3] = 1 - (1 - out[..., :3]) * (1 - np.clip(rgb * k, 0, 1))
    return out


def glow(layer, sigma, k=1.0):
    """Мягкое свечение слоя (premultiplied RGB размытый), возвращает RGB для add/screen."""
    ks = int(sigma * 3) * 2 + 1
    return cv2.GaussianBlur(layer[..., :3], (ks, ks), sigma) * k


def blur(layer, sigma):
    if sigma <= 0.05:
        return layer
    ks = int(sigma * 3) * 2 + 1
    return cv2.GaussianBlur(layer, (ks, ks), sigma)


def soft_mask(shape, polys=(), ellipses=(), feather=8):
    """Маска из многоугольников [(x, y), ...] и эллипсов (cx, cy, rx, ry) с мягким краем."""
    m = np.zeros(shape[:2], np.float32)
    for poly in polys:
        cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 1.0, lineType=cv2.LINE_AA)
    for cx, cy, rx, ry in ellipses:
        cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    return blur(m, feather) if feather else m


# ---------- деформация MLS (Schaefer et al. 2006, rigid) ----------
def mls_map(src_pts, dst_pts, w, h, step=8, alpha=1.0):
    """Обратная карта для cv2.remap: пиксель кадра → откуда брать в исходнике.
    src_pts — точки на исходной картинке, dst_pts — куда они переехали в этом кадре."""
    p = np.asarray(dst_pts, np.float64)   # для обратной карты меняем роли: «из» = куда переехали
    q = np.asarray(src_pts, np.float64)
    gx, gy = np.meshgrid(np.arange(0, w + step, step, dtype=np.float64), np.arange(0, h + step, step, dtype=np.float64))
    v = np.stack([gx.ravel(), gy.ravel()], 1)
    if np.allclose(p, q):
        fx, fy = v[:, 0], v[:, 1]
    else:
        d2 = ((p[None] - v[:, None]) ** 2).sum(-1)
        wgt = 1.0 / (d2 ** alpha + 1e-6)
        W = wgt.sum(1, keepdims=True)
        ps = (wgt[..., None] * p[None]).sum(1) / W
        qs = (wgt[..., None] * q[None]).sum(1) / W
        ph = p[None] - ps[:, None]
        qh = q[None] - qs[:, None]
        vp = v - ps
        c = (ph[..., 0] * vp[:, None, 0] + ph[..., 1] * vp[:, None, 1]) * wgt
        s = (ph[..., 0] * vp[:, None, 1] - ph[..., 1] * vp[:, None, 0]) * wgt
        frx = (qh[..., 0] * c - qh[..., 1] * s).sum(1)
        fry = (qh[..., 0] * s + qh[..., 1] * c).sum(1)
        norm = np.hypot(frx, fry) + 1e-9
        vlen = np.hypot(vp[:, 0], vp[:, 1])
        fx = vlen * frx / norm + qs[:, 0]
        fy = vlen * fry / norm + qs[:, 1]
        # точка точно на шарнире — без деления на ноль
        exact = d2.min(1) < 1e-6
        if exact.any():
            k = d2.argmin(1)
            fx[exact], fy[exact] = q[k[exact], 0], q[k[exact], 1]
    mx = fx.reshape(gy.shape).astype(np.float32)
    my = fy.reshape(gy.shape).astype(np.float32)
    # грубая сетка (узел i ↔ пиксель i·step) → полная карта точной интерполяцией
    key = (w, h, step)
    if key not in _grid:
        _grid[key] = np.meshgrid(np.arange(w, dtype=np.float32) / step, np.arange(h, dtype=np.float32) / step)
    xx, yy = _grid[key]
    return (cv2.remap(mx, xx, yy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE),
            cv2.remap(my, xx, yy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))


_grid = {}


def warp(layer, src_pts, dst_pts, step=8):
    h, w = layer.shape[:2]
    if np.allclose(src_pts, dst_pts):
        return layer
    mx, my = mls_map(src_pts, dst_pts, w, h, step)
    return cv2.remap(layer, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def rotate_pts(pts, pivot, deg, scale=(1.0, 1.0)):
    """Поворот точек вокруг pivot (градусы, по часовой при оси y вниз) + масштаб от pivot."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    out = []
    for x, y in pts:
        dx, dy = (x - pivot[0]) * scale[0], (y - pivot[1]) * scale[1]
        out.append((pivot[0] + dx * c - dy * s, pivot[1] + dx * s + dy * c))
    return out


def shift_pts(pts, dx, dy):
    return [(x + dx, y + dy) for x, y in pts]


class Rig:
    """Кости с родителями: у каждой — шарнир (pivot). pose = {кость: (угол°, dx, dy, масштаб)}.
    Точки деформации привязываются к костям: apply(точки, кость, pose) → где они в этом кадре."""

    def __init__(self, bones):
        self.bones = bones            # {имя: (родитель или None, (px, py))}

    def matrix(self, name, pose):
        parent, (px, py) = self.bones[name]
        ang, dx, dy, s = (tuple(pose.get(name, ())) + (0.0, 0.0, 0.0, 1.0)[len(pose.get(name, ())):])
        a = math.radians(ang)
        c, si = math.cos(a) * s, math.sin(a) * s
        local = np.array([[c, -si, px - c * px + si * py + dx], [si, c, py - si * px - c * py + dy], [0, 0, 1]])
        return (self.matrix(parent, pose) if parent else np.eye(3)) @ local

    def apply(self, pts, bone, pose):
        Mx = self.matrix(bone, pose)
        P = np.c_[np.asarray(pts, np.float64), np.ones(len(pts))]
        return [tuple(r) for r in (P @ Mx.T)[:, :2]]


def handles(rig, groups, pose):
    """groups = {кость: [точки]} → (исходные точки, точки в кадре) для warp()."""
    src, dst = [], []
    for bone, pts in groups.items():
        src += list(pts)
        dst += rig.apply(pts, bone, pose) if bone else list(pts)
    return src, dst


# ---------- плавность и пресеты тайминга ----------
def clamp(u, a=0.0, b=1.0):
    return max(a, min(b, u))


def smooth(u):
    u = np.clip(u, 0.0, 1.0) if isinstance(u, np.ndarray) else clamp(u)
    return u * u * (3 - 2 * u)


def ease_io(u):
    return 0.5 - 0.5 * math.cos(math.pi * clamp(u))


def ease_out(u):
    u = clamp(u)
    return 1 - (1 - u) ** 3


def ease_in(u):
    u = clamp(u)
    return u ** 3


def ease_back(u, s=1.4):
    """Выход с перелётом (overshoot) — для оседания после акцента."""
    u = clamp(u) - 1
    return 1 + u * u * ((s + 1) * u + s)


EASES = {"io": ease_io, "in": ease_in, "out": ease_out, "back": ease_back, "lin": clamp, "step": lambda u: 0.0}


def curve(t, keys):
    """keys = [(кадр, значение, плавность_до_следующего)], значение — число или кортеж."""
    if t <= keys[0][0]:
        return keys[0][1]
    for a, b in zip(keys, keys[1:]):
        if t <= b[0]:
            u = EASES[a[2] if len(a) > 2 else "io"]((t - a[0]) / max(1e-6, b[0] - a[0]))
            if isinstance(a[1], tuple):
                return tuple(x + (y - x) * u for x, y in zip(a[1], b[1]))
            return a[1] + (b[1] - a[1]) * u
    return keys[-1][1]


BLINK = {"normal": [0.55, 1.0, 1.0, 0.7, 0.35, 0.1],   # 2 закрытие + 1 закрыто + 3 открытие
         "fast": [0.6, 1.0, 0.6, 0.25, 0.05],
         "long": [0.4, 0.8, 1.0, 1.0, 1.0, 0.8, 0.55, 0.3, 0.1]}


def blink(t, starts, kind="normal"):
    """Закрытость век 0..1 в кадре t; starts — кадры начала морганий."""
    seq = BLINK[kind]
    for s in starts:
        if s <= t < s + len(seq):
            return seq[int(t - s)]
    return 0.0


def on_twos(t):
    """Позы на двойках: кадр округляется к чётному."""
    return t - (t % 2)


def noise(t, seed=0, freq=0.5):
    """Гладкий шум −1..1 (сумма синусов с несоизмеримыми частотами) — для moving hold."""
    r = np.random.default_rng(seed)
    ph = r.uniform(0, 2 * math.pi, 3)
    x = t / FPS * freq * 2 * math.pi
    return (math.sin(x + ph[0]) * 0.55 + math.sin(x * 1.73 + ph[1]) * 0.3 + math.sin(x * 2.91 + ph[2]) * 0.15)


def breath(t, period=3.6, phase=0.0):
    """Дыхание 0..1 (вдох медленнее выдоха)."""
    u = ((t / FPS) / period + phase) % 1.0
    return math.sin(math.pi * u) ** 2 if u < 0.55 else math.sin(math.pi * u) ** 1.5


# ---------- лицо ----------
def eyelid(layer, upper, lower, c, lash_w=2.0, skin_up=(3, 8), shade=0.96, skin_at=None):
    """Процедурное веко: опускает верхнее веко от линии upper к lower (закрытость c 0..1).
    Кожа века берётся из полосы над верхней линией, по новому краю — тёмная линия ресниц.
    upper/lower — ломаные одинаковой длины от внутреннего угла глаза к внешнему (в пикселях слоя)."""
    if c <= 0.01:
        return layer
    up = np.asarray(upper, np.float32)
    lo = np.asarray(lower, np.float32)
    edge = up + (lo - up) * c
    h, w = layer.shape[:2]
    x0, y0 = int(min(up[:, 0].min(), lo[:, 0].min())) - 12, int(up[:, 1].min()) - 16
    x1, y1 = int(max(up[:, 0].max(), lo[:, 0].max())) + 12, int(lo[:, 1].max()) + 12
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    rgb = layer[y0:y1, x0:x1, :3]
    # цвет кожи века: медиана полосы над верхней линией; цвет ресниц: самые тёмные точки на линии века
    samples, darks = [], []
    for (x, y) in up:
        for dy in range(skin_up[0], skin_up[1] + 1):
            yy, xx = int(y - dy) - y0, int(x) - x0
            if 0 <= yy < rgb.shape[0] and 0 <= xx < rgb.shape[1]:
                samples.append(rgb[yy, xx])
        yy, xx = int(round(y)) - y0, int(round(x)) - x0
        if 0 <= yy < rgb.shape[0] and 0 <= xx < rgb.shape[1]:
            darks.append(rgb[yy, xx])
    if skin_at:                       # явные точки кожи (если над глазом сразу бровь)
        samples = [layer[int(y), int(x), :3] for x, y in skin_at]
    skin = np.median(np.array(samples), 0) * shade
    lash = np.min(np.array(darks), 0) if darks else np.zeros(3, np.float32)
    S = 4
    m = np.zeros(((y1 - y0) * S, (x1 - x0) * S), np.float32)
    poly = np.concatenate([up, edge[::-1]]) - (x0, y0)
    cv2.fillPoly(m, [np.round(poly * S).astype(np.int32)], 1.0, cv2.LINE_AA, 0)
    line = np.zeros_like(m)
    cv2.polylines(line, [np.round((edge - (x0, y0)) * S).astype(np.int32)], False, 1.0, max(1, int(lash_w * S)), cv2.LINE_AA)
    m = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)[..., None]
    line = cv2.resize(line, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)[..., None]
    out = layer.copy()
    patch = rgb * (1 - m) + skin * m
    patch = patch * (1 - line) + lash * line
    out[y0:y1, x0:x1, :3] = patch
    return out


# ---------- частицы ----------
class Dust:
    """Пылинки в тёплом свете: медленный дрейф + мерцание, рисуются мягкими точками (аддитивно)."""

    def __init__(self, w, h, n=60, seed=1, region=None, size=(1.2, 3.5), color=(1.0, 0.82, 0.5)):
        r = np.random.default_rng(seed)
        x0, y0, x1, y1 = region or (0, 0, w, h)
        self.w, self.h = w, h
        self.p = np.stack([r.uniform(x0, x1, n), r.uniform(y0, y1, n)], 1)
        self.v = np.stack([r.normal(0, 3, n), r.normal(-2, 3, n)], 1)  # пикселей в секунду
        self.size = r.uniform(*size, n)
        self.ph = r.uniform(0, 2 * math.pi, n)
        self.blur = r.uniform(0, 1, n)
        self.color = np.array(color, np.float32)
        self.region = (x0, y0, x1, y1)

    def render(self, t):
        img = np.zeros((self.h, self.w, 3), np.float32)
        x0, y0, x1, y1 = self.region
        s = t / FPS
        for i in range(len(self.p)):
            x = x0 + (self.p[i, 0] - x0 + self.v[i, 0] * s + 6 * math.sin(s * 0.7 + self.ph[i])) % (x1 - x0)
            y = y0 + (self.p[i, 1] - y0 + self.v[i, 1] * s + 4 * math.sin(s * 0.9 + self.ph[i] * 2)) % (y1 - y0)
            a = 0.5 + 0.5 * math.sin(s * 2.2 + self.ph[i])
            rad = self.size[i] * (1 + 2 * self.blur[i])
            cv2.circle(img, (int(x * 4), int(y * 4)), max(1, int(rad * 4)), (a * (1 - 0.6 * self.blur[i]),) * 3, -1, cv2.LINE_AA, 2)
        img = blur(img, 1.2)
        return img * self.color


class Rain:
    """Дождь в 3 слоях глубины: дальний (мелкий, медленный, тусклый), средний, ближний (крупный, размытый).
    angle — наклон в градусах, ripples — зона луж (x0, y0, x1, y1) для кругов от капель."""

    def __init__(self, w, h, density=1.0, angle=8, seed=3, ripples=None, color=(0.75, 0.82, 0.95)):
        r = np.random.default_rng(seed)
        self.w, self.h, self.angle = w, h, math.radians(angle)
        self.layers = []
        for n, speed, length, width, alpha, bl in ((int(500 * density), 900, 22, 1, 0.22, 0.0),
                                                   (int(220 * density), 1500, 42, 1, 0.32, 0.6),
                                                   (int(45 * density), 2600, 110, 2, 0.30, 2.2)):
            self.layers.append(dict(x=r.uniform(-0.2 * w, 1.1 * w, n), y=r.uniform(0, h, n), speed=speed * r.uniform(0.85, 1.15, n),
                                    length=length, width=width, alpha=alpha * r.uniform(0.5, 1.0, n), blur=bl))
        self.color = np.array(color, np.float32)
        self.ripples = ripples
        if ripples:
            x0, y0, x1, y1 = ripples
            m = int(70 * density)
            self.rp = dict(x=r.uniform(x0, x1, m), y=r.uniform(y0, y1, m), t0=r.uniform(0, 0.6, m), period=r.uniform(0.45, 0.8, m))

    def render(self, t):
        out = np.zeros((self.h, self.w, 3), np.float32)
        s = t / FPS
        dx, dy = math.sin(self.angle), math.cos(self.angle)
        for L in self.layers:
            img = np.zeros((self.h, self.w), np.float32)
            span = self.h + L["length"]
            y = (L["y"] + L["speed"] * s) % span - L["length"]
            x = L["x"] + (y - L["y"]) * dx / dy
            for xi, yi, ai in zip(x, y, L["alpha"]):
                cv2.line(img, (int(xi * 4), int(yi * 4)), (int((xi + dx * L["length"]) * 4), int((yi + dy * L["length"]) * 4)),
                         float(ai), L["width"], cv2.LINE_AA, 2)
            out += blur(img, L["blur"])[..., None] * self.color
        if self.ripples:
            img = np.zeros((self.h, self.w), np.float32)
            for x, y, t0, per in zip(self.rp["x"], self.rp["y"], self.rp["t0"], self.rp["period"]):
                u = ((s - t0) % per) / per
                rad = 2 + 26 * ease_out(u)
                a = 0.35 * (1 - u)
                cv2.ellipse(img, (int(x * 4), int(y * 4)), (int(rad * 4), int(rad * 4 * 0.32)), 0, 0, 360, a, 1, cv2.LINE_AA, 2)
            out += blur(img, 0.6)[..., None] * self.color
        return out


# ---------- цвет ----------
def grade(rgb, sat, keep=None, keep_boost=1.15):
    """Насыщенность мира sat; keep — маска (0..1) того, что остаётся полноцветным (воспоминания)."""
    lum = (rgb @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    k = np.full(lum.shape, sat, np.float32)
    if keep is not None:
        k = k * (1 - keep[..., None]) + keep_boost * keep[..., None]
    return np.clip(lum + (rgb - lum) * k, 0, 1)


def vignette(w, h, strength=0.35):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot((xx / w - 0.5) * 1.2, (yy / h - 0.5))
    return (1 - strength * smooth((d - 0.35) / 0.45))[..., None]


def grain(w, h, t, k=0.018):
    r = np.random.default_rng(int(t) * 7919 + 13)
    g = r.normal(0, 1, (h // 2, w // 2)).astype(np.float32)
    g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
    return g[..., None] * k


# ---------- камера ----------
def camera(img, out_w, out_h, zoom=1.0, cx=0.5, cy=0.5, rot=0.0, shake=(0.0, 0.0)):
    """Кадрирование холста с субпиксельной точностью: zoom относительно «вписанного» кадра."""
    h, w = img.shape[:2]
    base = min(w / out_w, h / out_h)          # сколько пикселей холста в пикселе кадра при zoom=1
    s = 1.0 / (base / zoom)
    M = cv2.getRotationMatrix2D((cx * w, cy * h), rot, s)
    M[0, 2] += out_w / 2 - cx * w + shake[0]
    M[1, 2] += out_h / 2 - cy * h + shake[1]
    return cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC if s > 0.75 else cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE)


# ---------- вывод ----------
class Writer:
    """Кадры → H.264 1080p по правилам экспорта (BT.709, crf 16, tune animation, gop 12)."""

    def __init__(self, out, w=1920, h=1080, audio=None, audio_start=0.0, crf=16):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(FPS), "-i", "-"]
        if audio:
            cmd += ["-ss", f"{audio_start:.4f}", "-i", str(audio)]
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-tune", "animation",
                "-pix_fmt", "yuv420p", "-vf", "scale=out_color_matrix=bt709:out_range=tv",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-g", "12", "-bf", "2"]
        if audio:
            cmd += ["-c:a", "aac", "-b:a", "384k", "-ar", "48000", "-shortest"]
        cmd += ["-movflags", "+faststart", str(out)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb):
        self.p.stdin.write((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def save_png(rgb, path):
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
