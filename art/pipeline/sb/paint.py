"""Painted 2D layers: skies, haze, speed lines, impact bursts, comic text (numpy + PIL)."""
import math
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .comicfx import halftone_mask, smoothstep

FONTS = os.path.join(os.path.dirname(__file__), '../../fonts')


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size)


def vgrad(h, w, stops):
    """stops: list of (pos 0..1 from top, rgb)."""
    y = np.linspace(0, 1, h)
    out = np.zeros((h, 3), np.float32)
    ps = [s[0] for s in stops]
    for c in range(3):
        out[:, c] = np.interp(y, ps, [s[1][c] for s in stops])
    return np.repeat(out[:, None, :], w, axis=1)


def brush_streaks(h, w, n, ys, cols, seed=1, length=(0.15, 0.5), thick=(4, 18), alpha=0.5):
    rng = random.Random(seed)
    layer = np.zeros((h, w, 3), np.float32)
    mask = np.zeros((h, w), np.float32)
    for _ in range(n):
        y = rng.uniform(*ys) * h
        x = rng.uniform(-0.1, 1.1) * w
        L = rng.uniform(*length) * w
        t = rng.uniform(*thick) * h / 1080
        col = cols[rng.randrange(len(cols))]
        tmp = np.zeros((h, w), np.uint8)
        pts = []
        for k in range(12):
            u = k / 11
            px = x + (u - 0.5) * L
            py = y + math.sin(u * 3 + x) * t * 0.4 + (u - 0.5) * rng.uniform(-0.05, 0.05) * L
            pts.append((px, py))
        for k in range(11):
            tt = max(1, int(t * math.sin(math.pi * (k + 0.5) / 11) ** 0.6))
            cv2.line(tmp, (int(pts[k][0]), int(pts[k][1])), (int(pts[k + 1][0]), int(pts[k + 1][1])), 255, tt,
                     cv2.LINE_AA)
        m = tmp.astype(np.float32) / 255 * alpha * rng.uniform(0.5, 1)
        layer = layer * (1 - m[..., None]) + np.array(col, np.float32) * m[..., None]
        mask = np.maximum(mask, m)
    layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=6, sigmaY=0.8)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=6, sigmaY=0.8)
    return layer, mask


def sky_sunset(w, h, sun=(0.30, 0.56), seed=2):
    s = h / 1080
    img = vgrad(h, w, [(0.0, (0.42, 0.35, 0.70)), (0.28, (0.80, 0.48, 0.66)), (0.50, (0.99, 0.66, 0.55)),
                       (0.66, (1.0, 0.86, 0.60)), (1.0, (1.0, 0.93, 0.74))])
    # big halftone glow around the sun
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - sun[0] * w) / w) ** 2 + ((yy - sun[1] * h) / w) ** 2)
    glow = np.clip(1 - d / 0.35, 0, 1) ** 1.5
    img = img + np.array([0.15, 0.10, 0.0], np.float32) * glow[..., None]
    dots = halftone_mask(h, w, glow * 0.55, 9 * s, 20)
    img = img * (1 - dots[..., None] * 0.18) + np.array([1.0, 0.97, 0.80]) * dots[..., None] * 0.18
    disk = np.clip((0.045 - d) * w / 2, 0, 1)
    img = img * (1 - disk[..., None]) + np.array([1.0, 0.98, 0.86]) * disk[..., None]
    # painterly cloud streaks (ref: graded horizontal brush strokes)
    lay, m = brush_streaks(h, w, 140, (0.05, 0.62), [(1.0, 0.75, 0.62), (0.95, 0.60, 0.70), (0.75, 0.55, 0.80),
                                                      (1.0, 0.90, 0.75), (0.62, 0.48, 0.78)], seed=seed,
                           thick=(6, 26), alpha=0.55)
    img = img * (1 - m[..., None]) + lay * m[..., None] / np.clip(m[..., None], 1e-3, None) * m[..., None]
    # Ben-Day dots on the upper sky
    top = smoothstep(0.45, 0.0, yy / h)
    bd = halftone_mask(h, w, top * 0.35, 11 * s, 45)
    img = img * (1 - bd[..., None] * 0.12) + np.array([0.35, 0.25, 0.65]) * bd[..., None] * 0.12
    return np.clip(img, 0, 1)


def sky_night(w, h, seed=4):
    s = h / 1080
    img = vgrad(h, w, [(0.0, (0.05, 0.04, 0.14)), (0.45, (0.16, 0.10, 0.32)), (0.70, (0.45, 0.20, 0.45)),
                       (1.0, (0.85, 0.40, 0.45))])
    rng = np.random.default_rng(seed)
    st = (rng.random((h, w)) > 0.9993).astype(np.float32)
    st = cv2.dilate(st, np.ones((2, 2), np.uint8))
    img = img + st[..., None] * 0.8 * smoothstep(0.7, 0.1, np.linspace(0, 1, h))[:, None, None]
    lay, m = brush_streaks(h, w, 80, (0.1, 0.6), [(0.30, 0.18, 0.45), (0.50, 0.25, 0.50), (0.20, 0.15, 0.40)],
                           seed=seed, thick=(8, 30), alpha=0.6)
    img = img * (1 - m[..., None]) + lay * m[..., None] / np.clip(m[..., None], 1e-3, None) * m[..., None]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - 0.78 * w) / w) ** 2 + ((yy - 0.18 * h) / w) ** 2)
    moon = np.clip((0.04 - d) * w / 2, 0, 1)
    halo = halftone_mask(h, w, np.clip(1 - d / 0.2, 0, 1) * 0.5, 8 * s, 30)
    img = img * (1 - halo[..., None] * 0.25) + np.array([0.9, 0.8, 1.0]) * halo[..., None] * 0.25
    img = img * (1 - moon[..., None]) + np.array([1.0, 0.97, 0.88]) * moon[..., None]
    return np.clip(img, 0, 1)


def haze(color, depth, alpha, haze_col, near=25, far=260, amount=0.75):
    t = smoothstep(near, far, depth) * amount
    t = np.where(alpha > 0.01, t, 0)
    return color * (1 - t[..., None]) + np.array(haze_col, np.float32) * t[..., None]


# ----------------------------------------------------------------------------- comic FX

def speed_lines(h, w, center, n=90, inner=0.18, color=(1, 1, 1), seed=3, width=(1, 5), alpha=0.8):
    rng = random.Random(seed)
    im = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(im)
    cx, cy = center[0] * w, center[1] * h
    R = math.hypot(w, h)
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        r0 = (inner + rng.uniform(0, 0.25)) * min(w, h)
        wd = rng.uniform(*width) * h / 1080
        p0 = (cx + math.cos(a) * r0, cy + math.sin(a) * r0)
        p1 = (cx + math.cos(a - 0.004 * wd) * R, cy + math.sin(a - 0.004 * wd) * R)
        p2 = (cx + math.cos(a + 0.004 * wd) * R, cy + math.sin(a + 0.004 * wd) * R)
        d.polygon([p0, p1, p2], fill=int(255 * alpha))
    m = np.asarray(im, np.float32) / 255
    return m


def burst(h, w, center, r_in, r_out, spikes=18, seed=1, fill=(1.0, 0.92, 0.2), edge=(0.05, 0.03, 0.1), rot=0.0):
    rng = random.Random(seed)
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = center
    pts = []
    for k in range(spikes * 2):
        a = rot + math.pi * k / spikes
        r = r_out * rng.uniform(0.8, 1.15) if k % 2 == 0 else r_in * rng.uniform(0.85, 1.1)
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    st = max(3, int(r_out * 0.05))
    d.polygon(pts, fill=tuple(int(c * 255) for c in fill) + (255,), outline=tuple(int(c * 255) for c in edge) + (255,),
              width=st)
    return np.asarray(im, np.float32) / 255


def text_layer(w, h, items):
    """items: dicts(text, xy, font, size, fill, stroke, stroke_fill, anchor, rotate, shadow)."""
    out = np.zeros((h, w, 4), np.float32)
    for it in items:
        f = font(it.get('font', 'Bangers-Regular.ttf'), it['size'])
        pad = it['size'] * 2
        tw = int(f.getlength(it['text'])) + pad * 2
        th = it['size'] * 2 + pad
        im = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        sw = it.get('stroke', max(2, it['size'] // 12))
        sf = tuple(int(c * 255) for c in it.get('stroke_fill', (0.05, 0.03, 0.1))) + (255,)
        if it.get('shadow'):
            off = it['shadow']
            d.text((tw / 2 + off, th / 2 + off), it['text'], font=f, anchor='mm', fill=sf, stroke_width=sw,
                   stroke_fill=sf)
        if it.get('outer'):
            oc = tuple(int(c * 255) for c in it['outer']) + (255,)
            d.text((tw / 2, th / 2), it['text'], font=f, anchor='mm', fill=oc, stroke_width=sw * 2 + 4,
                   stroke_fill=oc)
        fill = tuple(int(c * 255) for c in it.get('fill', (1, 1, 1))) + (255,)
        d.text((tw / 2, th / 2), it['text'], font=f, anchor='mm', fill=fill, stroke_width=sw, stroke_fill=sf)
        if it.get('rotate'):
            im = im.rotate(it['rotate'], resample=Image.BICUBIC, expand=True)
        if it.get('skew'):
            k = it['skew']
            im = im.transform((im.width + int(abs(k) * im.height), im.height), Image.AFFINE,
                              (1, k, -abs(k) * im.height if k > 0 else 0, 0, 1, 0), resample=Image.BICUBIC)
        arr = np.asarray(im, np.float32) / 255
        x, y = it['xy']
        x0, y0 = int(x - arr.shape[1] / 2), int(y - arr.shape[0] / 2)
        out = paste_rgba(out, arr, x0, y0)
    return out


def paste_rgba(dst, src, x0, y0):
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return dst
    s = src[ya - y0:yb - y0, xa - x0:xb - x0]
    d = dst[ya:yb, xa:xb]
    a = s[..., 3:4]
    d[..., :3] = s[..., :3] * a + d[..., :3] * (1 - a)
    d[..., 3:4] = a + d[..., 3:4] * (1 - a)
    dst[ya:yb, xa:xb] = d
    return dst


def over_rgba(img, layer):
    a = layer[..., 3:4]
    return img * (1 - a) + layer[..., :3] * a


def sky_comic(w, h, glow=(0.5, 0.62), night=False, seed=3):
    """Reference-style sky: flat graded magenta -> orange with a big halftone screen whose dots
    swell toward the sun glow (no brush clouds, no blur)."""
    s = h / 1080
    if night:
        img = vgrad(h, w, [(0.0, (0.10, 0.08, 0.32)), (0.45, (0.30, 0.14, 0.52)), (0.75, (0.72, 0.22, 0.58)),
                           (1.0, (0.95, 0.42, 0.50))])
        dot_col = np.array([0.95, 0.45, 0.80])
    else:
        img = vgrad(h, w, [(0.0, (0.86, 0.22, 0.58)), (0.30, (0.96, 0.34, 0.50)), (0.55, (1.0, 0.52, 0.34)),
                           (0.75, (1.0, 0.70, 0.36)), (1.0, (1.0, 0.82, 0.48))])
        dot_col = np.array([1.0, 0.86, 0.40])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - glow[0] * w) / w) ** 2 + ((yy - glow[1] * h) / w) ** 2)
    cov = np.clip(1 - d / 0.55, 0, 1) ** 1.4 * 0.75
    dots = halftone_mask(h, w, cov, 15 * s, 45)
    img = img * (1 - dots[..., None]) + dot_col * dots[..., None]
    # upper sky: darker magenta dots fading out
    cov2 = smoothstep(0.38, 0.0, yy / h) * 0.28
    d2 = halftone_mask(h, w, cov2, 11 * s, 15)
    img = img * (1 - d2[..., None] * 0.9) + (np.array([0.55, 0.12, 0.55]) if not night else np.array([0.05, 0.04, 0.2])) * d2[..., None] * 0.9
    if night:
        rng = np.random.default_rng(seed)
        st = (rng.random((h, w)) > 0.9994).astype(np.float32)
        st = cv2.dilate(st, np.ones((max(2, int(2 * s)), max(2, int(2 * s))), np.uint8))
        img = img + st[..., None] * smoothstep(0.6, 0.0, yy / h)[..., None] * 0.9
    return np.clip(img, 0, 1)
