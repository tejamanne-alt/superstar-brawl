"""Comic FX in the reference look: halftone gradient SFX lettering, starburst impacts,
outlined smoke puffs, speed lines, debris shards. All return float RGBA layers (H, W, 4)."""
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .comicfx import halftone_mask, smoothstep
from .paint import font, paste_rgba

INK = (0.07, 0.04, 0.13)
HOT = [(1.0, 0.92, 0.25), (1.0, 0.62, 0.12), (0.95, 0.22, 0.20)]      # KRUNCH / SMASH
COOL = [(0.70, 1.0, 0.95), (0.30, 0.85, 0.85), (0.15, 0.45, 0.80)]    # TAKEDOWN
PINK = [(1.0, 0.85, 0.95), (1.0, 0.45, 0.75), (0.75, 0.15, 0.55)]
GOLD = [(1.0, 0.97, 0.60), (1.0, 0.84, 0.25), (0.95, 0.55, 0.10)]


def sfx(W, H, text, xy, size, colors=HOT, rotate=0.0, skew=-0.20, shadow=(0.10, 0.10), outline=None,
        fnt='Bangers-Regular.ttf', dot=None, white_edge=False):
    f = font(fnt, size)
    ol = outline or max(3, int(size * 0.075))
    tw = int(f.getlength(text)) + size * 2
    th = int(size * 1.9)
    pad = ol * 3
    cw, ch = tw + pad * 2, th + pad * 2

    def mask(stroke):
        m = Image.new('L', (cw, ch), 0)
        ImageDraw.Draw(m).text((cw / 2, ch / 2), text, font=f, fill=255, anchor='mm', stroke_width=stroke, stroke_fill=255)
        return np.asarray(m, np.float32) / 255
    fill = mask(0)
    edge = mask(ol)
    outer = mask(ol + max(3, int(size * 0.05))) if white_edge else None
    ys = np.where(fill.max(1) > 0)[0]
    y0, y1 = (ys.min(), ys.max()) if len(ys) else (0, ch)
    t = np.clip((np.arange(ch)[:, None] - y0) / max(1, y1 - y0), 0, 1)
    c0, c1, c2 = [np.array(c, np.float32) for c in colors]
    grad = np.where(t[..., None] < 0.5, c0 + (c1 - c0) * (t[..., None] * 2), c1 + (c2 - c1) * (t[..., None] * 2 - 1))
    grad = np.broadcast_to(grad, (ch, cw, 3)).copy()
    dots = halftone_mask(ch, cw, np.broadcast_to(smoothstep(0.15, 1.0, t) * 0.65, (ch, cw)), dot or max(5, size * 0.085), 45)
    grad = grad * (1 - dots[..., None]) + (c2 * 0.85 + c1 * 0.15) * dots[..., None]
    hi = halftone_mask(ch, cw, np.broadcast_to(smoothstep(0.45, 0.0, t) * 0.45, (ch, cw)), dot or max(5, size * 0.085), 15)
    grad = grad * (1 - hi[..., None]) + np.array([1.0, 1.0, 0.94]) * hi[..., None]
    out = np.zeros((ch, cw, 4), np.float32)
    dx, dy = int(shadow[0] * size), int(shadow[1] * size)
    sh = np.roll(np.roll(edge, dy, 0), dx, 1)
    out[..., :3] = np.array(INK)
    out[..., 3] = np.maximum(sh, edge)
    if outer is not None:
        o = np.zeros_like(out)
        o[..., :3] = 1.0
        o[..., 3] = outer
        out = _over(o, out)
    out[..., :3] = out[..., :3] * (1 - fill[..., None]) + grad * fill[..., None]
    out[..., 3] = np.maximum(out[..., 3], fill)
    im = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), 'RGBA')
    if skew:
        k = skew
        im = im.transform((im.width + int(abs(k) * im.height), im.height), Image.AFFINE,
                          (1, k, -abs(k) * im.height if k > 0 else 0, 0, 1, 0), resample=Image.BICUBIC)
    if rotate:
        im = im.rotate(rotate, resample=Image.BICUBIC, expand=True)
    arr = np.asarray(im, np.float32) / 255
    layer = np.zeros((H, W, 4), np.float32)
    return paste_rgba(layer, arr, int(xy[0] - arr.shape[1] / 2), int(xy[1] - arr.shape[0] / 2))


def _over(a, b):
    """b over a for float RGBA."""
    ab = b[..., 3:4]
    out = a.copy()
    out[..., :3] = b[..., :3] * ab + a[..., :3] * (1 - ab)
    out[..., 3:4] = ab + a[..., 3:4] * (1 - ab)
    return out


def burst(W, H, c, r_in, r_out, spikes=14, seed=1, rot=0.0, rays=26, ray_len=2.4):
    rng = random.Random(seed)
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = c
    for k in range(rays):  # thin radiating strokes
        a = rng.uniform(0, 2 * math.pi)
        r0 = r_out * rng.uniform(0.6, 1.0)
        r1 = r_out * rng.uniform(1.2, ray_len)
        col = rng.choice([(255, 230, 70), (255, 150, 40), (255, 250, 230)])
        wdt = max(2, int(r_out * rng.uniform(0.012, 0.03)))
        d.line([(cx + math.cos(a) * r0, cy + math.sin(a) * r0), (cx + math.cos(a) * r1, cy + math.sin(a) * r1)],
               fill=col + (255,), width=wdt)

    def star(rin, rout, fill, outline=None, width=0):
        pts = []
        for k in range(spikes * 2):
            a = rot + math.pi * k / spikes
            r = (rout * random.Random(seed * 100 + k).uniform(0.82, 1.18)) if k % 2 == 0 else rin * random.Random(seed * 7 + k).uniform(0.85, 1.1)
            pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
        d.polygon(pts, fill=fill, outline=outline, width=width)
    star(r_in, r_out, (255, 214, 40, 255), (18, 10, 33, 255), max(3, int(r_out * 0.05)))
    star(r_in * 0.55, r_out * 0.55, (255, 248, 210, 255))
    return np.asarray(im, np.float32) / 255


def puffs(W, H, c, n, r, seed=2, spread=(1.0, 0.45), tint=(0.62, 0.62, 0.95)):
    """Cartoon smoke balls: white fill, ink outline, shaded underside in halftone lavender."""
    rng = random.Random(seed)
    out = np.zeros((H, W, 4), np.float32)
    balls = []
    for i in range(n):
        rr = r * rng.uniform(0.55, 1.0)
        x = c[0] + rng.uniform(-1, 1) * r * 2.2 * spread[0]
        y = c[1] + rng.uniform(-1, 1) * r * 2.2 * spread[1]
        balls.append((x, y, rr))
    balls.sort(key=lambda b: b[1])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for (x, y, rr) in balls:
        x0, x1 = int(max(0, x - rr - 8)), int(min(W, x + rr + 8))
        y0, y1 = int(max(0, y - rr - 8)), int(min(H, y + rr + 8))
        if x0 >= x1 or y0 >= y1:
            continue
        X, Y = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
        dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        ol = max(2.5, rr * 0.07)
        disc = np.clip(rr - dist + 0.5, 0, 1)
        inner = np.clip(rr - ol - dist + 0.5, 0, 1)
        sdist = np.sqrt((X - x + rr * 0.35) ** 2 + (Y - y + rr * 0.40) ** 2)
        lit = np.clip(rr * 0.92 - sdist, 0, 1)
        shade = (1 - lit) * inner
        dots = halftone_mask(y1 - y0, x1 - x0, shade * 0.75, max(4, rr * 0.16), 45)
        col = np.ones((y1 - y0, x1 - x0, 3), np.float32) * np.array([1.0, 0.99, 0.95])
        col = col * (1 - dots[..., None]) + np.array(tint) * dots[..., None]
        col = col * inner[..., None] + np.array(INK) * (1 - inner[..., None])
        blk = out[y0:y1, x0:x1]
        blk[..., :3] = col * disc[..., None] + blk[..., :3] * (1 - disc[..., None])
        blk[..., 3] = disc + blk[..., 3] * (1 - disc)
        out[y0:y1, x0:x1] = blk
    return out


def speed_lines(W, H, center, n=70, inner=0.30, color=INK, seed=3, width=(1.5, 5), alpha=0.85, region=None):
    rng = random.Random(seed)
    im = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(im)
    cx, cy = center
    R = math.hypot(W, H)
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        r0 = (inner + rng.uniform(0, 0.25)) * min(W, H)
        wd = rng.uniform(*width) * H / 1080
        p0 = (cx + math.cos(a) * r0, cy + math.sin(a) * r0)
        p1 = (cx + math.cos(a - 0.0035 * wd) * R, cy + math.sin(a - 0.0035 * wd) * R)
        p2 = (cx + math.cos(a + 0.0035 * wd) * R, cy + math.sin(a + 0.0035 * wd) * R)
        d.polygon([p0, p1, p2], fill=int(255 * alpha))
    m = np.asarray(im, np.float32) / 255
    if region is not None:
        m = m * region
    out = np.zeros((H, W, 4), np.float32)
    out[..., :3] = np.array(color)
    out[..., 3] = m
    return out


def shards(W, H, c, n, size, seed=4, cols=((0.36, 0.82, 0.76), (1.0, 0.84, 0.25), (0.95, 0.30, 0.40))):
    rng = random.Random(seed)
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i in range(n):
        x = c[0] + rng.uniform(-1, 1) * size * 6
        y = c[1] + rng.uniform(-1, 1) * size * 4
        s = size * rng.uniform(0.4, 1.0)
        a = rng.uniform(0, 2 * math.pi)
        pts = [(x + math.cos(a + k * 2.1 + rng.uniform(-0.3, 0.3)) * s, y + math.sin(a + k * 2.1) * s * rng.uniform(0.5, 1))
               for k in range(3)]
        col = tuple(int(v * 255) for v in rng.choice(cols)) + (255,)
        d.polygon(pts, fill=col, outline=(18, 10, 33, 255), width=max(2, int(size * 0.08)))
    return np.asarray(im, np.float32) / 255


def over(img, layer):
    a = layer[..., 3:4]
    return img * (1 - a) + layer[..., :3] * a
