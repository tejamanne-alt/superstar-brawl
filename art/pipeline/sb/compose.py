"""High-level compositing: EXR passes -> finished comic frame; HUD / UI drawing helpers."""
import math
import os

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from . import exr, comicfx as fx, paint

SUNSET = dict(key_tint=(1.0, 0.90, 0.76), shadow_tint=(0.52, 0.42, 0.76), skin_shadow_tint=(0.72, 0.44, 0.52),
              rim_color=(1.0, 0.45, 0.55), fill_color=(0.45, 0.45, 1.0), fill_strength=0.10)
NIGHT = dict(key_tint=(0.70, 0.78, 1.0), shadow_tint=(0.30, 0.30, 0.62), skin_shadow_tint=(0.45, 0.34, 0.55),
             rim_color=(0.25, 0.95, 1.0), fill_color=(1.0, 0.3, 0.7), fill_strength=0.18,
             dark_lift=(0.10, 0.12, 0.30))


def render_frame(exr_path, sky, style=None, haze_col=(1.0, 0.84, 0.72), haze_rng=(30, 320), haze_amt=0.7,
                 far_misreg=(25, 90, 3.0), near_misreg=1.2, scale=1.0, line_fade=(25, 120), paper=0.035, out_size=None):
    """out_size=(W,H): the EXR was rendered supersampled; shade at full res, then downsample."""
    P, meta = exr.load(exr_path)
    if out_size is not None:
        ss = P['depth'].shape[1] / out_size[0]
        scale = scale * ss * 0.85
        sky = cv2.resize(sky.astype(np.float32), (P['depth'].shape[1], P['depth'].shape[0]), interpolation=cv2.INTER_CUBIC)
    st = dict(style or {})
    st.setdefault('line_fade_near', line_fade[0])
    st.setdefault('line_fade_far', line_fade[1])
    L = fx.shade(P, meta, st, scale=scale)
    col = paint.haze(L['color'], P['depth'], L['alpha'], haze_col, haze_rng[0], haze_rng[1], haze_amt)
    a = L['alpha']
    img = fx.over(sky, col, a)
    far = fx.smoothstep(far_misreg[0], far_misreg[1], np.where(a > 0.01, P['depth'], 1e4))
    img = fx.misregister(img, far_misreg[2] * scale, mask=far)
    img = fx.misregister(img, near_misreg * scale, mask=(1 - far) * (a > 0.01))
    img = fx.screen(img, L['glow'])
    if out_size is not None:
        img = cv2.resize(img.astype(np.float32), out_size, interpolation=cv2.INTER_AREA)
        L = dict(L, alpha=cv2.resize(L['alpha'], out_size, interpolation=cv2.INTER_AREA),
                 glow=cv2.resize(L['glow'], out_size, interpolation=cv2.INTER_AREA))
        an = meta.get('meta', {}).get('anchors')
        if an:
            f = out_size[0] / P['depth'].shape[1]
            meta['meta']['anchors'] = {k: [v[0] * f, v[1] * f, v[2]] for k, v in an.items()}
    if paper:
        img = fx.paper(img, paper)
    return img, P, meta, L


# ----------------------------------------------------------------------------- PIL helpers

def to_pil(img):
    return Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))


def to_np(im):
    return np.asarray(im.convert('RGB'), np.float32) / 255


def C(c, a=255):
    return tuple(int(x * 255) for x in c) + (a,)


INK = (0.05, 0.03, 0.10)


def slanted_bar(d, x, y, w, h, frac, fill, back=(0.15, 0.1, 0.2), slant=0.25, flip=False, edge=INK, stroke=4,
                ghost=None):
    """Angled health bar. flip mirrors (for P2, fills from the right)."""
    s = h * slant

    def poly(x0, x1):
        return [(x0 + s, y), (x1 + s, y), (x1 - s, y + h), (x0 - s, y + h)]
    d.polygon(poly(x, x + w), fill=C(back))
    if ghost is not None:
        gw = w * ghost
        d.polygon(poly(x + w - gw, x + w) if flip else poly(x, x + gw), fill=C((1.0, 0.95, 0.85)))
    fw = w * frac
    d.polygon(poly(x + w - fw, x + w) if flip else poly(x, x + fw), fill=C(fill))
    # highlight stripe
    hs = h * 0.28
    hp = [(x + s + 6, y + 5), ((x + w - fw if flip else x + fw) - 6 + s, y + 5),
          ((x + w - fw if flip else x + fw) - 6 + s - s * hs / h, y + 5 + hs), (x + s + 6 - s * hs / h, y + 5 + hs)]
    if flip:
        hp = [(x + w - fw + s + 6, y + 5), (x + w + s - 6, y + 5), (x + w + s - 6 - s * hs / h, y + 5 + hs),
              (x + w - fw + s + 6 - s * hs / h, y + 5 + hs)]
    d.polygon(hp, fill=C((1, 1, 1), 110))
    d.polygon(poly(x, x + w), outline=C(edge), width=stroke)


def portrait_disc(img_rgba, size, ring=(1.0, 0.85, 0.2), bg=(0.95, 0.35, 0.3), mirror=False, shape='circle'):
    """Crop a portrait into a framed badge."""
    p = img_rgba.resize((size, size), Image.LANCZOS)
    if mirror:
        p = p.transpose(Image.FLIP_LEFT_RIGHT)
    base = Image.new('RGBA', (size, size), C(bg))
    dots = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dots)
    step = max(6, size // 18)
    for yy in range(0, size, step):
        for xx in range(0, size, step):
            r = step * 0.22 * (1 + yy / size)
            ox = (yy // step % 2) * step / 2
            dd.ellipse([xx + ox - r, yy - r, xx + ox + r, yy + r], fill=C((1, 1, 1), 60))
    base = Image.alpha_composite(base, dots)
    base = Image.alpha_composite(base, p)
    mask = Image.new('L', (size, size), 0)
    md = ImageDraw.Draw(mask)
    if shape == 'circle':
        md.ellipse([0, 0, size - 1, size - 1], fill=255)
    else:
        md.polygon([(size * 0.12, 0), (size, 0), (size * 0.88, size), (0, size)], fill=255)
    out = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    out.paste(base, (0, 0), mask)
    d = ImageDraw.Draw(out)
    w = max(4, size // 22)
    if shape == 'circle':
        d.ellipse([w // 2, w // 2, size - w // 2 - 1, size - w // 2 - 1], outline=C(ring), width=w)
        d.ellipse([0, 0, size - 1, size - 1], outline=C(INK), width=max(2, w // 2))
    else:
        d.polygon([(size * 0.12, 0), (size - 1, 0), (size * 0.88, size - 1), (0, size - 1)], outline=C(ring), width=w)
    return out


def text(d, xy, s, size, fill=(1, 1, 1), font='Bangers-Regular.ttf', stroke=None, stroke_fill=INK, anchor='mm'):
    f = paint.font(font, size)
    d.text(xy, s, font=f, fill=C(fill), stroke_width=stroke if stroke is not None else max(2, size // 12),
           stroke_fill=C(stroke_fill), anchor=anchor)


def round_button(layer, cx, cy, r, label, fill, size=None, glow=False, sub=None, icon=None):
    d = ImageDraw.Draw(layer)
    if glow:
        g = Image.new('RGBA', layer.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(g)
        gd.ellipse([cx - r * 1.35, cy - r * 1.35, cx + r * 1.35, cy + r * 1.35], fill=C(fill, 150))
        g = g.filter(ImageFilter.GaussianBlur(r * 0.25))
        layer.alpha_composite(g)
        d = ImageDraw.Draw(layer)
    d.ellipse([cx - r + 6, cy - r + 8, cx + r + 6, cy + r + 8], fill=C(INK, 160))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=C(fill, 235), outline=C(INK), width=max(3, r // 14))
    d.ellipse([cx - r * 0.78, cy - r * 0.86, cx + r * 0.55, cy - r * 0.15], fill=C((1, 1, 1), 70))
    text(d, (cx, cy + (0 if not sub else -r * 0.12)), label, size or int(r * 0.52))
    if sub:
        text(d, (cx, cy + r * 0.42), sub, int(r * 0.22), font='Anton-Regular.ttf')


def paste(dst, src, xy):
    dst.alpha_composite(src, (int(xy[0]), int(xy[1])))
