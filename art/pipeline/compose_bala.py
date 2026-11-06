"""Bala Simha boards: 10 hero (cover look), 11 model sheet, 12 in-game (ink-dot look)."""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from sb import cover, inkdot, paint, fxart, exr
from sb import compose as cp
from sb.compose import C, text
from sb.comicfx import save, smoothstep, halftone_mask

B = os.path.join(os.path.dirname(__file__), '../build/bala')
OUT = os.path.join(os.path.dirname(__file__), '../concept')
INK = (0.13, 0.07, 0.09)
PAPER = (0.95, 0.92, 0.85)
ACC = (1.0, 0.56, 0.12)


def rgba_over(bg, rgba):
    return bg * (1 - rgba[..., 3:4]) + rgba[..., :3] * rgba[..., 3:4]


def neon(W, H, pts, col, width):
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(lay).line(pts, fill=C(col), width=int(width), joint='curve')
    out = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    for r in (width * 5, width * 2):
        out.alpha_composite(lay.filter(ImageFilter.GaussianBlur(r)))
    out.alpha_composite(lay)
    core = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(core).line(pts, fill=(255, 250, 245, 230), width=max(2, int(width * 0.35)), joint='curve')
    out.alpha_composite(core)
    return np.asarray(out, np.float32) / 255


def painted_backdrop(W, H, seed=3):
    """Single-screen cinema at night, painted loosely: teal sky, warm lobby glow, marquee bulbs,
    neon tubes and a Telugu neon sign, all softened like a background painting."""
    rng = random.Random(seed)
    yy = np.linspace(0, 1, H)[:, None, None]
    img = (np.array([0.09, 0.24, 0.33]) * (1 - yy) + np.array([0.20, 0.17, 0.34]) * yy) * np.ones((H, W, 3))
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    # cinema facade blocks + lobby doors (warm)
    d.rectangle([int(W * 0.05), int(H * 0.30), int(W * 0.95), H], fill=C((0.13, 0.13, 0.25)))
    d.rectangle([int(W * 0.58), int(H * 0.50), int(W * 0.92), H], fill=C((1.0, 0.66, 0.28)))
    d.rectangle([int(W * 0.62), int(H * 0.54), int(W * 0.88), H], fill=C((1.0, 0.86, 0.52)))
    d.rectangle([int(W * 0.04), int(H * 0.44), int(W * 0.96), int(H * 0.49)], fill=C((0.95, 0.30, 0.42)))
    for k in range(48):  # marquee bulbs
        x = W * (0.05 + 0.90 * k / 47)
        d.ellipse([x - 7, H * 0.465 - 7, x + 7, H * 0.465 + 7], fill=C((1.0, 0.92, 0.60)))
    lay = lay.filter(ImageFilter.GaussianBlur(W * 0.006))
    img = rgba_over(img, np.asarray(lay, np.float32) / 255)
    # neon tubes + sign
    tubes = [([(W * 0.08, H * 0.40), (W * 0.30, H * 0.40), (W * 0.30, H * 0.12)], (1.0, 0.35, 0.62)),
             ([(W * 0.92, H * 0.18), (W * 0.70, H * 0.18), (W * 0.70, H * 0.40)], (0.30, 0.95, 0.90))]
    for pts, col in tubes:
        img = rgba_over(img, neon(W, H, pts, col, W * 0.010))
    sign = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    f = paint.font('NotoSansTelugu.ttf', int(W * 0.13))
    ImageDraw.Draw(sign).text((W * 0.18, H * 0.24), 'సింహ', font=f, fill=C((1.0, 0.62, 0.20)), anchor='mm')
    glow = sign.filter(ImageFilter.GaussianBlur(W * 0.012))
    s2 = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    s2.alpha_composite(glow)
    s2.alpha_composite(glow)
    s2.alpha_composite(sign)
    img = rgba_over(img, np.asarray(s2, np.float32) / 255)
    # loose brush texture over everything (painted, not photographic)
    br = cover.brush_field(H, W, H, 30, 0.06, 0.006, seed=seed)
    img = img * (1 + 0.06 * br[..., None])
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), W * 0.0025)
    return np.clip(img, 0, 1)


# ----------------------------------------------------------------------------- 10 hero

def hero():
    W, H = 1500, 2000
    bg = painted_backdrop(W, H)
    rgba, P, meta = cover.layer(f'{B}/fbx_hero.exr')
    # light wrap: background colour bleeding onto the figure's silhouette (painted feel)
    a = rgba[..., 3]
    edge = np.clip(cv2.GaussianBlur(a, (0, 0), 6) - cv2.GaussianBlur(a, (0, 0), 1.5), 0, 1) * a
    blur_bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), 25)
    figure = rgba[..., :3] * (1 - edge[..., None] * 0.35) + blur_bg * edge[..., None] * 0.35
    img = bg * (1 - a[..., None]) + figure * a[..., None]
    # title lockup (own design): name + avatar, bottom left; game mark top left
    img = fxart.over(img, fxart.sfx(W, H, 'BALA SIMHA', (W * 0.36, H - 230), 165, fxart.GOLD, rotate=4, white_edge=True))
    pil = cp.to_pil(img).convert('RGBA')
    d = ImageDraw.Draw(pil)
    d.rectangle([0, H - 140, W, H], fill=C(INK, 235))
    text(d, (60, H - 98), 'FACTION SIMHA', 54, ACC, font='Anton-Regular.ttf', anchor='lm', stroke=0)
    text(d, (60, H - 46), 'Mass Brawler  •  “The Rayalaseema Roar”', 32, (1, 1, 1), font='Anton-Regular.ttf', anchor='lm',
         stroke=0)
    text(d, (W - 60, H - 72), 'FIGHTER 01 / 09', 34, (1, 1, 1), font='Anton-Regular.ttf', anchor='rm', stroke=0)
    text(d, (60, 70), 'SUPERSTAR BRAWL', 44, (1, 1, 1), anchor='lm', stroke=4)
    out = cover.paper(cp.to_np(pil), 0.8)
    save(out, f'{OUT}/10_bala_hero.png')
    print('hero')


# ----------------------------------------------------------------------------- 11 model sheet

def crop_alpha(rgba, pad=12):
    a = rgba[..., 3]
    ys, xs = np.where(a > 0.05)
    y0, y1 = max(0, ys.min() - pad), ys.max() + pad
    x0, x1 = max(0, xs.min() - pad), xs.max() + pad
    return rgba[y0:y1, x0:x1], x0, y0


def paste_rgba(sheet, crop, X, Y):
    h, w = crop.shape[:2]
    h = min(h, sheet.shape[0] - Y)
    w = min(w, sheet.shape[1] - X)
    sheet[Y:Y + h, X:X + w] = rgba_over(sheet[Y:Y + h, X:X + w], crop[:h, :w])


def model_sheet():
    W, H = 3600, 2300
    sheet = np.ones((H, W, 3)) * np.array(PAPER)
    for y in range(0, H, 60):
        sheet[y:y + 1] *= 0.985
    st = dict(rim_amt=0.5)
    # main 3/4 figure with callouts
    rgba, P, meta = cover.layer(f'{B}/fbx_turn_35.exr', style=st)
    crop, cx0, cy0 = crop_alpha(rgba)
    s = 1880 / crop.shape[0]
    crop = cv2.resize(crop, (int(crop.shape[1] * s), 1880), interpolation=cv2.INTER_AREA)
    MX, MY = 90, 320
    paste_rgba(sheet, crop, MX, MY)
    an = meta['meta']['anchors']
    # small turnaround: front / side / back
    tx0 = 1880
    for i, (yaw, lab) in enumerate(((0, 'FRONT'), (90, 'SIDE'), (180, 'BACK'))):
        r2, _, _ = cover.layer(f'{B}/fbx_turn_{yaw}.exr', style=st)
        c2, _, _ = crop_alpha(r2)
        s2 = 1020 / c2.shape[0]
        c2 = cv2.resize(c2, (int(c2.shape[1] * s2), 1020), interpolation=cv2.INTER_AREA)
        paste_rgba(sheet, c2, tx0 + i * 570 + (520 - c2.shape[1]) // 2, 320)
    # head close-ups
    hw = 400
    for i, yaw in enumerate((0, 35, 90)):
        r3, _, _ = cover.layer(f'{B}/fbx_head_{yaw}.exr', style=st)
        h, w = r3.shape[:2]
        c3 = cv2.resize(r3[int(h * 0.05):int(h * 0.85), int(w * 0.1):int(w * 0.9)], (hw, hw), interpolation=cv2.INTER_AREA)
        X, Y = tx0 + i * (hw + 25), 1500
        bgc = np.ones((hw, hw, 3)) * np.array([0.88, 0.86, 0.80])
        sheet[Y:Y + hw, X:X + hw] = rgba_over(bgc, c3)
    pil = cp.to_pil(sheet).convert('RGBA')
    d = ImageDraw.Draw(pil)
    for i, lab in enumerate(('FRONT', 'SIDE', 'BACK')):
        text(d, (tx0 + i * 570 + 260, 1385), lab, 40, INK, font='Anton-Regular.ttf', stroke=0)
    for i, lab in enumerate(('FACE', '3/4', 'PROFILE')):
        X = tx0 + i * (hw + 25)
        d.rectangle([X, 1500, X + hw, 1500 + hw], outline=C(INK), width=4)
        text(d, (X + hw / 2, 1500 + hw + 32), lab, 30, INK, font='Anton-Regular.ttf', stroke=0)
    text(d, (90, 120), 'BALA SIMHA', 120, INK, font='Bangers-Regular.ttf', anchor='lm', stroke=0)
    text(d, (790, 132), 'MODEL SHEET  •  FACTION SIMHA  •  from the supplied FBX sculpt', 40, INK, font='Anton-Regular.ttf',
         anchor='lm', stroke=0)
    text(d, (90, 215), 'FBX sculpt segmented into skin / hair / beard / mustache / aviators / gold / shirt / dhoti / sandals in '
                       'Blender, painted by the cover shader.', 30, (0.35, 0.28, 0.32), font='Anton-Regular.ttf', anchor='lm',
         stroke=0)

    def P_(name):
        v = an.get(name)
        if not v:
            return None
        return (MX + (v[0] - cx0) * s, MY + (v[1] - cy0) * s)
    notes = [('hair', 'Big wavy swept-back hair, dark with grey sides'), ('tilak', 'Red tilak on the forehead'),
             ('lens', 'Gold-wire aviators, dark olive lenses'), ('mustache', 'Thick dark mustache drooping into the beard'),
             ('beard', 'Short salt-and-pepper beard, greyest at the chin'), ('pocket', 'Black shirt, sleeves rolled to the elbow'),
             ('bracelet', 'Beaded bracelet (right wrist)'), ('watch', 'Gold watch (left wrist)'),
             ('dhoti', 'Cream dhoti gathered in hand: the mass walk'), ('sandal', 'Kolhapuri sandals')]
    tx = 1080
    k = 0
    for name, lab in notes:
        p = P_(name)
        if p is None:
            continue
        ty = 380 + k * 175
        k += 1
        d.line([p, (tx - 20, ty)], fill=C((0.85, 0.20, 0.25)), width=4)
        d.ellipse([p[0] - 9, p[1] - 9, p[0] + 9, p[1] + 9], fill=C((0.85, 0.20, 0.25)))
        d.rounded_rectangle([tx - 20, ty - 30, tx + 700, ty + 30], 10, fill=C((1, 1, 1), 215), outline=C(INK), width=3)
        text(d, (tx, ty), lab, 29, INK, font='Anton-Regular.ttf', anchor='lm', stroke=0)
    pal = [((0.80, 0.56, 0.42), 'skin'), ((0.12, 0.10, 0.11), 'hair'), ((0.64, 0.63, 0.66), 'grey'),
           ((0.17, 0.17, 0.21), 'shirt'), ((0.95, 0.91, 0.80), 'dhoti'), ((0.95, 0.76, 0.30), 'gold'),
           ((0.84, 0.08, 0.12), 'tilak'), ((0.16, 0.19, 0.14), 'lens'), (ACC, 'accent')]
    px0, py0 = 3185, 1500
    text(d, (px0, py0 - 40), 'PALETTE', 40, INK, font='Bangers-Regular.ttf', anchor='lm', stroke=0)
    for i, (c, lab) in enumerate(pal):
        X = px0 + (i % 3) * 125
        Y = py0 + (i // 3) * 150
        d.rounded_rectangle([X, Y, X + 105, Y + 95], 10, fill=C(c), outline=C(INK), width=3)
        text(d, (X + 52, Y + 118), lab.upper(), 22, INK, font='Anton-Regular.ttf', stroke=0)
    save(cover.paper(cp.to_np(pil), 0.6), f'{OUT}/11_bala_model_sheet.png')
    print('model sheet')


# ----------------------------------------------------------------------------- 12 in-game

def ingame():
    W, H = 2340, 1080
    P, meta = exr.load(f'{B}/fbx_ingame.exr')
    an = meta['meta']['anchors']
    rw, rh = P['depth'].shape[1], P['depth'].shape[0]
    vx, vy = an['vanish'][0] / rw, an['vanish'][1] / rh
    sky = paint.sky_comic(W, H, glow=(vx, vy))
    img, P, meta, S = inkdot.render_frame(f'{B}/fbx_ingame.exr', sky, 'sunset', out_size=(W, H),
                                          haze=((1.0, 0.66, 0.42), 40, 170, 0.55))
    an = meta['meta']['anchors']
    fx_, fy_ = an['sandal'][:2]
    img = fxart.over(img, fxart.puffs(W, H, (fx_ - 60, fy_ + 6), 8, 40, seed=4, spread=(1.6, 0.25)))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    reg = smoothstep(0.55, 0.95, yy / H) * (smoothstep(0.30, 0.0, xx / W) + smoothstep(0.70, 1.0, xx / W))
    img = fxart.over(img, fxart.speed_lines(W, H, (vx * W, vy * H), n=110, inner=0.2, alpha=0.9, width=(1, 3), region=reg))
    img = fxart.over(img, fxart.sfx(W, H, 'MASS ENTRY!', (W * 0.25, H * 0.30), 150, fxart.HOT, rotate=6, white_edge=True))
    img = fxart.over(img, fxart.sfx(W, H, 'జై బాలయ్య!', (W * 0.23, H * 0.47), 70, fxart.GOLD, rotate=6,
                                    fnt='NotoSansTelugu.ttf', skew=0, shadow=(0.06, 0.06)))
    pil = cp.to_pil(img).convert('RGBA')
    d = ImageDraw.Draw(pil)
    d.polygon([(W - 760, H - 170), (W - 40, H - 170), (W - 70, H - 50), (W - 790, H - 50)], fill=C((0.10, 0.06, 0.16), 235),
              outline=C((1.0, 0.86, 0.25)), width=6)
    text(d, (W - 410, H - 132), 'BALA SIMHA', 64, (1, 1, 1), stroke=5)
    text(d, (W - 410, H - 80), 'INTRO POSE  •  same model, in-game ink-dot look', 28, (1.0, 0.86, 0.25),
         font='Anton-Regular.ttf', stroke=0)
    save(cp.to_np(pil), f'{OUT}/12_bala_ingame.png')
    print('ingame')


ALL = dict(hero=hero, sheet=model_sheet, ingame=ingame)
if __name__ == '__main__':
    for n in (sys.argv[1:] or list(ALL)):
        ALL[n]()
