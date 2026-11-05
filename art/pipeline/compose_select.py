"""05 Character select screen mockup (2340x1080)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from sb import compose as cp, paint, comicfx as fx, roster
from sb.compose import C, INK, text
from compose_cards import layer_from_exr, CARD
from compose_keyart import film_strip

BUILD = os.path.join(os.path.dirname(__file__), '../build')
OUT = os.path.join(os.path.dirname(__file__), '../concept')


def tile(por, w, h, accent, deep, name, sel=None):
    """Slanted portrait tile like a cinema lobby card."""
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    sk = w * 0.12
    mask = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask).polygon([(sk, 0), (w, 0), (w - sk, h), (0, h)], fill=255)
    bg = Image.new('RGBA', (w, h), C(deep))
    d = ImageDraw.Draw(bg)
    for y in range(0, h, 14):
        for x in range(0, w, 14):
            r = 2.5 + 3.5 * y / h
            d.ellipse([x - r, y - r, x + r, y + r], fill=C(accent, 90))
    p = por.resize((int(w * 1.0), int(w * 1.0)), Image.LANCZOS)
    bg.alpha_composite(p, (0, int(h - w * 1.0 + h * 0.18)))
    d = ImageDraw.Draw(bg)
    d.rectangle([0, h - 56, w, h], fill=C(INK, 230))
    text(d, (w / 2, h - 28), name, 34, (1, 1, 1), stroke=3)
    im.paste(bg, (0, 0), mask)
    d = ImageDraw.Draw(im)
    if sel:
        col, lab = sel
        d.polygon([(sk, 0), (w - 1, 0), (w - sk, h - 1), (0, h - 1)], outline=C(col), width=10)
        d.polygon([(sk + 6, 6), (sk + 90, 6), (sk + 76, 52), (sk - 8, 52)], fill=C(col), outline=C(INK), width=3)
        text(d, (sk + 40, 29), lab, 36, INK, stroke=0)
    else:
        d.polygon([(sk, 0), (w - 1, 0), (w - sk, h - 1), (0, h - 1)], outline=C(INK), width=5)
    return im


def main():
    W, H = 2340, 1080
    yy = np.linspace(0, 1, H)[:, None, None]
    img = np.array([0.20, 0.08, 0.35]) * (1 - yy) + np.array([0.05, 0.03, 0.12]) * yy + np.zeros((H, W, 3))
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = np.arctan2(ys - H * 0.55, xs - W * 0.5)
    rays = (np.sin(ang * 22) > 0.2).astype(np.float32) * 0.07
    img = img + rays[..., None] * np.array([1.0, 0.6, 0.9])
    dots = fx.halftone_mask(H, W, fx.smoothstep(0.0, 1.0, yy[..., 0]) * 0.5, 12, 45)
    img = img * (1 - dots[..., None] * 0.25)
    # left / right player panels
    pl = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(pl)
    a1, d1 = CARD['chiru'][0], CARD['chiru'][1]
    a2, d2 = CARD['srk'][0], CARD['srk'][1]
    d.polygon([(0, 0), (720, 0), (560, H), (0, H)], fill=C(a1, 235))
    d.polygon([(W, 0), (W - 720, 0), (W - 560, H), (W, H)], fill=C(a2, 235))
    img = paint.over_rgba(img, np.asarray(pl, np.float32) / 255)
    img = paint.over_rgba(img, film_strip(W, H, (640, -40), (500, 1120), 90, 0.9))
    img = paint.over_rgba(img, film_strip(W, H, (W - 640, -40), (W - 500, 1120), 90, 0.9))
    style = cp.SUNSET
    for cid, x0, flip in (('chiru', -330, False), ('srk', W - 900, True)):
        body = layer_from_exr(f'{BUILD}/cards/{cid}.exr', style, (930, 1240))
        if flip:
            body = body[:, ::-1]
        canvas = np.zeros((H, W, 4), np.float32)
        canvas = paint.paste_rgba(canvas, body, int(x0), -60)
        sh = canvas.copy()
        sh[..., :3] = np.array(INK)
        sh[..., 3] *= 0.6
        sh = np.roll(sh, 16, 1)
        img = paint.over_rgba(img, sh)
        img = paint.over_rgba(img, canvas)
    pil = cp.to_pil(img).convert('RGBA')
    # grid of 9
    order = [c['id'] for c in roster.ROSTER]
    tw, th, gap = 282, 214, 12
    gx0 = W / 2 - (3 * tw + 2 * gap) / 2
    gy0 = 228
    for i, cid in enumerate(order):
        por = Image.open(f'{BUILD}/portraits/{cid}_comic.png').convert('RGBA')
        sel = ((1.0, 0.85, 0.15), '1P') if cid == 'chiru' else (((0.25, 0.9, 1.0), '2P') if cid == 'srk' else None)
        t = tile(por, tw, th, CARD[cid][0], CARD[cid][1], roster.BY_ID[cid]['name'], sel)
        cx = gx0 + (i % 3) * (tw + gap) - (i // 3) * 24 + 24
        cy = gy0 + (i // 3) * (th + gap)
        pil.alpha_composite(t, (int(cx), int(cy)))
    d = ImageDraw.Draw(pil)
    # titles
    tl = paint.text_layer(W, H, [
        dict(text='CHOOSE YOUR SUPERSTAR', xy=(W / 2, 92), size=104, fill=(1.0, 0.86, 0.15), stroke=8, outer=(1, 1, 1),
             shadow=8, skew=-0.1),
        dict(text='మీ సూపర్ స్టార్‌ను ఎంచుకోండి', xy=(W / 2, 192), size=40, font='NotoSansTelugu.ttf', fill=(1, 1, 1),
             stroke=4),
        dict(text='VS', xy=(W / 2, 985), size=110, fill=(1.0, 0.25, 0.25), stroke=10, outer=(1, 1, 1), shadow=8,
             rotate=-6),
        dict(text='MEGA CHIRAN', xy=(300, 900), size=96, fill=(1, 1, 1), stroke=8, rotate=4, skew=-0.12),
        dict(text='GANG LEADER', xy=(300, 985), size=44, font='Anton-Regular.ttf', fill=(1.0, 0.86, 0.15), stroke=5),
        dict(text='SHAH RAHUL', xy=(W - 300, 900), size=96, fill=(1, 1, 1), stroke=8, rotate=-4, skew=-0.12),
        dict(text='BANDAGE JAWAN', xy=(W - 300, 985), size=44, font='Anton-Regular.ttf', fill=(1.0, 0.86, 0.15), stroke=5),
    ])
    pil = cp.to_pil(paint.over_rgba(cp.to_np(pil), tl)).convert('RGBA')
    d = ImageDraw.Draw(pil)
    # stage chip + ready
    d.rounded_rectangle([W / 2 - 480, H - 140, W / 2 - 110, H - 60], 16, fill=C(INK, 230), outline=C((1, 1, 1)), width=4)
    text(d, (W / 2 - 295, H - 116), 'STAGE', 26, (1.0, 0.86, 0.15), font='Anton-Regular.ttf', stroke=0)
    text(d, (W / 2 - 295, H - 82), 'SINGLE SCREEN SHOWDOWN', 30, (1, 1, 1), stroke=0)
    d.rounded_rectangle([W / 2 + 110, H - 140, W / 2 + 480, H - 60], 16, fill=C((1.0, 0.86, 0.15)), outline=C(INK), width=5)
    text(d, (W / 2 + 295, H - 100), 'READY!', 54, INK, stroke=0)
    out = fx.paper(cp.to_np(pil), 0.03)
    fx.save(out, f'{OUT}/05_character_select.png')
    print('saved select')


if __name__ == '__main__':
    main()
