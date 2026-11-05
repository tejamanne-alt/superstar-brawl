"""03 Character cards (1500x2000 each) + a 3x3 contact sheet."""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from sb import compose as cp, paint, comicfx as fx, roster, exr
from sb.compose import C, INK, text

BUILD = os.path.join(os.path.dirname(__file__), '../build')
OUT = os.path.join(os.path.dirname(__file__), '../concept')

CARD = dict(  # accent, deep, stats (power, speed, style, mass) /10
    bala=((1.0, 0.55, 0.10), (0.55, 0.10, 0.08), (10, 4, 7, 10), 'Rayalaseema faction hero'),
    chiru=((0.95, 0.15, 0.20), (0.25, 0.05, 0.12), (7, 9, 10, 9), 'Street-dance gang leader'),
    nag=((0.25, 0.60, 1.0), (0.06, 0.12, 0.35), (7, 8, 7, 7), 'Angry campus rebel'),
    venky=((0.95, 0.80, 0.25), (0.35, 0.28, 0.10), (7, 6, 7, 8), 'Hard-boiled supercop'),
    ambar=((0.70, 0.80, 0.95), (0.10, 0.12, 0.22), (10, 4, 8, 10), 'Iron-armed vigilante emperor'),
    raja=((1.0, 0.80, 0.15), (0.12, 0.08, 0.16), (8, 8, 10, 10), 'Style-icon don with a past'),
    kamal=((0.20, 0.95, 0.80), (0.05, 0.22, 0.25), (6, 7, 9, 8), 'Veteran pressure-point master'),
    srk=((1.0, 0.30, 0.40), (0.35, 0.04, 0.10), (7, 8, 10, 9), 'Bandaged vigilante with a grin'),
    hriday=((0.30, 0.85, 1.0), (0.04, 0.06, 0.20), (8, 10, 9, 8), 'Masked superhero speedster'),
)


def layer_from_exr(path, style, out_size, glow_alpha=True):
    P, meta = exr.load(path)
    ss = P['depth'].shape[1] / out_size[0]
    L = fx.shade(P, meta, style, scale=ss * 0.9)
    col, a = fx.misregister_layer(L['color'], L['alpha'], 1.6 * ss)
    col = fx.screen(col, L['glow'])
    if glow_alpha:
        a = np.clip(a + fx.lum(L['glow']) * 0.9, 0, 1)
    rgba = np.dstack([col, a]).astype(np.float32)
    rgba = cv2.resize(rgba, out_size, interpolation=cv2.INTER_AREA)
    return rgba


def background(W, H, accent, deep, seed):
    yy = np.linspace(0, 1, H)[:, None, None]
    img = np.array(accent) * (1 - yy) * 0.85 + np.array(deep) * yy + np.zeros((H, W, 3))
    img = img + 0.15 * (1 - yy)
    # radial burst
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W * 0.62, H * 0.42
    ang = np.arctan2(ys - cy, xs - cx)
    rays = (np.sin(ang * 14) > 0).astype(np.float32)
    r = np.hypot(xs - cx, ys - cy) / H
    img = img * (1 - rays[..., None] * 0.10 * fx.smoothstep(1.0, 0.1, r)[..., None]) + \
        rays[..., None] * 0.10 * fx.smoothstep(1.0, 0.1, r)[..., None]
    dots = fx.halftone_mask(H, W, fx.smoothstep(0.2, 1.0, yy[..., 0]) * 0.55, 14, 45)
    img = img * (1 - dots[..., None] * 0.25) + np.array(deep) * 0.4 * dots[..., None] * 0.25
    # skyline silhouette
    sil = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(sil)
    rng = random.Random(seed)
    x = -20
    while x < W:
        w = rng.randint(70, 180)
        h = rng.randint(180, 620)
        d.rectangle([x, H - h - 260, x + w, H], fill=255)
        if rng.random() < 0.5:
            d.rectangle([x + w * 0.3, H - h - 300, x + w * 0.6, H - h - 260], fill=255)
        x += w + rng.randint(-10, 20)
    m = np.asarray(sil, np.float32)[..., None] / 255 * 0.55
    img = img * (1 - m) + np.array(deep) * 0.7 * m
    return np.clip(img, 0, 1)


def stat_bar(d, x, y, w, label, val, accent):
    text(d, (x, y + 18), label, 34, (1, 1, 1), font='Anton-Regular.ttf', anchor='lm', stroke=4)
    bx = x + 170
    for i in range(10):
        x0 = bx + i * (w / 10)
        fill = C(accent) if i < val else C((0.1, 0.08, 0.15), 200)
        d.polygon([(x0 + 10, y), (x0 + w / 10 - 4, y), (x0 + w / 10 - 14, y + 36), (x0, y + 36)], fill=fill,
                  outline=C(INK), width=3)


def card(cid, idx):
    W, H = 1500, 2000
    spec = roster.BY_ID[cid]
    accent, deep, stats, archetype = CARD[cid]
    img = background(W, H, accent, deep, idx)
    style = dict(cp.SUNSET, rim_color=tuple(min(1, c * 1.2) for c in accent))
    # giant ghost name behind the fighter
    ghost = paint.text_layer(W, H, [dict(text=spec['name'].split()[0], xy=(W * 0.55, H * 0.30), size=420,
                                         fill=tuple(min(1, c * 1.15) for c in accent), stroke=0, rotate=90)])
    ghost[..., 3] *= 0.30
    img = paint.over_rgba(img, ghost)
    body = layer_from_exr(f'{BUILD}/cards/{cid}.exr', style, (W, H))
    # lower info panel (behind the fighter so feet can break the frame)
    pil = cp.to_pil(img).convert('RGBA')
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.polygon([(0, H - 640), (W, H - 760), (W, H), (0, H)], fill=C(INK, 225))
    d.polygon([(0, H - 650), (W, H - 770), (W, H - 755), (0, H - 635)], fill=C(accent))
    pil.alpha_composite(ov)
    img = cp.to_np(pil)
    # drop shadow (offset ink silhouette) for the comic cut-out feel
    sh = np.zeros_like(body)
    sh[..., :3] = np.array(INK)
    sh[..., 3] = body[..., 3] * 0.55
    sh = np.roll(np.roll(sh, 18, 0), 18, 1)
    img = paint.over_rgba(img, sh)
    img = paint.over_rgba(img, body)
    pil = cp.to_pil(img).convert('RGBA')
    # portrait inset
    por = Image.open(f'{BUILD}/portraits/{cid}_comic.png').convert('RGBA')
    disc = cp.portrait_disc(por, 420, ring=accent, bg=deep)
    cp.paste(pil, disc, (60, 70))
    d = ImageDraw.Draw(pil)
    text(d, (W - 70, 90), f'{idx:02d}/09', 64, (1, 1, 1), anchor='rm')
    # name plate
    nl = paint.text_layer(W, H, [dict(text=spec['name'], xy=(W * 0.44, H - 560), size=150, fill=accent, stroke=10,
                                      outer=(1, 1, 1), shadow=10, rotate=3, skew=-0.12)])
    pil = cp.to_pil(paint.over_rgba(cp.to_np(pil), nl)).convert('RGBA')
    d = ImageDraw.Draw(pil)
    text(d, (80, H - 445), f"AVATAR:  {spec['avatar'].upper()}", 46, (1, 1, 1), font='Anton-Regular.ttf', anchor='lm',
         stroke=4)
    text(d, (80, H - 392), archetype.upper() + '   •   ' + spec['style'].upper(), 30, accent, font='Anton-Regular.ttf',
         anchor='lm', stroke=3)
    for i, (lab, v) in enumerate(zip(('POWER', 'SPEED', 'STYLE', 'MASS'), stats)):
        stat_bar(d, 80, H - 330 + i * 58, 520, lab, v, accent)
    bx = 800
    d.rounded_rectangle([bx, H - 340, W - 60, H - 205], 14, fill=C((1, 1, 1), 235), outline=C(accent), width=5)
    text(d, (bx + 24, H - 312), 'SIGNATURE', 28, (0.55, 0.05, 0.15), font='Anton-Regular.ttf', anchor='lm', stroke=0)
    text(d, (bx + 24, H - 258), spec['signature'].upper(), 40, INK, anchor='lm', stroke=0)
    d.rounded_rectangle([bx, H - 185, W - 60, H - 50], 14, fill=C(accent, 245), outline=C((1, 1, 1)), width=5)
    text(d, (bx + 24, H - 157), 'SUPER', 28, (0.2, 0.02, 0.08), font='Anton-Regular.ttf', anchor='lm', stroke=0)
    sup = spec['super'].split(' (')[0].upper()
    text(d, (bx + 24, H - 103), sup, 40, INK, anchor='lm', stroke=0)
    text(d, (W / 2, H - 22), '“' + spec['tagline'] + '”', 30, (1, 1, 1), font='Anton-Regular.ttf', stroke=3)
    out = fx.paper(cp.to_np(pil), 0.03)
    fx.save(out, f'{OUT}/03_card_{idx:02d}_{cid}.png')
    return out


def main(ids=None):
    os.makedirs(OUT, exist_ok=True)
    from compose_gameplay import portrait_rgba
    order = [c['id'] for c in roster.ROSTER]
    for cid in order:
        portrait_rgba(cid)
    cards = []
    for i, cid in enumerate(order):
        if ids and cid not in ids:
            continue
        cards.append(card(cid, i + 1))
        print('card', cid)
    if not ids:
        sheet = Image.new('RGB', (500 * 3 + 40, 667 * 3 + 40), (20, 12, 30))
        for i, c in enumerate(cards):
            im = cp.to_pil(c).resize((500, 667), Image.LANCZOS)
            sheet.paste(im, (10 + (i % 3) * 510, 10 + (i // 3) * 677))
        sheet.save(f'{OUT}/03_cards_sheet.png')


if __name__ == '__main__':
    main(sys.argv[1].split(',') if len(sys.argv) > 1 else None)
