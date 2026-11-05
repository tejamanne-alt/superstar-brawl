"""01 Key art + 02 roster lineup (same render, different overlays)."""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from sb import compose as cp, paint, comicfx as fx, roster
from sb.compose import C, INK, text

BUILD = os.path.join(os.path.dirname(__file__), '../build')
OUT = os.path.join(os.path.dirname(__file__), '../concept')


def film_strip(W, H, p0, p1, width, alpha=0.85):
    """Diagonal film-strip band with sprocket holes ("Reel-verse" glitch motif)."""
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    (x0, y0), (x1, y1) = p0, p1
    L = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    nx, ny = -uy, ux
    w = width / 2
    poly = [(x0 + nx * w, y0 + ny * w), (x1 + nx * w, y1 + ny * w), (x1 - nx * w, y1 - ny * w), (x0 - nx * w, y0 - ny * w)]
    d.polygon(poly, fill=C((0.06, 0.04, 0.10), int(255 * alpha)))
    step = width * 0.32
    n = int(L / step)
    for k in range(n):
        cx, cy = x0 + ux * (k + 0.5) * step, y0 + uy * (k + 0.5) * step
        for side in (-1, 1):
            hx, hy = cx + nx * side * w * 0.78, cy + ny * side * w * 0.78
            s1, s2 = step * 0.28, width * 0.08
            d.polygon([(hx - ux * s1 - nx * s2, hy - uy * s1 - ny * s2), (hx + ux * s1 - nx * s2, hy + uy * s1 - ny * s2),
                       (hx + ux * s1 + nx * s2, hy + uy * s1 + ny * s2), (hx - ux * s1 + nx * s2, hy - uy * s1 + ny * s2)],
                      fill=C((1, 0.95, 0.85), int(255 * alpha)))
        # frame cells
        if k % 3 == 0:
            fxc, fyc = cx, cy
            s1, s2 = step * 1.3, w * 0.55
            d.polygon([(fxc - ux * s1 - nx * s2, fyc - uy * s1 - ny * s2), (fxc + ux * s1 - nx * s2, fyc + uy * s1 - ny * s2),
                       (fxc + ux * s1 + nx * s2, fyc + uy * s1 + ny * s2), (fxc - ux * s1 + nx * s2, fyc - uy * s1 + ny * s2)],
                      fill=C((0.95, 0.35, 0.45), int(120 * alpha)))
    return np.asarray(im, np.float32) / 255


def glitch_blocks(img, n=14, seed=5, region=None):
    rng = random.Random(seed)
    H, W = img.shape[:2]
    out = img.copy()
    for _ in range(n):
        if region:
            x = rng.randint(region[0], region[2]); y = rng.randint(region[1], region[3])
        else:
            x = rng.randint(0, W - 200); y = rng.randint(0, H - 60)
        w, h = rng.randint(60, 260), rng.randint(10, 46)
        dx = rng.randint(-40, 40)
        x2, y2 = min(W, x + w), min(H, y + h)
        src = img[y:y2, max(0, x - dx):max(0, x - dx) + (x2 - x)]
        if src.shape[1] != x2 - x:
            continue
        blk = src.copy()
        blk[..., 0] = np.roll(blk[..., 0], 8, axis=1)
        blk[..., 2] = np.roll(blk[..., 2], -8, axis=1)
        out[y:y2, x:x2] = blk
    return out


def ground_fx(W, H, an):
    """Dust puffs at feet + crack lines under the hero landing."""
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    rng = random.Random(3)
    for k, v in an.items():
        if not k.endswith('_feet'):
            continue
        x, y = v[0], v[1]
        if y > H or y < 0:
            continue
        for i in range(9):
            r = rng.uniform(18, 46)
            ox, oy = rng.uniform(-90, 90), rng.uniform(-10, 18)
            d.ellipse([x + ox - r, y + oy - r * 0.6, x + ox + r, y + oy + r * 0.6], fill=C((1.0, 0.93, 0.85), 150),
                      outline=C(INK, 160), width=3)
    hx, hy = an['hriday_feet'][0] + 60, an['hriday_feet'][1] + 10
    for i in range(9):
        a = rng.uniform(math.pi * 0.05, math.pi * 0.95)
        pts = [(hx, hy)]
        r = 0
        for s in range(5):
            r += rng.uniform(20, 55)
            pts.append((hx + math.cos(a) * r * 1.8 + rng.uniform(-8, 8), hy + math.sin(a) * r * 0.35 + rng.uniform(-4, 4)))
        d.line(pts, fill=C(INK, 230), width=5)
    return np.asarray(lay, np.float32) / 255


def title_logo(W, H, cx, cy, scale=1.0):
    items = [
        dict(text='SUPERSTAR', xy=(cx, cy), size=int(210 * scale), fill=(1.0, 0.86, 0.15), stroke=int(14 * scale),
             outer=(1, 1, 1), shadow=int(16 * scale), rotate=3, skew=-0.12),
        dict(text='BRAWL', xy=(cx + 40 * scale, cy + 190 * scale), size=int(300 * scale), fill=(1.0, 0.24, 0.24),
             stroke=int(18 * scale), outer=(1, 1, 1), shadow=int(20 * scale), rotate=3, skew=-0.12),
    ]
    return paint.text_layer(W, H, items)


def main():
    W, H = 2560, 1440
    sky = paint.sky_sunset(W, H, sun=(0.55, 0.60), seed=11)
    style = dict(cp.SUNSET, rim_color=(1.0, 0.55, 0.35))
    img, P, meta, L = cp.render_frame(f'{BUILD}/keyart.exr', sky, style, haze_rng=(15, 200), haze_amt=0.82,
                                      out_size=(W, H))
    an = meta['meta']['anchors']
    # radial speed lines from behind the centre fighter (upper area only)
    sl = paint.speed_lines(H, W, (0.5, 0.62), n=110, inner=0.28, alpha=0.35, seed=8, width=(2, 6))
    yy = np.linspace(0, 1, H)[:, None]
    sl = sl * fx.smoothstep(0.66, 0.30, yy) * (1 - L['alpha'])
    img = img * (1 - sl[..., None]) + np.array([1.0, 0.96, 0.85]) * sl[..., None]
    img = paint.over_rgba(img, ground_fx(W, H, an))
    # film-strip glitch bands in the corners
    img = paint.over_rgba(img, film_strip(W, H, (-100, 520), (720, -60), 120, 0.9))
    img = paint.over_rgba(img, film_strip(W, H, (1900, 1500), (2700, 900), 120, 0.9))
    img = glitch_blocks(img, 18, 5, region=(0, 0, W - 300, 380))
    base = img.copy()

    # ---- 01 key art
    k = base.copy()
    k = paint.over_rgba(k, title_logo(W, H, W * 0.5, 210, 1.0))
    tl = paint.text_layer(W, H, [
        dict(text='సూపర్ స్టార్', xy=(W * 0.5 - 380, 612), size=44, font='NotoSansTelugu.ttf', fill=(1, 1, 1), stroke=5),
        dict(text='सुपरस्टार', xy=(W * 0.5, 606), size=50, font='TiroDevanagariHindi-Regular.ttf', fill=(1, 1, 1), stroke=5),
        dict(text='சூப்பர் ஸ்டார்', xy=(W * 0.5 + 400, 612), size=40, font='NotoSansTamil.ttf', fill=(1, 1, 1), stroke=5),
        dict(text='NINE LEGENDS.  ONE SCREEN.  NO RETAKES.', xy=(W * 0.5, 1370), size=58, font='Anton-Regular.ttf',
             fill=(1, 1, 1), stroke=6),
        dict(text='iOS  •  ANDROID', xy=(W - 230, 1370), size=40, font='Anton-Regular.ttf', fill=(1.0, 0.86, 0.15), stroke=4),
        dict(text='CONCEPT KEY ART', xy=(200, 1370), size=34, font='Anton-Regular.ttf', fill=(1, 1, 1), stroke=4),
    ])
    k = paint.over_rgba(k, tl)
    k = fx.vignette(k, 0.22)
    fx.save(k, f'{OUT}/01_key_art.png')

    # ---- 02 roster lineup with callouts
    r = base.copy()
    r = paint.over_rgba(r, paint.text_layer(W, H, [
        dict(text='THE ROSTER', xy=(W * 0.5, 120), size=170, fill=(1.0, 0.86, 0.15), stroke=12, outer=(1, 1, 1),
             shadow=12, rotate=2, skew=-0.1),
        dict(text='9 fighters  •  film-inspired avatars  •  parody names', xy=(W * 0.5, 260), size=46,
             font='Anton-Regular.ttf', fill=(1, 1, 1), stroke=5)]))
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for spec in roster.ROSTER:
        hx, hy = an[spec['id'] + '_head'][:2]
        back = spec['id'] in ('ambar', 'nag', 'venky', 'kamal')
        ty = hy - 170 if back else hy - 150
        if spec['id'] == 'hriday':
            ty = hy - 230
        tx = hx + (0 if spec['id'] not in ('chiru',) else -40)
        f = paint.font('Bangers-Regular.ttf', 46)
        tw = f.getlength(spec['name'])
        d.line([(hx, hy - 60), (tx, ty + 40)], fill=C(INK), width=4)
        d.polygon([(tx - tw / 2 - 24, ty - 32), (tx + tw / 2 + 30, ty - 32), (tx + tw / 2 + 18, ty + 44),
                   (tx - tw / 2 - 36, ty + 44)], fill=C((1.0, 0.86, 0.15)), outline=C(INK), width=5)
        text(d, (tx, ty - 4), spec['name'], 46, (0.08, 0.05, 0.12), stroke=0)
        text(d, (tx, ty + 30), spec['avatar'].upper(), 22, (0.55, 0.05, 0.15), font='Anton-Regular.ttf', stroke=0)
    r = paint.over_rgba(r, np.asarray(lay, np.float32) / 255)
    r = fx.vignette(r, 0.2)
    fx.save(r, f'{OUT}/02_roster_lineup.png')
    print('saved key art + roster')


if __name__ == '__main__':
    main()
