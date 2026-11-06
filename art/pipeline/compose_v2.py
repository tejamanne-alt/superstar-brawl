"""v2 concept boards in the INK-DOT look (flat colour, light as dots, ink line, slip + paper).

  python compose_v2.py [keyart cards gameplay select panels meter stages style]   (default: all)
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from sb import inkdot, paint, fxart, roster, exr
from sb import compose as cp
from sb.compose import C, INK, text
from sb.comicfx import halftone_mask, smoothstep, save

BUILD = os.path.join(os.path.dirname(__file__), '../build')
OUT = os.path.join(os.path.dirname(__file__), '../concept')
ORDER = [c['id'] for c in roster.ROSTER]

#           accent               deep                 stats (power, speed, style, mass)  neon word      sfx
CARD = dict(
    bala=((1.0, 0.56, 0.12), (0.45, 0.10, 0.20), (10, 4, 7, 10), 'Simha', 'DHAMM!', 'Rayalaseema faction hero'),
    chiru=((1.0, 0.22, 0.34), (0.30, 0.06, 0.24), (7, 9, 10, 9), 'Gang Leader', 'MEGA!', 'Street-dance gang leader'),
    nag=((0.36, 0.64, 1.0), (0.08, 0.14, 0.40), (7, 8, 7, 7), 'Campus King', 'KRAKK!', 'Angry campus rebel'),
    venky=((1.0, 0.82, 0.25), (0.32, 0.20, 0.10), (7, 6, 7, 8), 'Supercop', 'HALT!', 'Hard-boiled supercop'),
    ambar=((0.60, 0.80, 1.0), (0.12, 0.10, 0.30), (10, 4, 8, 10), 'Shahenshah', 'KLANG!', 'Iron-armed vigilante'),
    raja=((1.0, 0.80, 0.18), (0.16, 0.10, 0.28), (8, 8, 10, 10), 'Style King', 'FLIP!', 'Style-icon don with a past'),
    kamal=((0.25, 0.98, 0.82), (0.06, 0.22, 0.28), (6, 7, 9, 8), 'Senapathy', 'TAK-TAK!', 'Veteran pressure-point master'),
    srk=((1.0, 0.34, 0.50), (0.34, 0.04, 0.16), (7, 8, 10, 9), 'Jawan', 'ZING!', 'Bandaged vigilante with a grin'),
    hriday=((0.30, 0.90, 1.0), (0.04, 0.08, 0.26), (8, 10, 9, 8), 'Wonder', 'ZOOOM!', 'Masked superhero speedster'),
)
MINT = (0.50, 0.82, 0.70)
CREAM = (0.96, 0.93, 0.85)


def F(x):
    return np.clip(x, 0, 1)


def portrait(cid):
    path = f'{BUILD}/portraits/{cid}_v2.png'
    if os.path.exists(path):
        return Image.open(path).convert('RGBA')
    rgba = inkdot.layer(f'{BUILD}/portraits/{cid}.exr', 'neon')
    im = Image.fromarray((F(rgba) * 255).astype(np.uint8), 'RGBA')
    im.save(path)
    return im


def anchors_of(meta):
    return meta.get('meta', {}).get('anchors', {})


# ----------------------------------------------------------------------------- neon backdrop (cover art)

def neon_tube(W, H, pts, col, width, glow=1.0):
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    d.line(pts, fill=C(col), width=int(width), joint='curve')
    g = lay.filter(ImageFilter.GaussianBlur(width * 1.6))
    g2 = lay.filter(ImageFilter.GaussianBlur(width * 4))
    core = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(core).line(pts, fill=C((1, 1, 1), 230), width=max(2, int(width * 0.35)), joint='curve')
    out = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    for im, a in ((g2, glow), (g, glow), (lay, 1.0), (core, 1.0)):
        if a < 1:
            im = Image.fromarray((np.asarray(im).astype(np.float32) * np.array([1, 1, 1, a])).astype(np.uint8))
        out.alpha_composite(im)
    return out


def neon_backdrop(W, H, accent, deep, word, seed=1):
    rng = random.Random(seed)
    yy = np.linspace(0, 1, H)[:, None, None]
    top = np.array([0.08, 0.20, 0.30])
    mid = np.array([0.12, 0.34, 0.42])
    img = np.where(yy < 0.6, top + (mid - top) * (yy / 0.6), mid * (1 - (yy - 0.6) / 0.4 * 0.4) + np.array(deep) * 0.4 *
                   (yy - 0.6) / 0.4) * np.ones((H, W, 3))
    # warm doorway glow on one side
    dx = 0.72 if seed % 2 else 0.18
    x0, x1 = int(W * (dx - 0.16)), int(W * (dx + 0.16))
    y0 = int(H * 0.38)
    door = np.zeros((H, W), np.float32)
    door[y0:, x0:x1] = 1
    door = cv2.GaussianBlur(door, (0, 0), W * 0.02)
    warm = np.array([1.0, 0.62, 0.22])
    img = img * (1 - door[..., None] * 0.85) + warm * door[..., None] * 0.85
    inner = np.zeros((H, W), np.float32)
    inner[y0 + 30:, x0 + 30:x1 - 30] = 1
    img = img * (1 - inner[..., None] * 0.6) + np.array([1.0, 0.86, 0.45]) * inner[..., None] * 0.6
    # Ben-Day screen over the night
    dots = halftone_mask(H, W, np.broadcast_to(smoothstep(0.0, 1.0, yy[..., 0]) * 0.35 + 0.08, (H, W)), 13, 45)
    img = img * (1 - dots[..., None] * 0.35) + np.array([0.02, 0.08, 0.16]) * dots[..., None] * 0.35
    pil = cp.to_pil(img).convert('RGBA')
    # neon arch (two tubes) behind the hero
    cx, cy, r = W * 0.5, H * 0.62, W * 0.42
    for k, col in enumerate((accent, (0.40, 1.0, 0.85))):
        rr = r - k * W * 0.05
        pts = [(cx + math.cos(a) * rr, cy - math.sin(a) * rr * 1.15) for a in np.linspace(math.pi, 0.0, 60)]
        pts = [(cx - rr, H)] + pts + [(cx + rr, H)]
        pil.alpha_composite(neon_tube(W, H, pts, col, W * 0.012))
    # neon script sign
    f = paint.font('Pacifico-Regular.ttf', int(W * 0.085))
    sign = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sign)
    sx, sy = (W * 0.27, H * 0.085) if seed % 2 else (W * 0.73, H * 0.085)
    tw = f.getlength(word)
    sd.rounded_rectangle([sx - tw / 2 - 40, sy - W * 0.075, sx + tw / 2 + 40, sy + W * 0.075], 18, fill=C((0.06, 0.05, 0.12), 235),
                         outline=C((0.25, 0.25, 0.4)), width=6)
    pil.alpha_composite(sign)
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(lay).text((sx, sy), word, font=f, fill=C(accent), anchor='mm')
    glow = lay.filter(ImageFilter.GaussianBlur(W * 0.012))
    pil.alpha_composite(glow)
    pil.alpha_composite(glow)
    pil.alpha_composite(lay)
    core = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(core).text((sx, sy), word, font=f, fill=(255, 255, 255, 140), anchor='mm', stroke_width=0)
    core = core.filter(ImageFilter.GaussianBlur(1.2))
    pil.alpha_composite(core)
    return cp.to_np(pil)


# ----------------------------------------------------------------------------- 01 + 02 key art / roster

def keyart():
    W, H = 2560, 1440
    P, meta = exr.load(f'{BUILD}/keyart.exr')
    an = anchors_of(meta)
    rw = P['depth'].shape[1]
    vx, vy = an['vanish'][0] / rw, an['vanish'][1] / P['depth'].shape[0]
    sky = paint.sky_comic(W, H, glow=(vx, vy))
    img, P, meta, S = inkdot.render_frame(f'{BUILD}/keyart.exr', sky, 'sunset', out_size=(W, H),
                                          haze=((1.0, 0.66, 0.42), 40, 170, 0.55))
    an = anchors_of(meta)
    # white speed lines bursting from the street's vanishing point (behind the cast only)
    sl = fxart.speed_lines(W, H, (vx * W, vy * H), n=90, inner=0.06, color=(1, 0.97, 0.88), alpha=0.55, width=(1.5, 4))
    sl[..., 3] *= (1 - cv2.resize(S['alpha'], (W, H))) if S['alpha'].shape != (H, W) else (1 - S['alpha'])
    sl[..., 3] *= smoothstep(0.75, 0.35, np.linspace(0, 1, H))[:, None]
    img = fxart.over(img, sl)
    for k, v in an.items():
        if k.endswith('_feet') and 0 < v[1] < H:
            img = fxart.over(img, fxart.puffs(W, H, (v[0], v[1] + 8), 4, 26, seed=int(v[0]) % 97, spread=(1.0, 0.2)))
    base = img.copy()
    k = fxart.over(base, fxart.sfx(W, H, 'SUPERSTAR', (W / 2, 105), 140, fxart.GOLD, rotate=2, white_edge=True))
    k = fxart.over(k, fxart.sfx(W, H, 'BRAWL!', (W / 2 + 20, 245), 185, fxart.HOT, rotate=2, white_edge=True))
    tl = paint.text_layer(W, H, [
        dict(text='సూపర్ స్టార్', xy=(W * 0.5 - 400, H - 140), size=44, font='NotoSansTelugu.ttf', fill=(1, 1, 1), stroke=6),
        dict(text='सुपरस्टार', xy=(W * 0.5, H - 145), size=50, font='TiroDevanagariHindi-Regular.ttf', fill=(1, 1, 1), stroke=6),
        dict(text='சூப்பர் ஸ்டார்', xy=(W * 0.5 + 420, H - 140), size=40, font='NotoSansTamil.ttf', fill=(1, 1, 1), stroke=6)])
    k = paint.over_rgba(k, tl)
    pil = cp.to_pil(k).convert('RGBA')
    d = ImageDraw.Draw(pil)
    d.rectangle([0, H - 92, W, H], fill=C(INK, 235))
    text(d, (W / 2, H - 46), 'NINE LEGENDS  •  ONE STREET  •  NO RETAKES', 50, (1, 0.86, 0.25), font='Anton-Regular.ttf', stroke=0)
    text(d, (60, H - 46), 'CONCEPT KEY ART', 30, (1, 1, 1), font='Anton-Regular.ttf', anchor='lm', stroke=0)
    text(d, (W - 60, H - 46), 'iOS  •  ANDROID', 30, (1, 1, 1), font='Anton-Regular.ttf', anchor='rm', stroke=0)
    save(inkdot.paper(cp.to_np(pil), 0.5), f'{OUT}/01_key_art.png')

    r = fxart.over(base, fxart.sfx(W, H, 'THE ROSTER', (W / 2, 140), 170, fxart.GOLD, rotate=2, white_edge=True))
    pil = cp.to_pil(r).convert('RGBA')
    d = ImageDraw.Draw(pil)
    specs = sorted(roster.ROSTER, key=lambda sp: an[sp['id'] + '_head'][0])
    n = len(specs)
    for i, spec in enumerate(specs):
        hx, hy = an[spec['id'] + '_head'][:2]
        acc = CARD[spec['id']][0]
        tx = W * (i + 0.5) / n
        ty = 330 if i % 2 == 0 else 455
        f = paint.font('Bangers-Regular.ttf', 40)
        tw = max(f.getlength(spec['name']), paint.font('Anton-Regular.ttf', 19).getlength(spec['avatar'].upper()))
        d.line([(tx, ty + 44), (hx, hy - 60)], fill=C(INK), width=5)
        d.ellipse([hx - 8, hy - 68, hx + 8, hy - 52], fill=C(INK))
        d.polygon([(tx - tw / 2 - 22, ty - 32), (tx + tw / 2 + 28, ty - 32), (tx + tw / 2 + 16, ty + 44),
                   (tx - tw / 2 - 34, ty + 44)], fill=C((1.0, 0.95, 0.70)), outline=C(INK), width=5)
        d.rectangle([tx - tw / 2 - 27, ty + 17, tx + tw / 2 + 21, ty + 41], fill=C(acc))
        text(d, (tx, ty - 7), spec['name'], 40, INK, stroke=0)
        text(d, (tx, ty + 29), spec['avatar'].upper(), 19, INK, font='Anton-Regular.ttf', stroke=0)
    save(inkdot.paper(cp.to_np(pil), 0.5), f'{OUT}/02_roster_lineup.png')
    print('keyart + roster')


# ----------------------------------------------------------------------------- 03 cards (comic-cover layout)

def stat_bar(d, x, y, w, label, val, accent):
    text(d, (x, y + 16), label, 30, (1, 1, 1), font='Anton-Regular.ttf', anchor='lm', stroke=0)
    bx = x + 130
    for i in range(10):
        x0 = bx + i * (w / 10)
        fill = C(accent) if i < val else C((1, 1, 1), 40)
        d.polygon([(x0 + 8, y), (x0 + w / 10 - 3, y), (x0 + w / 10 - 11, y + 32), (x0, y + 32)], fill=fill,
                  outline=C(INK), width=3)


def card(cid, idx):
    W, H = 1500, 2000
    spec = roster.BY_ID[cid]
    accent, deep, stats, word, sfxw, arche = CARD[cid]
    band = 360
    art_h = H - band
    bg = neon_backdrop(W, art_h, accent, deep, word, seed=idx)
    body = inkdot.layer(f'{BUILD}/cards/{cid}.exr', 'neon', style=dict(rim=tuple(accent)), out_size=(1500, 2000))
    bh = int(art_h * 0.97)
    body = cv2.resize(body, (int(bh * 0.75), bh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((art_h, W, 4), np.float32)
    canvas = paint.paste_rgba(canvas, body, int((W - body.shape[1]) / 2) + 60, art_h - body.shape[0] + 10)
    sh = canvas.copy()
    sh[..., :3] = np.array(INK)
    sh[..., 3] *= 0.5
    sh = np.roll(np.roll(sh, 16, 0), 16, 1)
    art = fxart.over(fxart.over(bg, sh), canvas)
    art = fxart.over(art, fxart.sfx(W, art_h, sfxw, (W * (0.22 if idx % 2 == 0 else 0.78), art_h * 0.20), 130,
                                    fxart.HOT if idx % 3 else fxart.PINK, rotate=8 if idx % 2 else -8))
    art = inkdot.slip(art, 2.5)
    page = np.ones((H, W, 3)) * np.array(MINT)
    page[band:] = art
    pil = cp.to_pil(page).convert('RGBA')
    d = ImageDraw.Draw(pil)
    # title band, cover style
    d.ellipse([46, 34, 186, 104], outline=C(INK), width=6)
    text(d, (116, 69), 'SB', 40, INK, font='ArchivoBlack-Regular.ttf', stroke=0)
    text(d, (204, 52), 'FIGHTER FILE', 26, INK, font='ArchivoBlack-Regular.ttf', anchor='lm', stroke=0)
    text(d, (204, 86), f'No. {idx:02d} OF 09', 22, INK, font='ArchivoBlack-Regular.ttf', anchor='lm', stroke=0)
    text(d, (W - 60, 70), spec['avatar'].upper(), 34, INK, font='ArchivoBlack-Regular.ttf', anchor='rm', stroke=0)
    d.line([(W * 0.42, 108), (W - 60, 108)], fill=C(INK), width=3)
    f = paint.font('ArchivoBlack-Regular.ttf', 170)
    size = 170
    while f.getlength(spec['name']) > W - 110:
        size -= 6
        f = paint.font('ArchivoBlack-Regular.ttf', size)
    d.text((W / 2, 205), spec['name'], font=f, fill=C(INK), anchor='mm')
    sub = f"{spec['style'].upper()}   •   {arche.upper()}"
    text(d, (W / 2, 318), sub, 28, INK, font='ArchivoBlack-Regular.ttf', stroke=0)
    # face roundel, bottom left (the likeness read at a glance)
    disc = cp.portrait_disc(portrait(cid), 290, ring=accent, bg=deep)
    cp.paste(pil, disc, (28, H - 330))
    d = ImageDraw.Draw(pil)
    # stats + moves panel
    px0, py0 = 330, H - 300
    d.rounded_rectangle([px0, py0, W - 50, H - 40], 18, fill=C(INK, 225), outline=C(accent), width=5)
    for i, (lab, v) in enumerate(zip(('POWER', 'SPEED', 'STYLE', 'MASS'), stats)):
        stat_bar(d, px0 + 30, py0 + 26 + i * 58, 470, lab, v, accent)
    mx = px0 + 680
    text(d, (mx, py0 + 40), 'SIGNATURE', 24, accent, font='Anton-Regular.ttf', anchor='lm', stroke=0)
    text(d, (mx, py0 + 82), spec['signature'].upper(), 34, (1, 1, 1), anchor='lm', stroke=0)
    text(d, (mx, py0 + 140), 'SUPER', 24, accent, font='Anton-Regular.ttf', anchor='lm', stroke=0)
    text(d, (mx, py0 + 182), spec['super'].split(' (')[0].upper(), 34, (1, 1, 1), anchor='lm', stroke=0)
    text(d, (mx, py0 + 232), '“' + spec['tagline'] + '”', 24, (1, 1, 1), font='Anton-Regular.ttf', anchor='lm', stroke=0)
    out = inkdot.paper(cp.to_np(pil), 0.8)
    save(out, f'{OUT}/03_card_{idx:02d}_{cid}.png')
    return out


def cards():
    outs = [card(cid, i + 1) for i, cid in enumerate(ORDER)]
    sheet = Image.new('RGB', (500 * 3 + 40, 667 * 3 + 40), (24, 14, 36))
    for i, c in enumerate(outs):
        sheet.paste(cp.to_pil(c).resize((500, 667), Image.LANCZOS), (10 + (i % 3) * 510, 10 + (i // 3) * 677))
    sheet.save(f'{OUT}/03_cards_sheet.png')
    print('cards')


# ----------------------------------------------------------------------------- 04 gameplay

def hud(W, H, p1, p2, n1, a1, n2, a2, super_name, combo=7, timer=47):
    hudl = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(hudl)
    PS = 170
    d1 = cp.portrait_disc(portrait(p1), PS, ring=(1.0, 0.84, 0.2), bg=CARD[p1][1])
    d2 = cp.portrait_disc(portrait(p2), PS, ring=(1.0, 0.84, 0.2), bg=CARD[p2][1], mirror=True)
    cp.slanted_bar(d, 215, 46, 840, 56, 0.62, (1.0, 0.84, 0.18), back=(0.20, 0.12, 0.38), ghost=0.70, stroke=6)
    cp.slanted_bar(d, W - 215 - 840, 46, 840, 56, 0.81, (1.0, 0.84, 0.18), back=(0.20, 0.12, 0.38), flip=True,
                   ghost=0.81, stroke=6)
    cp.paste(hudl, d1, (30, 22))
    cp.paste(hudl, d2, (W - 30 - PS, 22))
    d = ImageDraw.Draw(hudl)
    text(d, (232, 138), n1, 52, (1, 1, 1), anchor='lm', stroke=6)
    text(d, (232, 182), a1, 28, (1.0, 0.86, 0.25), font='Anton-Regular.ttf', anchor='lm', stroke=4)
    text(d, (W - 232, 138), n2, 52, (1, 1, 1), anchor='rm', stroke=6)
    text(d, (W - 232, 182), a2, 28, (1.0, 0.86, 0.25), font='Anton-Regular.ttf', anchor='rm', stroke=4)
    for i, won in enumerate((True, False)):
        for cx in (1030 - i * 44, W - 1030 + i * 44):
            w2 = won and cx < W / 2
            d.ellipse([cx - 16, 122, cx + 16, 154], fill=C((1, 0.84, 0.2) if w2 else (0.25, 0.18, 0.45)), outline=C(INK), width=4)
            d.ellipse([cx - 5, 133, cx + 5, 143], fill=C(INK))
    tx, ty = W // 2, 92
    d.polygon([(tx - 92, ty - 70), (tx + 92, ty - 70), (tx + 80, ty + 70), (tx - 80, ty + 70)], fill=C((0.20, 0.12, 0.38)),
              outline=C(INK), width=7)
    for k in range(6):
        x0 = tx - 92 + k * 32
        d.polygon([(x0, ty - 70), (x0 + 16, ty - 70), (x0 + 8, ty - 48), (x0 - 8, ty - 48)], fill=C((1, 1, 1)))
    text(d, (tx, ty + 14), str(timer), 96, (1, 1, 1), stroke=7)
    text(d, (tx, ty + 98), 'ROUND 2', 32, (1.0, 0.86, 0.25), font='Anton-Regular.ttf', stroke=5)

    def meter(x, y, w, full, frac, label, flip=False, color=(1.0, 0.45, 0.15)):
        sw = w / 3
        for i in range(3):
            x0 = x + i * sw if not flip else x + w - (i + 1) * sw
            f = 1.0 if i < full else (frac if i == full else 0.0)
            poly = [(x0 + 10, y), (x0 + sw - 4, y), (x0 + sw - 14, y + 26), (x0, y + 26)]
            d.polygon(poly, fill=C((0.20, 0.12, 0.38), 230))
            if f > 0:
                fx0 = x0 if not flip else x0 + sw * (1 - f)
                d.polygon([(fx0 + 10, y), (fx0 + sw * f - 4, y), (fx0 + sw * f - 14, y + 26), (fx0, y + 26)], fill=C(color))
            d.polygon(poly, outline=C(INK), width=4)
        text(d, (x + (0 if not flip else w), y + 52), label, 30, (1, 1, 1), anchor='lm' if not flip else 'rm',
             font='Anton-Regular.ttf', stroke=4)
    meter(232, 214, 420, 2, 0.4, 'MASS METER  ×2')
    meter(W - 232 - 420, 214, 420, 1, 0.75, 'STYLE METER  ×1', flip=True, color=(1.0, 0.84, 0.18))
    jx, jy, jr = 250, 840, 150
    d.ellipse([jx - jr, jy - jr, jx + jr, jy + jr], fill=C((1, 1, 1), 50), outline=C(INK), width=7)
    d.ellipse([jx - jr + 7, jy - jr + 7, jx + jr - 7, jy + jr - 7], outline=C((1, 1, 1), 200), width=4)
    kx, ky = jx + 55, jy - 10
    d.ellipse([kx - 72, ky - 72, kx + 72, ky + 72], fill=C((1.0, 0.84, 0.2)), outline=C(INK), width=7)
    d.ellipse([kx - 50, ky - 58, kx + 20, ky - 12], fill=C((1, 1, 1), 110))
    cp.round_button(hudl, 1985, 955, 88, 'PUNCH', (0.95, 0.28, 0.32), size=40)
    cp.round_button(hudl, 2015, 760, 88, 'KICK', (0.30, 0.58, 0.98), size=44)
    cp.round_button(hudl, 1815, 860, 78, 'SPECIAL', (0.62, 0.40, 0.92), size=32)
    cp.round_button(hudl, 2215, 690, 70, 'BLOCK', (0.30, 0.80, 0.70), size=32)
    cp.round_button(hudl, 2190, 925, 112, 'SUPER', (1.0, 0.72, 0.12), size=50, glow=True, sub=super_name)
    d = ImageDraw.Draw(hudl)
    d.rounded_rectangle([W // 2 - 34, 214, W // 2 + 34, 270], 12, fill=C(INK, 200), outline=C((1, 1, 1), 220), width=3)
    d.rectangle([W // 2 - 14, 226, W // 2 - 4, 258], fill=C((1, 1, 1)))
    d.rectangle([W // 2 + 4, 226, W // 2 + 14, 258], fill=C((1, 1, 1)))
    lay = np.asarray(hudl, np.float32) / 255
    lay = fxart._over(lay, fxart.sfx(W, H, f'{combo} HITS!', (240, 440), 120, fxart.GOLD, rotate=6))
    lay = fxart._over(lay, fxart.sfx(W, H, 'MASS COMBO', (230, 540), 52, fxart.HOT, rotate=6, shadow=(0.06, 0.06)))
    return lay


def gameplay():
    W, H = 2340, 1080
    P, meta = exr.load(f'{BUILD}/gameplay.exr')
    an = anchors_of(meta)
    rw, rh = P['depth'].shape[1], P['depth'].shape[0]
    vx, vy = an['vanish'][0] / rw, an['vanish'][1] / rh
    sky = paint.sky_comic(W, H, glow=(vx, vy))
    img, P, meta, S = inkdot.render_frame(f'{BUILD}/gameplay.exr', sky, 'sunset', out_size=(W, H),
                                          haze=((1.0, 0.66, 0.42), 40, 170, 0.55))
    an = anchors_of(meta)
    gx, gy = an['glasses'][:2]
    hx, hy = an['raja_hand'][:2]
    bx, by = an['bala_fists'][:2]
    # dark speed lines in the lower corners converging on the vanishing point (reference look)
    reg = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    reg = smoothstep(0.55, 0.95, yy / H) * (smoothstep(0.30, 0.0, xx / W) + smoothstep(0.70, 1.0, xx / W))
    img = fxart.over(img, fxart.speed_lines(W, H, (vx * W, vy * H), n=120, inner=0.2, alpha=0.9, width=(1, 3), region=reg))
    # shades boomerang: smear + white speed streaks
    tr = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(tr)
    for k in range(8):
        t0 = k / 8
        x0 = hx + (gx - hx) * t0
        x1 = hx + (gx - hx) * (t0 + 0.16)
        y = hy + (gy - hy) * t0 - math.sin(t0 * math.pi) * 40
        td.line([(x0, y), (x1, y - 6)], fill=C((1.0, 0.86, 0.25), int(90 + 160 * t0)), width=int(10 + 26 * t0))
        td.line([(x0, y), (x1, y - 6)], fill=C((1, 1, 1), int(90 + 160 * t0)), width=int(3 + 8 * t0))
    for k in range(12):
        y = gy + (k - 6) * 12
        L = 110 + (k * 37) % 150
        td.line([(gx + 40, y), (gx + 40 + L, y)], fill=C(INK, 230), width=4)
    img = fxart.over(img, np.asarray(tr, np.float32) / 255)
    img = fxart.over(img, fxart.burst(W, H, (bx + 70, by + 10), 40, 110, seed=9))
    img = fxart.over(img, fxart.puffs(W, H, (an['bala_feet'][0] - 40, an['bala_feet'][1] + 10), 7, 34, seed=3))
    img = fxart.over(img, fxart.puffs(W, H, (an['raja_feet'][0], an['raja_feet'][1] + 10), 5, 30, seed=5))
    img = fxart.over(img, fxart.sfx(W, H, 'DHISHUM!', (bx - 10, by - 230), 150, fxart.HOT, rotate=8))
    img = fxart.over(img, fxart.sfx(W, H, 'ఢిష్యుం!', (bx + 210, by - 80), 66, fxart.GOLD, rotate=-6,
                                    fnt='NotoSansTelugu.ttf', skew=0, shadow=(0.06, 0.06)))
    img = fxart.over(img, fxart.sfx(W, H, 'SHADES BOOMERANG!', (gx + 60, gy - 130), 64, fxart.COOL, rotate=-3))
    img = fxart.over(img, hud(W, H, 'bala', 'raja', 'BALA SIMHA', 'FACTION SIMHA', 'RAJA KANTH', 'STYLE KING BAASHA',
                              'SUMO STORM'))
    save(img, f'{OUT}/04_gameplay_mockup.png')
    print('gameplay')


# ----------------------------------------------------------------------------- 05 select screen

def tile(por, w, h, accent, deep, name, sel=None):
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    sk = w * 0.12
    mask = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask).polygon([(sk, 0), (w, 0), (w - sk, h), (0, h)], fill=255)
    bgc = np.ones((h, w, 3)) * np.array(deep)
    dots = halftone_mask(h, w, np.broadcast_to(np.linspace(0.15, 0.6, h)[:, None], (h, w)), 12, 45)
    bgc = bgc * (1 - dots[..., None]) + np.array(accent) * dots[..., None]
    bg = cp.to_pil(bgc).convert('RGBA')
    p = por.resize((w, w), Image.LANCZOS)
    bg.alpha_composite(p, (0, int(h - w * 1.0 + h * 0.20)))
    d = ImageDraw.Draw(bg)
    d.rectangle([0, h - 54, w, h], fill=C(INK, 235))
    text(d, (w / 2, h - 27), name, 32, (1, 1, 1), stroke=3)
    im.paste(bg, (0, 0), mask)
    d = ImageDraw.Draw(im)
    if sel:
        col, lab = sel
        d.polygon([(sk, 0), (w - 1, 0), (w - sk, h - 1), (0, h - 1)], outline=C(col), width=12)
        d.polygon([(sk + 6, 6), (sk + 90, 6), (sk + 76, 52), (sk - 8, 52)], fill=C(col), outline=C(INK), width=4)
        text(d, (sk + 40, 29), lab, 36, INK, stroke=0)
    else:
        d.polygon([(sk, 0), (w - 1, 0), (w - sk, h - 1), (0, h - 1)], outline=C(INK), width=6)
    return im


def select():
    W, H = 2340, 1080
    img = paint.sky_comic(W, H, glow=(0.5, 0.95))
    pl = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(pl)
    a1, d1 = CARD['chiru'][:2]
    a2, d2 = CARD['srk'][:2]
    d.polygon([(0, 0), (720, 0), (560, H), (0, H)], fill=C(a1, 255), outline=C(INK), width=10)
    d.polygon([(W, 0), (W - 720, 0), (W - 560, H), (W, H)], fill=C(a2, 255), outline=C(INK), width=10)
    img = fxart.over(img, np.asarray(pl, np.float32) / 255)
    yy = np.linspace(0, 1, H)[:, None]
    for x0, x1, col in ((0, 720, d1), (W - 720, W, d2)):
        dots = halftone_mask(H, x1 - x0, np.broadcast_to(yy * 0.6, (H, x1 - x0)), 14, 45)
        img[:, x0:x1] = img[:, x0:x1] * (1 - dots[..., None] * 0.5) + np.array(col) * dots[..., None] * 0.5
    for cid, x0, flip in (('chiru', -300, False), ('srk', W - 920, True)):
        body = inkdot.layer(f'{BUILD}/cards/{cid}.exr', 'neon', style=dict(rim=CARD[cid][0]), out_size=(930, 1240))
        if flip:
            body = body[:, ::-1].copy()
        canvas = np.zeros((H, W, 4), np.float32)
        canvas = paint.paste_rgba(canvas, body, int(x0), -40)
        sh = canvas.copy()
        sh[..., :3] = np.array(INK)
        sh[..., 3] *= 0.6
        img = fxart.over(img, np.roll(sh, 16, 1))
        img = fxart.over(img, canvas)
    pil = cp.to_pil(img).convert('RGBA')
    tw, th, gap = 282, 214, 12
    gx0 = W / 2 - (3 * tw + 2 * gap) / 2
    for i, cid in enumerate(ORDER):
        sel = ((1.0, 0.86, 0.18), '1P') if cid == 'chiru' else (((0.30, 0.95, 1.0), '2P') if cid == 'srk' else None)
        t = tile(portrait(cid), tw, th, CARD[cid][0], CARD[cid][1], roster.BY_ID[cid]['name'], sel)
        pil.alpha_composite(t, (int(gx0 + (i % 3) * (tw + gap) - (i // 3) * 24 + 24), int(236 + (i // 3) * (th + gap))))
    lay = np.asarray(pil, np.float32) / 255
    out = lay[..., :3]
    out = fxart.over(out, fxart.sfx(W, H, 'CHOOSE YOUR SUPERSTAR!', (W / 2, 100), 104, fxart.GOLD, white_edge=True))
    out = paint.over_rgba(out, paint.text_layer(W, H, [
        dict(text='మీ సూపర్ స్టార్‌ను ఎంచుకోండి', xy=(W / 2, 196), size=38, font='NotoSansTelugu.ttf', fill=(1, 1, 1), stroke=5)]))
    out = fxart.over(out, fxart.sfx(W, H, 'VS!', (W / 2, 985), 120, fxart.HOT, rotate=-6, white_edge=True))
    out = fxart.over(out, fxart.sfx(W, H, 'MEGA CHIRAN', (300, 900), 92, fxart.GOLD, rotate=4))
    out = fxart.over(out, fxart.sfx(W, H, 'SHAH RAHUL', (W - 300, 900), 92, fxart.PINK, rotate=-4))
    pil = cp.to_pil(out).convert('RGBA')
    d = ImageDraw.Draw(pil)
    text(d, (300, 985), 'GANG LEADER', 42, (1, 1, 1), font='Anton-Regular.ttf', stroke=5)
    text(d, (W - 300, 985), 'BANDAGE JAWAN', 42, (1, 1, 1), font='Anton-Regular.ttf', stroke=5)
    d.rounded_rectangle([W / 2 - 480, H - 140, W / 2 - 110, H - 60], 16, fill=C(INK, 235), outline=C((1, 1, 1)), width=4)
    text(d, (W / 2 - 295, H - 116), 'STAGE', 26, (1.0, 0.86, 0.25), font='Anton-Regular.ttf', stroke=0)
    text(d, (W / 2 - 295, H - 82), 'CINEMA STREET', 32, (1, 1, 1), stroke=0)
    d.rounded_rectangle([W / 2 + 110, H - 140, W / 2 + 480, H - 60], 16, fill=C((1.0, 0.86, 0.18)), outline=C(INK), width=6)
    text(d, (W / 2 + 295, H - 100), 'READY!', 54, INK, stroke=0)
    save(inkdot.paper(inkdot.slip(cp.to_np(pil), 2.0), 0.8), f'{OUT}/05_character_select.png')
    print('select')


# ----------------------------------------------------------------------------- 06 action panels page

def panel_img(name, size):
    P, meta = exr.load(f'{BUILD}/panels/{name}.exr')
    sky = paint.sky_comic(size[0], size[1], glow=(0.5, 0.75))
    img, P, meta, S = inkdot.render_frame(f'{BUILD}/panels/{name}.exr', sky, 'sunset', out_size=size,
                                          haze=((1.0, 0.66, 0.42), 40, 170, 0.55), paper_amt=0)
    return img, anchors_of(meta)


def panels():
    W, H = 2560, 1600
    page = np.ones((H, W, 3)) * np.array([0.95, 0.92, 0.84])
    g = 26
    A = [(g, g), (1500, g), (1440, 760), (g, 760)]
    B = [(g, 760 + g), (1430, 760 + g), (1370, H - g), (g, H - g)]
    Cc = [(1500 + g, g), (W - g, g), (W - g, H - g), (1370 + g + 20, H - g), (1440 + g, 760 + g)]

    def place(poly, img):
        nonlocal page
        h, w = img.shape[:2]
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        x0, y0 = int(min(xs)), int(min(ys))
        m = Image.new('L', (W, H), 0)
        ImageDraw.Draw(m).polygon(poly, fill=255)
        m = np.asarray(m, np.float32)[..., None] / 255
        full = page.copy()
        full[y0:y0 + h, x0:x0 + w] = img[:min(h, H - y0), :min(w, W - x0)]
        page = page * (1 - m) + full * m

    # panel A: SWOOSH
    a, an = panel_img('a', (1500, 720))
    gx, gy = an['glasses'][:2]
    reg = np.ones((720, 1500), np.float32)
    reg[:, 900:] = 0.0  # keep the lines off Raja
    reg = cv2.GaussianBlur(reg, (0, 0), 60)
    a = fxart.over(a, fxart.speed_lines(1500, 720, (gx, gy), n=45, inner=0.14, color=(1, 1, 1), alpha=0.55, width=(1, 3),
                                        region=reg))
    a = fxart.over(a, fxart.sfx(1500, 720, 'SWOOSH!', (420, 560), 140, fxart.COOL, rotate=-5))
    place(A, a)
    # panel B: DHISHUM
    b, an = panel_img('b', (1420, 820))
    hx, hy = an['bala_wrR'][:2]
    b = fxart.over(b, fxart.speed_lines(1420, 820, (hx, hy), n=60, inner=0.16, color=(1, 0.97, 0.85), alpha=0.45, width=(1, 3)))
    b = fxart.over(b, fxart.burst(1420, 820, (hx + 30, hy), 40, 110, seed=4, rays=18, ray_len=1.8))
    b = fxart.over(b, fxart.puffs(1420, 820, (an['nag_feet'][0] + 60, an['nag_feet'][1]), 8, 44, seed=6))
    b = fxart.over(b, fxart.sfx(1420, 820, 'DHISHUM!', (430, 170), 160, fxart.HOT, rotate=-5))
    place(B, b)
    # panel C: KRUNCH
    c, an = panel_img('c', (1100, 1560))
    hx, hy = an['ambar_wrR'][:2]
    cx, cy = an['chiru_chest'][:2]
    c = fxart.over(c, fxart.speed_lines(1100, 1560, (cx, cy), n=60, inner=0.22, color=(1.0, 0.97, 0.88), alpha=0.40,
                                        width=(1, 3)))
    c = fxart.over(c, fxart.burst(1100, 1560, (hx * 0.4 + cx * 0.6, hy * 0.4 + cy * 0.6), 50, 140, seed=7, spikes=14,
                                  rays=20, ray_len=1.8))
    c = fxart.over(c, fxart.shards(1100, 1560, (cx + 60, cy - 120), 12, 34))
    c = fxart.over(c, fxart.puffs(1100, 1560, (an['ambar_feet'][0], an['ambar_feet'][1] + 10), 9, 60, seed=8))
    c = fxart.over(c, fxart.sfx(1100, 1560, 'KRUNCH!', (560, 230), 200, fxart.HOT, rotate=8))
    c = fxart.over(c, fxart.sfx(1100, 1560, 'K.O.!', (820, 1380), 150, fxart.COOL, rotate=-8))
    place(Cc, c)
    pil = cp.to_pil(page).convert('RGBA')
    d = ImageDraw.Draw(pil)
    for poly in (A, B, Cc):
        d.polygon(poly, outline=C(INK), width=9)
    # caption boxes
    for (x, y, s) in ((60, 60, 'MEANWHILE, ON CINEMA STREET...'), (60, 1500, 'ONE PUNCH. FULL MASS.')):
        f = paint.font('Anton-Regular.ttf', 34)
        tw = f.getlength(s)
        d.rectangle([x, y, x + tw + 40, y + 56], fill=C((1.0, 0.95, 0.60)), outline=C(INK), width=5)
        text(d, (x + 20, y + 28), s, 34, INK, font='Anton-Regular.ttf', anchor='lm', stroke=0)
    save(inkdot.paper(inkdot.slip(cp.to_np(pil), 2.2), 1.0), f'{OUT}/06_action_panels.png')
    print('panels')


# ----------------------------------------------------------------------------- 07 mass-meter strip

def meter():
    W, H = 3000, 900
    labs = ['MASS ×1', 'MASS ×2', 'FULL MASS!']
    strip = np.ones((H, W, 3)) * np.array([0.95, 0.92, 0.84])
    pw = (W - 4 * 24) // 3
    for i in range(3):
        sky = paint.sky_comic(pw, H - 48, glow=(0.5, 0.7))
        img, P, meta, S = inkdot.render_frame(f'{BUILD}/meter/level{i}.exr', sky, 'sunset', out_size=(pw, H - 48),
                                              haze=((1.0, 0.66, 0.42), 40, 170, 0.55), paper_amt=0,
                                              slip_scale=1.0 + i * 1.4)
        if i >= 1:
            img = fxart.over(img, fxart.speed_lines(pw, H - 48, (pw * 0.5, (H - 48) * 0.55), n=40 + 50 * i,
                                                    inner=0.35 - 0.08 * i, color=(1, 0.95, 0.8), alpha=0.35 + 0.2 * i))
        if i == 2:
            img = fxart.over(img, fxart.sfx(pw, H - 48, 'SUMO STORM!', (pw * 0.5, (H - 48) * 0.86), 92, fxart.HOT, rotate=-4))
        x0 = 24 + i * (pw + 24)
        strip[24:H - 24, x0:x0 + pw] = img
    pil = cp.to_pil(strip).convert('RGBA')
    d = ImageDraw.Draw(pil)
    for i, lab in enumerate(labs):
        x0 = 24 + i * (pw + 24)
        d.rectangle([x0, 24, x0 + pw, H - 24], outline=C(INK), width=9)
        f = paint.font('Anton-Regular.ttf', 52)
        tw = f.getlength(lab)
        d.rectangle([x0 + 24, 48, x0 + 24 + tw + 40, 48 + 80], fill=C((1.0, 0.86, 0.20)), outline=C(INK), width=6)
        text(d, (x0 + 44, 88), lab, 52, INK, font='Anton-Regular.ttf', anchor='lm', stroke=0)
    save(inkdot.paper(cp.to_np(pil), 1.0), f'{OUT}/07_mass_meter.png')
    print('meter')


# ----------------------------------------------------------------------------- 08 stages

NOTES = {
    1: ('STAGE 1', 'CINEMA STREET', 'Golden hour  •  old-city main road', [
        'Release day on a busy cinema street: the single-screen theatre closes the far end',
        'Shopfronts with awnings and hand-painted signs in English, Telugu and Hindi',
        'Black-and-yellow kerbs, bunting, overhead wires, auto-rickshaws, crowd',
        'Fight plane runs across the road; the camera looks down the street (one-point canyon)',
        'Hazard: a runaway auto-rickshaw crosses the plane between rounds',
        'Parallax: road + fighters / near facades / far street / painted sky']),
    2: ('STAGE 2', 'NECKLACE NIGHT', 'Night  •  sea-front promenade', [
        'Promenade sea wall, arc of lamps around the bay, lit skyline',
        'NIGHT preset: cool key, cyan neon rim, magenta bounce dots',
        'Painted night sky: indigo to magenta with a halftone glow',
        'Hazard: high-tide splash pushes fighters toward the corner',
        'Same ink-dot shader, re-lit: proves the look holds at night']),
}


def stages():
    BW, BH = 2560, 2460
    board = Image.new('RGB', (BW, BH), (24, 14, 36))
    d = ImageDraw.Draw(board)
    title = fxart.sfx(BW, 200, 'STAGE CONCEPTS', (430, 100), 110, fxart.GOLD, rotate=0)
    board = cp.to_pil(fxart.over(cp.to_np(board), np.pad(title, ((0, BH - 200), (0, 0), (0, 0)))))
    d = ImageDraw.Draw(board)
    text(d, (BW - 60, 110), '2.5D arenas  •  3D foreground, painted parallax sky', 40, (1, 1, 1), font='Anton-Regular.ttf',
         anchor='rm', stroke=0)
    for n, y in ((1, 220), (2, 1340)):
        W, H = 1920, 1080
        if n == 1:
            P, meta = exr.load(f'{BUILD}/stage1.exr')
            an = anchors_of(meta)
            vx, vy = an['vanish'][0] / P['depth'].shape[1], an['vanish'][1] / P['depth'].shape[0]
            sky = paint.sky_comic(W, H, glow=(vx, vy))
            img, *_ = inkdot.render_frame(f'{BUILD}/stage1.exr', sky, 'sunset', out_size=(W, H),
                                          haze=((1.0, 0.66, 0.42), 40, 170, 0.55))
        else:
            sky = paint.sky_comic(W, H, glow=(0.5, 0.55), night=True)
            img, P, meta, S = inkdot.render_frame(f'{BUILD}/stage2.exr', sky, 'night', out_size=(W, H),
                                                  haze=((0.55, 0.25, 0.60), 60, 420, 0.5))
            e = cv2.resize(np.clip(P['emit'], 0, 3).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
            e = np.roll(e, int(H * 0.012), 0)
            refl = cv2.GaussianBlur(e, (0, 0), sigmaX=2.5, sigmaY=34)
            sea = cv2.resize((P['obj'] == 53).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            stripes = (np.sin(np.arange(H)[:, None, None] * 0.9) > -0.2).astype(np.float32)
            refl = np.clip(refl * 1.4, 0, 1) * sea[..., None] * stripes
            img = 1 - (1 - img) * (1 - refl * 0.8)
        save(img, f'{OUT}/08_stage{n}.png')
        board.paste(cp.to_pil(img), (40, y))
        d = ImageDraw.Draw(board)
        d.rectangle([40, y, 40 + 1920, y + 1080], outline=(18, 10, 33), width=9)
        tag, name, sub, notes = NOTES[n]
        x0 = 2000
        text(d, (x0, y + 40), tag, 44, (1.0, 0.86, 0.25), font='Anton-Regular.ttf', anchor='lm', stroke=0)
        text(d, (x0, y + 110), name.split()[0], 74, (1, 1, 1), anchor='lm', stroke=5)
        text(d, (x0, y + 180), ' '.join(name.split()[1:]), 74, (1, 1, 1), anchor='lm', stroke=5)
        text(d, (x0, y + 250), sub, 26, (1.0, 0.55, 0.70), font='Anton-Regular.ttf', anchor='lm', stroke=0)
        f = paint.font('Anton-Regular.ttf', 28)
        for i, line in enumerate(notes):
            yy = y + 320 + i * 125
            d.ellipse([x0, yy - 9, x0 + 18, yy + 9], fill=(250, 220, 40))
            words, cur, rows = line.split(), '', []
            for w in words:
                if f.getlength(cur + ' ' + w) > 470:
                    rows.append(cur)
                    cur = w
                else:
                    cur = (cur + ' ' + w).strip()
            rows.append(cur)
            for r_i, r in enumerate(rows):
                text(d, (x0 + 34, yy + r_i * 36), r, 28, (0.92, 0.9, 0.95), font='Anton-Regular.ttf', anchor='lm', stroke=0)
    board.save(f'{OUT}/08_stage_concepts.png')
    print('stages')


# ----------------------------------------------------------------------------- 09 style guide

def style():
    P, meta = exr.load(f'{BUILD}/gameplay.exr')
    h, w = P['depth'].shape
    an = anchors_of(meta)
    vx, vy = an['vanish'][0] / w, an['vanish'][1] / h
    sky = paint.sky_comic(w, h, glow=(vx, vy))
    flat_sky = paint.vgrad(h, w, [(0.0, (0.86, 0.22, 0.58)), (0.55, (1.0, 0.52, 0.34)), (1.0, (1.0, 0.82, 0.48))])
    steps = []
    for k in (1, 2, 3, 4):
        S = inkdot.shade(P, meta, 'sunset', steps=min(k, 3))
        bg = flat_sky if k == 1 else sky
        img = S['color'] * S['alpha'][..., None] + bg * (1 - S['alpha'][..., None])
        if k >= 3:
            img = img * (1 - S['ink'][..., None]) + np.array(inkdot.BASE['ink']) * S['ink'][..., None]
        if k == 4:
            img = inkdot.finish(S['color'] * S['alpha'][..., None] + sky * (1 - S['alpha'][..., None]), S['ink'],
                                inkdot.BASE['slip_rel'] * h, glow=S['glow'])
        steps.append(img)
    # crop on Raja (right fighter)
    rx, ry = an['raja_head'][:2]
    cw, ch = int(h * 0.62), int(h * 0.62 * 1.25)
    x0 = int(np.clip(rx - cw * 0.5, 0, w - cw))
    y0 = int(np.clip(ry - ch * 0.22, 0, h - ch))
    BW, BH = 2560, 1880
    board = np.ones((BH, BW, 3)) * np.array([0.95, 0.92, 0.84])
    pw, ph = 600, 750
    labels = ['1  FLAT COLOUR', '2  LIGHT AS DOTS', '3  INK LINE', '4  SLIP + PAPER']
    for i, im in enumerate(steps):
        crop = cv2.resize(im[y0:y0 + ch, x0:x0 + cw].astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA)
        X = 40 + i * (pw + 26)
        board[220:220 + ph, X:X + pw] = crop
    pil = cp.to_pil(board).convert('RGBA')
    d = ImageDraw.Draw(pil)
    for i, lab in enumerate(labels):
        X = 40 + i * (pw + 26)
        d.rectangle([X, 220, X + pw, 220 + ph], outline=C(INK), width=8)
        f = paint.font('Anton-Regular.ttf', 40)
        d.rectangle([X + 16, 236, X + 16 + f.getlength(lab) + 30, 236 + 62], fill=C((1.0, 0.86, 0.20)), outline=C(INK), width=5)
        text(d, (X + 31, 267), lab, 40, INK, font='Anton-Regular.ttf', anchor='lm', stroke=0)
    out = fxart.over(cp.to_np(pil), fxart.sfx(BW, BH, 'THE INK-DOT LOOK', (520, 110), 110, fxart.HOT))
    pil = cp.to_pil(out).convert('RGBA')
    d = ImageDraw.Draw(pil)
    text(d, (1060, 110), 'Real 3D (Blender now, Unity URP in the build), printed like a comic. Characters drawn in a modern teen-comic style.',
         30, INK, font='Anton-Regular.ttf', anchor='lm', stroke=0)
    yb = 1020
    text(d, (40, yb), 'FIGHTER PALETTES', 54, INK, anchor='lm', stroke=0)
    for i, spec in enumerate(roster.ROSTER):
        pal = spec['palette']
        keys = [k for k in ('skin', 'hair', 'top', 'jacket', 'bottom', 'skirt', 'towel') if k in pal][:6]
        cols = [pal[k]['color'] if isinstance(pal[k], dict) else pal[k] for k in keys] + [CARD[spec['id']][0]]
        x = 40 + (i % 3) * 840
        yy = yb + 50 + (i // 3) * 100
        text(d, (x, yy + 30), spec['name'], 34, INK, anchor='lm', stroke=0)
        for j, c in enumerate(cols):
            cx = x + 270 + j * 76
            d.rounded_rectangle([cx, yy, cx + 66, yy + 62], 8, fill=C(c)[:3], outline=(18, 10, 33), width=4)
    yr = 1380
    text(d, (40, yr), 'RULES', 54, INK, anchor='lm', stroke=0)
    rules = [
        'FLAT COLOUR: saturated, no gradients on objects. "Black" is deep indigo so dots still read. Whites are warm cream.',
        'LIGHT AS DOTS: one hard key. Lit = flat colour; turning away = growing dots of the shadow colour; shadow = flat violet.',
        'INK LINE: 4 px @1080p, silhouettes ×1.35, seams ×0.7, creases ×0.6, slight hand wobble. Fades a little with distance.',
        'SLIP + PAPER: colour plates off-register by 2 px under a crisp ink plate; paper grain + flecks over everything.',
        'CHARACTERS: ~7.5 heads tall, sculpted pointed-lock hair, drawn comic faces (heavy lash line, big readable mouths).',
        'FX: halftone-gradient SFX lettering with ink outline + drop shadow, starburst hits, outlined smoke balls, speed lines.',
    ]
    for i, r in enumerate(rules):
        d.ellipse([40, yr + 50 + i * 72 - 10, 60, yr + 50 + i * 72 + 10], fill=C((0.95, 0.24, 0.30)))
        text(d, (80, yr + 50 + i * 72), r, 30, INK, font='Anton-Regular.ttf', anchor='lm', stroke=0)
    save(inkdot.paper(cp.to_np(pil), 0.6), f'{OUT}/09_style_guide.png')
    print('style')


ALL = dict(keyart=keyart, cards=cards, gameplay=gameplay, select=select, panels=panels, meter=meter, stages=stages,
           style=style)

if __name__ == '__main__':
    for name in (sys.argv[1:] or list(ALL)):
        ALL[name]()
