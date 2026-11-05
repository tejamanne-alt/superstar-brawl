"""Gameplay mockup: 2.5D fight on Stage 1 with full mobile HUD (2340x1080, 19.5:9)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sb import compose as cp, paint, comicfx as fx, roster
from sb.compose import C, INK, text

BUILD = os.path.join(os.path.dirname(__file__), '../build')
OUT = os.path.join(os.path.dirname(__file__), '../concept')


def portrait_rgba(cid, style=cp.SUNSET):
    path = f'{BUILD}/portraits/{cid}_comic.png'
    if os.path.exists(path):
        return Image.open(path).convert('RGBA')
    from sb import exr
    P, meta = exr.load(f'{BUILD}/portraits/{cid}.exr')
    L = fx.shade(P, meta, style)
    col, a = fx.misregister_layer(L['color'], L['alpha'], 1.5)
    col = fx.screen(col, L['glow'])
    rgba = np.dstack([col, np.clip(a + fx.lum(L['glow']), 0, 1)])
    im = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), 'RGBA')
    im.save(path)
    return im


def main():
    W, H = 2340, 1080
    sky = paint.sky_sunset(W, H, sun=(0.16, 0.52))
    img, P, meta, L = cp.render_frame(f'{BUILD}/gameplay.exr', sky, cp.SUNSET, haze_rng=(18, 220), haze_amt=0.8,
                                      out_size=(W, H))
    an = meta['meta']['anchors']
    gx, gy = an['glasses'][:2]
    hx, hy = an['raja_hand'][:2]
    bx, by = an['bala_fists'][:2]

    # --- shades boomerang: smear trail + speed lines (behind HUD, over scene)
    trail = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(trail)
    for k in range(7):
        t0 = k / 7
        x0 = hx + (gx - hx) * t0
        x1 = hx + (gx - hx) * (t0 + 0.18)
        yy = hy + (gy - hy) * t0 - math.sin(t0 * math.pi) * 40
        td.line([(x0, yy), (x1, yy - 6)], fill=C((1.0, 0.85, 0.25), int(90 + 150 * t0)), width=int(10 + 26 * t0))
        td.line([(x0, yy), (x1, yy - 6)], fill=C((1, 1, 1), int(90 + 150 * t0)), width=int(3 + 8 * t0))
    for k in range(14):
        y = gy + (k - 7) * 11
        L0 = 120 + (k * 37) % 160
        td.line([(gx + 40, y), (gx + 40 + L0, y)], fill=C((1, 1, 1), 200), width=3)
    trail = trail.filter(ImageFilter.GaussianBlur(1.2))
    base = cp.to_pil(img).convert('RGBA')
    base.alpha_composite(trail)
    img = cp.to_np(base)

    # --- impact on Bala's block: burst + radial lines + DHISHUM
    sl = paint.speed_lines(H, W, (bx / W, by / H), n=40, inner=0.13, alpha=0.45, seed=4, width=(1, 3))
    sl = sl * fx.smoothstep(0.32, 0.10, np.hypot((np.arange(W)[None, :] - bx) / W, (np.arange(H)[:, None] - by) / W))
    img = img * (1 - sl[..., None]) + np.array([1.0, 0.97, 0.85]) * sl[..., None]
    b = paint.burst(H, W, (bx + 85, by + 10), 40, 115, spikes=14, seed=9, fill=(1.0, 0.95, 0.35))
    img = paint.over_rgba(img, b)
    tl = paint.text_layer(W, H, [
        dict(text='DHISHUM!', xy=(bx - 40, by - 190), size=150, fill=(1.0, 0.25, 0.25), stroke=10,
             outer=(1, 1, 1), rotate=8, shadow=10),
        dict(text='ఢిష్యుం!', xy=(bx + 150, by - 60), size=64, font='NotoSansTelugu.ttf', fill=(1.0, 0.9, 0.2),
             stroke=6, rotate=-6),
        dict(text='SHADES BOOMERANG!', xy=(gx + 60, gy - 120), size=58, fill=(1.0, 0.85, 0.2), stroke=6,
             rotate=-3, shadow=6),
    ])
    img = paint.over_rgba(img, tl)
    img = fx.vignette(img, 0.18)

    # --- HUD
    hud = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(hud)
    PS = 170
    p1 = cp.portrait_disc(portrait_rgba('bala'), PS, ring=(1.0, 0.82, 0.2), bg=(0.95, 0.45, 0.15))
    p2 = cp.portrait_disc(portrait_rgba('raja'), PS, ring=(1.0, 0.82, 0.2), bg=(0.35, 0.3, 0.75), mirror=True)
    # top band
    d.polygon([(0, 0), (W, 0), (W, 70), (0, 70)], fill=C(INK, 120))
    cp.slanted_bar(d, 215, 46, 840, 56, 0.62, (1.0, 0.80, 0.12), ghost=0.70, stroke=5)
    cp.slanted_bar(d, W - 215 - 840, 46, 840, 56, 0.81, (1.0, 0.80, 0.12), flip=True, ghost=0.81, stroke=5)
    cp.paste(hud, p1, (30, 22))
    cp.paste(hud, p2, (W - 30 - PS, 22))
    text(d, (232, 136), 'BALA SIMHA', 50, (1, 1, 1), anchor='lm')
    text(d, (232, 178), 'FACTION SIMHA', 28, (1.0, 0.82, 0.2), font='Anton-Regular.ttf', anchor='lm', stroke=3)
    text(d, (W - 232, 136), 'RAJA KANTH', 50, (1, 1, 1), anchor='rm')
    text(d, (W - 232, 178), 'STYLE KING BAASHA', 28, (1.0, 0.82, 0.2), font='Anton-Regular.ttf', anchor='rm',
         stroke=3)
    # round wins (film reels)
    for i, won in enumerate((True, False)):
        cx = 1030 - i * 44
        d.ellipse([cx - 16, 122, cx + 16, 154], fill=C((1, 0.8, 0.2) if won else (0.2, 0.2, 0.3)), outline=C(INK),
                  width=3)
        d.ellipse([cx - 5, 133, cx + 5, 143], fill=C(INK))
    for i, won in enumerate((False, False)):
        cx = W - 1030 + i * 44
        d.ellipse([cx - 16, 122, cx + 16, 154], fill=C((1, 0.8, 0.2) if won else (0.2, 0.2, 0.3)), outline=C(INK),
                  width=3)
        d.ellipse([cx - 5, 133, cx + 5, 143], fill=C(INK))
    # timer clapperboard badge
    tx, ty = W // 2, 92
    d.polygon([(tx - 92, ty - 70), (tx + 92, ty - 70), (tx + 80, ty + 70), (tx - 80, ty + 70)], fill=C((0.12, 0.08, 0.2)),
              outline=C((1.0, 0.82, 0.2)), width=6)
    for k in range(6):
        x0 = tx - 92 + k * 32
        d.polygon([(x0, ty - 70), (x0 + 16, ty - 70), (x0 + 8, ty - 48), (x0 - 8, ty - 48)], fill=C((1, 1, 1)))
    text(d, (tx, ty + 14), '47', 96, (1, 1, 1), stroke=7)
    text(d, (tx, ty + 98), 'ROUND 2', 32, (1.0, 0.82, 0.2), font='Anton-Regular.ttf', stroke=4)
    # super meters
    def meter(x, y, w, fill_segments, frac, label, flip=False, color=(1.0, 0.45, 0.1)):
        seg = 3
        sw = w / seg
        for i in range(seg):
            x0 = x + i * sw if not flip else x + w - (i + 1) * sw
            f = 1.0 if i < fill_segments else (frac if i == fill_segments else 0.0)
            d.polygon([(x0 + 10, y), (x0 + sw - 4, y), (x0 + sw - 14, y + 26), (x0, y + 26)], fill=C((0.15, 0.1, 0.2), 220))
            if f > 0:
                fx0 = x0 if not flip else x0 + sw * (1 - f)
                d.polygon([(fx0 + 10, y), (fx0 + sw * f - 4, y), (fx0 + sw * f - 14, y + 26), (fx0, y + 26)],
                          fill=C(color))
            d.polygon([(x0 + 10, y), (x0 + sw - 4, y), (x0 + sw - 14, y + 26), (x0, y + 26)], outline=C(INK), width=3)
        text(d, (x + (0 if not flip else w), y + 52), label, 30, (1, 1, 1), anchor='lm' if not flip else 'rm',
             font='Anton-Regular.ttf', stroke=4)
    meter(232, 210, 420, 2, 0.4, 'MASS METER  ×2', color=(1.0, 0.45, 0.1))
    meter(W - 232 - 420, 210, 420, 1, 0.75, 'STYLE METER  ×1', flip=True, color=(1.0, 0.82, 0.15))
    # combo counter
    cl = paint.text_layer(W, H, [dict(text='7', xy=(150, 430), size=170, fill=(1.0, 0.82, 0.2), stroke=10,
                                      rotate=6, shadow=8, outer=(1, 1, 1)),
                                 dict(text='HITS!', xy=(270, 455), size=78, fill=(1, 1, 1), stroke=7, rotate=6),
                                 dict(text='MASS COMBO', xy=(220, 530), size=40, font='Anton-Regular.ttf',
                                      fill=(1.0, 0.45, 0.2), stroke=4, rotate=6)])
    # joystick
    jx, jy, jr = 250, 840, 150
    d.ellipse([jx - jr, jy - jr, jx + jr, jy + jr], fill=C((1, 1, 1), 40), outline=C((1, 1, 1), 170), width=6)
    for k in range(8):
        a = k * math.pi / 4
        d.polygon([(jx + math.cos(a) * (jr - 12), jy + math.sin(a) * (jr - 12)),
                   (jx + math.cos(a + 0.12) * (jr - 38), jy + math.sin(a + 0.12) * (jr - 38)),
                   (jx + math.cos(a - 0.12) * (jr - 38), jy + math.sin(a - 0.12) * (jr - 38))], fill=C((1, 1, 1), 150))
    kx, ky = jx + 55, jy - 10
    d.ellipse([kx - 72, ky - 72, kx + 72, ky + 72], fill=C((1.0, 0.82, 0.2), 230), outline=C(INK), width=6)
    d.ellipse([kx - 50, ky - 58, kx + 20, ky - 10], fill=C((1, 1, 1), 90))
    # action buttons
    cp.round_button(hud, 1985, 955, 88, 'PUNCH', (0.95, 0.30, 0.25), size=40)
    cp.round_button(hud, 2015, 760, 88, 'KICK', (0.25, 0.55, 0.95), size=44)
    cp.round_button(hud, 1815, 860, 78, 'SPECIAL', (0.65, 0.35, 0.90), size=32)
    cp.round_button(hud, 2215, 690, 70, 'BLOCK', (0.35, 0.75, 0.45), size=32)
    cp.round_button(hud, 2190, 925, 112, 'SUPER', (1.0, 0.70, 0.10), size=50, glow=True, sub='SUMO STORM')
    d = ImageDraw.Draw(hud)
    # pause
    d.rounded_rectangle([W // 2 - 34, 214, W // 2 + 34, 270], 12, fill=C(INK, 170), outline=C((1, 1, 1), 200), width=3)
    d.rectangle([W // 2 - 14, 226, W // 2 - 4, 258], fill=C((1, 1, 1)))
    d.rectangle([W // 2 + 4, 226, W // 2 + 14, 258], fill=C((1, 1, 1)))
    hud_np = np.asarray(hud, np.float32) / 255
    img = paint.over_rgba(img, cl)
    img = paint.over_rgba(img, hud_np)
    fx.save(img, f'{OUT}/04_gameplay_mockup.png')
    print('saved gameplay')


if __name__ == '__main__':
    main()
