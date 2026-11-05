"""07 Style guide / art-bible board: shader breakdown, palettes, rules."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
from PIL import Image, ImageDraw
from sb import compose as cp, paint, comicfx as fx, roster, exr
from sb.compose import C, INK, text
from compose_cards import CARD

BUILD = os.path.join(os.path.dirname(__file__), '../build')
OUT = os.path.join(os.path.dirname(__file__), '../concept')


def stage_img(P, meta, st, final=False, bg=(0.93, 0.88, 0.80)):
    L = fx.shade(P, meta, st, scale=1.4)
    col, a = L['color'], L['alpha']
    if final:
        col, a = fx.misregister_layer(col, a, 2.2)
    img = fx.over(np.ones_like(col) * np.array(bg), col, a)
    img = fx.screen(img, L['glow'])
    if final:
        img = fx.paper(img, 0.04)
    return img


def main():
    P, meta = exr.load(f'{BUILD}/cards/raja.exr')
    h, w = P['depth'].shape
    # raw "Blender viewport" look for comparison
    lin = P['albedo'] * (0.35 + 0.65 * np.clip(P['direct'] / (3 / np.pi), 0, 1))
    raw = fx.over(np.ones((h, w, 3)) * 0.85, fx.to_srgb(lin), P['alpha'])
    base = dict(cp.SUNSET)
    steps = [
        ('1  BLENDER RENDER', 'procedural mesh, one hard sun', raw),
        ('2  CEL BANDS', '3 tones + hue-shifted shadows',
         stage_img(P, meta, dict(base, use_halftone=False, use_hatch=False, use_lines=False, use_rim=False, use_spec=False))),
        ('3  HALFTONE + HATCH', 'dots in the terminator, hatching in AO',
         stage_img(P, meta, dict(base, use_lines=False, use_rim=False, use_spec=False))),
        ('4  INK LINES', 'depth / normal / ID edges, heavier in shadow',
         stage_img(P, meta, dict(base, use_rim=False))),
        ('5  FINAL', 'rim light, spec shapes, misregistration, paper', stage_img(P, meta, base, final=True)),
    ]
    BW, BH = 2560, 1900
    board = Image.new('RGB', (BW, BH), (245, 238, 225))
    d = ImageDraw.Draw(board)
    text(d, (60, 90), 'ART BIBLE  •  THE "FILMVERSE" COMIC SHADER', 92, (0.95, 0.25, 0.25), anchor='lm', stroke=6,
         stroke_fill=INK)
    text(d, (62, 170), 'Every frame = real 3D in Blender/Unity  →  painted like a printed comic.  Same 5 steps run as a URP shader + '
                       'post stack in Unity.', 34, INK, font='Anton-Regular.ttf', anchor='lm', stroke=0)
    # crop upper body
    y0, y1 = int(h * 0.10), int(h * 0.52)
    cw = int((y1 - y0) * 0.75)
    cols = np.where(P['alpha'][y0:y1].max(0) > 0.5)[0]
    cx = int((cols.min() + cols.max()) / 2)
    x0 = int(np.clip(cx - cw / 2, 0, w - cw))
    pw, ph = 470, 627
    for i, (title, sub, im) in enumerate(steps):
        crop = im[y0:y1, x0:x0 + cw]
        crop = cv2.resize(crop.astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA)
        px = 50 + i * (pw + 32)
        board.paste(cp.to_pil(crop), (px, 240))
        d.rectangle([px, 240, px + pw, 240 + ph], outline=(20, 10, 30), width=6)
        text(d, (px + 8, 240 + ph + 40), title, 40, (0.95, 0.25, 0.25) if i == 4 else INK, anchor='lm', stroke=0)
        text(d, (px + 8, 240 + ph + 82), sub, 25, (0.3, 0.25, 0.35), font='Anton-Regular.ttf', anchor='lm', stroke=0)
        if i < 4:
            text(d, (px + pw + 16, 240 + ph / 2), '›', 70, (0.95, 0.25, 0.25), stroke=0)
    # palettes
    yb = 1040
    text(d, (50, yb), 'FIGHTER PALETTES', 54, INK, anchor='lm', stroke=0)
    for i, spec in enumerate(roster.ROSTER):
        pal = spec['palette']
        keys = [k for k in ('skin', 'hair', 'top', 'jacket', 'bottom', 'skirt', 'towel', 'energy') if k in pal][:6]
        cols = [pal[k]['color'] if isinstance(pal[k], dict) else pal[k] for k in keys] + [CARD[spec['id']][0]]
        x = 50 + (i % 3) * 830
        y = yb + 50 + (i // 3) * 112
        text(d, (x, y + 30), spec['name'], 34, INK, anchor='lm', stroke=0)
        for j, c in enumerate(cols):
            cx = x + 270 + j * 76
            d.rounded_rectangle([cx, y, cx + 66, y + 66], 8, fill=C(c)[:3], outline=(20, 10, 30), width=3)
    # rules
    yr = 1440
    text(d, (50, yr), 'RULES', 54, INK, anchor='lm', stroke=0)
    rules = [
        'LIGHT: one hard key per shot; cel bands at 0.32 / 0.16 of N·L. Shadows are hue-shifted (purple; magenta-brown on skin), never grey.',
        'DOTS: 45° halftone, 7 px @1080p, only in the light→shadow terminator. HATCH: -35°, 6 px, deep shadow + AO creases only.',
        'INK: 1.8 px @1080p. Silhouettes ×1.25, inner seams ×0.6, +60% weight on the shadow side. Fades with depth (25–120 m).',
        'DEPTH: no blur. Far planes get print misregistration (red/blue offset) + sunset haze, like an off-register comic.',
        'COLOUR: each fighter owns one accent (cards, HUD, FX). Rim light colour follows the stage lighting preset (SUNSET / NIGHT).',
        'MOTION (in-engine): characters animate on 2s (12 fps) for the comic snap; FX, camera and UI run on 1s (60 fps).',
    ]
    for i, r in enumerate(rules):
        d.ellipse([50, yr + 50 + i * 64 - 9, 68, yr + 50 + i * 64 + 9], fill=(240, 60, 60))
        text(d, (84, yr + 50 + i * 64), r, 29, INK, font='Anton-Regular.ttf', anchor='lm', stroke=0)
    board.save(f'{OUT}/07_style_guide.png')
    print('saved style guide')


if __name__ == '__main__':
    main()
