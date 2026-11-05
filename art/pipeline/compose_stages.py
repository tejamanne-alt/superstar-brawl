"""06 Stage concepts board (Stage 1 sunset rooftop + Stage 2 night promenade)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw
from sb import compose as cp, paint, comicfx as fx
from sb.compose import C, INK, text

BUILD = os.path.join(os.path.dirname(__file__), '../build')
OUT = os.path.join(os.path.dirname(__file__), '../concept')

NOTES = {
    1: ('STAGE 1', 'SINGLE SCREEN SHOWDOWN', 'Golden hour  •  Hyderabad-style rooftop', [
        'Rooftop of an old single-screen cinema on release day',
        'Black water tanks, bunting, overhead wires, Sankranti kites',
        'Hand-painted "HOUSEFULL" billboard with marigold garlands',
        'Far skyline + four-minaret monument silhouette',
        'Hazards: tanks can be smashed for a water-burst stun',
        'Parallax: 4 layers (floor / props / city / painted sky)']),
    2: ('STAGE 2', 'NECKLACE NIGHT', 'Night  •  Sea-front promenade', [
        'Sea wall promenade with an arc of lamps on the far bay',
        'Lit window grid + neon colour fill (cyan rim, magenta fill)',
        'Painted night sky with halftone moon halo',
        'Hazard: high-tide wave splash pushes fighters to the corner',
        'Lighting preset: NIGHT (cool key, cyan rim, magenta bounce)',
        'Same city generator as Stage 1, re-seeded and re-lit']),
}


def panel(n):
    W, H = 1920, 1080
    if n == 1:
        sky = paint.sky_sunset(W, H, sun=(0.80, 0.50), seed=5)
        img, P, meta, L = cp.render_frame(f'{BUILD}/stage1.exr', sky, cp.SUNSET, haze_rng=(15, 230), haze_amt=0.8,
                                          out_size=(W, H))
    else:
        sky = paint.sky_night(W, H)
        img, P, meta, L = cp.render_frame(f'{BUILD}/stage2.exr', sky, cp.NIGHT, haze_col=(0.40, 0.20, 0.45),
                                          haze_rng=(30, 420), haze_amt=0.55, out_size=(W, H))
        # painted reflections of every light on the sea (vertical smears)
        import cv2
        e = np.clip(P['emit'], 0, 3).astype(np.float32)
        e = np.roll(e, int(e.shape[0] * 0.012), 0)
        refl = cv2.GaussianBlur(e, (0, 0), sigmaX=2.5, sigmaY=38)
        sea = (P['obj'] == 53).astype(np.float32)
        refl = fx.to_srgb(refl * 1.6) * sea[..., None]
        refl = cv2.resize(refl, (W, H), interpolation=cv2.INTER_AREA)
        stripes = (np.sin(np.arange(H)[:, None, None] * 0.9) > -0.2).astype(np.float32)
        img = fx.screen(img, refl * stripes * 0.8)
    return fx.vignette(img, 0.2)


def main():
    BW, BH = 2560, 2460
    board = Image.new('RGB', (BW, BH), (22, 12, 34))
    d = ImageDraw.Draw(board)
    text(d, (60, 100), 'STAGE CONCEPTS', 120, (1.0, 0.86, 0.15), anchor='lm', stroke=8)
    text(d, (BW - 60, 110), '2.5D arenas  •  3D foreground, painted parallax background', 40, (1, 1, 1),
         font='Anton-Regular.ttf', anchor='rm', stroke=0)
    for n, y in ((1, 220), (2, 1340)):
        p = panel(n)
        fx.save(p, f'{OUT}/06_stage{n}.png')
        board.paste(cp.to_pil(p), (40, y))
        d.rectangle([40, y, 40 + 1920, y + 1080], outline=(250, 220, 40), width=6)
        tag, name, sub, notes = NOTES[n]
        x0 = 2000
        text(d, (x0, y + 40), tag, 44, (1.0, 0.86, 0.15), font='Anton-Regular.ttf', anchor='lm', stroke=0)
        text(d, (x0, y + 110), name.split()[0], 74, (1, 1, 1), anchor='lm', stroke=5)
        text(d, (x0, y + 180), ' '.join(name.split()[1:]), 74, (1, 1, 1), anchor='lm', stroke=5)
        text(d, (x0, y + 250), sub, 26, (1.0, 0.55, 0.65), font='Anton-Regular.ttf', anchor='lm', stroke=0)
        for i, line in enumerate(notes):
            yy = y + 320 + i * 120
            d.ellipse([x0, yy - 9, x0 + 18, yy + 9], fill=(250, 220, 40))
            words, cur, rows = line.split(), '', []
            f = paint.font('Anton-Regular.ttf', 28)
            for w in words:
                if f.getlength(cur + ' ' + w) > 470:
                    rows.append(cur)
                    cur = w
                else:
                    cur = (cur + ' ' + w).strip()
            rows.append(cur)
            for r_i, r in enumerate(rows):
                text(d, (x0 + 34, yy + r_i * 36), r, 28, (0.92, 0.9, 0.95), font='Anton-Regular.ttf', anchor='lm',
                     stroke=0)
    board.save(f'{OUT}/06_stage_concepts.png')
    print('saved stages')


if __name__ == '__main__':
    main()
