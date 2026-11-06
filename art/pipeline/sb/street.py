"""Stage 1 v2: "Cinema Street", a sunset street canyon in the reference look.

The street runs along +Y; the fight plane is across the road (X), so the gameplay camera
looks straight down the street and gets the one-point-perspective canyon of the reference
shots. Saturated pastel facades, cyan shop glass, red/teal awnings, hand-painted signboards
(English / Telugu / Hindi), black-and-yellow kerbs, auto-rickshaws, low-poly crowd, lamp
posts, wires, bunting and the single-screen cinema closing the far end.
"""
import math
import os
import random

import bpy
from mathutils import Vector, Matrix
from PIL import Image, ImageDraw, ImageFont

from .geo import loft, blob, make_object, cylinder
from .environment import MeshAcc, catenary, ENV_INDEX, CITY_INDEX

FONTS = os.path.join(os.path.dirname(__file__), '../../fonts')

WALLS = [(1.0, 0.62, 0.30), (0.36, 0.82, 0.76), (0.98, 0.46, 0.60), (0.58, 0.44, 0.90), (1.0, 0.84, 0.40),
         (0.88, 0.36, 0.32), (0.56, 0.90, 0.70), (0.80, 0.66, 0.98), (1.0, 0.70, 0.55), (0.40, 0.66, 0.98)]
AWN = [(0.95, 0.24, 0.30), (0.20, 0.72, 0.70), (1.0, 0.78, 0.20), (0.62, 0.30, 0.85), (0.98, 0.45, 0.20)]
SIGNS = [('SRI LAKSHMI TIFFINS', 'Anton-Regular.ttf'), ('టిఫిన్స్', 'NotoSansTelugu.ttf'), ('CHAI  &  BUN', 'Bangers-Regular.ttf'),
         ('MOBILE REPAIRS', 'Anton-Regular.ttf'), ('चाय', 'TiroDevanagariHindi-Regular.ttf'), ('SWEETS', 'Bangers-Regular.ttf'),
         ('KIRANA STORE', 'Anton-Regular.ttf'), ('TAILORS', 'Bangers-Regular.ttf'), ('MEDICAL', 'Anton-Regular.ttf'),
         ('LASSI CORNER', 'Bangers-Regular.ttf'), ('బేకరీ', 'NotoSansTelugu.ttf'), ('FILM POSTERS', 'Anton-Regular.ttf'),
         ('GOLD PALACE', 'Bangers-Regular.ttf'), ('BIRYANI HOUSE', 'Anton-Regular.ttf'), ('मिठाई', 'TiroDevanagariHindi-Regular.ttf')]


def sign_texture(path, text, font, bg, fg, w=1024, h=256):
    im = Image.new('RGB', (w, h), tuple(int(c * 255) for c in bg))
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, w - 9, h - 9], outline=tuple(int(c * 255) for c in fg), width=10)
    size = 150
    f = ImageFont.truetype(os.path.join(FONTS, font), size)
    while f.getlength(text) > w - 90 and size > 40:
        size -= 6
        f = ImageFont.truetype(os.path.join(FONTS, font), size)
    d.text((w / 2 + 6, h / 2 + 6), text, font=f, fill=(25, 15, 40), anchor='mm')
    d.text((w / 2, h / 2), text, font=f, fill=tuple(int(c * 255) for c in fg), anchor='mm', stroke_width=4,
           stroke_fill=(25, 15, 40))
    im.save(path)
    return path


def textured_quad(ml, name, img, corners, pass_index=ENV_INDEX, emit=0.0):
    if name not in ml.mats:
        m = ml.get(name, (1, 1, 1), emit=emit)
        nt = m.node_tree
        bsdf = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.image = bpy.data.images.load(img, check_existing=True)
        nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        if emit:
            nt.links.new(tex.outputs['Color'], bsdf.inputs['Emission Color'])
        ml.props[m.pass_index]['textured'] = True
    m = ml.mats[name]
    ob = make_object(name, corners, [(0, 1, 2, 3)], [m], [0], subsurf=0, smooth=False, pass_index=pass_index)
    uv = ob.data.uv_layers.new()
    for li, (u, v) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        uv.data[li].uv = (u, v)
    return ob


def person(ml, name, p, rng, facing=0.0, scale=1.0):
    """Low-poly street figure in the reference style (capsule body, ball head)."""
    shirts = [(0.95, 0.30, 0.35), (0.30, 0.70, 0.95), (1.0, 0.80, 0.25), (0.55, 0.85, 0.50), (0.90, 0.50, 0.85),
              (0.98, 0.98, 0.92), (0.35, 0.45, 0.85), (1.0, 0.55, 0.20)]
    skins = [(0.86, 0.60, 0.44), (0.68, 0.45, 0.32), (0.55, 0.36, 0.25), (0.92, 0.70, 0.55)]
    sh = ml.get(f'env.shirt{rng.randrange(len(shirts))}', shirts[rng.randrange(len(shirts))])
    pants = ml.get('env.pants', (0.22, 0.22, 0.45))
    sk = ml.get(f'env.pskin{rng.randrange(4)}', skins[rng.randrange(4)])
    hair = ml.get('env.phair', (0.18, 0.15, 0.32))
    s = scale * rng.uniform(0.92, 1.06)
    x, y = p
    obs = []
    for dx in (-0.09, 0.09):
        obs.append(loft(f'{name}_leg{dx}', [(x + dx * s, y, 0.0), (x + dx * s, y, 0.85 * s)], 0.07 * s, ring=6,
                        samples=1, mats=[pants], subsurf=0, pass_index=CITY_INDEX + 6))
    obs.append(loft(f'{name}_body', [(x, y, 0.80 * s), (x, y, 1.45 * s)], lambda t: (0.20 * s, 0.13 * s), ring=8,
                    samples=2, mats=[sh], subsurf=1, pass_index=CITY_INDEX + 6))
    obs.append(blob(f'{name}_head', lambda q: q, segments=10, rings=6, mats=[sk], subsurf=1, pass_index=CITY_INDEX + 6,
                    matrix=Matrix.Translation((x, y, 1.62 * s)) @ Matrix.Diagonal((0.12 * s, 0.12 * s, 0.14 * s, 1))))
    obs.append(blob(f'{name}_hair', lambda q: q, segments=10, rings=6, mats=[hair], subsurf=1, pass_index=CITY_INDEX + 6,
                    keep_fn=lambda q: q.z > 0.1,
                    matrix=Matrix.Translation((x, y + 0.01, 1.64 * s)) @ Matrix.Diagonal((0.13 * s, 0.13 * s, 0.14 * s, 1))))
    return obs


def auto_rickshaw(ml, name, loc, yaw=0.0):
    """Three-wheeler: green body, yellow canopy, black seat area."""
    body = ml.get('env.auto_body', (0.22, 0.78, 0.42), spec=0.6)
    canopy = ml.get('env.auto_top', (1.0, 0.84, 0.18), spec=0.4)
    dark = ml.get('env.auto_dark', (0.14, 0.13, 0.28))
    glass = ml.get('env.glass', (0.66, 0.96, 0.94), glass=True)
    tyre = ml.get('env.tyre', (0.12, 0.11, 0.22))
    R = Matrix.Translation(Vector(loc)) @ Matrix.Rotation(math.radians(yaw), 4, 'Z')
    obs = []
    acc = MeshAcc(name + '_body', [body, dark, canopy, glass])
    acc.box(-0.65, -1.35, 0.30, 0.65, 0.95, 0.90, 0)        # floor pan + lower body
    acc.box(-0.45, -1.60, 0.30, 0.45, -1.35, 1.35, 0)       # nose / front shield
    acc.box(-0.62, -0.20, 0.90, 0.62, 0.95, 1.45, 1)        # rear seat cabin
    acc.box(-0.70, -1.45, 1.75, 0.70, 1.05, 1.95, 2)        # canopy roof
    acc.box(-0.70, 0.85, 0.90, 0.70, 1.05, 1.80, 2)         # canopy back
    acc.box(-0.42, -1.50, 1.30, 0.42, -1.42, 1.78, 3)       # windscreen
    ob = acc.build(CITY_INDEX + 7)
    ob.data.transform(R)
    b = ob.modifiers.new('bevel', 'BEVEL')
    b.width, b.segments, b.limit_method = 0.06, 2, 'NONE'
    obs.append(ob)
    for (x, y) in ((0.0, -1.25), (-0.62, 0.60), (0.62, 0.60)):
        p0 = R @ Vector((x - 0.08, y, 0.26))
        p1 = R @ Vector((x + 0.08, y, 0.26))
        obs.append(cylinder(f'{name}_w{x}{y}', p0, p1, 0.26, sides=12, mats=[tyre], pass_index=CITY_INDEX + 7))
    return obs


def street(ml, build_dir, length=150.0, seed=4, night=False):
    rng = random.Random(seed)
    os.makedirs(os.path.join(build_dir, 'signs'), exist_ok=True)
    obs = []
    road = ml.get('env.road', (0.56, 0.46, 0.92))
    yel = ml.get('env.lane_yellow', (1.0, 0.84, 0.25))
    wht = ml.get('env.lane_white', (0.98, 0.96, 0.92))
    walk = ml.get('env.sidewalk', (0.84, 0.72, 0.96))
    kerb_y = ml.get('env.kerb_y', (1.0, 0.84, 0.20))
    kerb_k = ml.get('env.kerb_k', (0.16, 0.14, 0.30))
    skid = ml.get('env.skid', (0.36, 0.28, 0.66))
    acc = MeshAcc('road', [road, yel, wht, walk, kerb_y, kerb_k, skid])
    acc.box(-6, -20, -0.3, 6, length + 30, 0.0, 0, faces=('top',))
    for x in (-0.16, 0.16):
        acc.box(x - 0.06, -20, 0.0, x + 0.06, length, 0.005, 1, faces=('top',))
    y = -20
    while y < length:
        for x in (-3.0, 3.0):
            acc.box(x - 0.08, y, 0.0, x + 0.08, y + 2.2, 0.005, 2, faces=('top',))
        y += 6
    for x0 in (-1.6, 1.3):  # skid marks across the fight zone (reference: dark tyre tracks)
        acc.box(x0 - 0.10, -6, 0.0, x0 + 0.10, 18, 0.006, 6, faces=('top',))
        acc.box(x0 + 0.35, -6, 0.0, x0 + 0.52, 18, 0.006, 6, faces=('top',))
    for s in (-1, 1):
        x0, x1 = (6.0, 9.6) if s > 0 else (-9.6, -6.0)
        acc.box(x0, -20, 0.0, x1, length + 30, 0.18, 3, faces=('top', 'front'))
        kx0, kx1 = (6.0, 6.18) if s > 0 else (-6.18, -6.0)
        y = -20.0
        k = 0
        while y < length:
            acc.box(kx0, y, 0.0, kx1, y + 1.0, 0.19, 4 if k % 2 else 5)
            y += 1.0
            k += 1
    obs.append(acc.build(ENV_INDEX))

    # ---------------------------------------------------------------- buildings
    win_d = ml.get('env.win_dark', (0.24, 0.24, 0.55))
    win_l = ml.get('env.win_lit', (1.0, 0.88, 0.42), emit=2.0 if night else 0.0)
    glass = ml.get('env.glass', (0.66, 0.96, 0.94), glass=True)
    frame = ml.get('env.frame', (0.98, 0.95, 0.88))
    roofm = ml.get('env.roof2', (0.70, 0.58, 0.85))
    tank = ml.get('env.tank_black', (0.14, 0.13, 0.30), spec=0.5)
    acm = ml.get('env.ac', (0.92, 0.92, 0.95))
    walls = [ml.get(f'env.wall_{i}', c) for i, c in enumerate(WALLS)]
    awns = [ml.get(f'env.awning{i}', c) for i, c in enumerate(AWN)]
    mats = walls + awns + [win_d, win_l, glass, frame, roofm, tank, acm]
    iW = len(walls)
    iA = iW
    iD, iL, iG, iF, iR, iT, iC = [iW + len(awns) + k for k in range(7)]
    sign_i = 0
    for s in (-1, 1):
        A = MeshAcc(f'facades{s}', mats)
        y = -18.0
        while y < length - 6:
            w = rng.uniform(6.5, 12.0)
            floors = rng.choice((2, 3, 3, 4, 4, 5, 6))
            H = 4.9 + floors * 3.1
            wi = rng.randrange(iW)
            xf = 9.6 * s
            xb = xf + 14 * s
            x0, x1 = (min(xf, xb), max(xf, xb))
            A.box(x0, y, 0, x1, y + w - 0.15, H, wi, top_mi=iR)
            H0 = 4.9  # ground floor (shop + sign) height
            # cornices (thin ledges = clean ink lines in the reference look)
            for k in range(floors + 1):
                z = H0 + k * 3.1 - 0.15
                A.box(xf - s * 0.25 if s > 0 else xf, y - 0.1, z, xf if s > 0 else xf + 0.25, y + w - 0.05, z + 0.22, iF)
            # ground-floor shopfront: glass, pillars, awning
            gx = xf - s * 0.06
            A.box(min(gx, gx - s * 0.02), y + 0.6, 0.4, max(gx, gx - s * 0.02), y + w - 0.75, 3.0, iG)
            ai = iA + rng.randrange(len(awns))
            ax0, ax1 = (xf - s * 1.8, xf)
            A.box(min(ax0, ax1), y + 0.3, 3.15, max(ax0, ax1), y + w - 0.45, 3.45, ai)
            A.box(min(ax0, ax1) - 0.02, y + 0.3, 3.0, max(ax0, ax1) + 0.02, y + w - 0.45, 3.15, iF)
            # signboard above the awning
            text, font = SIGNS[sign_i % len(SIGNS)]
            sign_i += 1
            bg = WALLS[(wi + 3) % len(WALLS)] if rng.random() < 0.6 else (0.16, 0.14, 0.32)
            fg = (1.0, 0.95, 0.55) if sum(bg) < 1.2 or bg == (0.16, 0.14, 0.32) else (0.95, 0.22, 0.30)
            img = sign_texture(os.path.join(build_dir, 'signs', f'sign{sign_i}.png'), text, font, bg, fg)
            sx = xf - s * 0.08
            sy0, sy1 = y + 0.6, y + w - 0.75
            pts = [(sx, sy0, 3.55), (sx, sy1, 3.55), (sx, sy1, 4.45 + 0.25 * (w / 10)), (sx, sy0, 4.45 + 0.25 * (w / 10))]
            if s > 0:
                pts = [pts[1], pts[0], pts[3], pts[2]]
            obs.append(textured_quad(ml, f'env.sign{sign_i}', img, pts, emit=1.5 if night else 0.0))
            # upper-floor windows, balconies, AC boxes
            cols = max(2, int(w / 2.6))
            for k in range(floors):
                z0 = H0 + k * 3.1 + 0.6
                for c in range(cols):
                    cy0 = y + 0.8 + c * (w - 1.4) / cols
                    cy1 = cy0 + (w - 1.4) / cols * 0.62
                    lit = rng.random() < (0.55 if night else 0.32)
                    wx = xf - s * 0.05
                    A.box(min(wx, wx - s * 0.02), cy0, z0, max(wx, wx - s * 0.02), cy1, z0 + 1.6, iL if lit else iD)
                    A.box(min(wx, wx - s * 0.12), cy0 - 0.08, z0 - 0.12, max(wx, wx - s * 0.12), cy1 + 0.08, z0, iF)
                    if rng.random() < 0.12:
                        A.box(min(wx, wx - s * 0.5), cy1 - 0.6, z0 + 0.1, max(wx, wx - s * 0.5), cy1 + 0.2, z0 + 0.6, iC)
                if rng.random() < 0.35 and k < floors - 1:  # balcony slab + rail
                    bx0, bx1 = xf - s * 1.1, xf
                    A.box(min(bx0, bx1), y + 1.0, z0 - 0.7, max(bx0, bx1), y + w - 1.2, z0 - 0.5, iF)
                    A.box(min(bx0, bx0 + s * 0.08), y + 1.0, z0 - 0.5, max(bx0, bx0 + s * 0.08), y + w - 1.2, z0 + 0.4,
                          iA + rng.randrange(len(awns)))
            if rng.random() < 0.6:  # rooftop water tank
                tx, ty = xf + s * rng.uniform(3, 8), y + rng.uniform(1.5, w - 1.5)
                A.cyl((tx, ty), 0.8, 1.5, iT, sides=10, z0=H)
            y += w
        obs.append(A.build(ENV_INDEX + 2 + (s > 0)))
    obs += cinema(ml, build_dir, length, night)
    obs += street_props(ml, length, rng, night)
    return obs


def cinema(ml, build_dir, y, night=False):
    """Single-screen cinema closing the street: facade, marquee, giant hoarding."""
    from .environment import poster_texture
    obs = []
    wall = ml.get('env.cinema', (1.0, 0.92, 0.70))
    trim = ml.get('env.cinema_trim', (0.95, 0.30, 0.40))
    bulb = ml.get('env.bulb', (1.0, 0.92, 0.55), emit=5.0)
    A = MeshAcc('cinema', [wall, trim, ml.get('env.glass', (0.66, 0.96, 0.94), glass=True)])
    A.box(-24, y, 0, 24, y + 18, 14, 0)
    A.box(-9, y - 0.2, 0, 9, y, 3.6, 2)                    # entrance glass
    A.box(-11, y - 2.5, 3.6, 11, y, 4.6, 1)               # marquee canopy
    A.box(-13, y - 0.4, 13.4, 13, y + 0.2, 15.6, 1)       # crown
    for k in range(7):
        A.box(-10.5 + k * 3.5 - 0.35, y - 0.45, 0, -10.5 + k * 3.5 + 0.35, y, 13.4, 1)  # art-deco fins
    obs.append(A.build(CITY_INDEX + 1))
    tex = poster_texture(os.path.join(build_dir, 'poster.png'))
    obs.append(textured_quad(ml, 'env.hoarding', tex, [(-10, y - 0.6, 5.4), (10, y - 0.6, 5.4), (10, y - 0.6, 12.8),
                                                         (-10, y - 0.6, 12.8)], pass_index=CITY_INDEX + 1))
    for k in range(40):  # marquee bulbs
        x = -10.6 + k * (21.2 / 39)
        obs.append(blob(f'bulb{k}', lambda q: q, segments=6, rings=4, mats=[bulb], subsurf=0, pass_index=CITY_INDEX + 1,
                        matrix=Matrix.Translation((x, y - 2.6, 4.1)) @ Matrix.Diagonal((0.12, 0.12, 0.12, 1))))
    return obs


def street_props(ml, length, rng, night):
    obs = []
    pole = ml.get('env.pole2', (0.30, 0.30, 0.55), spec=0.5)
    lamp = ml.get('env.lamp2', (1.0, 0.95, 0.70), emit=6.0 if night else 0.0)
    wire = ml.get('env.wire', (0.05, 0.05, 0.07), ink=True)
    tree = ml.get('env.tree', (0.30, 0.72, 0.68))
    trunk = ml.get('env.trunk', (0.50, 0.30, 0.45))
    for s in (-1, 1):
        for k in range(10):
            y = -8 + k * 15 + (7 if s > 0 else 0)
            x = 6.5 * s
            obs.append(cylinder(f'pole{s}{k}', (x, y, 0.18), (x, y, 7.5), 0.09, mats=[pole], pass_index=ENV_INDEX + 1))
            arm = [Vector((x, y, 7.3)), Vector((x - s * 0.8, y, 7.8)), Vector((x - s * 1.8, y, 7.7))]
            obs.append(loft(f'arm{s}{k}', arm, 0.06, ring=6, samples=4, mats=[pole], subsurf=0, pass_index=ENV_INDEX + 1))
            obs.append(blob(f'lamp{s}{k}', lambda q: q, segments=10, rings=6, mats=[lamp], subsurf=0,
                            pass_index=ENV_INDEX + 1,
                            matrix=Matrix.Translation((x - s * 1.85, y, 7.55)) @ Matrix.Diagonal((0.42, 0.22, 0.14, 1))))
        for k in range(3):
            y = 30 + k * 34 + (12 if s > 0 else 0)
            x = 8.2 * s
            obs.append(cylinder(f'trunk{s}{k}', (x, y, 0.18), (x, y, 3.5), 0.16, mats=[trunk], pass_index=ENV_INDEX + 1))
            obs.append(blob(f'tree{s}{k}', lambda q: Vector((q.x, q.y, q.z * 0.85)), segments=14, rings=9, mats=[tree],
                            subsurf=0, pass_index=ENV_INDEX + 1,
                            matrix=Matrix.Translation((x, y, 4.6)) @ Matrix.Diagonal((1.7, 1.7, 1.7, 1))))
    # wires along and across the street
    for k in range(4):
        y0 = 10 + k * 32
        for (a, b, sag) in (((-9.6, y0, 9.5), (9.6, y0 + 4, 9.0), 1.2), ((-9.6, y0 + 8, 11.0), (9.6, y0 + 12, 10.5), 1.6)):
            obs.append(loft(f'xwire{k}{a[1]}', catenary(a, b, sag, 30), 0.035, ring=4, samples=1, mats=[wire], subsurf=0,
                            pass_index=ENV_INDEX + 1))
    for s in (-1, 1):
        for k in range(3):
            pts = [(6.5 * s, -10 + 0.0, 7.0 + 0.4 * k), (6.5 * s, length, 7.0 + 0.4 * k)]
            obs.append(loft(f'lwire{s}{k}', catenary(pts[0], pts[1], 0.6, 60), 0.03, ring=4, samples=1, mats=[wire],
                            subsurf=0, pass_index=ENV_INDEX + 1))
    # bunting across the street
    fc = [(0.95, 0.25, 0.35), (1.0, 0.82, 0.18), (0.25, 0.75, 0.55), (0.30, 0.55, 0.98), (0.95, 0.50, 0.85),
          (0.98, 0.98, 0.92)]
    fm = [ml.get(f'env.flagv{i}', c) for i, c in enumerate(fc)]
    for k, y0 in enumerate((16, 42, 74, 110)):
        line = catenary((-9.6, y0, 8.4), (9.6, y0 + 2, 8.2), 1.4, 28)
        obs.append(loft(f'bline{k}', line, 0.02, ring=4, samples=1, mats=[wire], subsurf=0, pass_index=ENV_INDEX + 1))
        acc = MeshAcc(f'flags{k}', fm)
        for i in range(len(line) - 1):
            p, q = line[i], line[i + 1]
            m = (p + q) / 2
            acc.v += [p, q, m + Vector((0, 0, -0.75))]
            j = len(acc.v) - 3
            acc.f.append((j, j + 1, j + 2))
            acc.m.append(i % len(fm))
        obs.append(acc.build(ENV_INDEX + 1))
    # traffic + crowd
    obs += auto_rickshaw(ml, 'auto0', (-4.3, 26, 0), yaw=180)
    obs += auto_rickshaw(ml, 'auto1', (4.4, 47, 0), yaw=0)
    obs += auto_rickshaw(ml, 'auto2', (-2.0, 78, 0), yaw=170)
    for i in range(46):
        s = rng.choice((-1, 1))
        y = rng.uniform(4, length - 15)
        x = s * rng.uniform(7.0, 9.2)
        obs += person(ml, f'p{i}', (x, y), rng)
    return obs
