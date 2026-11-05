"""Procedural stages for Superstar Brawl.

Stage 1  "Single Screen Showdown" - rooftop of an old single-screen cinema at golden hour,
         water tanks, bunting, overhead wires, kites, a hand-painted release billboard and a
         dense Hyderabad-style skyline with a far monument silhouette.
Stage 2  "Necklace Night" - the same city generator at night: lit windows, a sea-front
         promenade with an arc of street lamps and neon.
Geometry is merged into a handful of meshes so a whole city renders in seconds.
"""
import math
import os
import random

import bpy
from mathutils import Vector, Matrix

from .geo import loft, blob, make_object, catmull_rom, cylinder, box

ENV_INDEX = 40
CITY_INDEX = 50


class MeshAcc:
    """Accumulate many boxes / quads into one mesh with per-face materials."""

    def __init__(self, name, mats):
        self.name, self.mats = name, mats
        self.v, self.f, self.m = [], [], []

    def quad(self, a, b, c, d, mi):
        i = len(self.v)
        self.v += [Vector(a), Vector(b), Vector(c), Vector(d)]
        self.f.append((i, i + 1, i + 2, i + 3))
        self.m.append(mi)

    def box(self, x0, y0, z0, x1, y1, z1, mi, faces='all', top_mi=None):
        V = [Vector((x, y, z)) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
        i = len(self.v)
        self.v += V
        F = {'bottom': (0, 2, 3, 1), 'top': (4, 5, 7, 6), 'front': (0, 1, 5, 4), 'back': (2, 6, 7, 3),
             'left': (0, 4, 6, 2), 'right': (1, 3, 7, 5)}
        for k, fc in F.items():
            if faces != 'all' and k not in faces:
                continue
            if k == 'bottom' and faces == 'all':
                continue
            self.f.append(tuple(i + q for q in fc))
            self.m.append(top_mi if (k == 'top' and top_mi is not None) else mi)

    def cyl(self, c, r, h, mi, sides=10, z0=0.0, cap_mi=None):
        i = len(self.v)
        for z in (z0, z0 + h):
            for k in range(sides):
                a = 2 * math.pi * k / sides
                self.v.append(Vector((c[0] + r * math.cos(a), c[1] + r * math.sin(a), z)))
        for k in range(sides):
            k2 = (k + 1) % sides
            self.f.append((i + k, i + k2, i + sides + k2, i + sides + k))
            self.m.append(mi)
        self.f.append(tuple(i + sides + k for k in range(sides)))
        self.m.append(cap_mi if cap_mi is not None else mi)

    def build(self, index, subsurf=0, smooth=False):
        ob = make_object(self.name, self.v, self.f, self.mats, self.m, subsurf=subsurf, smooth=smooth,
                         pass_index=index)
        return ob


# ----------------------------------------------------------------------------- materials

def env_mats(ml, night=False):
    pastel = [(0.95, 0.78, 0.62), (0.92, 0.60, 0.55), (0.98, 0.88, 0.60), (0.62, 0.82, 0.80),
              (0.80, 0.70, 0.88), (0.95, 0.95, 0.88), (0.85, 0.55, 0.40), (0.70, 0.78, 0.92),
              (0.96, 0.70, 0.75), (0.80, 0.86, 0.62)]
    if night:
        pastel = [tuple(c * 0.55 for c in p) for p in pastel]
    walls = [ml.get(f'env.wall{i}', c) for i, c in enumerate(pastel)]
    win_lit = ml.get('env.window_lit', (1.0, 0.82, 0.45), emit=2.5 if night else 0.0)
    win = ml.get('env.window', (0.30, 0.35, 0.55) if not night else (0.08, 0.09, 0.16))
    roof = ml.get('env.roof', (0.62, 0.58, 0.55) if not night else (0.2, 0.2, 0.25))
    tank = ml.get('env.tank', (0.10, 0.10, 0.12), hatch=True)
    return dict(walls=walls, win=win, win_lit=win_lit, roof=roof, tank=tank)


# ----------------------------------------------------------------------------- city

def city(ml, seed=7, x_range=(-160, 160), y_range=(40, 320), night=False, monument=True, z0=-30.0):
    rng = random.Random(seed)
    M = env_mats(ml, night)
    mats = M['walls'] + [M['win'], M['win_lit'], M['roof'], M['tank']]
    W0 = len(M['walls'])
    I_WIN, I_LIT, I_ROOF, I_TANK = W0, W0 + 1, W0 + 2, W0 + 3
    near = MeshAcc('city_near', mats)
    far = MeshAcc('city_far', mats)
    y = y_range[0]
    while y < y_range[1]:
        depth = rng.uniform(7, 14)
        x = x_range[0] * (1 + (y - y_range[0]) / 200)
        xmax = -x
        acc = near if y < 110 else far
        while x < xmax:
            w = rng.uniform(5, 13)
            tall = rng.random()
            dist_f = (y - y_range[0]) / (y_range[1] - y_range[0])
            if tall > 0.94 and y > 90:
                h = rng.uniform(42, 62) * (0.75 + 0.5 * dist_f)
            elif tall > 0.6:
                h = rng.uniform(18, 28) + dist_f * 10
            else:
                h = rng.uniform(10, 22)
            mi = rng.randrange(W0)
            acc.box(x, y, z0, x + w * 0.94, y + depth, z0 + h, mi, top_mi=I_ROOF)
            # windows on the camera-facing (-Y) face and the side facing the centre
            if y < (650 if night else 200):
                cols = max(1, int(w / 2.2))
                rows = max(1, int(h / 3.2))
                for r in range(rows):
                    for cidx in range(cols):
                        if rng.random() < 0.18:
                            continue
                        wx0 = x + (cidx + 0.28) * (w * 0.94 / cols)
                        wx1 = wx0 + (w * 0.94 / cols) * 0.45
                        wz0 = z0 + 1.2 + r * (h - 1.5) / rows
                        wz1 = wz0 + (h - 1.5) / rows * 0.5
                        lit = night and rng.random() < 0.55
                        if not night and rng.random() < 0.15:
                            lit = False
                        acc.quad((wx0, y - 0.06, wz0), (wx1, y - 0.06, wz0), (wx1, y - 0.06, wz1),
                                 (wx0, y - 0.06, wz1), I_LIT if lit else I_WIN)
            # rooftop water tanks + parapet on low-rise
            if h < 25 and rng.random() < 0.7:
                for k in range(rng.randint(1, 3)):
                    tx = x + rng.uniform(1, w - 2)
                    ty = y + rng.uniform(1.5, depth - 1.5)
                    acc.cyl((tx, ty), rng.uniform(0.5, 0.9), rng.uniform(1.0, 1.6), I_TANK, sides=8, z0=z0 + h)
            x += w
        y += depth + rng.uniform(2, 9)
    obs = [near.build(CITY_INDEX), far.build(CITY_INDEX + 1)]
    if monument:
        obs += charminar(ml, (-55, 190, z0), scale=2.4, night=night)
    return obs


def charminar(ml, loc, scale=1.0, night=False):
    """Simplified four-minaret gate monument silhouette (far background landmark)."""
    stone = ml.get('env.stone', (0.93, 0.83, 0.66) if not night else (0.45, 0.42, 0.5))
    dome = ml.get('env.dome', (0.90, 0.78, 0.60) if not night else (0.4, 0.38, 0.46))
    arch = ml.get('env.arch', (0.35, 0.25, 0.30) if not night else (0.08, 0.06, 0.12))
    x, y, z = loc
    s = scale * 10
    acc = MeshAcc('charminar', [stone, dome, arch])
    acc.box(x - s, y - s, z, x + s, y + s, z + s * 1.25, 0)
    acc.box(x - s * 1.02, y - s * 1.02, z + s * 1.25, x + s * 1.02, y + s * 1.02, z + s * 1.45, 0)
    acc.box(x - s * 0.95, y - s * 0.95, z + s * 1.45, x + s * 0.95, y + s * 0.95, z + s * 1.75, 0)
    # arch openings on the front face
    acc.quad((x - s * 0.45, y - s - 0.1, z), (x + s * 0.45, y - s - 0.1, z), (x + s * 0.45, y - s - 0.1, z + s * 0.95),
             (x - s * 0.45, y - s - 0.1, z + s * 0.95), 2)
    for k in range(5):
        ax = x - s * 0.8 + k * s * 0.4
        acc.quad((ax - s * 0.12, y - s * 1.03, z + s * 1.5), (ax + s * 0.12, y - s * 1.03, z + s * 1.5),
                 (ax + s * 0.12, y - s * 1.03, z + s * 1.68), (ax - s * 0.12, y - s * 1.03, z + s * 1.68), 2)
    ob = acc.build(CITY_INDEX + 2)
    obs = [ob]
    rings = MeshAcc('minaret_rings', [stone, dome])
    for cx in (-1, 1):
        for cy in (-1, 1):
            px, py = x + cx * s * 0.95, y + cy * s * 0.95
            top = z + s * 2.75
            obs.append(loft(f'minaret{cx}{cy}', [(px, py, z), (px, py, top)],
                            lambda t: s * (0.10 - 0.03 * t), ring=10, samples=2, mats=[stone], subsurf=0,
                            pass_index=CITY_INDEX + 2))
            for bz in (0.42, 0.62, 0.80, 0.93):
                rings.cyl((px, py), s * 0.15, s * 0.05, 0, sides=10, z0=z + s * 2.75 * bz)
            rings.cyl((px, py), s * 0.004 + 0.15, s * 0.35, 1, sides=6, z0=top + s * 0.12)

            def onion(p):
                q = Vector((p.x, p.y, p.z * 1.5 if p.z > 0 else p.z * 0.5))
                k = 1 - 0.75 * max(0.0, p.z) ** 1.5
                return Vector((q.x * k, q.y * k, q.z))
            obs.append(blob(f'minaret_dome{cx}{cy}', onion, segments=12, rings=10, mats=[dome], subsurf=0,
                            pass_index=CITY_INDEX + 2,
                            matrix=Matrix.Translation((px, py, top + s * 0.02)) @ Matrix.Diagonal((s * 0.12,) * 3 + (1,))))
    obs.append(rings.build(CITY_INDEX + 2))
    return obs


# ----------------------------------------------------------------------------- rooftop stage

def poster_texture(path, w=1600, h=900, night=False):
    """Hand-painted single-screen release banner (PIL)."""
    from PIL import Image, ImageDraw, ImageFont
    fonts = os.path.join(os.path.dirname(__file__), '../../fonts')
    im = Image.new('RGB', (w, h), (250, 200, 40))
    d = ImageDraw.Draw(im)
    cx, cy = w * 0.5, h * 0.62
    for k in range(36):
        a0 = 2 * math.pi * k / 36
        a1 = a0 + math.pi / 36
        col = (235, 60, 40) if k % 2 else (255, 150, 30)
        d.polygon([(cx, cy), (cx + 2000 * math.cos(a0), cy + 2000 * math.sin(a0)),
                   (cx + 2000 * math.cos(a1), cy + 2000 * math.sin(a1))], fill=col)
    big = ImageFont.truetype(os.path.join(fonts, 'Bangers-Regular.ttf'), 230)
    mid = ImageFont.truetype(os.path.join(fonts, 'Anton-Regular.ttf'), 90)
    tel = ImageFont.truetype(os.path.join(fonts, 'NotoSansTelugu.ttf'), 110)
    hin = ImageFont.truetype(os.path.join(fonts, 'TiroDevanagariHindi-Regular.ttf'), 100)

    def outlined(xy, text, font, fill, stroke=10, anchor='mm'):
        d.text(xy, text, font=font, fill=fill, stroke_width=stroke, stroke_fill=(20, 10, 30), anchor=anchor)
    outlined((w / 2, h * 0.30), 'SUPERSTAR', big, (255, 255, 255), 12)
    outlined((w / 2, h * 0.56), 'BRAWL', big, (255, 230, 60), 12)
    outlined((w * 0.30, h * 0.80), 'త్వరలో', tel, (255, 255, 255), 8)
    outlined((w * 0.70, h * 0.80), 'जल्द आ रहा है', hin, (255, 255, 255), 8)
    d.rectangle([0, 0, w, 70], fill=(20, 10, 30))
    d.rectangle([0, h - 70, w, h], fill=(20, 10, 30))
    outlined((w / 2, 35), 'NOW SHOWING  •  4 SHOWS DAILY  •  HOUSEFULL', ImageFont.truetype(
        os.path.join(fonts, 'Anton-Regular.ttf'), 46), (255, 220, 80), 0)
    im.save(path)
    return path


def tex_material(ml, name, image_path, **props):
    if name in ml.mats:
        return ml.mats[name]
    m = ml.get(name, (1, 1, 1), **props)
    nt = m.node_tree
    bsdf = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(image_path)
    nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    ml.props[m.pass_index]['textured'] = True
    return m


def catenary(p0, p1, sag, n=24):
    p0, p1 = Vector(p0), Vector(p1)
    pts = []
    for i in range(n + 1):
        t = i / n
        p = p0.lerp(p1, t)
        p.z -= sag * 4 * t * (1 - t)
        pts.append(p)
    return pts


def rooftop(ml, build_dir, night=False, seed=3, width=34, billboard=True):
    rng = random.Random(seed)
    obs = []
    floor_a = ml.get('env.floor_a', (0.74, 0.62, 0.55) if not night else (0.25, 0.23, 0.30))
    floor_b = ml.get('env.floor_b', (0.66, 0.54, 0.50) if not night else (0.20, 0.19, 0.26))
    grout = ml.get('env.grout', (0.45, 0.36, 0.36) if not night else (0.12, 0.11, 0.16))
    wall = ml.get('env.parapet', (0.92, 0.72, 0.52) if not night else (0.30, 0.28, 0.38))
    wall_top = ml.get('env.parapet_top', (0.98, 0.88, 0.72) if not night else (0.40, 0.38, 0.48))
    rail = ml.get('env.rail', (0.25, 0.55, 0.55), spec=0.6)
    tank = ml.get('env.tank_big', (0.08, 0.08, 0.10), hatch=True, spec=0.5)
    steel = ml.get('env.steel', (0.55, 0.58, 0.62), metal=True, spec=0.8)
    acc = MeshAcc('roof_floor', [floor_a, floor_b, grout])
    half = width / 2
    # tiles
    ts = 1.2
    y0, y1 = -8.0, 7.0
    nx, ny = int(width / ts), int((y1 - y0) / ts)
    for i in range(nx):
        for j in range(ny):
            x0 = -half + i * ts
            yy = y0 + j * ts
            acc.box(x0 + 0.03, yy + 0.03, -0.05, x0 + ts - 0.03, yy + ts - 0.03, 0.0,
                    (i + j) % 2, faces=('top', 'front', 'left'))
    acc.box(-half, y0, -0.4, half, y1, -0.04, 2, faces=('top', 'front'))
    obs.append(acc.build(ENV_INDEX))
    # parapet wall with pillars at the back
    pw = MeshAcc('parapet', [wall, wall_top])
    pw.box(-half, y1, -0.4, half, y1 + 0.35, 1.05, 0, top_mi=1)
    for k in range(int(width / 3.4) + 1):
        px = -half + k * 3.4
        pw.box(px - 0.25, y1 - 0.05, -0.4, px + 0.25, y1 + 0.45, 1.35, 0, top_mi=1)
    obs.append(pw.build(ENV_INDEX))
    # rail pipes
    obs.append(loft('rail', [(-half, y1 + 0.1, 1.6), (half, y1 + 0.1, 1.6)], 0.045, ring=8, samples=1,
                    mats=[rail], pass_index=ENV_INDEX, subsurf=0))
    # big black water tanks on stands
    for (tx, ty, s) in ((-10.5, 4.6, 1.0), (11.5, 5.2, 0.85), (-13.5, 5.6, 0.7)):
        st = MeshAcc(f'stand{tx}', [steel])
        for dx in (-0.8, 0.8):
            for dy in (-0.8, 0.8):
                st.box(tx + dx * s - 0.06, ty + dy * s - 0.06, 0, tx + dx * s + 0.06, ty + dy * s + 0.06, 1.6, 0)
        st.box(tx - 1.0 * s, ty - 1.0 * s, 1.6, tx + 1.0 * s, ty + 1.0 * s, 1.75, 0)
        obs.append(st.build(ENV_INDEX))
        obs.append(loft(f'tank{tx}', [(tx, ty, 1.75), (tx, ty, 1.75 + 2.2 * s)],
                        lambda t: 1.05 * s * (1 - 0.05 * t), ring=24, samples=4, mats=[tank],
                        offset_fn=lambda t, a, d: 1.0 + 0.035 * (int(t * 7) % 2), pass_index=ENV_INDEX,
                        subsurf=1, cap_depth=0.25))
    # satellite dish
    obs.append(blob('dish', lambda p: Vector((p.x, p.y * 0.25 + (p.x * p.x + p.z * p.z) * 0.25, p.z)) if p.y < 0 else
                    Vector((p.x, -0.0, p.z)), segments=20, rings=10, mats=[ml.get('env.dish', (0.92, 0.92, 0.9))],
                    matrix=Matrix.Translation((-12.5, 6.0, 2.0)) @ Matrix.Rotation(-0.5, 4, 'X') @
                    Matrix.Rotation(0.4, 4, 'Z') @ Matrix.Diagonal((0.8, 0.8, 0.8, 1)), pass_index=ENV_INDEX))
    obs.append(cylinder('dish_pole', (-12.5, 6.2, 0), (-12.5, 6.1, 1.9), 0.05, mats=[steel], pass_index=ENV_INDEX))
    # billboard on scaffold
    if billboard:
        tex = poster_texture(os.path.join(build_dir, 'poster.png'), night=night)
        pm = tex_material(ml, 'env.poster', tex)
        bx0, bx1, bz0, bz1, by = 3.0, 13.0, 3.4, 9.0, 14.0
        bb = make_object('billboard', [(bx0, by, bz0), (bx1, by, bz0), (bx1, by, bz1), (bx0, by, bz1)],
                         [(0, 1, 2, 3)], [pm], [0], subsurf=0, smooth=False, pass_index=ENV_INDEX)
        uv = bb.data.uv_layers.new()
        for li, (u, v) in zip(range(4), ((0, 0), (1, 0), (1, 1), (0, 1))):
            uv.data[li].uv = (u, v)
        obs.append(bb)
        fr = MeshAcc('bb_frame', [steel])
        fr.box(bx0 - 0.2, by + 0.01, bz0 - 0.25, bx1 + 0.2, by + 0.25, bz0, 0)
        fr.box(bx0 - 0.2, by + 0.01, bz1, bx1 + 0.2, by + 0.25, bz1 + 0.25, 0)
        for k in range(6):
            px = bx0 + k * (bx1 - bx0) / 5
            fr.box(px - 0.08, by + 0.3, 0, px + 0.08, by + 0.46, bz1, 0)
        obs.append(fr.build(ENV_INDEX))
        # marigold garlands hanging across the billboard
        mari = ml.get('env.marigold', (1.0, 0.55, 0.05))
        mari2 = ml.get('env.marigold2', (1.0, 0.82, 0.10))
        for g in range(4):
            p0 = Vector((bx0 + g * 2.5, by - 0.15, bz1 - 0.15))
            p1 = Vector((bx0 + g * 2.5 + 2.5, by - 0.15, bz1 - 0.15))
            for k, p in enumerate(catenary(p0, p1, 0.9, 18)):
                obs.append(blob(f'mg{g}_{k}', lambda q: q, segments=8, rings=5, mats=[mari if k % 3 else mari2],
                                matrix=Matrix.Translation(p) @ Matrix.Diagonal((0.16, 0.16, 0.16, 1)), subsurf=0,
                                pass_index=ENV_INDEX))
    # bunting flags strung from the tanks to the billboard pole
    flag_cols = [(0.95, 0.25, 0.25), (1.0, 0.8, 0.1), (0.2, 0.7, 0.4), (0.25, 0.5, 0.95), (0.95, 0.45, 0.8)]
    fmats = [ml.get(f'env.flag{i}', c) for i, c in enumerate(flag_cols)]
    for (a, b, sag) in (((-10.5, 4.6, 4.3), (3.0, 14.0, 8.9), 1.3),):
        line = catenary(a, b, sag, 40)
        obs.append(loft(f'bunting_line{a[0]}', line, 0.015, ring=5, samples=1, mats=[steel], subsurf=0,
                        pass_index=ENV_INDEX))
        acc = MeshAcc(f'flags{a[0]}', fmats)
        for k in range(1, len(line) - 1, 1):
            p, q = line[k], line[k + 1]
            m = (p + q) / 2
            acc.v += [p, q, m + Vector((0, 0, -0.55))]
            i = len(acc.v) - 3
            acc.f.append((i, i + 1, i + 2))
            acc.m.append(k % len(fmats))
        obs.append(acc.build(ENV_INDEX))
    # overhead wires (a very Indian skyline signature)
    wire = ml.get('env.wire', (0.05, 0.05, 0.07), ink=True)
    for k in range(5):
        a = (-60, 18 + k * 3, 9 + k * 1.5 + rng.uniform(-1, 1))
        b = (60, 26 + k * 4, 11 + k * 1.2 + rng.uniform(-1, 1))
        obs.append(loft(f'wire{k}', catenary(a, b, rng.uniform(3, 6), 50), 0.035, ring=4, samples=1, mats=[wire],
                        subsurf=0, pass_index=ENV_INDEX + 1))
    return obs


def kites(ml, n=7, seed=5):
    rng = random.Random(seed)
    cols = [(0.95, 0.2, 0.3), (1.0, 0.75, 0.1), (0.2, 0.6, 0.95), (0.3, 0.85, 0.4), (0.9, 0.3, 0.85)]
    obs = []
    for i in range(n):
        c = Vector((rng.uniform(-40, 40), rng.uniform(45, 90), rng.uniform(12, 30)))
        s = rng.uniform(0.9, 1.4)
        m1 = ml.get(f'env.kite{i % 5}', cols[i % 5])
        m2 = ml.get(f'env.kite{(i + 2) % 5}', cols[(i + 2) % 5])
        rot = Matrix.Rotation(rng.uniform(-0.5, 0.5), 3, 'Y')
        pts = [Vector((0, 0, 1.0)), Vector((0.75, 0, 0.2)), Vector((0, 0, -1.0)), Vector((-0.75, 0, 0.2))]
        pts = [c + rot @ (p * s) for p in pts]
        ob = make_object(f'kite{i}', pts + [c + rot @ Vector((0, 0, 0.2)) * s],
                         [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)], [m1, m2], [0, 1, 0, 1], subsurf=0,
                         smooth=False, pass_index=CITY_INDEX + 5)
        obs.append(ob)
        tail = [pts[2] + Vector((math.sin(k) * 0.4, 0, -k * 0.6)) for k in range(6)]
        obs.append(loft(f'kitetail{i}', tail, 0.03, ring=4, samples=3, mats=[m2], subsurf=0,
                        pass_index=CITY_INDEX + 5))
        # string falls to the city
        obs.append(loft(f'kitestring{i}', [pts[2], pts[2] + Vector((rng.uniform(-8, 8), -10, -25))], 0.01, ring=3,
                        samples=1, mats=[ml.get('env.wire', (0.05, 0.05, 0.07), ink=True)], subsurf=0,
                        pass_index=CITY_INDEX + 5))
    return obs


def promenade(ml, night=True):
    """Sea-front stage: seawall, tetrapods, arc of lamps, sea plane."""
    obs = []
    sea = ml.get('env.sea', (0.10, 0.16, 0.32) if night else (0.25, 0.45, 0.70), spec=1.0)
    wall = ml.get('env.seawall', (0.55, 0.52, 0.58) if night else (0.85, 0.80, 0.72))
    path = ml.get('env.walk', (0.30, 0.28, 0.36) if night else (0.75, 0.70, 0.65))
    lamp = ml.get('env.lamp', (1.0, 0.85, 0.5), emit=8.0 if night else 0.0)
    pole = ml.get('env.pole', (0.15, 0.15, 0.2), spec=0.6)
    acc = MeshAcc('walkway', [path, wall])
    acc.box(-40, -8, -0.3, 40, 6, 0.0, 0, faces=('top', 'front'))
    acc.box(-40, 6, -1.5, 40, 6.6, 0.8, 1, faces=('top', 'front', 'back'))
    obs.append(acc.build(ENV_INDEX))
    obs.append(make_object('sea', [(-400, 6.6, -1.2), (400, 6.6, -1.2), (400, 600, -1.2), (-400, 600, -1.2)],
                           [(0, 1, 2, 3)], [sea], [0], subsurf=0, smooth=False, pass_index=CITY_INDEX + 3))
    for k in range(-6, 7):
        x = k * 6.5
        obs.append(cylinder(f'lamppole{k}', (x, 5.6, 0), (x, 5.6, 5.5), 0.08, mats=[pole], pass_index=ENV_INDEX))
        obs.append(blob(f'lamp{k}', lambda p: p, segments=10, rings=6, mats=[lamp], subsurf=0,
                        matrix=Matrix.Translation((x, 5.3, 5.6)) @ Matrix.Diagonal((0.32, 0.32, 0.22, 1)),
                        pass_index=ENV_INDEX))
    # the far arc of lamps ("Queen's Necklace")
    for k in range(90):
        a = math.radians(-75 + 150 * k / 89)
        p = Vector((math.sin(a) * 300, 300 - math.cos(a) * 190, 2.5))
        obs.append(blob(f'neck{k}', lambda q: q, segments=6, rings=4, mats=[lamp], subsurf=0,
                        matrix=Matrix.Translation(p) @ Matrix.Diagonal((0.75, 0.75, 0.75, 1)), pass_index=CITY_INDEX + 4))
    return obs
