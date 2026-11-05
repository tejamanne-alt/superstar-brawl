"""Signature props. Each function takes the Character and keyword args from the roster/pose."""
import math
from mathutils import Vector, Matrix

from .geo import loft, blob, catmull_rom, look_matrix, cylinder


def _N(v):
    return Vector(v).normalized()


def towel(ch, side='L', color_key='towel'):
    """Rayalaseema-style shoulder towel (kanduva) draped over one shoulder."""
    H, J = ch.H, ch.J
    sgn = 1 if side == 'L' else -1
    S = J['sh' + side]
    C = J['chest']
    f = ch.fwd
    up = ch.spine_dir
    lat = ch.lat_c * sgn
    pts = [C + lat * 0.05 * H - f * -0.075 * H - up * 0.16 * H,
           C + lat * 0.07 * H + f * 0.09 * H - up * 0.02 * H,
           S + up * 0.035 * H + lat * -0.02 * H,
           C + lat * 0.08 * H - f * 0.095 * H - up * 0.06 * H,
           C + lat * 0.09 * H - f * 0.10 * H - up * 0.24 * H]
    ob = loft(ch.nm('towel'), pts, lambda t: (0.045 * H, 0.008 * H), ring=10, samples=6, side_hint=lat,
              mats=[ch.mat(color_key), ch.mat(color_key + '_trim')],
              mat_fn=lambda t, a, d: 1 if (t < 0.06 or t > 0.94) else 0,
              offset_fn=lambda t, a, d: 1.0 + 0.08 * math.sin(t * 30))
    ch.track(ob)


def chain(ch, side='R', swing=(0.0, -0.6, 1.0), length=0.75, links=34):
    """Bicycle chain swung from the fist."""
    H = ch.H
    M = ch.hand_frames[side]
    start = M @ Vector((0, -0.02 * H, 0.06 * H))
    d = ch.local_dir(swing)
    side_v = d.orthogonal().normalized()
    pts = [start, start + d * length * H * 0.35 + side_v * 0.08 * H,
           start + d * length * H * 0.7 - side_v * 0.05 * H, start + d * length * H + side_v * 0.12 * H]
    path = catmull_rom(pts, 20)
    acc = [0.0]
    for a, b in zip(path, path[1:]):
        acc.append(acc[-1] + (b - a).length)
    total = acc[-1]
    mat = ch.mat('steel')
    import bmesh
    import bpy
    for i in range(links):
        s = total * i / (links - 1)
        j = max(k for k in range(len(acc)) if acc[k] <= s + 1e-9)
        j = min(j, len(path) - 2)
        u = (s - acc[j]) / max(acc[j + 1] - acc[j], 1e-9)
        p = path[j].lerp(path[j + 1], u)
        t = (path[j + 1] - path[j]).normalized()
        n = t.orthogonal().normalized()
        if i % 2:
            n = t.cross(n)
        m = look_matrix(t, n, p)
        # link = stretched torus
        me = bpy.data.meshes.new(ch.nm(f'link{i}'))
        bm = bmesh.new()
        R, r = 0.016 * H, 0.0045 * H
        segs, rs = 12, 6
        vs = []
        for a in range(segs):
            th = 2 * math.pi * a / segs
            for b in range(rs):
                ph = 2 * math.pi * b / rs
                x = (R + r * math.cos(ph)) * math.cos(th) * 0.6
                y = (R + r * math.cos(ph)) * math.sin(th) * 1.25
                z = r * math.sin(ph)
                vs.append(bm.verts.new(m @ Vector((x, y, z))))
        for a in range(segs):
            for b in range(rs):
                a2, b2 = (a + 1) % segs, (b + 1) % rs
                bm.faces.new((vs[a * rs + b], vs[a2 * rs + b], vs[a2 * rs + b2], vs[a * rs + b2]))
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new(me.name, me)
        bpy.context.scene.collection.objects.link(ob)
        me.materials.append(mat)
        for poly in me.polygons:
            poly.use_smooth = True
        ch.track(ob)


def lathi(ch, side='R', direction=(0.3, -0.2, 1.0), length=0.62):
    H = ch.H
    M = ch.hand_frames[side]
    c = M @ Vector((0, -0.01 * H, 0.045 * H))
    d = ch.local_dir(direction)
    p0 = c - d * length * H * 0.25
    p1 = c + d * length * H * 0.75
    ob = loft(ch.nm('lathi'), [p0, p1], 0.012 * H, ring=10, samples=2, mats=[ch.mat('lathi'), ch.mat('lathi_band')],
              mat_fn=lambda t, a, dd: 1 if (t < 0.06 or 0.92 < t or 0.3 < t < 0.33) else 0)
    ch.track(ob)


def flying_glasses(ch, offset=(0.10, -0.12, 0.10), spin=(25, 0, 40), size=1.0):
    """Sunglasses mid-flip, hovering near the hand/face."""
    H = ch.H
    hd = ch.head
    c = ch.J['head'] + (ch.R3 @ Vector(offset)) * H
    s = hd.scale * size
    rx, ry, rz = [math.radians(a) for a in spin]
    R = (Matrix.Rotation(rz, 3, 'Z') @ Matrix.Rotation(rx, 3, 'X') @ Matrix.Rotation(ry, 3, 'Y'))
    M = Matrix.Translation(c) @ R.to_4x4() @ Matrix.Diagonal((s, s, s, 1))
    frame, lens = ch.mat('frame'), ch.mat('lens')
    for side in (-1, 1):
        cc = Vector((side * 0.29, 0, 0))
        pts = []
        for k in range(13):
            a = 2 * math.pi * k / 12
            pts.append(cc + Vector((math.cos(a) * 0.20, 0, math.sin(a) * 0.13 - (0.03 if math.sin(a) < 0 else 0))))
        ch.track(loft(ch.nm(f'fg_frame{side}'), [M @ p for p in pts], 0.014 * s, ring=6, samples=3,
                      mats=[frame], caps=(False, False)))
        lm = M @ Matrix.Translation(cc) @ Matrix.Diagonal((0.195, 0.02, 0.135, 1))
        ch.track(blob(ch.nm(f'fg_lens{side}'), lambda p: p, segments=16, rings=8, mats=[lens], matrix=lm))
        ch.track(loft(ch.nm(f'fg_temple{side}'), [M @ (cc + Vector((side * 0.2, 0, 0.03))),
                                                   M @ (cc + Vector((side * 0.24, 0.6, 0.02)))],
                      0.012 * s, ring=6, samples=2, mats=[frame]))
    ch.track(loft(ch.nm('fg_bridge'), [M @ Vector((-0.1, 0, 0.05)), M @ Vector((0, 0, 0.08)), M @ Vector((0.1, 0, 0.05))],
                  0.013 * s, ring=6, samples=3, mats=[frame]))


def watch(ch, side='L', key='gold'):
    H, J = ch.H, ch.J
    W, E = J['wr' + side], J['el' + side]
    d = (W - E).normalized()
    c = W - d * 0.025 * H
    ch.track(loft(ch.nm('watch' + side), [c - d * 0.01 * H, c + d * 0.01 * H], 0.029 * H, ring=14, samples=2,
                  mats=[ch.mat(key)], caps=(False, False)))


def epaulettes(ch):
    H, J = ch.H, ch.J
    for s, sg in (('L', 1), ('R', -1)):
        S = J['sh' + s]
        c = S + ch.spine_dir * 0.045 * H - ch.lat_c * sg * 0.005 * H
        m = look_matrix(ch.fwd, ch.spine_dir, c) @ Matrix.Diagonal((0.06 * H, 0.03 * H, 0.008 * H, 1))
        ch.track(blob(ch.nm('epaulette' + s), lambda p: p, segments=12, rings=6, mats=[ch.mat('capband')], matrix=m))
        for k in range(3):
            m2 = look_matrix(ch.fwd, ch.spine_dir, c + ch.lat_c * sg * (0.015 * H * (k - 1)) + ch.spine_dir * 0.009 * H) @ \
                Matrix.Diagonal((0.006 * H, 0.006 * H, 0.004 * H, 1))
            ch.track(blob(ch.nm(f'star{s}{k}'), lambda p: p, segments=8, rings=4, mats=[ch.mat('gold')], matrix=m2))


def chest_badge(ch, key='gold', side='L', size=0.02):
    H, J = ch.H, ch.J
    c = J['chest'] + ch.lat_c * (0.06 * H if side == 'L' else -0.06 * H) - ch.spine_dir * 0.06 * H + ch.fwd * 0.083 * H
    m = look_matrix(ch.fwd, ch.spine_dir, c) @ Matrix.Diagonal((size * H, 0.004 * H, size * 1.2 * H, 1))
    ch.track(blob(ch.nm('chestbadge'), lambda p: p, segments=10, rings=6, mats=[ch.mat(key)], matrix=m))


def energy(ch, side='R', radius=0.09, key='energy', rings=3):
    """Glowing energy halo around a fist (emission pass -> comic glow + speed lines)."""
    H = ch.H
    M = ch.hand_frames[side]
    c = M @ Vector((0, 0, 0.05 * H))
    z = (M.to_3x3() @ Vector((0, 0, 1))).normalized()
    for i in range(rings):
        r = radius * H * (1 + 0.45 * i)
        x = z.orthogonal().normalized()
        y = z.cross(x)
        pts = []
        for k in range(25):
            a = 2 * math.pi * k / 24
            jitter = 1 + 0.12 * math.sin(a * 5 + i)
            pts.append(c + (x * math.cos(a) + y * math.sin(a)) * r * jitter + z * (i * 0.03 * H))
        ch.track(loft(ch.nm(f'energy{side}{i}'), pts, 0.006 * H * (1 - i * 0.2), ring=6, samples=3,
                      mats=[ch.mat(key)], caps=(False, False)))
    ob = blob(ch.nm(f'energycore{side}'), lambda p: p, segments=16, rings=10, mats=[ch.mat(key)],
              matrix=Matrix.Translation(c) @ Matrix.Diagonal((radius * H * 0.35,) * 3 + (1,)))
    ch.track(ob)


def varma_lines(ch, side='R', key='energy'):
    """Pressure-point energy streaks from the two-finger strike."""
    H = ch.H
    M = ch.hand_frames[side]
    tip = M @ Vector((0, -0.01 * H, 0.16 * H))
    z = (M.to_3x3() @ Vector((0, 0, 1))).normalized()
    for i in range(5):
        a = i * 1.3
        off = z.orthogonal().normalized()
        off = Matrix.Rotation(a, 3, z) @ off
        pts = [tip + z * 0.02 * H, tip + z * (0.15 + 0.03 * i) * H + off * 0.04 * H,
               tip + z * (0.32 + 0.05 * i) * H + off * 0.10 * H]
        ch.track(loft(ch.nm(f'varma{i}'), pts, lambda t: 0.006 * H * (1 - t) + 0.001 * H, ring=6, samples=4,
                      mats=[ch.mat(key)]))


def trishul(ch, side='R', direction=(0, 0, 1), length=1.1):
    H = ch.H
    M = ch.hand_frames[side]
    c = M @ Vector((0, -0.01 * H, 0.045 * H))
    d = ch.local_dir(direction)
    p0 = c - d * 0.35 * H
    p1 = c + d * length * H * 0.65
    ch.track(loft(ch.nm('trishul_shaft'), [p0, p1], 0.011 * H, ring=10, samples=2, mats=[ch.mat('steel')]))
    side_v = ch.fwd.cross(d).normalized()
    for k in (-1, 0, 1):
        base = p1 + side_v * k * 0.0
        pts = [p1, p1 + side_v * k * 0.09 * H + d * 0.05 * H, p1 + side_v * k * 0.08 * H + d * (0.16 + 0.05 * (k == 0)) * H]
        ch.track(loft(ch.nm(f'trishul_prong{k}'), pts, lambda t: 0.014 * H * (1 - 0.85 * t), ring=8, samples=4,
                      mats=[ch.mat('steel')]))
