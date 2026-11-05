"""Procedural stylized heads.

The head is authored in a local unit space (sphere radius 1, face toward -Y, up +Z), then
mapped to the world with a 4x4 matrix that already contains the head scale. Features are
projected onto the deformed skull with a BVH ray-cast so eyes, brows, noses, mustaches etc.
sit on the surface whatever the skull shape. Every caricature knob is in the `face` dict of
the roster spec.
"""
import math
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
import bmesh

from .geo import loft, blob, make_object, lerp, smoothstep, catmull_rom


def _gauss(x, mu, sig):
    return math.exp(-((x - mu) / sig) ** 2)


class Head:
    def __init__(self, ch, matrix, face, pass_index):
        self.ch = ch                  # Character (for materials / naming)
        self.M = matrix               # local -> world (includes scale)
        self.R = matrix.to_3x3().normalized()
        self.scale = matrix.to_3x3().col[0].length
        self.f = face
        self.pi = pass_index
        self.parts = []
        self._build_skull()

    # ------------------------------------------------------------------ skull
    def deform(self, p):
        f = self.f
        x, y, z = p.x, p.y, p.z
        X = x * 0.80 * f.get('width', 1.0)
        Y = y * 0.93 * f.get('depth', 1.0)
        Z = z * 1.08 * f.get('height', 1.0)
        j = smoothstep(0.05, -0.95, z)
        X *= lerp(1.0, f.get('jaw', 0.80), j)
        if y < 0:
            Y -= f.get('chin', 0.10) * j * j * (-y)
        Z -= f.get('chin_len', 0.10) * j
        if y > 0 and z > -0.3:
            Y *= 1.0 + 0.10 * smoothstep(-0.3, 0.4, z)
        front = -0.56 * f.get('depth', 1.0)
        if Y < front:
            Y = front + (Y - front) * 0.5
        if y < 0:
            Y -= f.get('brow', 0.05) * _gauss(z, 0.25, 0.09) * (-y)
            X *= 1.0 + f.get('cheek', 0.05) * _gauss(z, -0.05, 0.16) * (-y)
        X *= 1.0 + f.get('temple', 0.0) * _gauss(z, 0.45, 0.2)
        return Vector((X, Y, Z))

    def _build_skull(self):
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=28, radius=1.0)
        bm.verts.ensure_lookup_table()
        loc = [self.deform(v.co.copy()) for v in bm.verts]
        faces = [tuple(v.index for v in f.verts) for f in bm.faces]
        bm.free()
        self.bvh = BVHTree.FromPolygons(loc, faces)
        verts = [self.M @ p for p in loc]
        ch = self.ch
        self.skull = make_object(ch.nm('head'), verts, faces, [ch.mat('skin')], [0] * len(faces),
                                 subsurf=1, pass_index=self.pi)
        self.parts.append(self.skull)

    # ------------------------------------------------------------------ helpers
    def surf(self, x, z, out=0.0):
        """Project (x, z) onto the face front; returns local point pushed `out` along normal."""
        hit, n, _, _ = self.bvh.ray_cast(Vector((x, -3.0, z)), Vector((0, 1, 0)))
        if hit is None:
            hit, n = Vector((x, -0.6, z)), Vector((0, -1, 0))
        return hit + n * out

    def surf_dir(self, d, out=0.0):
        d = Vector(d).normalized()
        hit, n, _, _ = self.bvh.ray_cast(d * 3.0, -d)
        if hit is None:
            return d, d
        return hit + n * out, n

    def W(self, p):
        return self.M @ Vector(p)

    def Wd(self, d):
        return (self.R @ Vector(d)).normalized()

    def floft(self, name, pts, radius, mat, **kw):
        s = self.scale
        r = radius
        if callable(radius):
            r = (lambda t, _r=radius: tuple(q * s for q in (_r(t) if isinstance(_r(t), tuple) else (_r(t), _r(t)))))
        elif isinstance(radius, tuple):
            r = (radius[0] * s, radius[1] * s)
        else:
            r = radius * s
        hint = kw.pop('hint', (0, -1, 0))
        ob = loft(self.ch.nm(name), [self.W(p) for p in pts], r, mats=[mat],
                  side_hint=self.Wd(hint), pass_index=self.pi, **kw)
        self.parts.append(ob)
        return ob

    def fblob(self, name, center, size, mat, rot=None, deform=None, subsurf=1, keep_fn=None,
              segments=20, rings=12):
        s = Matrix.Diagonal((size[0], size[1], size[2], 1.0))
        r = rot.to_4x4() if rot is not None else Matrix.Identity(4)
        local = Matrix.Translation(Vector(center)) @ r @ s
        ob = blob(self.ch.nm(name), deform or (lambda p: p), segments=segments, rings=rings,
                  mats=[mat], matrix=self.M @ local, subsurf=subsurf, pass_index=self.pi,
                  keep_fn=keep_fn)
        self.parts.append(ob)
        return ob

    def shell(self, name, keep, thick, mat, segments=48, rings=32, subsurf=1):
        """Offset part of the skull outward (hair, beards, masks, bandages)."""
        def deform(p):
            q = self.deform(p)
            return q + q.normalized() * thick(p)
        ob = blob(self.ch.nm(name), deform, segments=segments, rings=rings, mats=[mat],
                  matrix=self.M, subsurf=subsurf, pass_index=self.pi, keep_fn=keep)
        sol = ob.modifiers.new('solid', 'SOLIDIFY')
        sol.thickness = 0.012 * self.scale
        sol.offset = -1
        ob.modifiers.move(1, 0)
        self.parts.append(ob)
        return ob

    # ------------------------------------------------------------------ features
    def build(self):
        f = self.f
        expr = f.get('expr', 'angry')
        self.eyes(expr)
        self.brows(expr)
        self.nose()
        self.mouth(expr)
        self.ears()
        if f.get('lines'):
            self.age_lines(f['lines'])
        hair = f.get('hair')
        if hair:
            self.hair(hair)
        if f.get('beard'):
            self.beard(f['beard'])
        if f.get('mustache'):
            self.mustache(f['mustache'])
        g = f.get('glasses')
        if g:
            self.glasses(g)
        hg = f.get('headgear')
        if hg:
            getattr(self, 'hg_' + hg)()
        return self.parts

    def eyes(self, expr):
        ch, f = self.ch, self.f
        ez = f.get('eye_z', 0.11)
        ex = f.get('eye_x', 0.29)
        open_ = f.get('eye_open', 1.0) * (0.75 if expr in ('angry', 'smirk') else 1.0)
        look = f.get('look', (0.0, 0.0))
        for side in (-1, 1):
            c = self.surf(side * ex, ez, -0.015)
            n = (c - Vector((0, 0, ez * 0.3))).normalized()
            rot = (Vector((0, -1, 0)).rotation_difference(Vector((n.x * 0.6, n.y, 0)).normalized())).to_matrix()
            self.fblob(f'eye{side}', c, (0.15, 0.05, 0.085 * open_), ch.mat('eyewhite'), rot=rot)
            ic = c + Vector((look[0] * 0.05 - side * 0.01, -0.04, look[1] * 0.03 - 0.005))
            self.fblob(f'iris{side}', ic, (0.065, 0.02, 0.065 * min(1, open_ * 1.05)), ch.mat('iris'), rot=rot)
            hi = ic + Vector((0.025, -0.02, 0.025))
            self.fblob(f'catch{side}', hi, (0.018, 0.01, 0.018), ch.mat('eyewhite'), segments=10, rings=6)
            # upper lid: the main comic eye line
            lift = 0.02 if expr in ('shout', 'calm') else 0.0
            inner_drop = 0.03 if expr == 'angry' else 0.0
            pts = [self.surf(side * (ex - 0.16), ez + 0.01 - inner_drop, 0.02),
                   self.surf(side * (ex - 0.05), ez + 0.07 * open_ + lift, 0.03),
                   self.surf(side * (ex + 0.08), ez + 0.065 * open_ + lift, 0.03),
                   self.surf(side * (ex + 0.17), ez + 0.0, 0.015)]
            self.floft(f'lid{side}', pts, lambda t: 0.022 * (0.5 + math.sin(math.pi * t)),
                       ch.mat('ink'), ring=8, samples=5)
            # lower lid hint
            pts = [self.surf(side * (ex - 0.10), ez - 0.06 * open_, 0.012),
                   self.surf(side * (ex + 0.06), ez - 0.07 * open_, 0.012)]
            self.floft(f'llid{side}', pts, lambda t: 0.008 * math.sin(math.pi * t) + 0.003,
                       ch.mat('ink'), ring=6, samples=4)
            if f.get('eye_bags'):
                pts = [self.surf(side * (ex - 0.06), ez - 0.12, 0.01),
                       self.surf(side * (ex + 0.10), ez - 0.11, 0.01)]
                self.floft(f'bag{side}', pts, lambda t: 0.006 * math.sin(math.pi * t) + 0.002,
                           ch.mat('ink'), ring=6, samples=4)

    def brows(self, expr):
        ch, f = self.ch, self.f
        bz = f.get('brow_z', 0.29)
        th = f.get('brow_thick', 1.0)
        mat = ch.mat('brow')
        ang = {'angry': -0.07, 'shout': -0.04, 'smirk': -0.03, 'calm': 0.01, 'grin': 0.02}.get(expr, 0)
        smirk_raise = 0.04 if expr == 'smirk' else 0.0
        for side in (-1, 1):
            r = smirk_raise if side > 0 else 0.0
            pts = [self.surf(side * 0.08, bz + ang + r, 0.02),
                   self.surf(side * 0.25, bz + 0.03 + r, 0.03),
                   self.surf(side * 0.46, bz - ang * 0.3 + r - 0.01, 0.02)]
            self.floft(f'brow{side}', pts,
                       lambda t: (0.045 * th * (1.0 - 0.55 * t), 0.022 * th * (1 - 0.4 * t)),
                       mat, ring=8, samples=6, hint=(0, 0, 1))

    def nose(self):
        ch, f = self.ch, self.f
        nl = f.get('nose_len', 1.0)
        nw = f.get('nose_w', 1.0)
        tip_z = 0.11 - 0.30 * nl
        out = f.get('nose_out', 1.0)
        hook = f.get('nose_hook', 0.0)
        pts = [self.surf(0, 0.17, -0.02),
               self.surf(0, 0.06, 0.045 * out),
               self.surf(0, tip_z + 0.07, (0.13 + hook * 0.04) * out),
               self.surf(0, tip_z, 0.12 * out)]
        self.floft('nose', pts, lambda t: (lerp(0.045, 0.075 * nw, t ** 1.5), lerp(0.04, 0.06, t)),
                   ch.mat('skin'), ring=12, samples=6, hint=(1, 0, 0))
        for side in (-1, 1):
            c = self.surf(side * 0.085 * nw, tip_z + 0.015, 0.045 * out)
            self.fblob(f'nostril{side}', c, (0.055 * nw, 0.05, 0.045), ch.mat('skin'))
        # nose shadow line (stylized nostril crease)
        for side in (-1, 1):
            pts = [self.surf(side * 0.12 * nw, tip_z + 0.06, 0.055 * out),
                   self.surf(side * 0.11 * nw, tip_z - 0.005, 0.06 * out),
                   self.surf(side * 0.04, tip_z - 0.03, 0.07 * out)]
            self.floft(f'ncrease{side}', pts, lambda t: 0.007 * math.sin(math.pi * t) + 0.002,
                       ch.mat('ink'), ring=6, samples=4)

    def mouth(self, expr):
        ch, f = self.ch, self.f
        mz = f.get('mouth_z', -0.42)
        mw = f.get('mouth_w', 0.20)
        if expr == 'shout':
            c = self.surf(0, mz - 0.03, -0.01)
            self.fblob('mouth_in', c, (mw * 0.95, 0.06, 0.11), ch.mat('mouth'))
            t = self.surf(0, mz + 0.045, 0.005)
            self.fblob('teeth', t, (mw * 0.75, 0.04, 0.03), ch.mat('eyewhite'))
            pts = [self.surf(-mw, mz, 0.01), self.surf(0, mz + 0.085, 0.03), self.surf(mw, mz, 0.01)]
            self.floft('lipU', pts, lambda t: 0.018, ch.mat('lip'), ring=8, samples=5)
            pts = [self.surf(-mw, mz - 0.02, 0.01), self.surf(0, mz - 0.15, 0.03), self.surf(mw, mz - 0.02, 0.01)]
            self.floft('lipL', pts, lambda t: 0.024, ch.mat('lip'), ring=8, samples=5)
            return
        curve = {'smirk': (0.0, 0.035), 'grin': (0.03, 0.03), 'calm': (0.0, 0.0),
                 'angry': (-0.02, -0.02)}.get(expr, (0, 0))
        pts = [self.surf(-mw, mz + curve[0], 0.01),
               self.surf(-mw * 0.4, mz - 0.005, 0.02),
               self.surf(mw * 0.4, mz - 0.005, 0.02),
               self.surf(mw, mz + curve[1], 0.01)]
        self.floft('mouthline', pts, lambda t: 0.012 + 0.006 * math.sin(math.pi * t), ch.mat('ink'),
                   ring=8, samples=5)
        if expr == 'grin':
            t = self.surf(0, mz - 0.015, 0.012)
            self.fblob('teeth', t, (mw * 0.8, 0.03, 0.028), ch.mat('eyewhite'))
        pts = [self.surf(-mw * 0.55, mz - 0.06, 0.015), self.surf(mw * 0.55, mz - 0.06, 0.015)]
        self.floft('lowerlip', pts, lambda t: (0.035 * math.sin(math.pi * t) + 0.004, 0.018), ch.mat('lip'),
                   ring=8, samples=5, hint=(1, 0, 0))
        # chin dimple / cleft
        if f.get('chin_cleft'):
            pts = [self.surf(0, mz - 0.25, 0.012), self.surf(0, mz - 0.33, 0.012)]
            self.floft('cleft', pts, lambda t: 0.007 * math.sin(math.pi * t) + 0.002, ch.mat('ink'), ring=6,
                       samples=3)

    def ears(self):
        ch = self.ch
        for side in (-1, 1):
            p, n = self.surf_dir((side, 0.05, 0.02), 0.0)
            c = p + Vector((side * 0.02, 0.02, 0.0))
            rot = Matrix.Rotation(side * 0.25, 3, 'Z')
            self.fblob(f'ear{side}', c, (0.07, 0.15, 0.20), ch.mat('skin'), rot=rot)

    def age_lines(self, lines):
        ch = self.ch
        if 'smile' in lines:
            for side in (-1, 1):
                pts = [self.surf(side * 0.17, -0.12, 0.03), self.surf(side * 0.25, -0.30, 0.025),
                       self.surf(side * 0.27, -0.46, 0.015)]
                self.floft(f'smile{side}', pts, lambda t: 0.010 * math.sin(math.pi * t) + 0.002,
                           ch.mat('ink'), ring=6, samples=5)
        if 'forehead' in lines:
            for i, z in enumerate((0.50, 0.58, 0.66)):
                pts = [self.surf(-0.28, z - 0.01, 0.012), self.surf(0, z + 0.015, 0.014),
                       self.surf(0.28, z - 0.01, 0.012)]
                self.floft(f'fh{i}', pts, lambda t: 0.006 * math.sin(math.pi * t) + 0.0015,
                           ch.mat('ink'), ring=6, samples=5)
        if 'crow' in lines:
            for side in (-1, 1):
                for k in range(2):
                    pts = [self.surf(side * 0.47, 0.13 - k * 0.06, 0.01),
                           self.surf(side * 0.56, 0.15 - k * 0.1, 0.008)]
                    self.floft(f'crow{side}{k}', pts, lambda t: 0.005 * math.sin(math.pi * t) + 0.001,
                               ch.mat('ink'), ring=6, samples=3)

    # ------------------------------------------------------------------ hair
    def hair(self, h):
        ch = self.ch
        mat = ch.mat('hair')
        style = h.get('style', 'short')
        vol = h.get('volume', 0.12)
        front = h.get('hairline', 0.45) + 0.17
        side_l = h.get('side', 0.12)
        back_l = h.get('back', -0.45)
        quiff = h.get('quiff', 0.0)
        curl = h.get('curl', 0.0)
        bald = style == 'bald_sides'

        def zt(p):
            phi = math.atan2(p.x, -p.y)  # 0 = front
            c = math.cos(phi)
            if c >= 0:
                return lerp(side_l, front, c ** 1.5)
            return lerp(side_l, back_l, (-c) ** 1.2)

        def keep(p):
            if p.z < zt(p):
                return False
            # cut around ears
            if abs(p.x) > 0.55 and abs(p.y) < 0.35 and p.z < side_l + 0.2:
                return False
            if bald and p.z > 0.25 and not (abs(p.x) > 0.62 or p.y > 0.55):
                return False
            return True

        def thick(p):
            edge = smoothstep(zt(p) - 0.02, zt(p) + 0.16, p.z)
            t = 0.035 + vol * smoothstep(-0.2, 0.9, p.z)
            if quiff:
                t += quiff * smoothstep(0.2, 0.75, p.z) * smoothstep(0.3, -0.9, p.y)
            if curl:
                n = (math.sin(13 * p.x + 1.3) * math.sin(11 * p.y + 0.7) * math.sin(12 * p.z))
                t += curl * max(0.0, n) * smoothstep(-0.3, 0.6, p.z)
            if h.get('part'):
                t -= 0.02 * _gauss(p.x, h['part'], 0.05) * smoothstep(0.4, 0.8, p.z)
            return (t * edge + 0.006) * (0.5 if bald else 1.0)

        self.shell('hair', keep, thick, mat)
        # chunky comic clumps: tapered strands that sweep from the hairline over the crown.
        n = h.get('clumps', 0)
        sweep = h.get('sweep', 0.0)
        for i in range(n):
            u = (i + 0.5) / n * 2 - 1
            x0 = u * 0.62
            pts = []
            path = ((front + 0.03, -0.80), (front + 0.30, -0.62), (0.97, -0.15), (0.82, 0.45), (0.45, 0.85))
            for k, (zz, yy) in enumerate(path):
                spread = 1 - 0.18 * k
                d = Vector((x0 * spread + sweep * 0.10 * k, yy, zz)).normalized()
                p, nrm = self.surf_dir(d)
                lift = 0.02 + vol * (0.55 + 0.35 * math.sin(math.pi * k / 4)) + quiff * (0.9 if k == 1 else 0.45 if k == 2 else 0.0)
                pts.append(p + nrm * lift)
            w = 0.62 / n + 0.07
            self.floft(f'clump{i}', pts,
                       lambda t, w=w: (0.02 + w * math.sin(math.pi * min(1, 0.08 + t * 1.05)) ** 0.6,
                                       0.02 + 0.09 * (1 + vol) * math.sin(math.pi * min(1, 0.08 + t * 1.05)) ** 0.6),
                       mat, ring=8, samples=5, hint=(1, 0, 0))
        if h.get('sideburns', True) and not bald:
            for side in (-1, 1):
                pts = [self.surf_dir((side, -0.25, 0.25), 0.03)[0], self.surf_dir((side, -0.3, -0.05), 0.02)[0]]
                self.floft(f'sideburn{side}', pts, lambda t: (0.05 * (1 - 0.5 * t), 0.03), mat, ring=8, samples=4,
                           hint=(0, 1, 0))
        if h.get('long'):
            # hair falling to the nape / shoulders
            for side in (-1, 0, 1):
                pts = [self.surf_dir((side * 0.6, 0.8, 0.2), 0.05)[0], self.surf_dir((side * 0.7, 0.9, -0.5), 0.1)[0],
                       self.surf_dir((side * 0.8, 1.0, -1.2), 0.25)[0]]
                self.floft(f'long{side}', pts, lambda t: (0.25 * (1 - 0.6 * t), 0.12), mat, ring=10, samples=5,
                           hint=(1, 0, 0))

    def beard(self, b):
        ch = self.ch
        style = b.get('style', 'full')
        mat = ch.mat('beard')
        th = b.get('thick', 0.06)
        mz = self.f.get('mouth_z', -0.42)

        def keep(p):
            if p.y > 0.3:
                return False
            if style == 'goatee':
                return abs(p.x) < 0.32 and p.z < -0.47 and p.y < 0.0
            if style == 'french':
                return abs(p.x) < 0.30 and p.z < -0.47 and p.y < 0.05
            zlim = -0.05 if abs(p.x) > 0.55 else lerp(-0.05, -0.28, smoothstep(0.55, 0.2, abs(p.x)))
            if p.z > zlim:
                return False
            if abs(p.x) < 0.22 and mz - 0.12 < p.z < mz + 0.1 and p.y < -0.3:
                return False
            return True

        def thick(p):
            t = th
            if style in ('full', 'goatee', 'french'):
                t += th * 0.6 * smoothstep(-0.5, -0.95, p.z)
            return t

        self.shell('beard', keep, thick, mat)

    def mustache(self, m):
        ch = self.ch
        style = m.get('style', 'thick')
        mz = self.f.get('mouth_z', -0.42)
        mat = ch.mat('beard') if m.get('use_beard_mat', True) else ch.mat('hair')
        w = m.get('width', 0.27)
        th = m.get('thick', 1.0)
        if style == 'handlebar':
            pts = [(-w - 0.06, mz + 0.06), (-w, mz - 0.02), (-0.12, mz + 0.06), (0, mz + 0.085),
                   (0.12, mz + 0.06), (w, mz - 0.02), (w + 0.06, mz + 0.06)]
            prof = lambda t: (0.035 + 0.05 * th * math.sin(math.pi * t) ** 0.6, 0.035)
        elif style == 'walrus':
            pts = [(-w, mz - 0.10), (-0.16, mz + 0.03), (0, mz + 0.08), (0.16, mz + 0.03), (w, mz - 0.10)]
            prof = lambda t: (0.04 + 0.06 * th * math.sin(math.pi * t) ** 0.5, 0.045)
        elif style == 'trim':
            pts = [(-w * 0.85, mz + 0.0), (0, mz + 0.075), (w * 0.85, mz + 0.0)]
            prof = lambda t: (0.015 + 0.025 * th * math.sin(math.pi * t) ** 0.7, 0.02)
        else:  # thick, slightly drooping
            pts = [(-w, mz - 0.04), (-0.14, mz + 0.04), (0, mz + 0.08), (0.14, mz + 0.04), (w, mz - 0.04)]
            prof = lambda t: (0.03 + 0.05 * th * math.sin(math.pi * t) ** 0.6, 0.035)
        wp = [self.surf(x, z, 0.045) for x, z in pts]
        self.floft('mustache', wp, prof, mat, ring=10, samples=5, hint=(0, 0, 1))

    # ------------------------------------------------------------------ eyewear & headgear
    def glasses(self, g):
        ch = self.ch
        kind = g if isinstance(g, str) else g.get('kind', 'aviator')
        frame = ch.mat('frame')
        lens = ch.mat('lens')
        ex, ez = self.f.get('eye_x', 0.29), self.f.get('eye_z', 0.11)
        rw, rh = (0.19, 0.15) if kind == 'aviator' else ((0.20, 0.12) if kind == 'shades' else (0.15, 0.13))
        for side in (-1, 1):
            c = self.surf(side * ex, ez - 0.01, 0.13)
            drop = 0.04 if kind == 'aviator' else 0.0
            pts = []
            for k in range(13):
                a = 2 * math.pi * k / 12
                x = math.cos(a) * rw
                z = math.sin(a) * rh - (drop if math.sin(a) < 0 and x * side > -0.05 else 0)
                pts.append(Vector((c.x + x, c.y + 0.012 * (x * side) / rw, c.z + z)))
            self.floft(f'frame{side}', pts, 0.014, frame, ring=6, samples=3, caps=(False, False))
            self.fblob(f'lens{side}', c + Vector((0, 0.005, -drop * 0.35)),
                       (rw * 0.98, 0.02, rh * 0.98 + drop * 0.4), lens)
            # temple arm to ear
            p_ear, _ = self.surf_dir((side, 0.1, 0.12), 0.04)
            self.floft(f'temple{side}', [c + Vector((side * rw, 0, 0.03)), p_ear], 0.012, frame, ring=6,
                       samples=3)
        pts = [self.surf(-ex + rw * 0.9, ez + 0.04, 0.14), self.surf(0, ez + 0.08, 0.15),
               self.surf(ex - rw * 0.9, ez + 0.04, 0.14)]
        self.floft('bridge', pts, 0.013, frame, ring=6, samples=4)
        if kind == 'aviator':
            pts = [self.surf(-ex + rw * 0.8, ez + 0.12, 0.14), self.surf(ex - rw * 0.8, ez + 0.12, 0.14)]
            self.floft('bridge2', pts, 0.011, frame, ring=6, samples=3)

    def hg_bandage(self):
        ch = self.ch
        mat = ch.mat('bandage')
        bands = [((0.15, 0.3, 1.0), 0.55, 0.13), ((0.0, -0.2, 1.0), 0.20, 0.10), ((-0.25, 0.1, 1.0), 0.80, 0.10),
                 ((0.35, -0.6, 1.0), 0.05, 0.08), ((-0.1, 0.9, 0.6), 0.10, 0.11), ((0.2, -0.95, 0.35), -0.05, 0.08),
                 ((-0.3, -0.4, 1.0), -0.40, 0.09), ((0.25, 0.2, 1.0), -0.62, 0.08)]
        ex, ez = self.f.get('eye_x', 0.29), self.f.get('eye_z', 0.11)
        mz = self.f.get('mouth_z', -0.42)
        for i, (n, d, w) in enumerate(bands):
            n = Vector(n).normalized()

            def keep(p, n=n, d=d, w=w):
                if abs(p.dot(n) - d) > w:
                    return False
                if p.y < -0.3 and abs(p.x) < 0.58 and ez - 0.09 < p.z < ez + 0.12:
                    return False  # eye slit
                if p.y < -0.5 and abs(p.x) < 0.25 and mz - 0.14 < p.z < mz + 0.1:
                    return False  # mouth
                return True
            self.shell(f'band{i}', keep, lambda p, i=i: 0.035 + 0.012 * i, mat, segments=96, rings=64)

    def hg_mask(self):
        ch = self.ch
        ex, ez = self.f.get('eye_x', 0.29), self.f.get('eye_z', 0.11)

        def keep(p):
            if p.y > 0.35:
                return False
            if not (-0.02 < p.z < 0.44):
                return False
            for side in (-1, 1):
                if ((p.x - side * ex) / 0.17) ** 2 + ((p.z - ez) / 0.10) ** 2 < 1.0 and p.y < 0:
                    return False
            if p.y < -0.6 and abs(p.x) < 0.09 and p.z < 0.05:
                return False
            return True
        self.shell('mask', keep, lambda p: 0.04 + 0.03 * smoothstep(0.1, -0.9, p.y), ch.mat('mask'),
                   segments=80, rings=56)

    def hg_cap_police(self):
        ch = self.ch
        crown = ch.mat('cap')
        pts = [Vector((0, 0.05, 0.40)), Vector((0, 0.07, 0.62)), Vector((0, 0.10, 0.80))]
        self.floft('cap_crown', pts, lambda t: (lerp(0.87, 1.10, t ** 1.5), lerp(0.99, 1.20, t ** 1.5)), crown,
                   ring=24, samples=4, hint=(1, 0, 0), cap_depth=0.18)
        self.floft('cap_band', [Vector((0, 0.05, 0.42)), Vector((0, 0.05, 0.6))], (0.88, 1.0),
                   ch.mat('capband'), ring=24, samples=2, caps=(False, False), hint=(1, 0, 0))

        def vis(p):
            return Vector((p.x * 0.78, -0.95 + p.y * 0.40, 0.44 + p.z * 0.05 - 0.10 * (p.y < 0) * abs(p.y)))
        self.fblob('visor', (0, 0, 0), (1, 1, 1), ch.mat('visor'), deform=vis,
                   keep_fn=lambda p: True)
        self.fblob('badge', self.surf(0, 0.62, 0.0) + Vector((0, -0.42, 0)), (0.13, 0.03, 0.12),
                   ch.mat('gold'))

    def hg_headband(self):
        ch = self.ch
        self.shell('headband', lambda p: abs(p.z - 0.40 + 0.25 * p.y) < 0.09, lambda p: 0.06,
                   ch.mat('headband'))
        # knot tails at the back
        p, n = self.surf_dir((0.15, 1.0, 0.25), 0.05)
        self.floft('hb_tail', [p, p + Vector((0.25, 0.55, -0.2)), p + Vector((0.45, 1.05, -0.05))],
                   lambda t: (0.10 * (1 - 0.5 * t), 0.02), ch.mat('headband'), ring=8, samples=4,
                   hint=(0, 0, 1))
