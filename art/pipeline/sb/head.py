"""Archie-styled heads: sculpted skull + projected 2D comic face + pointed-lock hair.

Head local space: sphere radius 1, face toward -Y, up +Z, +X = character's left.
The 4x4 matrix passed in maps local -> world and already contains the head scale.

  * skull   : deformed sphere (long face, angular jaw, defined chin, flat face plane)
  * face    : drawn by facepaint.py and projected front-on through UVs, so eyes, brows,
              mouth, mustaches and stubble stay crisp comic artwork
  * nose    : real 3D (gives the silhouette and the light/shadow split), shares the face decal
  * hair    : a cap shell plus 15-40 tapered ribbons ("locks") with pointed tips, which is
              what gives the modern teen-comic hair its sculpted, spiky read
"""
import math
import os

import bmesh
import bpy
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

from .geo import loft, blob, make_object, lerp, smoothstep, srgb_to_linear
from . import facepaint

FACE_DIR = os.path.join(os.path.dirname(__file__), '../../build/faces')


def _gauss(x, mu, sig):
    return math.exp(-((x - mu) / sig) ** 2)


def sph(az, el):
    a, e = math.radians(az), math.radians(el)
    return Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))


class Head:
    def __init__(self, ch, matrix, face, pass_index):
        self.ch = ch
        self.M = matrix
        self.Minv = matrix.inverted()
        self.R = matrix.to_3x3().normalized()
        self.scale = matrix.to_3x3().col[0].length
        self.f = face
        self.pi = pass_index
        self.parts = []
        self.expr = face.get('expr', 'grin')
        self.decal_mat = self._decal_material()
        self._build_skull()

    # ------------------------------------------------------------------ skull
    def deform(self, p):
        f = self.f
        x, y, z = p.x, p.y, p.z
        X = x * 0.80 * f.get('width', 1.0)
        Y = y * 0.94 * f.get('depth', 1.0)
        Z = z * 1.08 * f.get('height', 1.0)
        # angular jaw: full width down to the cheekbones, then a V to a defined chin
        j = smoothstep(-0.25, -1.0, z)
        X *= lerp(1.0, f.get('jaw', 0.72), j ** 1.2)
        if y < 0:
            Y -= f.get('chin', 0.12) * j * j * (-y)
        Z -= f.get('chin_len', 0.12) * smoothstep(0.0, -1.0, z)
        if y > 0 and z > -0.3:
            Y *= 1.0 + 0.12 * smoothstep(-0.3, 0.4, z)
        front = -0.58 * f.get('depth', 1.0)
        if Y < front:  # flat face plane keeps the projected artwork undistorted
            Y = front + (Y - front) * 0.45
        if y < 0:
            Y -= f.get('brow', 0.05) * _gauss(z, 0.27, 0.08) * (-y)
            X *= 1.0 + f.get('cheek', 0.06) * _gauss(z, -0.02, 0.14) * (-y)
            X *= 1.0 - 0.05 * _gauss(z, -0.38, 0.12) * (-y)  # cheek hollow
        X *= 1.0 + f.get('temple', 0.0) * _gauss(z, 0.45, 0.2)
        return Vector((X, Y, Z))

    def _decal_material(self):
        ch = self.ch
        os.makedirs(FACE_DIR, exist_ok=True)
        face = dict(self.f)
        pal = ch.spec['palette']
        face.setdefault('brow_col', pal.get('brow', pal.get('hair', (0.06, 0.04, 0.05))))
        face.setdefault('iris_col', pal.get('iris', (0.32, 0.18, 0.08)))
        face.setdefault('lip_col', pal.get('lip', tuple(c * 0.78 for c in pal['skin'])))
        if face.get('mustache_2d') is not None:
            face['mustache_2d'] = dict(face['mustache_2d'])
            face['mustache_2d'].setdefault('col', pal.get('beard', face['brow_col']))
        if face.get('beard_2d') is not None:
            face['beard_2d'] = dict(face['beard_2d'])
            face['beard_2d'].setdefault('col', pal.get('beard', pal.get('hair', (0.05, 0.04, 0.05))))
        path = os.path.join(FACE_DIR, f'{ch.cid}_{self.expr}.png')
        facepaint.paint_face(face, self.expr, path)
        name = f'{ch.cid}.face.{self.expr}'
        if name in ch.ml.mats:
            return ch.ml.mats[name]
        skin = pal['skin']
        m = ch.ml.get(name, skin, char=ch.cid, skin=True)
        nt = m.node_tree
        bsdf = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.image = bpy.data.images.load(path, check_existing=False)
        tex.image.alpha_mode = 'STRAIGHT'
        tex.extension = 'CLIP'
        tex.interpolation = 'Cubic'
        mix = nt.nodes.new('ShaderNodeMix')
        mix.data_type = 'RGBA'
        cols = [s for s in mix.inputs if s.type == 'RGBA']
        fac = [s for s in mix.inputs if s.type == 'VALUE'][0]
        out = [s for s in mix.outputs if s.type == 'RGBA'][0]
        cols[0].default_value = srgb_to_linear(skin) + (1.0,)
        nt.links.new(tex.outputs['Color'], cols[1])
        nt.links.new(tex.outputs['Alpha'], fac)
        nt.links.new(out, bsdf.inputs['Base Color'])
        return m

    def apply_decal_uv(self, ob, local=None, front_only=True):
        me = ob.data
        uv = me.uv_layers.new(name='UVMap')
        loc = local if local is not None else [self.Minv @ v.co for v in me.vertices]
        blank = (0.002, 0.998)
        for poly in me.polygons:
            vs = [loc[i] for i in poly.vertices]
            c = sum(vs, Vector()) / len(vs)
            front = (c.y < -0.02) or (c.z < -0.55 and c.y < 0.30)
            for li, vi in zip(poly.loop_indices, poly.vertices):
                if front or not front_only:
                    p = loc[vi]
                    uv.data[li].uv = facepaint.uv_of(p.x, p.z)
                else:
                    uv.data[li].uv = blank

    def _build_skull(self):
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=32, radius=1.0)
        bm.verts.ensure_lookup_table()
        loc = [self.deform(v.co.copy()) for v in bm.verts]
        faces = [tuple(v.index for v in f.verts) for f in bm.faces]
        bm.free()
        self.bvh = BVHTree.FromPolygons(loc, faces)
        verts = [self.M @ p for p in loc]
        self.skull = make_object(self.ch.nm('head'), verts, faces, [self.decal_mat], [0] * len(faces),
                                 subsurf=1, pass_index=self.pi)
        self.apply_decal_uv(self.skull, loc)
        self.parts.append(self.skull)

    # ------------------------------------------------------------------ helpers
    def surf(self, x, z, out=0.0):
        hit, n, _, _ = self.bvh.ray_cast(Vector((x, -3.0, z)), Vector((0, 1, 0)))
        if hit is None:
            hit, n = Vector((x, -0.6, z)), Vector((0, -1, 0))
        return hit + n * out

    def surf_dir(self, d, out=0.0):
        d = Vector(d).normalized()
        hit, n, _, _ = self.bvh.ray_cast(d * 3.0, -d)
        if hit is None:
            return d * 1.0, d
        return hit + n * out, n

    def W(self, p):
        return self.M @ Vector(p)

    def Wd(self, d):
        return (self.R @ Vector(d)).normalized()

    def floft(self, name, pts, radius, mat, **kw):
        s = self.scale
        if callable(radius):
            def r(t, _r=radius):
                v = _r(t)
                v = v if isinstance(v, tuple) else (v, v)
                return (v[0] * s, v[1] * s)
        elif isinstance(radius, tuple):
            r = (radius[0] * s, radius[1] * s)
        else:
            r = radius * s
        hint = kw.pop('hint', (0, -1, 0))
        side = (lambda t, _h=hint: self.Wd(_h(t))) if callable(hint) else self.Wd(hint)
        ob = loft(self.ch.nm(name), [self.W(p) for p in pts], r, mats=[mat], side_hint=side,
                  pass_index=self.pi, **kw)
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

    def shell(self, name, keep, thick, mat, segments=56, rings=40, subsurf=1, solid=0.012):
        def deform(p):
            q = self.deform(p)
            return q + q.normalized() * thick(p)
        ob = blob(self.ch.nm(name), deform, segments=segments, rings=rings, mats=[mat],
                  matrix=self.M, subsurf=subsurf, pass_index=self.pi, keep_fn=keep)
        sol = ob.modifiers.new('solid', 'SOLIDIFY')
        sol.thickness = solid * self.scale
        sol.offset = -1
        ob.modifiers.move(1, 0)
        self.parts.append(ob)
        return ob

    # ------------------------------------------------------------------ build
    def build(self):
        self.nose()
        self.ears()
        if self.f.get('hair'):
            self.hair(self.f['hair'])
        g = self.f.get('glasses')
        if g:
            self.glasses(g)
        hg = self.f.get('headgear')
        if hg:
            getattr(self, 'hg_' + hg)()
        return self.parts

    def nose(self):
        f = self.f
        nl, nw, out = f.get('nose_len', 1.0), f.get('nose_w', 1.0), f.get('nose_out', 1.0)
        tip_z = 0.11 - 0.30 * nl
        hook = f.get('nose_hook', 0.0)
        pts = [self.surf(0, 0.20, -0.03), self.surf(0, 0.08, 0.035 * out),
               self.surf(0, tip_z + 0.07, (0.115 + hook * 0.04) * out), self.surf(0, tip_z, 0.105 * out)]
        ob = self.floft('nose', pts, lambda t: (lerp(0.034, 0.068 * nw, t ** 1.8), lerp(0.03, 0.055, t)),
                        self.decal_mat, ring=12, samples=6, hint=(1, 0, 0), cap_depth=0.8)
        self.apply_decal_uv(ob, front_only=False)
        for side in (-1, 1):
            c = self.surf(side * 0.075 * nw, tip_z + 0.012, 0.035 * out)
            b = self.fblob(f'wing{side}', c, (0.048 * nw, 0.045, 0.040), self.decal_mat)
            self.apply_decal_uv(b, front_only=False)

    def ears(self):
        for side in (-1, 1):
            p, n = self.surf_dir((side, 0.08, 0.04), 0.0)
            c = p + Vector((side * 0.01, 0.03, -0.02))
            rot = Matrix.Rotation(side * 0.35, 3, 'Z') @ Matrix.Rotation(-0.12, 3, 'X')

            def ear(q):
                r = Vector((q.x * (0.55 if q.x * side < 0 else 1.0), q.y, q.z))
                if q.y < 0:  # cupped front
                    r.x *= 0.7
                return r
            self.fblob(f'ear{side}', c, (0.07, 0.14, 0.20), self.ch.mat('skin'), rot=rot, deform=ear)

    # ------------------------------------------------------------------ hair
    def lock(self, i, az0, el0, az1, el1, w=0.16, th=0.07, lift=(0.03, 0.06), vol=0.0, vol_at=0.45,
             tip=0.05, steps=8, mat=None):
        pts, nrms = [], []
        for k in range(steps):
            t = k / (steps - 1)
            d = sph(lerp(az0, az1, t), lerp(el0, el1, t))
            p, n = self.surf_dir(d)
            hump = math.sin(math.pi * min(1.0, t / (2 * vol_at))) if t < 2 * vol_at else 0.0
            if t >= vol_at:
                hump = max(hump, 1.0 - 0.75 * smoothstep(vol_at, 1.0, t))
            lf = lerp(lift[0], lift[1], t) + vol * hump + tip * smoothstep(0.65, 1.0, t)
            pts.append(p + n * lf)
            nrms.append(n)

        def hint(t):
            return nrms[min(int(round(t * (steps - 1))), steps - 1)]

        def radius(t):
            body = math.sin(math.pi * min(1.0, 0.18 + t * 0.95)) ** 0.5
            taper = (1.0 - t) ** 0.55
            return (th * taper * (0.7 + 0.3 * body) + 0.004, w * body * taper + 0.005)
        return self.floft(f'lock{i}', pts, radius, mat, ring=8, samples=4, hint=hint, cap_depth=0.6)

    def spike(self, i, az, el, back, up, length, width, mat, out=0.0):
        """Fat pointed lock end. back/up steer the tip in the local tangent frame."""
        p, n = self.surf_dir(sph(az, el))
        t_back = (sph(az, el - 8) - sph(az, el)).normalized() * -1  # toward increasing elevation (over the top)
        t_side = n.cross(t_back).normalized()
        root = p + n * (out * 0.6)
        d = (n * up + t_back * back).normalized()
        mid = root + d * length * 0.55 + n * length * 0.10
        tip = root + d * length
        return self.floft(f'spike{i}', [root - n * 0.04, root, mid, tip],
                          lambda t: (width * 0.55 * (1 - t) ** 0.8 + 0.003, width * (1 - t) ** 0.7 + 0.004),
                          mat, ring=8, samples=4, hint=lambda t: n, cap_depth=0.4)

    def hair(self, h):
        """Sculpted hair volume (one shell) + pointed spikes + forelock flips."""
        mat = self.ch.mat('hair')
        style = h.get('style', 'swept_back')
        V = h.get('V', 0.2)
        hl = h.get('hl', 48.0)
        sweep = h.get('sweep', 0.0)
        spacing = h.get('lock_deg', 24.0)
        import random
        rng = random.Random(h.get('seed', 3))
        bald = style == 'bald_sides'
        part = h.get('part', 36.0)

        def hairline(az):
            a = abs(((az + 180) % 360) - 180)
            if a <= 55:
                return lerp(hl, hl - 9, (a / 55) ** 1.5)
            if a <= 100:
                return lerp(hl - 9, 22, (a - 55) / 45)
            return lerp(22, -14, (a - 100) / 80)

        def el_az(p):
            q = self.deform(p)
            el = math.degrees(math.atan2(q.z, math.hypot(q.x, q.y)))
            az = math.degrees(math.atan2(q.x, -q.y))
            return el, az

        def keep(p):
            el, az = el_az(p)
            if el < hairline(az) - 1:
                return False
            if abs(p.x) > 0.55 and -0.35 < p.y < 0.30 and p.z < 0.30:
                return False
            if bald and el > hl + 4 and abs(az) < 125:
                return False
            return True

        def thick(p):
            el, az = el_az(p)
            hlz = hairline(az)
            a = abs(((az + 180) % 360) - 180)
            front = math.exp(-(a / 70.0) ** 2)
            rise = smoothstep(hlz, hlz + 7, el)  # steep rounded front face of the hair
            t = 0.065 + 0.04 * smoothstep(0, 60, el)
            if style in ('pompadour', 'swept_back', 'quiff'):
                wave = math.exp(-((el - (hl + 20)) / 24.0) ** 2) * math.exp(-((az - sweep * 0.4) / 62.0) ** 2)
                top = smoothstep(hl - 10, hl + 35, el) * (1 - smoothstep(110, 175, el))
                t += V * (wave * (1.0 if style == 'pompadour' else 0.7) + 0.45 * top * front + 0.18 * top)
            elif style == 'side_part':
                wave = math.exp(-((el - (hl + 14)) / 26.0) ** 2) * math.exp(-((az - (part - 70)) / 55.0) ** 2)
                t += V * (wave + 0.25 * smoothstep(hl, hl + 30, el) * (1 - smoothstep(110, 175, el)))
                t *= 1 - 0.55 * math.exp(-((az - part) / 6.0) ** 2) * smoothstep(hl, hl + 15, el)
            elif style in ('spiky_up', 'messy'):
                t += V * 0.45 * smoothstep(hl - 5, hl + 30, el) * (1 - smoothstep(120, 175, el))
            elif bald:
                t = 0.06 + (0.10 + V) * (1 - front)
            side = math.exp(-((a - 80) / 30.0) ** 2) * smoothstep(hlz, hlz + 25, el) * (1 - smoothstep(60, 110, el))
            t += (0.07 + V * 0.45) * side * (0 if bald else 1)
            # lock grooves: locks bend toward the sweep as they travel back over the head
            u = (az - sweep * smoothstep(hl, hl + 70, el) - (el - hl) * 0.15) / spacing
            m = abs(math.sin(math.pi * u)) ** 0.45
            t *= 0.80 + 0.20 * m
            return 0.004 + rise * t
        self.shell('hairvol', keep, thick, mat, segments=144, rings=110, solid=0.02)
        n = 0
        # pointed lock ends breaking the silhouette over the crown and at the back
        crown = {'pompadour': 5, 'swept_back': 5, 'quiff': 5, 'side_part': 4, 'spiky_up': 3, 'messy': 6,
                 'bald_sides': 0}.get(style, 4)
        for k in range(crown):
            az = lerp(-60, 60, k / max(1, crown - 1)) + sweep * 0.5 + rng.uniform(-6, 6)
            el = hl + 62 + rng.uniform(-6, 10)
            self.spike(n, az, el, back=1.0, up=0.30, length=0.26 + V * 0.4, width=0.14, mat=mat, out=V * 0.4)
            n += 1
        for k in range(5 if not bald else 0):
            az = lerp(130, 230, k / 4) + rng.uniform(-8, 8)
            self.spike(n, az, 20 + rng.uniform(-6, 6), back=1.0, up=0.15, length=0.22, width=0.12, mat=mat, out=0.04)
            n += 1
        if bald:
            for s in (-1, 1):
                for k in range(3):
                    self.spike(n, s * (85 + 22 * k), hl - 8, back=-0.6, up=0.6, length=0.16, width=0.11, mat=mat,
                               out=V * 0.5)
                    n += 1
            return
        if style == 'pompadour':  # the wave crest: tips standing up at the front
            for k in range(4):
                az = (k - 1.5) * 20 + sweep * 0.4
                self.spike(n, az, hl + 26, back=0.9, up=0.65, length=0.14 + V * 0.35, width=0.15, mat=mat, out=V * 1.5)
                n += 1
        if style == 'spiky_up':
            for k in range(7):
                az = lerp(-50, 50, k / 6) + rng.uniform(-5, 5)
                self.spike(n, az, hl + 8 + rng.uniform(0, 14), back=0.35, up=1.0, length=0.26 + V * 0.6, width=0.12,
                           mat=mat, out=V * 0.5)
                n += 1
        if style == 'messy':
            for k in range(9):
                az = rng.uniform(-110, 110)
                el = rng.uniform(hl, hl + 70)
                self.spike(n, az, el, back=rng.uniform(-0.5, 1.0), up=rng.uniform(0.2, 0.9), length=0.18 + V * 0.5,
                           width=0.11, mat=mat, out=V * 0.45)
                n += 1
        # forelock flips hanging out over the forehead
        nfl = h.get('forelocks', 2 if style in ('pompadour', 'side_part', 'messy') else 1)
        fl_az = (part - 70) if style == 'side_part' else sweep * 0.5
        for k in range(nfl):
            az0 = fl_az + (k - (nfl - 1) / 2) * 24
            p0, n0 = self.surf_dir(sph(az0, hl + 18))
            p1, n1 = self.surf_dir(sph(az0 + 10 + 8 * k, hl - 2))
            p2, n2 = self.surf_dir(sph(az0 + 26 + 10 * k, hl - 16 - 6 * k))
            lift = V * 0.9 + 0.06
            pts = [p0 + n0 * (lift * 0.8), p1 + n1 * (lift * 0.55), p2 + n2 * 0.10]
            self.floft(f'forelock{k}', pts, lambda t: (0.06 * (1 - t) ** 0.7 + 0.004, 0.13 * (1 - t) ** 0.6 + 0.005),
                       mat, ring=8, samples=6, hint=lambda t, _n=n1: _n, cap_depth=0.5)

    # ------------------------------------------------------------------ eyewear & headgear
    def glasses(self, g):
        ch = self.ch
        kind = g if isinstance(g, str) else g.get('kind', 'aviator')
        frame = ch.mat('frame')
        lens = ch.mat('lens')
        ex, ez = self.f.get('eye_x', 0.29), self.f.get('eye_z', 0.11)
        rw, rh = (0.20, 0.155) if kind == 'aviator' else (0.21, 0.12)
        for side in (-1, 1):
            c = self.surf(side * ex, ez - 0.01, 0.12)
            drop = 0.045 if kind == 'aviator' else 0.0
            pts = []
            for k in range(13):
                a = 2 * math.pi * k / 12
                x = math.cos(a) * rw
                z = math.sin(a) * rh - (drop if math.sin(a) < 0 and x * side > -0.05 else 0)
                pts.append(Vector((c.x + x, c.y + 0.012 * (x * side) / rw, c.z + z)))
            self.floft(f'frame{side}', pts, 0.015, frame, ring=6, samples=3, caps=(False, False))
            self.fblob(f'lens{side}', c + Vector((0, 0.005, -drop * 0.35)),
                       (rw * 0.98, 0.02, rh * 0.98 + drop * 0.4), lens)
            p_ear, _ = self.surf_dir((side, 0.1, 0.12), 0.04)
            self.floft(f'temple{side}', [c + Vector((side * rw, 0, 0.03)), p_ear], 0.013, frame, ring=6, samples=3)
        pts = [self.surf(-ex + rw * 0.9, ez + 0.04, 0.13), self.surf(0, ez + 0.08, 0.14),
               self.surf(ex - rw * 0.9, ez + 0.04, 0.13)]
        self.floft('bridge', pts, 0.013, frame, ring=6, samples=4)

    def strip(self, name, pts, nrms, width, thick, mat):
        def hint(t):
            return nrms[min(int(round(t * (len(nrms) - 1))), len(nrms) - 1)]
        return self.floft(name, pts, (thick, width), mat, ring=8, samples=3, hint=hint, caps=(True, True),
                          cap_depth=0.3)

    def hg_bandage(self):
        """Jawan-style wraps: smooth strips following plane/skull intersections, broken around
        the character's left eye and the grin so the face stays readable."""
        mat = self.ch.mat('bandage')
        ex, ez = self.f.get('eye_x', 0.29), self.f.get('eye_z', 0.11)
        mz = self.f.get('mouth_z', -0.42)
        bands = [((0.30, 0.25, 1.0), 0.55, 0.11), ((-0.45, -0.55, 1.0), 0.22, 0.10), ((0.55, -0.45, 1.0), -0.12, 0.085),
                 ((-0.15, 0.1, 1.0), 0.80, 0.09), ((0.0, -0.35, 1.0), -0.62, 0.075)]
        k = 0
        for bi, (n, d, w) in enumerate(bands):
            n = Vector(n).normalized()
            u = n.orthogonal().normalized()
            v = n.cross(u)
            r = math.sqrt(max(0.0, 1 - d * d))
            segs, cur, curn = [], [], []
            for i in range(97):
                th = 2 * math.pi * i / 96
                q = n * d + (u * math.cos(th) + v * math.sin(th)) * r
                p, nrm = self.surf_dir(q)
                hole = (p.y < -0.25 and abs(p.x - ex) < 0.22 and abs(p.z - ez) < 0.13) or \
                       (p.y < -0.3 and abs(p.x) < 0.32 and mz - 0.17 < p.z < mz + 0.07)
                if hole:
                    if len(cur) > 3:
                        segs.append((cur, curn))
                    cur, curn = [], []
                    continue
                cur.append(p + nrm * (0.022 + 0.008 * bi))
                curn.append(nrm)
            if len(cur) > 3:
                segs.append((cur, curn))
            for pts, nrms in segs:
                self.strip(f'band{k}', pts, nrms, w * 1.25, 0.018, mat)
                k += 1

    def hg_mask(self):
        ex, ez = self.f.get('eye_x', 0.29), self.f.get('eye_z', 0.11)

        def keep(p):
            q = self.deform(p)
            if p.y > 0.40:
                return False
            if not (-0.04 < q.z < 0.42):
                return False
            for side in (-1, 1):
                if ((q.x - side * ex) / 0.165) ** 2 + ((q.z - ez) / 0.095) ** 2 < 1.0 and p.y < 0:
                    return False
            if q.y < -0.5 and abs(q.x) < 0.07 and q.z < 0.05:
                return False
            return True
        self.shell('mask', keep, lambda p: 0.035 + 0.025 * smoothstep(0.1, -0.9, p.y), self.ch.mat('mask'),
                   segments=96, rings=64)

    def hg_headband(self):
        ch = self.ch
        self.shell('headband', lambda p: abs(p.z - 0.40 + 0.25 * p.y) < 0.09, lambda p: 0.06, ch.mat('headband'))
        p, n = self.surf_dir((0.15, 1.0, 0.25), 0.05)
        self.floft('hb_tail', [p, p + Vector((0.25, 0.55, -0.2)), p + Vector((0.45, 1.05, -0.05))],
                   lambda t: (0.10 * (1 - 0.5 * t), 0.02), ch.mat('headband'), ring=8, samples=4, hint=(0, 0, 1))
