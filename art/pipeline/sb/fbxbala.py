"""Bala Simha from the supplied FBX sculpt (Meshy "Desert King": walking pose, geometry only).

The FBX has no UVs, colours or textures, so this module paints it: every vertex is classified
into a region (skin, hair, beard, mustache, lips, lens, gold, shirt, dhoti, sandal) from its
position in body space, measured on orthographic views of the sculpt. Each region becomes its
own material slot (so the comic shaders ink region borders and shade skin / hair / cloth / metal
/ glass correctly) and vertices get colours (salt-and-pepper hair and beard). A red tilak disc is
ray-cast onto the forehead.

Body space after load(): metres, feet at z = 0, facing -Y, +X = character's left; 1.78 m to the
top of the hair.
"""
import math
import os

import numpy as np

from .bala import vcol_material, _vnoise, _mix
from .sdf import smoothstep

FBX = os.path.join(os.path.dirname(__file__), '../../source/bala_desert_king.fbx')
SCALE = 0.935
LIFT = 0.952
HC = np.array([0.045, 0.085, 1.625])          # head centre (eye level)
REGIONS = ['skin', 'lips', 'hair', 'beard', 'mustache', 'lens', 'gold', 'shirt', 'dhoti', 'sandal']
R = {k: i for i, k in enumerate(REGIONS)}
COL = dict(skin=(0.80, 0.56, 0.42), lips=(0.62, 0.34, 0.32), shirt=(0.12, 0.12, 0.15), dhoti=(0.95, 0.91, 0.80),
           sandal=(0.42, 0.24, 0.13), gold=(0.95, 0.76, 0.30), lens=(0.16, 0.19, 0.14))

HL_AZ = [0, 40, 70, 95, 115, 140, 180]
HL_EL = [31, 28, 20, 4, 15, -22, -46]
BEARD_AZ = [0, 25, 50, 75, 100, 125]
BEARD_EL = [-38, -33, -22, -10, 2, 2]


def classify(V):
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    lab = np.full(len(V), R['shirt'], np.int32)
    xf = x - HC[0]
    # ---- lower body
    hem = 0.94 + 0.015 * np.sin(x * 9.0)
    lab[z < hem] = R['dhoti']
    lab[z < 0.085] = R['skin']                     # feet
    lab[z < 0.026] = R['sandal']
    # ---- forearms / hands below the rolled cuffs (outside the torso silhouette)
    arm = ((x > 0.238) | (x < -0.178)) & (z < 1.165) & (z > 0.84)
    lab[arm] = R['skin']
    hand_r = (x > -0.29) & (x < -0.09) & (z > 0.90) & (z < 1.07) & (y < -0.035)   # right hand gathering the dhoti
    lab[hand_r & (np.abs(x + 0.19) < 0.10)] = R['skin']
    for (x0, x1, z0, z1, xo) in ((-0.278, -0.200, 1.035, 1.105, -0.255), (0.240, 0.300, 0.965, 1.008, 0.265)):
        box = (x > x0) & (x < x1) & (z > z0) & (z < z1)                            # bracelet / watch, on the wrist only
        core = box & (np.sign(xo) * (x - xo) > 0)
        if core.any():
            yw = np.median(y[core])
            lab[box & (np.abs(y - yw) < 0.032)] = R['gold']
    # ---- neck + open collar (skin inside the V)
    rn = np.hypot(x - 0.035, y - 0.105)
    neck = (z > 1.43) & (z < 1.53) & (rn < 0.072)
    lab[neck] = R['skin']
    vw = 0.012 + 0.040 * smoothstep(1.39, 1.47, z)
    vee = (z > 1.38) & (z < 1.48) & (np.abs(x - 0.040) < vw) & (y < 0.06)
    lab[vee] = R['skin']
    # ---- head
    v = V - HC
    az = np.degrees(np.arctan2(v[:, 0], -v[:, 1]))
    el = np.degrees(np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1])))
    head = ((z > 1.53) & (((v[:, 0] / 0.135) ** 2 + (v[:, 1] / 0.15) ** 2) < 1.0)) | ((z > 1.47) & (rn < 0.085))
    head |= (z > 1.488) & (np.abs(xf) < 0.065) & (y < 0.05) & (y > -0.06)             # the jutting chin
    lab[head] = R['skin']
    a = np.abs(az)
    hair = head & (el > np.interp(a, HL_AZ, HL_EL))
    lab[hair] = R['hair']
    beard = head & (el < np.interp(a, BEARD_AZ, BEARD_EL)) & (a < 128) & ~hair
    lab[beard] = R['beard']
    lab[neck & (z > 1.47) & (y < 0.07) & (el < -40)] = R['beard']                 # under the chin
    ztop = 1.576 - 5.0 * xf ** 2
    zbot = 1.548 - 9.5 * xf ** 2
    mus = head & (y < 0.04) & (np.abs(xf) < 0.050) & (z > zbot) & (z < ztop)
    lab[mus] = R['mustache']
    lips = head & (y < 0.005) & (np.abs(xf) < 0.024) & (z > 1.534) & (z < 1.547)
    lab[lips] = R['lips']
    # ---- aviator lenses: front-most layer inside the lens outlines (z-buffer from the front)
    band = (z > 1.584) & (z < 1.650) & (x > -0.050) & (x < 0.140)
    cell = 0.002
    ix = np.floor((x - (-0.06)) / cell).astype(int)
    iz = np.floor((z - 1.58) / cell).astype(int)
    ok = band & (ix >= 0) & (ix < 110) & (iz >= 0) & (iz < 40)
    front = np.full((110, 40), 9.0)
    np.minimum.at(front, (ix[ok], iz[ok]), y[ok])
    inl = np.zeros(len(V), bool)
    for cx, rx in ((-0.006, 0.038), (0.087, 0.042)):
        q = ((x - cx) / rx) ** 2 + ((z - 1.617) / 0.030) ** 2
        q = np.where(z < 1.617, ((x - cx - 0.006 * np.sign(cx - 0.045)) / rx) ** 2 + ((z - 1.617) / 0.032) ** 2, q)
        inl |= q < 1.0
    nose = (np.abs(xf) < 0.017) & (z < 1.622)
    fy = np.where(ok, front[np.clip(ix, 0, 109), np.clip(iz, 0, 39)], 9.0)
    lens = ok & inl & ~nose & (y < fy + 0.0035)
    lab[lens] = R['lens']
    rim = ok & ~nose & (y < fy + 0.0035) & ~inl
    near = np.zeros(len(V), bool)
    for cx, rx in ((-0.006, 0.038), (0.087, 0.042)):
        q = ((x - cx) / rx) ** 2 + ((z - 1.617) / 0.031) ** 2
        near |= (q > 0.82) & (q < 1.22)
    lab[rim & near & (z > 1.60)] = R['gold']
    return lab


def find_glasses(ob, V):
    """Thin shells in front of the face: a ray cast inward hits the shell's own back face within
    a few mm and then the face behind it within a few cm."""
    from mathutils import Vector
    eye = np.array([HC[0], HC[1], 1.617])
    v = V - HC
    az = np.abs(np.degrees(np.arctan2(v[:, 0], -v[:, 1])))
    cand = np.where((V[:, 2] > 1.575) & (V[:, 2] < 1.660) & (az < 80))[0]
    out = np.zeros(len(V), bool)
    for i in cand:
        p = V[i]
        d = eye - p
        d = d / np.linalg.norm(d)
        o = Vector((p + d * 0.0004).tolist())
        dv = Vector(d.tolist())
        hit, loc, n, idx = ob.ray_cast(o, dv, distance=0.007)
        if not hit:
            continue
        hit2, loc2, n2, idx2 = ob.ray_cast(loc + dv * 0.0004, dv, distance=0.035)
        if hit2:
            out[i] = True
    return out


def colours(V, lab):
    C = np.zeros((len(V), 3), np.float32)
    for k, c in COL.items():
        C[lab == R[k]] = c
    m = lab == R['hair']
    if m.any():
        v = V[m] - HC
        az = np.degrees(np.arctan2(v[:, 0], -v[:, 1]))
        el = np.degrees(np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1])))
        g = 0.10 + 0.36 * smoothstep(55, 100, np.abs(az)) * (1 - smoothstep(25, 60, el))
        g = g + 0.40 * np.clip(_vnoise(V[m] * np.array([2.5, 0.6, 0.9]), 0.006, 3) - 0.45, 0, 1)
        C[m] = _mix((0.12, 0.10, 0.11), (0.64, 0.63, 0.66), g)
    m = lab == R['beard']
    if m.any():
        z = V[m, 2]
        g = 0.36 + 0.34 * smoothstep(1.555, 1.505, z) + 0.14 * _vnoise(V[m] * np.array([1, 1, 0.35]), 0.0018, 7)
        C[m] = _mix((0.17, 0.155, 0.165), (0.86, 0.86, 0.87), g)
    m = lab == R['mustache']
    if m.any():
        C[m] = _mix((0.10, 0.085, 0.09), (0.55, 0.55, 0.57), 0.12 + 0.12 * _vnoise(V[m], 0.0018, 11))
    m = lab == R['skin']
    if m.any():
        v = V[m]
        flush = np.exp(-(((np.abs(v[:, 0] - HC[0]) - 0.045) / 0.025) ** 2 + ((v[:, 2] - 1.595) / 0.022) ** 2))
        C[m] = C[m] + np.array([0.05, -0.03, -0.03]) * flush[:, None] * (v[:, 1] < 0.06)[:, None]
    return np.clip(C, 0, 1)


def materials(ml):
    return [
        vcol_material(ml, 'fb.skin', COL['skin'], skin=True, char='bala'),
        vcol_material(ml, 'fb.lips', COL['lips'], skin=True, char='bala'),
        vcol_material(ml, 'fb.hair', (0.12, 0.10, 0.11), hair=True, fur=True, spec=0.7, char='bala'),
        vcol_material(ml, 'fb.beard', (0.5, 0.5, 0.5), fur=True, beard=True, char='bala'),
        vcol_material(ml, 'fb.mustache', (0.1, 0.09, 0.09), fur=True, char='bala'),
        vcol_material(ml, 'fb.lens', COL['lens'], glass=True, spec=1.0, char='bala'),
        vcol_material(ml, 'fb.gold', COL['gold'], metal=True, spec=1.0, char='bala'),
        vcol_material(ml, 'fb.shirt', COL['shirt'], cloth=True, spec=0.25, char='bala'),
        vcol_material(ml, 'fb.dhoti', COL['dhoti'], cloth=True, char='bala'),
        vcol_material(ml, 'fb.sandal', COL['sandal'], char='bala', spec=0.3),
    ]


def load(ml, path=FBX, pass_index=1, tilak=True):
    """Import, normalise, segment and paint. Returns (objects, anchors_world_dict)."""
    import bpy
    from mathutils import Matrix, Vector
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    ob = [o for o in new if o.type == 'MESH'][0]
    for o in new:
        if o is not ob:
            bpy.data.objects.remove(o)
    me = ob.data
    ob.matrix_world = Matrix.Identity(4)
    me.transform(Matrix.Translation((0, 0, LIFT)))
    me.transform(Matrix.Diagonal((SCALE, SCALE, SCALE, 1)))
    ob.name = 'bala_fbx'
    n = len(me.vertices)
    V = np.empty(n * 3, np.float32)
    me.vertices.foreach_get('co', V)
    V = V.reshape(-1, 3)
    lab = classify(V)
    C = colours(V, lab)
    me.materials.clear()
    for m in materials(ml):
        me.materials.append(m)
    loops = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get('vertex_index', loops)
    starts = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get('loop_start', starts)
    # a face takes the most "specific" label among its corners (lens/gold/lips/mustache win over skin)
    prio = np.array([0, 6, 2, 3, 5, 9, 8, 1, 1, 4])[lab]
    l3 = loops[starts[:, None] + np.arange(3)[None, :]]
    best = np.argmax(prio[l3], axis=1)
    flab = lab[l3[np.arange(len(starts)), best]]
    me.polygons.foreach_set('material_index', flab.astype(np.int32))
    me.polygons.foreach_set('use_smooth', np.ones(len(starts), bool))
    glasses = find_glasses(ob, V)
    if glasses.sum() > 500:
        lab[lab == R['lens']] = R['skin']
        lab[(lab == R['gold']) & (V[:, 2] > 1.5)] = R['skin']
        g = glasses
        xg, zg = V[:, 0], V[:, 2]
        vv = V - HC
        azg = np.abs(np.degrees(np.arctan2(vv[:, 0], -vv[:, 1])))
        lens = np.zeros(len(V), bool)
        for side in (xg < HC[0], xg >= HC[0]):
            m = g & side & (azg < 55)
            if m.sum() < 50:
                continue
            x0, x1 = np.percentile(xg[m], [1, 99])
            z0, z1 = np.percentile(zg[m], [1, 99])
            q = ((xg - (x0 + x1) / 2) / ((x1 - x0) / 2 - 0.0035)) ** 2 + ((zg - (z0 + z1) / 2) / ((z1 - z0) / 2 - 0.0035)) ** 2
            lens |= m & (q < 1.0)
        lab[g & ~lens] = R['gold']
        lab[lens] = R['lens']
        C = colours(V, lab)
        prio = np.array([0, 6, 2, 3, 5, 9, 8, 1, 1, 4])[lab]
        best = np.argmax(prio[l3], axis=1)
        flab = lab[l3[np.arange(len(starts)), best]]
        me.polygons.foreach_set('material_index', flab.astype(np.int32))
    print('glasses vertices:', int(glasses.sum()))
    lin = np.where(C <= 0.04045, C / 12.92, ((C + 0.055) / 1.055) ** 2.4)
    ca = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
    ca.data.foreach_set('color', np.concatenate([lin, np.ones((n, 1))], 1).astype(np.float32).ravel())
    me.update()
    ob.pass_index = pass_index
    objs = [ob]
    anchors = {}
    for k in ('lens', 'beard', 'hair', 'mustache'):
        m = lab == R[k]
        if m.any():
            anchors[k] = Vector(V[m].mean(0).tolist())
    anchors['watch'] = Vector((0.266, float(np.median(V[(lab == R['gold']) & (V[:, 0] > 0.2), 1])) if ((lab == R['gold']) & (V[:, 0] > 0.2)).any() else 0.0, 0.986))
    anchors['bracelet'] = Vector((-0.24, float(np.median(V[(lab == R['gold']) & (V[:, 0] < -0.15), 1])) if ((lab == R['gold']) & (V[:, 0] < -0.15)).any() else 0.0, 1.07))
    anchors['pocket'] = Vector((0.12, -0.05, 1.30))
    anchors['dhoti'] = Vector((0.05, -0.05, 0.55))
    anchors['sandal'] = Vector(V[lab == R['sandal']].mean(0).tolist()) if (lab == R['sandal']).any() else Vector((0, 0, 0.02))
    anchors['shirt'] = Vector((0.0, -0.08, 1.20))
    if tilak:
        org = Vector((HC[0], -0.5, 1.661))
        hit, loc, nrm, idx = ob.ray_cast(org, Vector((0, 1, 0)))
        if hit:
            red = ml.get('fb.tilak', (0.84, 0.08, 0.12), flat=True, char='bala')
            from . import geo
            T = geo.look_matrix(-nrm, Vector((0, 0, 1)), loc + nrm * 0.0012)
            t = geo.blob('bala_tilak', lambda p: p, segments=16, rings=8, mats=[red], subsurf=1, pass_index=pass_index,
                         matrix=T @ Matrix.Diagonal((0.0052, 0.0015, 0.0066, 1)))
            objs.append(t)
            anchors['tilak'] = loc
    anchors['head'] = Vector(HC.tolist())
    return objs, anchors, lab
