"""Bala Simha, textured: the supplied Meshy textured FBX (base colour, normal, roughness and
metallic maps, 309k triangles with its own UVs) rendered directly.

Fallback for low-poly textured files: the sculpt FBX (~1M triangles, no texture) lines up with
them to within millimetres, so every sculpt triangle can borrow UVs from the nearest low-poly one:
all three corners are projected onto that one triangle's plane (so a sculpt face never
straddles a UV seam) and their barycentric coordinates give the UVs.

Per-pixel material information for the comic shaders comes from render passes:
  * Diffuse Color pass = the texture's base colour (what the painter sees)
  * AOV 'metal'  = metallic map (gold watch, bracelet, ring, glasses frames)
  * AOV 'gloss'  = 1 - roughness (glass lenses, glossy hair)
  * AOV 'region' = coarse body region id (from fbxbala.classify): head / shirt / dhoti / feet ...
The cover shader decides skin vs. hair/beard inside the head region from the base colour.
"""
import os

import numpy as np

from . import fbxbala

HERE = os.path.dirname(__file__)
SCULPT = os.path.join(HERE, '../../source/bala_desert_king.fbx')
TEXTURED = os.path.join(HERE, '../../source/bala_desert_king_textured.fbx')
REGION = dict(head=1, shirt=2, dhoti=3, skin=4, sandal=5)


def _import(path):
    import bpy
    from mathutils import Matrix
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    ob = [o for o in new if o.type == 'MESH'][0]
    for o in new:
        if o is not ob:
            bpy.data.objects.remove(o)
    me = ob.data
    me.transform(ob.matrix_world)
    ob.matrix_world = Matrix.Identity(4)
    me.transform(Matrix.Translation((0, 0, fbxbala.LIFT)))
    me.transform(Matrix.Diagonal((fbxbala.SCALE, fbxbala.SCALE, fbxbala.SCALE, 1)))
    return ob


def _arrays(me):
    V = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get('co', V)
    loops = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get('vertex_index', loops)
    return V.reshape(-1, 3), loops


def transfer_uvs(hi, lo):
    """Give the sculpt mesh `hi` UVs from the textured low-poly `lo` (both triangulated)."""
    from mathutils.bvhtree import BVHTree
    Vh, Lh = _arrays(hi.data)
    Vl, Ll = _arrays(lo.data)
    Th = Lh.reshape(-1, 3)
    Tl = Ll.reshape(-1, 3)
    uvl = np.empty(len(lo.data.loops) * 2, np.float32)
    lo.data.uv_layers.active.data.foreach_get('uv', uvl)
    uvl = uvl.reshape(-1, 3, 2)                      # per low-poly triangle, per corner
    bvh = BVHTree.FromPolygons([tuple(v) for v in Vl], [tuple(t) for t in Tl])
    C = Vh[Th].mean(1)
    idx = np.empty(len(Th), np.int64)
    for i, c in enumerate(C):
        loc, nrm, k, dist = bvh.find_nearest(c)
        idx[i] = k if k is not None else 0
    A, B, Cc = Vl[Tl[idx, 0]], Vl[Tl[idx, 1]], Vl[Tl[idx, 2]]
    e0, e1 = B - A, Cc - A
    d00 = (e0 * e0).sum(1)
    d01 = (e0 * e1).sum(1)
    d11 = (e1 * e1).sum(1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-14)
    uv = np.empty((len(Th), 3, 2), np.float32)
    for c in range(3):
        p = Vh[Th[:, c]] - A
        d20 = (p * e0).sum(1)
        d21 = (p * e1).sum(1)
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        u = 1 - v - w
        bary = np.stack([u, v, w], 1)
        bary = np.clip(bary, -0.15, 1.15)            # stay inside the island's bleed padding
        bary /= bary.sum(1, keepdims=True)
        uv[:, c] = (bary[:, :, None] * uvl[idx]).sum(1)
    layer = hi.data.uv_layers.new(name='UVMap')
    layer.data.foreach_set('uv', uv.ravel())
    return uv


def tex_material(ml, name, images, region, **props):
    """Material: base colour from the texture, plus AOVs metal / gloss / region."""
    import bpy
    if name in ml.mats:
        return ml.mats[name]
    m = ml.get(name, (0.8, 0.8, 0.8), char='bala', **props)
    nt = m.node_tree
    bsdf = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
    tc = nt.nodes.new('ShaderNodeTexImage')
    tc.image = images['color']
    tc.interpolation = 'Cubic'
    nt.links.new(tc.outputs['Color'], bsdf.inputs['Base Color'])
    for key, aov, invert in (('metallic', 'metal', False), ('roughness', 'gloss', True)):
        t = nt.nodes.new('ShaderNodeTexImage')
        t.image = images[key]
        t.image.colorspace_settings.name = 'Non-Color'
        out = nt.nodes.new('ShaderNodeOutputAOV')
        out.aov_name = aov
        if invert:
            inv = nt.nodes.new('ShaderNodeMath')
            inv.operation = 'SUBTRACT'
            inv.inputs[0].default_value = 1.0
            nt.links.new(t.outputs['Color'], inv.inputs[1])
            nt.links.new(inv.outputs['Value'], out.inputs['Value'])
        else:
            nt.links.new(t.outputs['Color'], out.inputs['Value'])
    ro = nt.nodes.new('ShaderNodeOutputAOV')
    ro.aov_name = 'region'
    ro.inputs['Value'].default_value = float(region)
    geo_n = nt.nodes.new('ShaderNodeNewGeometry')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(geo_n.outputs['Position'], sep.inputs[0])
    ho = nt.nodes.new('ShaderNodeOutputAOV')
    ho.aov_name = 'height'
    nt.links.new(sep.outputs['Z'], ho.inputs['Value'])
    return m


def ensure_aovs():
    import bpy
    vl = bpy.context.scene.view_layers[0]
    have = {a.name for a in vl.aovs}
    for n in ('metal', 'gloss', 'region', 'height'):
        if n not in have:
            a = vl.aovs.add()
            a.name = n
            a.type = 'VALUE'


def load(ml, pass_index=1, tilak=False, keep_lowpoly=False):
    """Import both FBX files, transfer the texture onto the sculpt and assign region materials.
    Returns (objects, anchors, labels)."""
    import bpy
    from mathutils import Vector
    ensure_aovs()
    lo = _import(TEXTURED)
    mat0 = lo.data.materials[0]
    imgs = {}
    for n in mat0.node_tree.nodes:
        if n.type == 'TEX_IMAGE' and n.image is not None:
            nm = n.image.name.lower()
            key = 'normal' if 'normal' in nm else 'roughness' if 'rough' in nm else 'metallic' if 'metal' in nm else 'color'
            imgs[key] = n.image
    if len(lo.data.vertices) > 50000:      # dense textured mesh: use it as is
        hi = lo
    else:                                   # low-poly: move its texture onto the sculpt
        hi = _import(SCULPT)
        transfer_uvs(hi, lo)
        if not keep_lowpoly:
            bpy.data.objects.remove(lo)
    hi.name = 'bala_fbx'
    me = hi.data
    V, loops = _arrays(me)
    lab = fbxbala.classify(V)
    R = fbxbala.R
    reg = np.full(len(V), REGION['shirt'], np.int32)
    head_labels = [R['hair'], R['beard'], R['mustache'], R['lips'], R['lens'], R['gold']]
    reg[np.isin(lab, head_labels)] = REGION['head']
    reg[(lab == R['skin']) & (V[:, 2] > 1.40)] = REGION['head']
    reg[(lab == R['gold']) & (V[:, 2] < 1.40)] = REGION['skin']
    reg[(lab == R['skin']) & (V[:, 2] <= 1.40)] = REGION['skin']
    reg[lab == R['dhoti']] = REGION['dhoti']
    reg[lab == R['sandal']] = REGION['sandal']
    mats = [tex_material(ml, 'tx.head', imgs, REGION['head'], skin=True, headmix=True),
            tex_material(ml, 'tx.shirt', imgs, REGION['shirt'], cloth=True, spec=0.25),
            tex_material(ml, 'tx.dhoti', imgs, REGION['dhoti'], cloth=True),
            tex_material(ml, 'tx.skin', imgs, REGION['skin'], skin=True),
            tex_material(ml, 'tx.sandal', imgs, REGION['sandal'], spec=0.3)]
    me.materials.clear()
    for m in mats:
        me.materials.append(m)
    T = loops.reshape(-1, 3)
    order = {REGION['head']: 0, REGION['shirt']: 1, REGION['dhoti']: 2, REGION['skin']: 3, REGION['sandal']: 4}
    rv = reg[T]
    # majority region per face
    fr = np.where(rv[:, 0] == rv[:, 1], rv[:, 0], np.where(rv[:, 1] == rv[:, 2], rv[:, 1], rv[:, 0]))
    me.polygons.foreach_set('material_index', np.vectorize(order.get)(fr).astype(np.int32))
    me.polygons.foreach_set('use_smooth', np.ones(len(T), bool))
    me.update()
    hi.pass_index = pass_index
    anchors = {}
    for k in ('beard', 'hair', 'mustache'):
        m = lab == R[k]
        if m.any():
            anchors[k] = Vector(V[m].mean(0).tolist())
    anchors['lens'] = Vector((0.087, -0.01, 1.617))
    anchors['tilak'] = Vector((fbxbala.HC[0], 0.0, 1.661))
    anchors['watch'] = Vector((0.266, 0.0, 0.986))
    anchors['bracelet'] = Vector((-0.24, 0.0, 1.07))
    anchors['pocket'] = Vector((0.12, -0.05, 1.30))
    anchors['dhoti'] = Vector((0.05, -0.05, 0.55))
    anchors['shirt'] = Vector((0.0, -0.08, 1.20))
    sm = lab == R['sandal']
    anchors['sandal'] = Vector(V[sm].mean(0).tolist()) if sm.any() else Vector((0, 0, 0.02))
    anchors['head'] = Vector(fbxbala.HC.tolist())
    return [hi], anchors, lab
