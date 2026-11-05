"""Blender scene setup + render-to-passes for the comic shader.

Lighting strategy: a single hard sun ("key") is the only light in the scene. Its Diffuse
Direct pass gives N.L x cast-shadow with no sampling noise, which the comic shader turns
into cel bands / halftone / hatching. Fill, rim and bounce are painted in post from the
Normal pass, so 16 samples are enough for clean AA on every pass.
"""
import json
import math
import os
import time

import bpy
from mathutils import Vector, Euler

from .geo import MatLib

PASSES = ['use_pass_normal', 'use_pass_z', 'use_pass_diffuse_direct', 'use_pass_diffuse_color',
          'use_pass_object_index', 'use_pass_material_index', 'use_pass_ambient_occlusion',
          'use_pass_emit']


def reset(width=1920, height=1080, samples=16, ao_distance=0.25):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.cycles.max_bounces = 2
    sc.cycles.diffuse_bounces = 1
    sc.cycles.glossy_bounces = 0
    sc.cycles.transmission_bounces = 0
    sc.cycles.transparent_max_bounces = 2
    sc.render.resolution_x = width
    sc.render.resolution_y = height
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.threads_mode = 'AUTO'
    try:
        sc.cycles.use_auto_tile = False
    except AttributeError:
        pass
    vl = sc.view_layers[0]
    for p in PASSES:
        setattr(vl, p, True)
    world = bpy.data.worlds.new('World')
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Color'].default_value = (0, 0, 0, 1)
        bg.inputs['Strength'].default_value = 0.0
    world.light_settings.distance = ao_distance
    sc.view_settings.view_transform = 'Standard'
    ims = sc.render.image_settings
    ims.media_type = 'MULTI_LAYER_IMAGE'
    ims.file_format = 'OPEN_EXR_MULTILAYER'
    ims.color_depth = '32'
    return sc


def add_sun(direction=(-0.55, 0.45, -0.7), energy=3.0, angle_deg=0.4):
    """direction = where the light travels (from light toward scene)."""
    sc = bpy.context.scene
    L = bpy.data.lights.new('key', 'SUN')
    L.energy = energy
    L.angle = math.radians(angle_deg)
    ob = bpy.data.objects.new('key', L)
    sc.collection.objects.link(ob)
    d = Vector(direction).normalized()
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    return ob, d


def add_camera(location, target, lens=50.0, ortho_scale=None, shift=(0, 0)):
    sc = bpy.context.scene
    cam = bpy.data.cameras.new('cam')
    cam.lens = lens
    cam.clip_start = 0.05
    cam.clip_end = 2000
    if ortho_scale:
        cam.type = 'ORTHO'
        cam.ortho_scale = ortho_scale
    cam.shift_x, cam.shift_y = shift
    ob = bpy.data.objects.new('cam', cam)
    sc.collection.objects.link(ob)
    ob.location = Vector(location)
    d = (Vector(target) - Vector(location)).normalized()
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    sc.camera = ob
    return ob


def fx_only_camera(matlib):
    """Objects using emissive materials are seen by the camera only (no light / shadow)."""
    emissive = {m.name for m in matlib.mats.values() if matlib.props[m.pass_index].get('emit', 0) > 0}
    for ob in bpy.context.scene.objects:
        if ob.type == 'MESH' and any(ms.material and ms.material.name in emissive for ms in ob.material_slots):
            ob.visible_diffuse = False
            ob.visible_glossy = False
            ob.visible_shadow = False
            ob.visible_transmission = False
            ob.visible_volume_scatter = False


def project(points):
    """World points -> pixel coords (x right, y down) for the active camera."""
    from bpy_extras.object_utils import world_to_camera_view
    sc = bpy.context.scene
    bpy.context.view_layer.update()
    W, H = sc.render.resolution_x, sc.render.resolution_y
    out = {}
    for k, p in points.items():
        c = world_to_camera_view(sc, sc.camera, Vector(p))
        out[k] = [c.x * W, (1 - c.y) * H, c.z]
    return out


def render(path, matlib: MatLib, meta=None):
    sc = bpy.context.scene
    fx_only_camera(matlib)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sc.render.filepath = path
    t = time.time()
    bpy.ops.render.render(write_still=True)
    cam = sc.camera
    info = dict(materials=matlib.table(), meta=meta or {})
    if cam:
        mw = cam.matrix_world
        info['camera'] = dict(
            location=list(mw.translation),
            forward=list((mw.to_3x3() @ Vector((0, 0, -1))).normalized()),
            up=list((mw.to_3x3() @ Vector((0, 1, 0))).normalized()),
            right=list((mw.to_3x3() @ Vector((1, 0, 0))).normalized()),
            lens=cam.data.lens, type=cam.data.type)
    key = bpy.data.objects.get('key')
    if key:
        info['key_dir'] = list((key.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized())
        info['key_energy'] = key.data.energy
    with open(os.path.splitext(path)[0] + '.json', 'w') as f:
        json.dump(info, f, indent=1)
    print(f'[render] {path} {time.time() - t:.1f}s')
    return path
