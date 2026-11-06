"""Head-and-shoulders portrait per fighter (select screen, HUD, card inset)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import bpy  # noqa: F401  (must precede mathutils)
from mathutils import Vector
from sb import scene, roster
from sb.geo import MatLib
from sb.character import Character
ids = sys.argv[1].split(',') if len(sys.argv) > 1 and sys.argv[1] != 'all' else [c['id'] for c in roster.ROSTER]
S = int(sys.argv[2]) if len(sys.argv) > 2 else 900
build = os.path.join(os.path.dirname(__file__), '../build/portraits')
for cid in ids:
    spec = roster.BY_ID[cid]
    scene.reset(S, S, samples=16)
    ml = MatLib()
    pose = dict(spec['pose'])
    pose['props'] = []
    # portrait pose: keep spine / head / expression, relax the arms so hands never cover the face
    pose['armL'] = dict(upper=(0.30, 0.05, -1), fore=(0.15, -0.35, -1), hand='fist')
    pose['armR'] = dict(upper=(-0.30, 0.05, -1), fore=(-0.15, -0.35, -1), hand='fist')
    pose['twist'] = 0
    pose['spine'] = (0, 0.0, 1)
    ch = Character(spec, pose, ml, 1)
    ch.build()
    hc = ch.J['head']
    f = (ch.head_matrix.to_3x3() @ Vector((0, -1, 0))).normalized()
    f.z = 0
    f.normalize()
    side = Vector((-f.y, f.x, 0))
    d = (f * math.cos(math.radians(22)) + side * math.sin(math.radians(22))).normalized()
    hs = 0.071 * spec['H'] * spec.get('head_scale', 1.0)
    cam_t = hc + Vector((0, 0, -hs * 1.3))
    scene.add_sun((0.42, 0.62, -0.66), 3.0)
    scene.add_camera(cam_t + d * hs * 10.5 + Vector((0, 0, hs * 0.8)), cam_t, lens=60)
    scene.render(f'{build}/{cid}.exr', ml, meta=dict(id=cid))
