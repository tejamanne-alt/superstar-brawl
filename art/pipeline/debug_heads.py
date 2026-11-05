"""Render a head close-up per fighter and montage them (geometry check)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sb import scene, roster
from sb.geo import MatLib
from sb.character import Character
from mathutils import Vector
ids = sys.argv[1].split(',') if len(sys.argv) > 1 and sys.argv[1] != 'all' else [c['id'] for c in roster.ROSTER]
outdir = os.path.join(os.path.dirname(__file__), '../build/heads')
for cid in ids:
    spec = roster.BY_ID[cid]
    scene.reset(512, 512, samples=8)
    ml = MatLib()
    pose = dict(spec['pose'])
    pose.update(dict(head_rot=(0, 0, 0)))
    ch = Character(spec, pose, ml, 1)
    ch.build()
    hc = ch.J['head']
    f = ch.head_matrix.to_3x3() @ Vector((0, -1, 0))
    f.normalize()
    side = Vector((f.y, -f.x, 0)).normalized()
    scene.add_sun((-0.5, 0.6, -0.65), 3.0)
    scene.add_camera(hc + f * 1.1 + side * 0.35 + Vector((0, 0, 0.05)), hc, lens=60)
    scene.render(f'{outdir}/{cid}.exr', ml)
