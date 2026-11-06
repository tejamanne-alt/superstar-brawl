"""Action comic page renders: three shots of moves landing (1.5x supersampled)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import bpy  # noqa
from mathutils import Vector
from sb import scene, roster, street
from sb.geo import MatLib
from sb.character import Character
build = os.path.join(os.path.dirname(__file__), '../build/panels')
os.makedirs(build, exist_ok=True)
which = sys.argv[1].split(',') if len(sys.argv) > 1 else ['a', 'b', 'c']
A = roster.ALT_POSES


def shot(name, W, H, cast, cam, target, lens, sun=(0.45, 0.55, -0.70), anchors=None):
    scene.reset(W, H, samples=16, ao_distance=0.3)
    ml = MatLib()
    street.street(ml, os.path.dirname(build))
    an = {}
    for i, (cid, pose, root) in enumerate(cast):
        ch = Character(roster.BY_ID[cid], pose, ml, i + 1, root=root)
        ch.build()
        an[cid + '_head'] = ch.J['head']
        an[cid + '_wrR'] = ch.J['wrR']
        an[cid + '_chest'] = ch.J['chest']
        an[cid + '_feet'] = (ch.J['anL'] + ch.J['anR']) / 2
    if anchors:
        an.update(anchors(bpy))
    scene.add_sun(sun, 3.0)
    scene.add_camera(cam, target, lens=lens)
    scene.render(os.path.join(build, f'{name}.exr'), ml, meta=dict(shot=name, anchors=scene.project(an)))


if 'a' in which:  # SWOOSH: the shades boomerang leaves the hand
    def gl(bpy):
        o = bpy.data.objects['raja_fg_bridge']
        return dict(glasses=sum((Vector(c) for c in o.bound_box), Vector()) / 8)
    shot('a', 2250, 1080, [('raja', dict(A['raja_throw'], yaw=-62), (0.6, 0.0, 0))],
         (-1.2, -3.6, 1.35), (-0.4, 0.0, 1.45), 30, anchors=gl)
if 'b' in which:  # DHISHUM: Bala's straight right lands on Naga
    shot('b', 2250, 1200, [('bala', dict(A['punch'], yaw=90), (-0.95, 0.0, 0)),
                           ('nag', dict(A['hit'], yaw=-90), (0.80, 0.05, 0))],
         (0.0, -5.6, 1.30), (0.0, 0.0, 1.32), 30, sun=(0.30, 0.60, -0.74))
if 'c' in which:  # KRUNCH: the iron sleeve smash launches Chiran
    shot('c', 1500, 2250, [('ambar', dict(A['smash'], yaw=80), (-0.60, 0.0, 0)),
                           ('chiru', dict(A['hit'], yaw=-95, air=0.30, head_rot=(0, 30, 15)), (0.75, 0.1, 0))],
         (0.05, -3.5, 0.95), (0.08, 0.2, 1.32), 25, sun=(0.08, 0.78, -0.62))
