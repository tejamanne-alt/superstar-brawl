"""2.5D gameplay shot on Stage 1 (side camera, fighters rotated 3/4 toward camera)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sb import scene, roster, environment as env
from sb.geo import MatLib
from sb.character import Character
W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (3510, 1620)  # 1.5x supersampled
build = os.path.join(os.path.dirname(__file__), '../build')
scene.reset(W, H, samples=16, ao_distance=0.3)
ml = MatLib()
env.rooftop(ml, build)
env.city(ml)
env.kites(ml)
p1 = dict(roster.ALT_POSES['bala_block'], yaw=62)
p2 = dict(roster.ALT_POSES['raja_throw'], yaw=-58)
c1 = Character(roster.BY_ID['bala'], p1, ml, 1, root=(-2.1, 0.4, 0))
c1.build()
c2 = Character(roster.BY_ID['raja'], p2, ml, 2, root=(2.0, 0.2, 0))
c2.build()
import bpy
gl = bpy.data.objects['raja_fg_bridge']
from mathutils import Vector
anchor_pts = dict(glasses=sum((Vector(c) for c in gl.bound_box), Vector()) / 8,
                  raja_hand=c2.J['wrR'], bala_fists=(c1.J['wrR'] + c1.J['wrL']) / 2,
                  bala_head=c1.J['head'], raja_head=c2.J['head'], bala_feet=c1.J['anL'], raja_feet=c2.J['anR'])
scene.add_sun((0.62, 0.55, -0.55), 3.0)   # low sun from screen-left / front
scene.add_camera((0, -6.6, 1.4), (0, 0.5, 1.62), lens=26)
scene.render(os.path.join(build, 'gameplay.exr'), ml, meta=dict(shot='gameplay', anchors=scene.project(anchor_pts)))
