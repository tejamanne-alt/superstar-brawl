"""Stage concept renders: Stage 1 establishing shot + Stage 2 night promenade (1.5x supersampled)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import bpy  # noqa
from sb import scene, roster, environment as env
from sb.geo import MatLib
from sb.character import Character
build = os.path.join(os.path.dirname(__file__), '../build')
which = sys.argv[1] if len(sys.argv) > 1 else 'both'
W, H = 2880, 1620
if which in ('1', 'both'):
    scene.reset(W, H, samples=16, ao_distance=0.3)
    ml = MatLib()
    env.rooftop(ml, build)
    env.city(ml, seed=7)
    env.kites(ml, n=9)
    scene.add_sun((0.62, 0.55, -0.55), 3.0)
    cam = scene.add_camera((-9.0, -9.5, 3.2), (1.0, 4.0, 2.2), lens=24)
    scene.render(os.path.join(build, 'stage1.exr'), ml, meta=dict(shot='stage1'))
if which in ('2', 'both'):
    scene.reset(W, H, samples=16, ao_distance=0.3)
    ml = MatLib()
    env.promenade(ml, night=True)
    env.city(ml, seed=21, night=True, monument=False, y_range=(330, 620), x_range=(-420, 420), z0=-1.0)
    a = Character(roster.BY_ID['nag'], dict(roster.BY_ID['nag']['pose'], yaw=55), ml, 1, root=(-2.2, 0.5, 0))
    a.build()
    b = Character(roster.BY_ID['hriday'], dict(roster.BY_ID['hriday']['pose'], yaw=-60), ml, 2, root=(2.4, 0.4, 0))
    b.build()
    scene.add_sun((-0.45, 0.70, -0.55), 3.0)
    scene.add_camera((0.0, -10.5, 4.6), (0.0, 40.0, 2.2), lens=24)
    scene.render(os.path.join(build, 'stage2.exr'), ml, meta=dict(shot='stage2'))
