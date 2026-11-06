"""Mass-meter strip: the same Bala shot at 1 bar / 2 bars / super (energy grows)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import bpy  # noqa
from sb import scene, roster, street
from sb.geo import MatLib
from sb.character import Character
build = os.path.join(os.path.dirname(__file__), '../build/meter')
os.makedirs(build, exist_ok=True)
LEVELS = [(0.045, 1), (0.09, 2), (0.16, 3)]
for i, (r, rings) in enumerate(LEVELS):
    scene.reset(1500, 1200, samples=16, ao_distance=0.3)
    ml = MatLib()
    street.street(ml, os.path.dirname(build))
    spec = roster.BY_ID['bala']
    pose = dict(spec['pose'], yaw=-18, props=[dict(type='energy', side='R', radius=r, rings=rings)])
    if i == 2:
        pose['props'].append(dict(type='energy', side='L', radius=0.08, rings=2))
    ch = Character(spec, pose, ml, 1, root=(0, 0, 0))
    ch.build()
    scene.add_sun((0.45, 0.55, -0.70), 3.0)
    scene.add_camera((0.4, -3.9, 0.9), (0.0, 0.0, 1.15), lens=30)
    scene.render(os.path.join(build, f'level{i}.exr'), ml, meta=dict(shot=f'level{i}'))
