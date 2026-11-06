"""Read Blender multilayer EXR passes into numpy arrays (H, W[, C]) with top row first."""
import json
import os
import numpy as np
import OpenEXR


def load(path):
    f = OpenEXR.File(path)
    out = {}
    for part in f.parts:
        for name, ch in part.channels.items():
            key = name.replace('ViewLayer.', '')
            out[key] = np.asarray(ch.pixels, dtype=np.float32)
    P = {}
    P['rgba'] = out.get('Combined')
    P['albedo'] = out['Diffuse Color'][..., :3]
    P['direct'] = out['Diffuse Direct'][..., :3]
    P['normal'] = np.stack([out['Normal.X'], out['Normal.Y'], out['Normal.Z']], -1)
    P['depth'] = out['Depth.Z']
    P['obj'] = np.rint(out['Object Index.X']).astype(np.int32)
    P['mat'] = np.rint(out['Material Index.X']).astype(np.int32)
    P['ao'] = out['Ambient Occlusion'][..., 0]
    P['emit'] = out['Emission'][..., :3]
    P['alpha'] = P['rgba'][..., 3] if P['rgba'] is not None else (P['obj'] > 0).astype(np.float32)
    for n in ('metal', 'gloss', 'region', 'height', 'fur'):
        if f'{n}.X' in out:
            P[n] = out[f'{n}.X']
    meta = {}
    js = os.path.splitext(path)[0] + '.json'
    if os.path.exists(js):
        meta = json.load(open(js))
    return P, meta
