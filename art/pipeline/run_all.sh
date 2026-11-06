#!/bin/bash
# Regenerate every v2 concept-art board from scratch.
#   VP=/path/to/python-with-bpy ./run_all.sh
# Needs: bpy==5.2.3 (Blender as a Python module), numpy, pillow, opencv-python-headless, OpenEXR.
set -e
cd "$(dirname "$0")"
VP=${VP:-python3}
export VP
./render_v2_all.sh
rm -f ../build/portraits/*_v2.png
$VP compose_v2.py
echo "done -> art/concept/"
