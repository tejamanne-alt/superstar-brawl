#!/bin/bash
# Regenerate every concept-art board from scratch.
#   VP=/path/to/python-with-bpy ./run_all.sh
# Needs: bpy==5.2.3 (Blender as a Python module), numpy, pillow, opencv-python-headless, OpenEXR.
set -e
cd "$(dirname "$0")"
VP=${VP:-python3}
export VP
$VP render_portraits.py all 900          # head & shoulders, select screen / HUD / card inset
./render_all_cards.sh                    # 9 full-body card renders (1.5x supersampled)
$VP render_keyart.py                     # full roster on Stage 1
$VP render_gameplay.py                   # 2.5D gameplay camera
$VP render_stages.py both                # Stage 1 establishing + Stage 2 night
rm -f ../build/portraits/*_comic.png
$VP compose_keyart.py                    # 01 key art, 02 roster lineup
$VP compose_cards.py                     # 03 cards + sheet
$VP compose_gameplay.py                  # 04 gameplay HUD mockup
$VP compose_select.py                    # 05 character select
$VP compose_stages.py                    # 06 stages
$VP compose_styleguide.py                # 07 art bible board
echo "done -> art/concept/"
