#!/bin/bash
# All Blender renders for the v2 concept boards.
VP=${VP:-python}
cd "$(dirname "$0")"
export VP
$VP render_portraits.py all 900 2>&1 | grep "\[render\]\|rror"
./render_all_cards.sh
$VP render_gameplay.py 2>&1 | grep "\[render\]\|rror"
$VP render_keyart.py 2>&1 | grep "\[render\]\|rror"
$VP render_panels.py 2>&1 | grep "\[render\]\|rror"
$VP render_meter.py 2>&1 | grep "\[render\]\|rror"
$VP render_stages.py both 2>&1 | grep "\[render\]\|rror"
echo ALLDONE
