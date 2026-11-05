#!/bin/bash
# Render every fighter card pass set (EXR) into art/build/cards
VP=${VP:-python}
cd "$(dirname "$0")"
for c in ${@:-bala chiru nag venky ambar raja kamal srk hriday}; do
  $VP render_card.py $c 2>&1 | grep "\[render\]\|rror"
done
