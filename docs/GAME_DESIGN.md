# Superstar Brawl — Game Design (concept phase)

**Genre:** 2.5D one-on-one fighter (3D characters on a 2D fighting plane, parallax backgrounds)
**Platforms:** iOS + Android (Unity 6 LTS, URP), landscape 19.5:9, touch-first
**Pitch:** Nine larger-than-life "superstars" from South and North Indian cinema, rebuilt as comic-book avatars, brawl across a film-reel multiverse ("the Filmverse").
**Look:** Hand-printed comic, built on real 3D. See `ART_BIBLE.md` and `art/concept/07_style_guide.png`.

## Story hook

A cursed projector on the roof of an old single-screen cinema starts splicing film reels together.
Iconic heroes from different film universes spill onto the same screen and have to fight their way back
to their own reels. Stages are "reels". The glitch motif is a film strip with sprocket holes, not a pixel glitch.

## Roster

Names are parody names (roughly half the real name kept, the rest changed). Faces in the concept art are caricatures
built from signature traits (mustache, hair, build, wardrobe). **Read `LEGAL_NOTE.md` before taking any of this public.**

| # | Fighter | Inspired by | Avatar (film archetype) | Style | Signature | Super |
|---|---------|-------------|--------------------------|-------|-----------|-------|
| 1 | **Bala Simha** | Balakrishna | Faction Simha: Rayalaseema faction hero, white shirt and dhoti, shoulder towel | Mass brawler | Thigh-Slap Shockwave | Sumo Storm (summons flying jeeps) |
| 2 | **Mega Chiran** | Chiranjeevi | Gang Leader: red leather jacket, dance-fighter | Dance-fu rushdown | Breakdance Cyclone | Mega Mass Medley (rhythm combo) |
| 3 | **Naga Raja** | Nagarjuna | Cycle-Chain Shiva: angry campus rebel, denim, bicycle chain | Chain zoner | Cycle-Chain Lash | Campus Riot (chain whirlwind) |
| 4 | **Venkat Vijay** | Venkatesh | Supercop: khaki, aviators, lathi | Counter-fighter | Lathi Charge | Encounter Special (slow-mo takedown) |
| 5 | **Ambar Bachan** | Amitabh Bachchan | Iron-Arm Shahenshah: long coat, steel sleeve, white beard | Heavy grappler | Iron Sleeve Smash | Shahenshah Verdict (justice slam) |
| 6 | **Raja Kanth** | Rajinikanth | Style King Baasha: black shirt, gold shades, impossible flips | Style trickster | Shades Boomerang | Thalaivar Time (world slows down) |
| 7 | **Kamal Nayak** | Kamal Haasan | Varma Master Senapathy: veteran pressure-point master, white walrus mustache | Technician | Varma Kalai Nerve Strike | Dasa-Avatar Shift (cycles 10 forms) |
| 8 | **Shah Rahul** | Shah Rukh Khan | Bandage Jawan: bandaged face, long crimson coat, open-arms pose | Charm rushdown | Open-Arms Counter | Zinda Banda Rampage |
| 9 | **Hriday Roshan** | Hrithik Roshan | Masked Wonder: masked speedster, long coat, cyan lightning | Aerial speedster | Sky-Step Dash | Lightning Pulse (cosmic sprint) |

Alternate avatars (skins / move-set variants) for later seasons: Bala, an Aghora mystic with a trident; Raja, a robot;
Kamal, a secret agent; Shah Rahul, a long-haired spy; Ambar, a 70s dock-worker "angry young man"; Chiran, a comedy doctor.
Each alternate avatar is a full outfit + intro + super, not just a recolour.

## Core systems

- **Controls (touch):** virtual stick + Punch / Kick / Special / Block + context **SUPER** button (see `04_gameplay_mockup.png`).
  Simple-input mode maps specials to swipe gestures; Pro mode keeps motion inputs for controller players.
- **Mass Meter:** 3 segments, filled by landing hits *and* by "whistle moments" (taunts, perfect blocks, crowd-pleasing finishes).
  1 segment = EX special, 3 = Super. Each fighter renames it (Mass / Style / Victory...) for flavour.
- **Entry & Punch Dialogue:** every round opens with a slow-motion "hero entry" (walk-in, shades on, wind, petals).
  Hold Block during it to skip.
- **Onomatopoeia system:** hit sparks spawn comic SFX text (DHISHUM!, DHISHKIYAAN!, ఢిష్యుం!, धिशूम!), localised per region.
- **Stages as reels:** each stage has one interactive hazard (water tank burst, high-tide splash) and a stage finisher.

## Modes (launch)

Story ("Reel Rift", 9 chapters, motion-comic cutscenes rendered in the same shader) · Arcade ladder · Versus (local and online,
rollback netcode) · Training · Daily "First Day First Show" challenge.

## Stages (concept)

1. **Single Screen Showdown:** cinema rooftop at golden hour, Hyderabad-style skyline (`06_stage1.png`).
2. **Necklace Night:** sea-front promenade at night, arc of lamps on the bay (`06_stage2.png`).
3. Planned: Temple-Town Festival, Film-City Backlot, Rain-Song Railway Platform, Interval Canteen.

## Tech plan (Unity)

- Unity 6 LTS + URP, IL2CPP, ARM64; target 60 fps on mid-range Android (Snapdragon 7-series) and iPhone 12+.
- Characters: Blender → FBX with a shared humanoid rig (Mecanim), blend shapes for expressions, 15–25k tris each.
- Comic look in real time: URP toon lit shader (ramp + hue-shifted shadow + screen-space halftone), depth/normal edge
  detection renderer feature for ink, misregistration + paper grain as a final post pass. Animation on 2s for characters.
- Netcode: deterministic fixed-point combat sim (60 Hz) + GGPO-style rollback.
- Addressables for per-fighter downloads; cosmetics-only monetisation.

## Next steps

1. Decide the likeness strategy (licence vs. original archetypes); see `LEGAL_NOTE.md`.
2. Lock 2–3 fighters for a vertical slice (suggest Bala Simha, Raja Kanth, Mega Chiran) and one stage.
3. Character sculpts (ZBrush/Blender) from licensed reference, retopology, rig, facial blend shapes.
4. Unity project: URP comic shader prototype matching `07_style_guide.png`, then the combat sandbox.
