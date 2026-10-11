# Tall mold rendering (40-160 mm)

## What was wrong
After the `skip()`/`adz` ray-skipping fix, geometry was correct, but tall molds were still badly lit:
- **Vertical walls almost black (dark maroon).** The key light (az 150°, el 52°) sits behind the
  objects, so the walls the camera sees get no key light at all, only sky ambient. That ambient
  was scaled by `ao²`, and the AO pass counted every below-horizon direction as fully occluded.
  A vertical wall's hemisphere is half below the horizon, so every wall node had AO ≈ 0.5,
  `ao²` ≈ 0.25, and an ambient of about 0.06: near black. On flat trays this only touches a
  thin edge, which is why flats looked fine.
- **Cast piece lost its colour** (ivory candle → dark grey, concrete → near black): same cause on the piece's sides.
- **Large heavy shadow**: the 52° key throws a shadow 0.78× object height toward the camera, at near-full darkness.

## What changed (render_mold.py)
Everything is gated by `set_tall()`. It runs only when the mold or piece is at least `TALL_CM = 4.0` cm tall.
The tallest mold in order_cloud.txt (the flat production list) is 3.95 cm, so no production flat mold switches over.
- AO: below-horizon directions now hit the paper floor, weighted by `DOWN_OCC` (1.0 flat, 0.35 tall), instead of counting as black.
- A soft camera-side fill light (az −15°, el 15°, 0.40) and a paper bounce term `(0.5−0.5·nz)` (0.22) light vertical faces of both mold and piece.
- Key light raised to el 60° and the light disc widened to 9° (shorter, softer shadow). Ground shadow strength is scaled by 0.78.
- Hero framing includes 60% of the shadow footprint, so the shadow body stays inside the frame.
- Flat molds go through the same code with fill=0, bounce=0, shadow_k=1 and DOWN_OCC=1.

## Flat molds
`worker_img.py --start 0 --count 3` before vs after: all 6 JPEGs are **byte-identical** (max pixel diff 0).

## Speed (4 vCPU, numba parallel, JIT cached)
Total per tall mold is 34–47 s (hero 21–29 s, top 13–18 s), about the same as before the change.
Two molds (planter, kiss-drop soap) are a little over the 40 s target.

## Still not perfect
- top.jpg: the steep outer draft walls appear as a duller, slightly grey rim around the mold outline.
  There is also a thin pale line between the mold base and its shadow. Both were there before the change.
- top.jpg frames the mold only, so a tall mold's soft shadow runs off the right/bottom edge.
  The mold itself is never clipped.
- The ivory candle's lit top is close to white. It still reads as ivory against the paper, but there is little headroom.
- Shading on tall pieces is flatter than on flats, because fill and bounce dominate the side faces.
  The faces read clearly but have less contrast.
