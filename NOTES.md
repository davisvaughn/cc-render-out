# render_mold.py fix

## Root cause
1. Not numba/threads/fastmath: `render.py_func` (pure Python) and NUMBA_NUM_THREADS=1 give the same wrong pixels.
2. `skip()` returned a fixed 60-cell (above M64) or 7-cell (above M8) step along the ray. That only bounds *horizontal* travel.
3. Primary rays start just above H.max(), so the first step drops z by 60*cell*|dz|, which is more than a thin mold's height (megaphone: 1.36 cm tall, 60 cells = 1.85 cm).
4. The ray passes through the mold to z<0 and is shaded as paper (wg=1) using the mold's normals/AO/shadow, so you get a white mold with correct shadows. Hero rays at the far edge happen to land inside, which gave the coral strip.
5. It depends on mold height vs. cell size (grid resolution), not on the machine: letter-a (thick) looked right, megaphone (thin) was white, and oval-freshie had a white lens through its 0.45 cm floor.

## Diff summary
- `skip(M8, M64, x, y, z, cell, adz)`: step = min(block radius, (z - blockmax) / |dz|). It drops to the next level (M64 -> M8 -> fine 0.4-cell march) when the vertical clearance is too small.
- `render()`: computes `adz = max(abs(dz), 1e-6)` per ray and passes it to `skip`.
- No changes to the camera, look, geometry, sizes, CLI or GLB code. Still parallel at 4 threads with fastmath. A rerun gives a byte-identical hero.jpg.

## Timing after fix (4 CPUs, whole mold incl. GLB, worker.py)
- letter-a-mold-3-inch: 22.6 s (includes the one-time numba recompile)
- megaphone-candy-mold-1-inch-30-cavity: 15.1 s
- oval-freshie-mold-3-inch: 15.3 s
(before the fix: 27 / 13 / 15 s)

verify/ holds <handle>-hero.jpg and <handle>-top.jpg for the three molds.
