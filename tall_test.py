"""Render a list of tall molds (hero.jpg + top.jpg, no GLB) into tall_out/<handle>/.
    python tall_test.py [handle ...]          (defaults to the TALL list below)
Mirrors worker.py: catalog entry -> gen/build.assemble() -> npz + in.json -> render_mold.py.
"""
import sys, os, json, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gen"))
import build as B

TALL = [
    "pillar-candle-mold-1-5-x-3-inch",                 # line pillar, candle
    "round-concrete-planter-mold-3-5-inch",            # line planter, concrete
    "octagon-candle-mold-4-x-2-inch",                  # line candle
    "pillar-votive-candle-mold-1-5-x-3-inch-2-cavity", # line votive, 2 cavities
    "cube-ice-mold-3-inch",                            # shape cube, ice
    "kiss-drop-soap-mold-4-inch",                      # extra: soap, 8 cm deep
]

def main():
    hs = sys.argv[1:] or TALL
    by = {p["handle"]: p for p in json.load(open(os.path.join(HERE, "catalog5k.json")))}
    arr = os.path.join(HERE, "arrays"); os.makedirs(arr, exist_ok=True); B.ARR = arr
    for h in hs:
        p = by[h]; o = os.path.join(HERE, "tall_out", h); os.makedirs(o, exist_ok=True)
        t = time.time()
        sp = B.assemble(p)
        j = os.path.join(o, "in.json"); json.dump(dict(catalog=p, spec=sp), open(j, "w"))
        npz = os.path.join(arr, h + ".npz")
        r = subprocess.run([sys.executable, os.path.join(HERE, "render_mold.py"), npz, j, o],
                           capture_output=True, text=True, timeout=1500, env=dict(os.environ, CC_NO_GLB="1"))
        os.remove(j); os.remove(npz)
        if r.returncode:
            print("FAIL", h, r.stderr.strip()[-400:], flush=True); continue
        print(f"DONE {h} {time.time() - t:.1f}s | " + r.stdout.strip().replace("\n", " | "), flush=True)

if __name__ == "__main__":
    main()
