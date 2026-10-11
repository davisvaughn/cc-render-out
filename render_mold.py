#!/usr/bin/env python3
"""
render_mold.py - product renders + GLB for a silicone mold from height maps.

    pip install numpy scipy pillow numba trimesh fast-simplification

Usage:
    python render_mold.py <in.npz> <in.json> <outdir>

Writes <outdir>/hero.jpg (1000x1000), <outdir>/top.jpg (900x900), <outdir>/mold.glb
and prints the wall-clock seconds for each.

Inputs
  npz : mold, piece  - height maps in cm on a regular grid, row 0 = +y
        cell         - grid spacing in cm
        (master is ignored)
  json: catalog.piece selects the cast-piece material;
        spec.cavities (or catalog.nx*catalog.ny) gives the cavity count.

Renderer: CPU height-field ray marcher (numba). Per-node ambient occlusion and
soft key-light shadows are precomputed; primary rays are supersampled 4x4 and
use a two-level max-height hierarchy, so tall molds stay fast.
The first run includes ~10-20 s of numba JIT compilation (cached afterwards).
"""
import sys, os, json, time, math
import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter, map_coordinates, binary_dilation, maximum_filter, zoom
from PIL import Image

# ============================================================== colour utils
def srgb_to_lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)

def hexc(h):
    h = h.lstrip('#')
    return srgb_to_lin([int(h[i:i + 2], 16) for i in (0, 2, 4)])

BG = hexc('F6F2EA')
CORAL = hexc('FF5A36')

# piece materials: colour, GGX roughness, environment reflection strength, wrap (translucent look)
PIECE_MATS = {
    'gummy':     ('C8142C', 0.14, 1.00, 0.55),
    'hardcandy': ('F07AAA', 0.10, 1.10, 0.35),
    'chocolate': ('3B1F12', 0.16, 1.00, 0.00),
    'cake':      ('3B1F12', 0.16, 1.00, 0.00),
    'caramel':   ('C27318', 0.20, 0.90, 0.35),
    'butter':    ('F2E19A', 0.60, 0.12, 0.10),
    'wax':       ('C3A0D6', 0.55, 0.15, 0.15),
    'soap':      ('F3C3CC', 0.55, 0.15, 0.10),
    'freshie':   ('2AA79B', 0.65, 0.10, 0.00),
    'candle':    ('F2EDDC', 0.55, 0.15, 0.25),
    'concrete':  ('9C9C98', 0.85, 0.05, 0.00),
    'treat':     ('B98A57', 0.75, 0.08, 0.00),
    'ice':       ('CFE7F4', 0.08, 1.20, 0.50),
    'dumpling':  ('F1E6CC', 0.65, 0.10, 0.20),
}

# ============================================================== numba core
@njit(cache=True, fastmath=True)
def hb(H, x, y):
    """bilinear height at grid coords (x=col, y=row); 0 outside."""
    ny, nx = H.shape
    if x < 0 or y < 0 or x > nx - 1.001 or y > ny - 1.001:
        return 0.0
    ix = int(x); iy = int(y); fx = x - ix; fy = y - iy
    return ((H[iy, ix] * (1 - fx) + H[iy, ix + 1] * fx) * (1 - fy)
            + (H[iy + 1, ix] * (1 - fx) + H[iy + 1, ix + 1] * fx) * fy)

@njit(cache=True, fastmath=True)
def skip(M8, M64, x, y, z, cell, adz):
    """safe 3D step length from the max-height hierarchy (0 = must fine-step).
    The step is bounded both horizontally (block radius) and vertically
    (clearance above the block max / |dz|), so it cannot pass through thin geometry."""
    ny, nx = M8.shape
    ix = int(x / cell + 0.5); iy = int(y / cell + 0.5)
    if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
        return 4.0 * cell
    m = M64[iy, ix]
    if z > m:
        s = min(60.0 * cell, (z - m) / adz)
        if s >= 7.0 * cell:
            return s
    m = M8[iy, ix]
    if z > m:
        s = min(7.0 * cell, (z - m) / adz)
        if s >= 0.4 * cell:
            return s
    return 0.0

@njit(parallel=True, cache=True, fastmath=True)
def precompute(H, N, cell, L, light_rad, nao, ao_len):
    ny, nx = H.shape
    AO = np.ones((ny, nx)); SH = np.ones((ny, nx))
    lel = math.asin(L[2]); laz = math.atan2(L[1], L[0])
    maxh = H.max()
    tmin = math.tan(lel - 1.5 * light_rad)
    for iy in prange(ny):
        for ix in range(nx):
            h0 = H[iy, ix]
            n0 = N[iy, ix, 0]; n1 = N[iy, ix, 1]; n2 = N[iy, ix, 2]
            # ---- soft key shadow: horizon angle along 15 azimuths across the light disc
            vis = 0.0
            for k in range(15):
                az = laz + (k - 7) / 7.0 * light_rad
                cx = math.cos(az); cy = math.sin(az)
                best = -1.0
                s = 1.0
                while True:
                    dist = s * cell
                    if (maxh - h0) / dist < tmin:
                        break
                    x = ix + cx * s; y = iy + cy * s
                    if x < 0 or y < 0 or x > nx - 1 or y > ny - 1:
                        break
                    t = (hb(H, x, y) - h0) / dist
                    if t > best:
                        best = t
                    s += 0.7 if s < 20 else 0.035 * s
                hz = math.atan(best)
                v = (lel - hz) / (2 * light_rad) + 0.5
                v = min(1.0, max(0.0, v))
                vis += v * v * (3 - 2 * v)
            SH[iy, ix] = vis / 15.0
            # ---- cosine-weighted AO around the normal
            if abs(n2) < 0.999:
                tx = -n1; ty = n0; tz = 0.0
            else:
                tx = 1.0; ty = 0.0; tz = 0.0
            tl = math.sqrt(tx * tx + ty * ty + tz * tz); tx /= tl; ty /= tl; tz /= tl
            bx = n1 * tz - n2 * ty; by = n2 * tx - n0 * tz; bz = n0 * ty - n1 * tx
            occ = 0.0
            rot = ((ix * 0.7548776662 + iy * 0.5698402910) % 1.0) * 6.283185307
            px = ix * cell + n0 * cell * 0.4; py = iy * cell + n1 * cell * 0.4; pz = h0 + n2 * cell * 0.4
            for k in range(nao):
                u = (k + 0.5) / nao
                r = math.sqrt(u); phi = k * 2.399963229728653 + rot
                lx = r * math.cos(phi); ly = r * math.sin(phi); lz = math.sqrt(max(0.0, 1 - u))
                dx = lx * tx + ly * bx + lz * n0
                dy = lx * ty + ly * by + lz * n1
                dz = lx * tz + ly * bz + lz * n2
                if dz <= 0.0:
                    occ += 1.0
                    continue
                t = cell * 0.5
                hit = 0.0
                while t < ao_len:
                    qx = px + dx * t; qy = py + dy * t; qz = pz + dz * t
                    if qz > maxh:
                        break
                    if qz < hb(H, qx / cell, qy / cell):
                        hit = 1.0 - (t / ao_len) ** 2
                        break
                    t += max(cell * 0.6, 0.06 * t)
                occ += hit
            AO[iy, ix] = 1.0 - occ / nao
    return AO, SH

@njit(cache=True, fastmath=True)
def bil(A, x, y, out):
    ny, nx = A.shape[0], A.shape[1]
    x = min(max(x, 0.0), nx - 1.001); y = min(max(y, 0.0), ny - 1.001)
    ix = int(x); iy = int(y); fx = x - ix; fy = y - iy
    for k in range(A.shape[2]):
        out[k] = ((A[iy, ix, k] * (1 - fx) + A[iy, ix + 1, k] * fx) * (1 - fy)
                  + (A[iy + 1, ix, k] * (1 - fx) + A[iy + 1, ix + 1, k] * fx) * fy)

@njit(cache=True, fastmath=True)
def ggx(nh, a):
    a2 = a * a
    dd = nh * nh * (a2 - 1) + 1
    return a2 / (math.pi * dd * dd)

@njit(cache=True, fastmath=True)
def env(rx, ry, rz, L):
    """studio environment: floor bounce, soft sky gradient, large softbox around the key."""
    if rz < 0:
        return 0.35
    base = 0.25 + 0.35 * rz
    c = rx * L[0] + ry * L[1] + rz * L[2]
    box = 0.0
    if c > 0.80:
        v = min(1.0, (c - 0.80) / 0.12)
        box = 3.2 * v * v * (3 - 2 * v)
    return base + box

@njit(parallel=True, cache=True, fastmath=True)
def render(H, M8, M64, F, cell, W, Hh, ss, cam, fwd, right, up, ortho, fovt, L, bg,
           coral, pcol, p_rough, p_refl, p_wrap, key, amb, mold_spec, mold_rough):
    # F[..] = nx ny nz ao sh w_mold w_piece
    ny, nx = H.shape
    img = np.zeros((Hh, W, 3))
    maxh = H.max() + 1e-3
    X1 = (nx - 1) * cell; Y1 = (ny - 1) * cell
    for py in prange(Hh):
        f = np.zeros(7)
        for px in range(W):
            a0 = 0.0; a1 = 0.0; a2 = 0.0
            for sy in range(ss):
                for sx in range(ss):
                    jx = (sx + 0.5 + 0.25 * ((sy * 7 + px * 3 + sx) % 3 - 1)) / ss
                    jy = (sy + 0.5 + 0.25 * ((sx * 5 + py * 3 + sy) % 3 - 1)) / ss
                    u = (px + jx) / W * 2 - 1
                    v = 1 - (py + jy) / Hh * 2
                    if ortho > 0:
                        ox = cam[0] + (right[0] * u + up[0] * v) * ortho
                        oy = cam[1] + (right[1] * u + up[1] * v) * ortho
                        oz = cam[2] + (right[2] * u + up[2] * v) * ortho
                        dx = fwd[0]; dy = fwd[1]; dz = fwd[2]
                    else:
                        ox = cam[0]; oy = cam[1]; oz = cam[2]
                        dx = fwd[0] + (right[0] * u + up[0] * v) * fovt
                        dy = fwd[1] + (right[1] * u + up[1] * v) * fovt
                        dz = fwd[2] + (right[2] * u + up[2] * v) * fovt
                        dl = math.sqrt(dx * dx + dy * dy + dz * dz); dx /= dl; dy /= dl; dz /= dl
                    t0 = (oz - maxh) / -dz
                    t1 = oz / -dz
                    hit = False
                    adz = max(abs(dz), 1e-6)
                    t = t0; prev = t
                    while t < t1:
                        x = ox + dx * t; y = oy + dy * t; z = oz + dz * t
                        st = skip(M8, M64, x, y, z, cell, adz)
                        if st == 0.0:
                            if x >= 0 and y >= 0 and x <= X1 and y <= Y1:
                                if z < hb(H, x / cell, y / cell):
                                    hit = True
                                    break
                            st = 0.4 * cell
                        prev = t
                        t += st
                    if hit:
                        lo = prev; hi = t
                        for _ in range(12):
                            mid = 0.5 * (lo + hi)
                            if oz + dz * mid < hb(H, (ox + dx * mid) / cell, (oy + dy * mid) / cell):
                                hi = mid
                            else:
                                lo = mid
                        t = hi
                    else:
                        t = t1
                    x = ox + dx * t; y = oy + dy * t; zh = oz + dz * t
                    gx = x / cell; gy = y / cell
                    if gx < 0 or gy < 0 or gx > nx - 1 or gy > ny - 1:
                        f[0] = 0; f[1] = 0; f[2] = 1; f[3] = 1; f[4] = 1; f[5] = 0; f[6] = 0
                    else:
                        bil(F, gx, gy, f)
                    nn = math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
                    n0 = f[0] / nn; n1 = f[1] / nn; n2 = f[2] / nn
                    ao = f[3]; sh = f[4]; wm = f[5]; wc = f[6]
                    if zh > 0.004 and wm + wc > 1e-6:
                        sw = wm + wc; wm /= sw; wc /= sw; wg = 0.0
                    else:
                        wm = 0.0; wc = 0.0; wg = 1.0
                    vx = -dx; vy = -dy; vz = -dz
                    ndlr = n0 * L[0] + n1 * L[1] + n2 * L[2]
                    ndl = max(0.0, ndlr)
                    ndv = max(1e-3, n0 * vx + n1 * vy + n2 * vz)
                    hx = L[0] + vx; hy = L[1] + vy; hz = L[2] + vz
                    hl = math.sqrt(hx * hx + hy * hy + hz * hz)
                    ndh = max(0.0, (n0 * hx + n1 * hy + n2 * hz) / hl)
                    vdh = max(0.0, (vx * hx + vy * hy + vz * hz) / hl)
                    aos = ao * ao
                    sky = amb * (0.55 + 0.45 * n2) * aos
                    # ground (paper): exactly bg where unoccluded
                    gvis = (0.55 * sh + 0.45 * ao) ** 1.15
                    # mold: matte silicone with a soft highlight
                    diff = key * ndl * sh + sky
                    fr = 0.04 + 0.96 * (1 - vdh) ** 5
                    spec = mold_spec * ggx(ndh, mold_rough * mold_rough) * fr * ndl * sh / (4 * ndv * max(ndl, 1e-3) + 1e-3) * key
                    sheen = 0.06 * (1 - ndv) ** 3 * ao
                    # piece
                    wl = max(0.0, (ndlr + p_wrap) / (1 + p_wrap))
                    dch = key * wl * (sh * (1 - 0.5 * p_wrap) + 0.5 * p_wrap) + sky * (1 + 0.6 * p_wrap)
                    frc = 0.045 + 0.955 * (1 - vdh) ** 5
                    sp2 = ggx(ndh, p_rough * p_rough) * frc * ndl * sh / (4 * ndv * max(ndl, 1e-3) + 1e-3) * key * 0.6 * min(1.0, p_refl + 0.3)
                    vdn = vx * n0 + vy * n1 + vz * n2
                    rx = 2 * vdn * n0 - vx; ry = 2 * vdn * n1 - vy; rz = 2 * vdn * n2 - vz
                    fre = 0.045 + 0.955 * (1 - ndv) ** 5
                    refl = p_refl * env(rx, ry, rz, L) * fre * ao * (0.4 + 0.6 * sh)
                    a0 += wg * bg[0] * gvis + wm * (coral[0] * diff + spec + sheen) + wc * (pcol[0] * dch + sp2 + refl)
                    a1 += wg * bg[1] * gvis + wm * (coral[1] * diff + spec + sheen) + wc * (pcol[1] * dch + sp2 + refl * 0.97)
                    a2 += wg * bg[2] * gvis + wm * (coral[2] * diff + spec + sheen) + wc * (pcol[2] * dch + sp2 + refl * 0.93)
            img[py, px, 0] = a0 / (ss * ss)
            img[py, px, 1] = a1 / (ss * ss)
            img[py, px, 2] = a2 / (ss * ss)
    return img

# ============================================================== scene
LIGHT_AZ, LIGHT_EL, LIGHT_RAD = 150.0, 52.0, math.radians(6.0)
PAR = dict(key=0.82, amb=0.42, mold_spec=0.35, mold_rough=0.55)
MAX_RENDER_NODES = 2_200_000       # render grid is resampled above this (GLB always uses full data)

def light_dir():
    a, e = math.radians(LIGHT_AZ), math.radians(LIGHT_EL)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])

def norm(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)

def crop_to(a):
    ys, xs = np.where(a > 0)
    if len(ys) == 0:
        return a
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def plan_pieces(mold, piece, cell, cavities):
    """returns list of (cx, cy, rot_deg) in mold-grid coords (iy up)."""
    if piece is None or piece.max() <= 0:
        return []
    ys, xs = np.where(mold > 0)
    mx0, mx1, my0, my1 = xs.min(), xs.max(), ys.min(), ys.max()
    py, px = piece.shape
    pw_cm = max(py, px) * cell
    half = 0.5 * math.hypot(py, px)               # rotated footprint radius (cells)
    if cavities <= 1 or pw_cm > 6.0:
        gap = max(1.0 / cell, 0.12 * max(py, px))
        cx = mx1 + gap + half
        cy = my0 + 0.42 * (my1 - my0)
        return [(cx, cy, 12.0)]
    s = max(py, px)
    spacing = 1.45 * s + 0.4 / cell
    gap0 = max(0.6 / cell, 0.35 * s)
    mcx = 0.5 * (mx0 + mx1) - 0.08 * (mx1 - mx0)
    out = []
    for k, (stag, rot) in enumerate(((0.05, -17.0), (0.45, 9.0), (0.0, 28.0))):
        cx = mcx + (k - 1) * spacing
        cy = my0 - gap0 - half - stag * s
        out.append((cx, cy, rot))
    return out

def stamp_piece(piece, cx, cy, rot):
    """rotated piece heights sampled on integer grid around (cx,cy); returns (h, iy0, ix0)."""
    py, px = piece.shape
    pcy, pcx = (py - 1) / 2, (px - 1) / 2
    r = int(math.ceil(0.5 * math.hypot(py, px))) + 2
    iy0, ix0 = int(round(cy)) - r, int(round(cx)) - r
    iy, ix = np.mgrid[iy0:iy0 + 2 * r, ix0:ix0 + 2 * r]
    dy, dx = iy - cy, ix - cx
    R = math.radians(rot)
    lx = math.cos(R) * dx + math.sin(R) * dy
    ly = -math.sin(R) * dx + math.cos(R) * dy
    h = map_coordinates(piece, [ly + pcy, lx + pcx], order=1, cval=0.0)
    return h, iy0, ix0

def build_world(mold, piece, cell, placements, L, ao_len):
    """union height field + material map, padded so shadows/AO stay on the grid."""
    stamps = [stamp_piece(piece, *p) for p in placements]
    y0, x0 = 0, 0
    y1, x1 = mold.shape
    for h, iy0, ix0 in stamps:
        y0 = min(y0, iy0); x0 = min(x0, ix0)
        y1 = max(y1, iy0 + h.shape[0]); x1 = max(x1, ix0 + h.shape[1])
    maxh = max(mold.max(), max((s[0].max() for s in stamps), default=0))
    shadow = maxh / math.tan(math.asin(L[2]) - 1.5 * LIGHT_RAD)
    base = ao_len + 0.3
    lx, ly = -L[0] / math.hypot(L[0], L[1]), -L[1] / math.hypot(L[0], L[1])   # shadow direction
    pad_r = int((base + max(0, lx) * shadow) / cell) + 4
    pad_l = int((base + max(0, -lx) * shadow) / cell) + 4
    pad_t = int((base + max(0, ly) * shadow) / cell) + 4
    pad_b = int((base + max(0, -ly) * shadow) / cell) + 4
    oy, ox = pad_b - y0, pad_l - x0
    H = np.zeros((y1 - y0 + pad_b + pad_t, x1 - x0 + pad_l + pad_r))
    MAT = np.zeros(H.shape, np.int8)
    H[oy:oy + mold.shape[0], ox:ox + mold.shape[1]] = mold
    MAT[oy:oy + mold.shape[0], ox:ox + mold.shape[1]] = (mold > 0)
    for h, iy0, ix0 in stamps:
        sl = (slice(oy + iy0, oy + iy0 + h.shape[0]), slice(ox + ix0, ox + ix0 + h.shape[1]))
        H[sl] = np.maximum(H[sl], h)
        MAT[sl] = np.where(h > 0, 2, MAT[sl])
    # keep render grid bounded
    n = H.size
    if n > MAX_RENDER_NODES:
        f = math.sqrt(MAX_RENDER_NODES / n)
        H = np.maximum(zoom(H, f, order=1), 0)
        MAT = zoom(MAT, f, order=0)
        return H, MAT, 1.0 / f              # cell-size multiplier
    return H, MAT, 1.0

def features(H, MAT, cell, L, ao_len):
    Hs = gaussian_filter(H, 1.1)
    gy, gx = np.gradient(Hs, cell)
    N = np.dstack([-gx, -gy, np.ones_like(H)])
    N /= np.linalg.norm(N, axis=2, keepdims=True)
    AO, SH = precompute(H, N, cell, L, LIGHT_RAD, 72, ao_len)
    AO = gaussian_filter(AO, 1.0)
    F = np.dstack([N, AO, SH,
                   binary_dilation(MAT == 1, iterations=2).astype(float),
                   binary_dilation(MAT == 2, iterations=2).astype(float)])
    M8 = maximum_filter(H, size=17) + 1e-4
    M64 = maximum_filter(H, size=129) + 1e-4
    return F, M8, M64

def save_jpg(img, path):
    out = lin_to_srgb(img)
    Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(path, quality=90)

def prepare(mold, piece, cell, placements, L):
    size_cm = max(mold.shape) * cell
    ao_len = float(np.clip(0.07 * size_cm, 0.9, 2.5))
    H, MAT, f = build_world(mold, piece, cell, placements, L, ao_len)
    c = cell * f
    F, M8, M64 = features(H, MAT, c, L, ao_len)
    return H, MAT, c, F, M8, M64

def render_hero(mold, piece, cell, placements, pmat, L, path, size=1000):
    H, MAT, c, F, M8, M64 = prepare(mold, piece, cell, placements, L)
    ys, xs = np.where(MAT > 0)
    stride = max(1, len(ys) // 150000)
    ys, xs = ys[::stride], xs[::stride]
    pts = np.c_[xs * c, ys * c, H[ys, xs]]
    pts = np.r_[pts, np.c_[pts[:, :2], np.zeros(len(pts))]]
    target = np.r_[0.5 * (pts[:, :2].min(0) + pts[:, :2].max(0)), 0.25 * H.max()]
    el, az = math.radians(44), math.radians(24)
    vdir = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
    fovt = math.tan(math.radians(24) / 2)
    fwd = -vdir
    right = norm(np.cross(fwd, [0, 0, 1])); up = np.cross(right, fwd)
    span = float(np.ptp(pts, axis=0).max())
    lo, hi = 0.3 * span, 60.0 * span
    for _ in range(60):
        dist = 0.5 * (lo + hi)
        cam = target + vdir * dist
        rel = pts - cam; zc = rel @ fwd
        if zc.min() <= 0:
            lo = dist; continue
        ext = max(np.abs((rel @ right) / zc).max(), np.abs((rel @ up) / zc).max()) / fovt
        if ext > 0.86: lo = dist
        else: hi = dist
    dist = hi
    cam = target + vdir * dist
    rel = pts - cam; zc = rel @ fwd
    u = (rel @ right) / zc / fovt; v = (rel @ up) / zc / fovt
    du, dv = 0.5 * (u.max() + u.min()), 0.5 * (v.max() + v.min())
    fwd = norm(fwd + right * du * fovt + up * dv * fovt)
    right = norm(np.cross(fwd, [0, 0, 1])); up = np.cross(right, fwd)
    pcol, prough, prefl, pwrap = pmat
    img = render(H, M8, M64, F, c, size, size, 4, cam, fwd, right, up, 0.0, fovt, L, BG,
                 CORAL, pcol, prough, prefl, pwrap, **PAR)
    save_jpg(img, path)

def render_top(mold, cell, pmat, L, path, size=900):
    H, MAT, c, F, M8, M64 = prepare(mold, None, cell, [], L)
    ys, xs = np.where(MAT > 0)
    cx, cy = 0.5 * (xs.min() + xs.max()) * c, 0.5 * (ys.min() + ys.max()) * c
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) * c / 2 / 0.86
    cam = np.array([cx, cy, H.max() + 50.0])
    fwd = np.array([0, 0, -1.0]); right = np.array([1.0, 0, 0]); up = np.array([0, 1.0, 0])
    pcol, prough, prefl, pwrap = pmat
    img = render(H, M8, M64, F, c, size, size, 4, cam, fwd, right, up, half, 0.0, L, BG,
                 CORAL, pcol, prough, prefl, pwrap, **PAR)
    save_jpg(img, path)

# ============================================================== GLB
def build_glb(mold, cell, path, max_tris=160000, target=150000):
    import trimesh, fast_simplification
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals
    M = np.pad(crop_to(mold), 2)
    # memory guard for very large grids: resample (bilinear) to <= 1.6M nodes
    if M.size > 1_600_000:
        f = math.sqrt(1_600_000 / M.size)
        M = np.maximum(zoom(M, f, order=1), 0)
        M[M < 1e-6] = 0
        cell = cell / f
    ny, nx = M.shape
    Mr = M.ravel()
    cin = (M[:-1, :-1] > 0) | (M[1:, :-1] > 0) | (M[:-1, 1:] > 0) | (M[1:, 1:] > 0)
    idx = np.arange(ny * nx).reshape(ny, nx)
    cy, cx = np.where(cin)
    a = idx[cy, cx]; b = idx[cy, cx + 1]; c = idx[cy + 1, cx]; d = idx[cy + 1, cx + 1]
    top = np.r_[np.c_[a, b, d], np.c_[a, d, c]]           # CCW from +z
    top = top[(Mr[top] > 0).any(1)]                       # drop flat z=0 triangles
    gx, gy = np.meshgrid(np.arange(nx) * cell, np.arange(ny) * cell)
    used = np.zeros(ny * nx, bool); used[np.unique(top)] = True
    interior = used & (Mr > 0)                            # boundary (h=0) nodes are shared with the bottom
    remap = np.arange(ny * nx)
    remap[interior] = ny * nx + np.arange(interior.sum())
    V = np.r_[np.c_[gx.ravel(), gy.ravel(), Mr],
              np.c_[gx.ravel()[interior], gy.ravel()[interior], np.zeros(interior.sum())]]
    Fc = np.r_[top, remap[top][:, ::-1]]
    m = trimesh.Trimesh(V, Fc, process=True)
    m.remove_unreferenced_vertices()
    if len(m.faces) > max_tris:
        best = None; fallback = None
        for tgt in (target, int(target * 0.93), int(target * 0.85)):
            for agg in (3, 2, 5):
                v2, f2 = fast_simplification.simplify(m.vertices, m.faces,
                                                      target_reduction=1 - tgt / len(m.faces), agg=agg)
                mm = trimesh.Trimesh(v2, f2, process=True)
                if len(mm.faces) <= max_tris and mm.is_watertight:
                    best = mm; break
                if best is None and len(mm.faces) <= max_tris:
                    fallback = mm
            if best is not None:
                break
        m = best if best is not None else (fallback if fallback is not None else mm)
    v = m.vertices.copy()
    v[:, :2] -= 0.5 * (v[:, :2].min(0) + v[:, :2].max(0))
    v /= 100.0                                            # cm -> m
    m = trimesh.Trimesh(np.c_[v[:, 0], v[:, 2], -v[:, 1]], m.faces, process=False)   # Z-up -> Y-up
    m.fix_normals()
    m = trimesh.graph.smooth_shade(m, angle=math.radians(35))
    m.visual = TextureVisuals(material=PBRMaterial(
        name='silicone', baseColorFactor=[1.0, 0.105, 0.03, 1.0],
        roughnessFactor=0.5, metallicFactor=0.0))
    m.export(path)
    return len(m.faces)

# ============================================================== main
def main():
    if len(sys.argv) != 4:
        print(__doc__); sys.exit(1)
    npz, js, outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)
    d = np.load(npz)
    cell = float(d['cell'])
    mold = np.flipud(np.asarray(d['mold'], np.float64))           # [iy, ix], iy -> +y
    piece = np.flipud(np.asarray(d['piece'], np.float64)) if 'piece' in d.files else None
    if piece is not None:
        piece = crop_to(np.maximum(piece, 0))
        piece = np.pad(piece, 1)
    meta = json.load(open(js))
    cat, spec = meta.get('catalog', {}), meta.get('spec', {})
    cavities = int(spec.get('cavities') or (cat.get('nx', 1) * cat.get('ny', 1)) or 1)
    kind = str(cat.get('piece', 'chocolate')).strip().lower()
    hx, rough, refl, wrap = PIECE_MATS.get(kind, PIECE_MATS['chocolate'])
    pmat = (hexc(hx), rough, refl, wrap)
    # mold cropped with a 1-cell zero border (placements use this frame)
    mold = np.pad(crop_to(np.maximum(mold, 0)), 1)
    L = light_dir()
    placements = plan_pieces(mold, piece, cell, cavities)

    t = time.time()
    render_hero(mold, piece, cell, placements, pmat, L, os.path.join(outdir, 'hero.jpg'))
    print(f'hero.jpg  {time.time() - t:6.1f} s  ({len(placements)} piece(s), material={kind})')
    t = time.time()
    render_top(mold, cell, pmat, L, os.path.join(outdir, 'top.jpg'))
    print(f'top.jpg   {time.time() - t:6.1f} s')
    t = time.time()
    nt = build_glb(mold, cell, os.path.join(outdir, 'mold.glb'))
    print(f'mold.glb  {time.time() - t:6.1f} s  ({nt} triangles)')

if __name__ == '__main__':
    main()
