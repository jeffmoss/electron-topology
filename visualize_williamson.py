#!/usr/bin/env python3
"""
Visualize Williamson/van der Mark constrained search results.

Renders electron (p_e, 2) and muon (p_mu, q_mu) paths on the same torus
with physical annotations from the Williamson (1997) model.

Outputs:
  1. williamson_electron_muon.png  — main result showing both paths
  2. williamson_families.png       — grid of top candidates across p_e values
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Dark theme ──────────────────────────────────────────────────────────
BG_COLOR = '#0a0a12'
plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#888888',
    'ytick.color': '#888888',
    'font.family': 'monospace',
    'font.size': 10,
})

TARGET_RATIO = 206.7682843
DISPLAY_RHO = 0.22  # exaggerated for visibility


# ═══════════════════════════════════════════════════════════════════════
# Geometry primitives (shared with visualize.py)
# ═══════════════════════════════════════════════════════════════════════

def torus_surface(R, rho, n_theta=80, n_phi=200):
    """Generate torus mesh."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)
    x = (R + rho * np.cos(theta)) * np.cos(phi)
    y = (R + rho * np.cos(theta)) * np.sin(phi)
    z = rho * np.sin(theta)
    return x, y, z, theta, phi


def torus_knot_curve(p, q, rho, R=1.0, n_pts=12000, lift=1.04):
    """(p,q) torus knot slightly lifted above surface."""
    t = np.linspace(0, 2 * np.pi, n_pts)
    theta = p * t
    phi = q * t
    r = rho * lift
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z


def compute_shading(theta_grid, phi_grid, light_dir=None):
    """Phong-like pseudo-shading from surface normals."""
    if light_dir is None:
        light_dir = np.array([0.35, -0.45, 0.82])
    light_dir = light_dir / np.linalg.norm(light_dir)
    nx = np.cos(theta_grid) * np.cos(phi_grid)
    ny = np.cos(theta_grid) * np.sin(phi_grid)
    nz = np.sin(theta_grid)
    diffuse = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
    diffuse = np.clip((diffuse + 1) / 2, 0, 1)
    view_dir = np.array([0.0, -0.3, 0.95])
    view_dir = view_dir / np.linalg.norm(view_dir)
    halfway = light_dir + view_dir
    halfway = halfway / np.linalg.norm(halfway)
    spec = nx * halfway[0] + ny * halfway[1] + nz * halfway[2]
    spec = np.clip(spec, 0, 1) ** 32
    shade = 0.18 + 0.62 * diffuse + 0.20 * spec
    return shade


def make_torus_facecolors(shade, base_color=np.array([0.10, 0.10, 0.30]),
                          alpha=0.32):
    """Convert shade array to RGBA facecolors."""
    facecolors = np.zeros((*shade.shape, 4))
    for ch in range(3):
        facecolors[:, :, ch] = base_color[ch] * shade
    rim = np.clip(1.0 - shade, 0, 1) ** 2
    facecolors[:, :, 0] += 0.04 * rim
    facecolors[:, :, 1] += 0.02 * rim
    facecolors[:, :, 2] += 0.10 * rim
    facecolors[:, :, :3] = np.clip(facecolors[:, :, :3], 0, 1)
    facecolors[:, :, 3] = alpha
    return facecolors


def plot_colored_curve(ax, x, y, z, cmap_name='cool', cmap_range=(0.1, 0.9),
                       lw=0.9, alpha=0.95, glow=True, glow_lw_mult=3.5,
                       glow_alpha=0.08, seg_size=60, zorder=3):
    """Plot a 3D curve with color gradient and optional glow."""
    n = len(x)
    cmap = matplotlib.colormaps[cmap_name]
    if glow:
        for i in range(0, n - seg_size, seg_size):
            j = min(i + seg_size + 1, n)
            frac = i / n
            c = cmap(cmap_range[0] + (cmap_range[1] - cmap_range[0]) * frac)
            ax.plot(x[i:j], y[i:j], z[i:j],
                    color=c, linewidth=lw * glow_lw_mult,
                    alpha=glow_alpha, zorder=zorder - 1, solid_capstyle='round')
    for i in range(0, n - seg_size, seg_size):
        j = min(i + seg_size + 1, n)
        frac = i / n
        c = cmap(cmap_range[0] + (cmap_range[1] - cmap_range[0]) * frac)
        ax.plot(x[i:j], y[i:j], z[i:j],
                color=c, linewidth=lw, alpha=alpha,
                zorder=zorder, solid_capstyle='round')


def setup_3d_ax(fig, rect, elev=25, azim=-60):
    """Create a clean dark 3D axes."""
    ax = fig.add_axes(rect, projection='3d', computed_zorder=False)
    ax.set_facecolor(BG_COLOR)
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    return ax


# ═══════════════════════════════════════════════════════════════════════
# Load results
# ═══════════════════════════════════════════════════════════════════════

def load_williamson_results():
    """Load results from williamson_results.json."""
    try:
        with open('/home/jmoss/code/physics/williamson_results.json') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: williamson_results.json not found, using defaults")
        return [{
            "p_electron": 1, "p_muon": 15, "q_muon": 3,
            "rho": 0.05, "ratio": 206.768, "score": 1e-4,
            "l_electron": 12.56, "l_muon": 2597.0,
            "physical_r_major": 1.93e-13, "physical_r_tube": 9.65e-15,
            "model_charge_ratio": 0.91, "g_factor": 2.0019,
            "p_tau": 30, "q_tau": 5, "tau_ratio": 3477.0,
            "tau_score": 1.0,
        }]


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Electron + Muon on same torus
# ═══════════════════════════════════════════════════════════════════════

def render_electron_muon(results):
    """Main visualization: electron and muon paths on the same torus."""
    best = results[0]
    p_e = best['p_electron']
    p_mu = best['p_muon']
    q_mu = best['q_muon']

    fig = plt.figure(figsize=(18, 11), dpi=300)

    # ── Large 3D view ──
    ax = setup_3d_ax(fig, [0.0, 0.05, 0.60, 0.85], elev=24, azim=-55)

    R = 1.0
    rho = DISPLAY_RHO
    n_theta, n_phi = 80, 200

    # Surface
    x_t, y_t, z_t, theta_grid, phi_grid = torus_surface(R, rho, n_theta, n_phi)
    shade = compute_shading(theta_grid, phi_grid)
    fc = make_torus_facecolors(shade, alpha=0.28)
    ax.plot_surface(x_t, y_t, z_t, facecolors=fc, edgecolor='none',
                    rcount=n_theta, ccount=n_phi, antialiased=True, zorder=1)

    # Wireframe
    x_w, y_w, z_w, _, _ = torus_surface(R, rho, n_theta=20, n_phi=60)
    ax.plot_wireframe(x_w, y_w, z_w, color='#3a3a7a', linewidth=0.12,
                      alpha=0.18, rcount=20, ccount=60, zorder=2)

    # Electron path: gold (p_e, 2) double-loop
    x_e, y_e, z_e = torus_knot_curve(p_e, 2, rho, R, n_pts=8000, lift=1.06)
    # Glow
    ax.plot(x_e, y_e, z_e, color='#ffd966', linewidth=4.0,
            alpha=0.08, zorder=4, solid_capstyle='round')
    ax.plot(x_e, y_e, z_e, color='#ffc800', linewidth=2.0,
            alpha=0.20, zorder=5, solid_capstyle='round')
    # Core
    ax.plot(x_e, y_e, z_e, color='#ffdd44', linewidth=1.2,
            alpha=0.95, zorder=6, solid_capstyle='round')

    # Muon path: cyan-magenta gradient (p_mu, q_mu)
    x_m, y_m, z_m = torus_knot_curve(p_mu, q_mu, rho, R, n_pts=16000, lift=1.04)
    plot_colored_curve(ax, x_m, y_m, z_m, cmap_name='cool',
                       cmap_range=(0.12, 0.88), lw=0.7, alpha=0.92,
                       glow=True, glow_lw_mult=3.5, glow_alpha=0.08,
                       seg_size=80, zorder=3)

    # Axis limits
    lim = R + rho + 0.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    zlim = rho * 1.8
    ax.set_zlim(-zlim, zlim)
    ax.set_box_aspect([1, 1, rho * 2.2])

    # ── Title ──
    fig.text(0.30, 0.97,
             "Williamson/van der Mark  --  Electron & Muon on Same Torus",
             ha='center', fontsize=17, fontweight='bold', color='#ffffff')
    fig.text(0.30, 0.935,
             f"Electron = ({p_e}, 2) double-loop  |  "
             f"Muon = ({p_mu}, {q_mu})  |  "
             f"L_mu / L_e = {TARGET_RATIO}",
             ha='center', fontsize=10, color='#888888')

    # ── Legend ──
    fig.text(0.02, 0.04,
             f"Gold: electron ({p_e},2) double-loop   |   "
             f"Cyan-magenta: muon ({p_mu},{q_mu})   |   "
             "Torus tube exaggerated for visibility",
             fontsize=8, color='#555555')

    # ── Right panel: Physics summary ──
    x0, y0 = 0.63, 0.88
    dy = 0.028

    def info(text, color='#cccccc', size=9, bold=False):
        nonlocal y0
        weight = 'bold' if bold else 'normal'
        fig.text(x0, y0, text, fontsize=size, color=color,
                 fontweight=weight, fontfamily='monospace')
        y0 -= dy

    info("=== Williamson Model ===", '#00ddff', 12, True)
    info("")
    info(f"Electron mode:  ({p_e}, 2) double-loop", '#ffdd44', 10, True)
    info(f"Muon mode:      ({p_mu}, {q_mu})", '#00ddff', 10, True)
    info("")
    info(f"Torus rho =     {best['rho']:.12f}")
    info(f"L_electron =    {best['l_electron']:.8f}")
    info(f"L_muon =        {best['l_muon']:.8f}")
    info(f"Ratio L_mu/L_e= {best['ratio']:.12f}")
    info(f"Score =         {best['score']:.6e}")
    info("")
    info("=== Physical Dimensions ===", '#aaaaaa', 10, True)
    info(f"Major radius R ={best['physical_r_major']:.4e} m")
    info(f"Tube radius r = {best['physical_r_tube']:.4e} m")
    info("")
    info("=== Williamson Constants ===", '#aaaaaa', 10, True)
    info(f"Charge q/e =    {best['model_charge_ratio']:.6f}  (paper: ~0.91)")
    info(f"g-factor =      {best['g_factor']:.6f}  (QED: 2.002319)")
    info("")

    if best.get('tau_score', 999) < 100:
        info("=== Tau Search ===", '#aaaaaa', 10, True)
        info(f"Tau mode:       ({best['p_tau']}, {best['q_tau']})")
        info(f"Tau ratio:      {best['tau_ratio']:.6f}")
        info(f"Tau score:      {best['tau_score']:.6e}")

    fig.savefig('/home/jmoss/code/physics/williamson_electron_muon.png',
                dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    print("  Saved: williamson_electron_muon.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Families grid
# ═══════════════════════════════════════════════════════════════════════

def render_families(results):
    """Grid showing diverse candidates across different rho values."""
    # Pick candidates that span different rho ranges for visual diversity
    sorted_by_rho = sorted(results, key=lambda r: r['rho'])
    seen_bins = set()
    family_list = []
    for r in sorted_by_rho:
        rho_bin = round(r['rho'], 1)
        if rho_bin not in seen_bins:
            seen_bins.add(rho_bin)
            family_list.append(r)
    family_list = family_list[:6]
    n = len(family_list)
    if n == 0:
        print("  No families to render")
        return

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(7 * cols, 7 * rows + 1), dpi=200)

    fig.text(0.5, 0.98,
             "Williamson Solutions  --  Same Mass Ratio at Different Torus Shapes",
             ha='center', fontsize=18, fontweight='bold', color='#ffffff')

    for i, r in enumerate(family_list):
        row = i // cols
        col = i % cols
        x0 = col / cols + 0.02
        y0 = 1.0 - (row + 1) / rows * 0.92 + 0.02
        w = 1.0 / cols - 0.04
        h = 0.92 / rows - 0.06

        ax = setup_3d_ax(fig, [x0, y0, w, h], elev=26, azim=-55 + i * 8)

        R = 1.0
        rho = r['rho']  # use actual rho for this candidate
        p_e = r['p_electron']
        p_mu = r['p_muon']
        q_mu = r['q_muon']

        # Surface
        x_t, y_t, z_t, tg, pg = torus_surface(R, rho, 60, 150)
        shade = compute_shading(tg, pg)
        fc = make_torus_facecolors(shade, alpha=0.25)
        ax.plot_surface(x_t, y_t, z_t, facecolors=fc, edgecolor='none',
                        rcount=60, ccount=150, antialiased=True, zorder=1)

        # Electron
        x_e, y_e, z_e = torus_knot_curve(p_e, 2, rho, R, n_pts=6000, lift=1.06)
        ax.plot(x_e, y_e, z_e, color='#ffdd44', linewidth=1.5,
                alpha=0.90, zorder=5, solid_capstyle='round')

        # Muon — cap n_pts to avoid very slow rendering for high winding
        mu_pts = min(q_mu * 40, 16000)
        x_m, y_m, z_m = torus_knot_curve(p_mu, q_mu, rho, R, n_pts=mu_pts, lift=1.04)
        plot_colored_curve(ax, x_m, y_m, z_m, cmap_name='cool',
                           cmap_range=(0.15, 0.85), lw=0.6, alpha=0.90,
                           glow=True, glow_lw_mult=3.0, glow_alpha=0.06,
                           seg_size=70, zorder=3)

        lim = R + rho + 0.12
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        zlim = max(rho * 1.8, 0.15)
        ax.set_zlim(-zlim, zlim)
        ax.set_box_aspect([1, 1, max(rho * 2.2, 0.3)])

        # Labels
        ax.text2D(0.5, 0.06,
                  f"e=({p_e},2)  mu=({p_mu},{q_mu})",
                  transform=ax.transAxes, ha='center',
                  fontsize=11, fontweight='bold', color='#00ddff')
        ax.text2D(0.5, 0.00,
                  f"\u03c1={r['rho']:.4f}  score={r['score']:.2e}",
                  transform=ax.transAxes, ha='center',
                  fontsize=8, color='#888888')

    fig.savefig('/home/jmoss/code/physics/williamson_families.png',
                dpi=200, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    print("  Saved: williamson_families.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading Williamson results...")
    results = load_williamson_results()
    print(f"  Loaded {len(results)} candidates\n")

    print("[1/2] Electron + muon on same torus...")
    render_electron_muon(results)

    print("[2/2] Families grid...")
    render_families(results)

    print("\nDone! Images saved in /home/jmoss/code/physics/:")
    print("  williamson_electron_muon.png  -- main result")
    print("  williamson_families.png       -- families grid")
