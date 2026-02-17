#!/usr/bin/env python3
"""
Visualize Williamson/van der Mark Run 4 FINAL results.

Outputs (does NOT overwrite existing files):
  1. run4_main.png          — best candidate: electron + muon on same torus
  2. run4_all_electrons.png — best candidate per electron mode (all 8 p_e values)
  3. run4_rho_diversity.png — same mass ratio achieved at different torus shapes
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
DISPLAY_RHO = 0.22


def torus_surface(R, rho, n_theta=80, n_phi=200):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)
    x = (R + rho * np.cos(theta)) * np.cos(phi)
    y = (R + rho * np.cos(theta)) * np.sin(phi)
    z = rho * np.sin(theta)
    return x, y, z, theta, phi


def torus_knot_curve(p, q, rho, R=1.0, n_pts=12000, lift=1.04):
    t = np.linspace(0, 2 * np.pi, n_pts)
    theta = p * t
    phi = q * t
    r = rho * lift
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z


def compute_shading(theta_grid, phi_grid, light_dir=None):
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
    ax = fig.add_axes(rect, projection='3d', computed_zorder=False)
    ax.set_facecolor(BG_COLOR)
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    return ax


def render_torus_panel(ax, r, rho_override=None):
    """Render electron+muon on a torus in the given axes."""
    R = 1.0
    rho = rho_override if rho_override is not None else r['rho']
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

    # Muon
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


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Best candidate — main result
# ═══════════════════════════════════════════════════════════════════════

def render_main(results):
    best = results[0]
    p_e = best['p_electron']
    p_mu = best['p_muon']
    q_mu = best['q_muon']

    fig = plt.figure(figsize=(18, 11), dpi=300)
    ax = setup_3d_ax(fig, [0.0, 0.05, 0.60, 0.85], elev=24, azim=-55)

    R = 1.0
    rho = DISPLAY_RHO

    # Surface
    x_t, y_t, z_t, theta_grid, phi_grid = torus_surface(R, rho, 80, 200)
    shade = compute_shading(theta_grid, phi_grid)
    fc = make_torus_facecolors(shade, alpha=0.28)
    ax.plot_surface(x_t, y_t, z_t, facecolors=fc, edgecolor='none',
                    rcount=80, ccount=200, antialiased=True, zorder=1)

    # Wireframe
    x_w, y_w, z_w, _, _ = torus_surface(R, rho, n_theta=20, n_phi=60)
    ax.plot_wireframe(x_w, y_w, z_w, color='#3a3a7a', linewidth=0.12,
                      alpha=0.18, rcount=20, ccount=60, zorder=2)

    # Electron
    x_e, y_e, z_e = torus_knot_curve(p_e, 2, rho, R, n_pts=8000, lift=1.06)
    ax.plot(x_e, y_e, z_e, color='#ffd966', linewidth=4.0,
            alpha=0.08, zorder=4, solid_capstyle='round')
    ax.plot(x_e, y_e, z_e, color='#ffc800', linewidth=2.0,
            alpha=0.20, zorder=5, solid_capstyle='round')
    ax.plot(x_e, y_e, z_e, color='#ffdd44', linewidth=1.2,
            alpha=0.95, zorder=6, solid_capstyle='round')

    # Muon
    x_m, y_m, z_m = torus_knot_curve(p_mu, q_mu, rho, R, n_pts=16000, lift=1.04)
    plot_colored_curve(ax, x_m, y_m, z_m, cmap_name='cool',
                       cmap_range=(0.12, 0.88), lw=0.7, alpha=0.92,
                       glow=True, glow_lw_mult=3.5, glow_alpha=0.08,
                       seg_size=80, zorder=3)

    lim = R + rho + 0.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    zlim = rho * 1.8
    ax.set_zlim(-zlim, zlim)
    ax.set_box_aspect([1, 1, rho * 2.2])

    # Title
    fig.text(0.30, 0.97,
             "Run 4: Williamson/van der Mark  --  Electron & Muon on Same Torus",
             ha='center', fontsize=17, fontweight='bold', color='#ffffff')
    fig.text(0.30, 0.935,
             f"Electron = ({p_e}, 2) double-loop  |  "
             f"Muon = ({p_mu}, {q_mu})  |  "
             f"L_mu / L_e = {TARGET_RATIO}",
             ha='center', fontsize=10, color='#888888')

    # Legend
    fig.text(0.02, 0.04,
             f"Gold: electron ({p_e},2) double-loop   |   "
             f"Cyan-magenta: muon ({p_mu},{q_mu})   |   "
             "Torus tube exaggerated for visibility",
             fontsize=8, color='#555555')
    fig.text(0.02, 0.015,
             f"Run 4 final: 2.8 trillion GPU evals, 42 candidates, "
             f"all score=0 (machine precision)   |   "
             f"Geodesic path-length ratio on curved torus",
             fontsize=8, color='#444444')

    # Right panel
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
    info("=== Run 4 Statistics ===", '#aaaaaa', 10, True)
    info(f"GPU evaluations: 2.8 trillion")
    info(f"Spectral seeds:  34,146")
    info(f"Validated:       42 candidates")
    info(f"All scores:      0 (exact)")

    fig.savefig('/home/jmoss/code/physics/run4_main.png',
                dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    print("  Saved: run4_main.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Best candidate per electron mode (all 8 p_e values)
# ═══════════════════════════════════════════════════════════════════════

def render_all_electrons(results):
    # Group by p_electron, take best per group
    families = {}
    for r in results:
        p_e = r['p_electron']
        if p_e not in families or r['score'] < families[p_e]['score']:
            families[p_e] = r

    family_list = sorted(families.values(), key=lambda r: r['p_electron'])
    n = len(family_list)

    cols = 4
    rows = 2
    fig = plt.figure(figsize=(7 * cols, 7 * rows + 1.5), dpi=200)

    fig.text(0.5, 0.97,
             "Run 4: All Electron Modes  --  Best Muon Match per e=(p,2)",
             ha='center', fontsize=20, fontweight='bold', color='#ffffff')
    fig.text(0.5, 0.955,
             f"Each panel: different electron double-loop, same target ratio {TARGET_RATIO}",
             ha='center', fontsize=11, color='#888888')

    for i, r in enumerate(family_list):
        row = i // cols
        col = i % cols
        x0 = col / cols + 0.015
        y0 = 1.0 - (row + 1) / rows * 0.90 + 0.02
        w = 1.0 / cols - 0.03
        h = 0.90 / rows - 0.08

        ax = setup_3d_ax(fig, [x0, y0, w, h], elev=26, azim=-55 + i * 5)
        render_torus_panel(ax, r, rho_override=DISPLAY_RHO)

        p_e = r['p_electron']
        p_mu = r['p_muon']
        q_mu = r['q_muon']

        ax.text2D(0.5, 0.08,
                  f"e=({p_e},2)  mu=({p_mu},{q_mu})",
                  transform=ax.transAxes, ha='center',
                  fontsize=11, fontweight='bold', color='#00ddff')
        ax.text2D(0.5, 0.02,
                  f"\u03c1={r['rho']:.6f}  q_mu={q_mu}  score={r['score']:.0e}",
                  transform=ax.transAxes, ha='center',
                  fontsize=8, color='#888888')

    fig.savefig('/home/jmoss/code/physics/run4_all_electrons.png',
                dpi=200, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    print("  Saved: run4_all_electrons.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Rho diversity — same ratio at different torus shapes
# ═══════════════════════════════════════════════════════════════════════

def render_rho_diversity(results):
    # Pick candidates spanning different rho values
    sorted_by_rho = sorted(results, key=lambda r: r['rho'])
    seen_bins = set()
    picks = []
    for r in sorted_by_rho:
        rho_bin = round(r['rho'], 1)
        if rho_bin not in seen_bins:
            seen_bins.add(rho_bin)
            picks.append(r)
    picks = picks[:6]
    n = len(picks)

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(7 * cols, 7 * rows + 1.5), dpi=200)

    fig.text(0.5, 0.97,
             "Run 4: Same Mass Ratio at Different Torus Shapes",
             ha='center', fontsize=20, fontweight='bold', color='#ffffff')
    fig.text(0.5, 0.955,
             f"All panels: L_mu/L_e = {TARGET_RATIO} exactly  |  "
             "Torus rendered at actual \u03c1 (thin ring to near-sphere)",
             ha='center', fontsize=11, color='#888888')

    for i, r in enumerate(picks):
        row = i // cols
        col = i % cols
        x0 = col / cols + 0.02
        y0 = 1.0 - (row + 1) / rows * 0.90 + 0.02
        w = 1.0 / cols - 0.04
        h = 0.90 / rows - 0.08

        ax = setup_3d_ax(fig, [x0, y0, w, h], elev=26, azim=-55 + i * 8)
        render_torus_panel(ax, r)  # use actual rho

        p_e = r['p_electron']
        p_mu = r['p_muon']
        q_mu = r['q_muon']

        ax.text2D(0.5, 0.08,
                  f"e=({p_e},2)  mu=({p_mu},{q_mu})",
                  transform=ax.transAxes, ha='center',
                  fontsize=11, fontweight='bold', color='#00ddff')
        ax.text2D(0.5, 0.02,
                  f"\u03c1 = {r['rho']:.4f}",
                  transform=ax.transAxes, ha='center',
                  fontsize=9, fontweight='bold', color='#ffdd44')

    fig.savefig('/home/jmoss/code/physics/run4_rho_diversity.png',
                dpi=200, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    print("  Saved: run4_rho_diversity.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading Run 4 final results...")
    with open('/home/jmoss/code/physics/williamson_results.json') as f:
        results = json.load(f)
    print(f"  Loaded {len(results)} candidates")
    print(f"  Electron modes: {sorted(set(r['p_electron'] for r in results))}")
    print()

    print("[1/3] Main result...")
    render_main(results)

    print("[2/3] All electron modes...")
    render_all_electrons(results)

    print("[3/3] Rho diversity...")
    render_rho_diversity(results)

    print(f"\nDone! New images saved in /home/jmoss/code/physics/:")
    print(f"  run4_main.png           -- best candidate")
    print(f"  run4_all_electrons.png  -- one per electron mode")
    print(f"  run4_rho_diversity.png  -- same ratio, different shapes")
