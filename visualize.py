#!/usr/bin/env python3
"""
Visualize the toroidal electron geometries from the resonance search.

Produces publication-quality 3D renders of the top 3 torus-knot candidates
plus 3 alternative topological surfaces that could confine photons.

Based on Williamson & van der Mark (1997): the electron as a
Compton-wavelength photon confined in a toroidal double-helix.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# ── Dark theme ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0a0a12',
    'axes.facecolor': '#0a0a12',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#888888',
    'ytick.color': '#888888',
    'font.family': 'monospace',
    'font.size': 10,
})

# ── The three best-fit geometries from Run 2 ───────────────────────────
GEOMETRIES = [
    {"p": 21, "q": 10, "rho": 0.048614995, "score": "1.0e-8",
     "label": "Best CPU-verified  (0.05 ppb)", "rank": 1},
    {"p": 22, "q": 13, "rho": 0.063231962, "score": "6.8e-8",
     "label": "Second family", "rank": 2},
    {"p": 29, "q": 18, "rho": 0.087926397, "score": "6.9e-8",
     "label": "Third family", "rank": 3},
]

TARGET_RATIO = 206.7682843

# Visual exaggeration: the real rho values (0.05-0.09) make the tube
# invisible at plot scale.  We render with display_rho for visual clarity,
# but label with the true mathematical value.
DISPLAY_RHO = 0.22


def torus_surface(R, rho, n_theta=100, n_phi=300):
    """Generate torus mesh with shading-friendly resolution."""
    theta = np.linspace(0, 2*np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)
    x = (R + rho * np.cos(theta)) * np.cos(phi)
    y = (R + rho * np.cos(theta)) * np.sin(phi)
    z = rho * np.sin(theta)
    return x, y, z, theta


def torus_knot_curve(p, q, rho, R=1.0, n_pts=10000, lift=1.03):
    """(p,q) torus knot on torus surface, slightly lifted for visibility."""
    t = np.linspace(0, 2*np.pi, n_pts)
    theta = p * t
    phi = q * t
    r = rho * lift
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z


def electron_loop(rho, R=1.0, n_pts=2000, lift=1.06):
    """(1,0) mode: single poloidal loop — the electron ground state."""
    t = np.linspace(0, 2*np.pi, n_pts)
    r = rho * lift
    x = (R + r * np.cos(t)) * np.ones_like(t)
    y = np.zeros_like(t)
    z = r * np.sin(t)
    return x, y, z


def setup_ax(fig, pos, elev=25, azim=-60):
    """Create dark 3D axes."""
    ax = fig.add_subplot(*pos, projection='3d', computed_zorder=False)
    ax.set_facecolor('#0a0a12')
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    return ax


def plot_torus(ax, geom, show_electron=True, display_rho=DISPLAY_RHO,
               knot_lw=0.9, elec_lw=3.0):
    """Render torus surface + knot + electron loop."""
    p, q = geom["p"], geom["q"]
    R = 1.0
    rho = display_rho

    # ── Torus surface with shading ──
    x_t, y_t, z_t, theta_grid = torus_surface(R, rho, n_theta=80, n_phi=200)

    # Compute pseudo-shading via surface normal dot light direction
    # Light from upper-left-front
    light_dir = np.array([0.3, -0.5, 0.8])
    light_dir /= np.linalg.norm(light_dir)

    # Surface normals (outward from torus center)
    phi_grid = np.linspace(0, 2*np.pi, 200)
    _, phi_2d = np.meshgrid(np.linspace(0, 2*np.pi, 80), phi_grid)
    nx = np.cos(theta_grid) * np.cos(phi_2d)
    ny = np.cos(theta_grid) * np.sin(phi_2d)
    nz = np.sin(theta_grid)

    shade = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
    shade = 0.3 + 0.7 * np.clip((shade + 1) / 2, 0, 1)

    # Map to RGBA
    base_color = np.array([0.12, 0.12, 0.28])
    facecolors = np.zeros((*shade.shape, 4))
    for i in range(shade.shape[0]):
        for j in range(shade.shape[1]):
            s = shade[i, j]
            facecolors[i, j, :3] = base_color * s
            facecolors[i, j, 3] = 0.35

    ax.plot_surface(x_t, y_t, z_t,
                    facecolors=facecolors,
                    edgecolor='none',
                    rcount=80, ccount=200,
                    antialiased=True, zorder=1)

    # Subtle wireframe on top for tube definition
    x_w, y_w, z_w, _ = torus_surface(R, rho, n_theta=24, n_phi=80)
    ax.plot_wireframe(x_w, y_w, z_w,
                      color='#3a3a6a', linewidth=0.15, alpha=0.25,
                      rcount=24, ccount=80, zorder=2)

    # ── Muon mode: (p,q) torus knot ──
    x_k, y_k, z_k = torus_knot_curve(p, q, rho, R)
    n = len(x_k)

    # Color gradient along the knot (cyan → magenta via cool colormap)
    seg = 80
    for i in range(0, n - seg, seg):
        j = min(i + seg + 1, n)
        frac = i / n
        c = cm.cool(0.15 + 0.75 * frac)
        ax.plot(x_k[i:j], y_k[i:j], z_k[i:j],
                color=c, linewidth=knot_lw, alpha=0.92, zorder=3)

    # ── Electron mode: (1,0) poloidal loop — gold ──
    if show_electron:
        x_e, y_e, z_e = electron_loop(rho, R)
        ax.plot(x_e, y_e, z_e,
                color='#ffcc00', linewidth=elec_lw, alpha=0.95, zorder=4)
        # Add a glow effect with wider, more transparent line
        ax.plot(x_e, y_e, z_e,
                color='#ffcc00', linewidth=elec_lw * 2.5, alpha=0.15, zorder=3)

    # Axis limits
    lim = R + rho + 0.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    zlim = rho * 1.8
    ax.set_zlim(-zlim, zlim)
    ax.set_box_aspect([1, 1, rho * 2.2])


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: The three torus-knot electron candidates (side by side)
# ═══════════════════════════════════════════════════════════════════════

def render_torus_candidates():
    fig = plt.figure(figsize=(21, 9), dpi=200)

    fig.text(0.5, 0.97,
             "The Shape of the Electron",
             ha='center', fontsize=20, fontweight='bold', color='#ffffff')
    fig.text(0.5, 0.935,
             "Toroidal Resonance Candidates — Williamson & van der Mark (1997)",
             ha='center', fontsize=11, color='#888888')
    fig.text(0.5, 0.905,
             f"Target: m_μ / m_e = {TARGET_RATIO}  ±  0.0000052  (25 ppb)",
             ha='center', fontsize=10, color='#666666')

    for i, geom in enumerate(GEOMETRIES):
        ax = setup_ax(fig, (1, 3, i+1), elev=30, azim=-60 + i*12)
        plot_torus(ax, geom)

        p, q, rho = geom["p"], geom["q"], geom["rho"]

        ax.text2D(0.5, 0.10,
                  f"({p},{q}) / (1,0)",
                  transform=ax.transAxes, ha='center',
                  fontsize=14, fontweight='bold', color='#00ccff')
        ax.text2D(0.5, 0.04,
                  f"ρ = {rho:.6f}    score = {geom['score']}",
                  transform=ax.transAxes, ha='center',
                  fontsize=9, color='#888888')
        ax.text2D(0.5, -0.01,
                  geom['label'],
                  transform=ax.transAxes, ha='center',
                  fontsize=9, color='#666666', style='italic')

    fig.text(0.5, 0.01,
             "Cyan–magenta: muon mode (p,q) torus knot   ·   Gold: electron ground state (1,0) poloidal loop"
             "   ·   Torus tube exaggerated for visibility",
             ha='center', fontsize=8, color='#555555')

    plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.05,
                        wspace=0.02)
    fig.savefig('/home/jmoss/code/physics/electron_torus_candidates.png',
                dpi=200, bbox_inches='tight',
                facecolor='#0a0a12', edgecolor='none')
    print("  Saved: electron_torus_candidates.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Hero view of the #1 candidate
# ═══════════════════════════════════════════════════════════════════════

def render_hero_view():
    fig = plt.figure(figsize=(16, 10), dpi=200)
    geom = GEOMETRIES[0]
    p, q, rho = geom["p"], geom["q"], geom["rho"]

    # Large main 3D view (left 65%)
    ax1 = fig.add_axes([0.0, 0.05, 0.65, 0.85], projection='3d',
                       computed_zorder=False)
    ax1.set_facecolor('#0a0a12')
    ax1.grid(False)
    ax1.set_axis_off()
    ax1.view_init(elev=22, azim=-55)
    plot_torus(ax1, geom, knot_lw=1.0, elec_lw=3.5)

    fig.text(0.325, 0.96,
             "The Electron as a Confined Photon",
             ha='center', fontsize=18, fontweight='bold', color='#ffffff')
    fig.text(0.325, 0.925,
             "Williamson & van der Mark (1997) toroidal topology",
             ha='center', fontsize=10, color='#888888')

    # Top view (upper right)
    ax2 = fig.add_axes([0.62, 0.52, 0.37, 0.40], projection='3d',
                       computed_zorder=False)
    ax2.set_facecolor('#0a0a12')
    ax2.grid(False)
    ax2.set_axis_off()
    ax2.view_init(elev=88, azim=0)
    plot_torus(ax2, geom, show_electron=False, knot_lw=0.6)
    ax2.text2D(0.5, 0.92, "Top view", transform=ax2.transAxes,
               ha='center', fontsize=9, color='#666666')

    # Side view (lower right)
    ax3 = fig.add_axes([0.62, 0.08, 0.37, 0.40], projection='3d',
                       computed_zorder=False)
    ax3.set_facecolor('#0a0a12')
    ax3.grid(False)
    ax3.set_axis_off()
    ax3.view_init(elev=2, azim=0)
    plot_torus(ax3, geom, show_electron=False, knot_lw=0.6)
    ax3.text2D(0.5, 0.92, "Side view", transform=ax3.transAxes,
               ha='center', fontsize=9, color='#666666')

    # Info text (bottom left)
    lines = [
        (f"Best-fit geometry:  ({p},{q}) / (1,0)", '#00ccff', 'bold'),
        (f"Aspect ratio  ρ = r/R = {rho:.9f}", '#cccccc', 'normal'),
        ("", '#0a0a12', 'normal'),
        (f"Muon mode:     {p} poloidal × {q} toroidal windings", '#cccccc', 'normal'),
        ("Electron mode:  single poloidal loop (1,0)", '#cccccc', 'normal'),
        ("", '#0a0a12', 'normal'),
        (f"Path length ratio = {TARGET_RATIO}", '#ffcc00', 'bold'),
        (f"Residual error:  {geom['score']}  (0.05 ppb)", '#cccccc', 'normal'),
        ("", '#0a0a12', 'normal'),
        ("110 billion evaluations  ·  CUDA + Rust", '#888888', 'normal'),
        ("GPU: 64-pt Gauss-Legendre  |  CPU: 10,000-pt f64", '#888888', 'normal'),
    ]
    for j, (text, color, weight) in enumerate(lines):
        fig.text(0.02, 0.17 - j * 0.022, text,
                 fontsize=8.5, color=color, fontweight=weight,
                 fontfamily='monospace')

    fig.savefig('/home/jmoss/code/physics/electron_hero.png',
                dpi=200, bbox_inches='tight',
                facecolor='#0a0a12', edgecolor='none')
    print("  Saved: electron_hero.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Alternative topological surfaces
# ═══════════════════════════════════════════════════════════════════════

def trefoil_knot_tube(n_pts=2000, tube_radius=0.22, tube_res=30):
    """Trefoil knot as a tube — a (2,3) torus knot in 3-space."""
    t = np.linspace(0, 2*np.pi, n_pts)
    x = np.sin(t) + 2*np.sin(2*t)
    y = np.cos(t) - 2*np.cos(2*t)
    z = -np.sin(3*t)

    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    dz = np.gradient(z, t)

    mag = np.sqrt(dx**2 + dy**2 + dz**2)
    tx_, ty_, tz_ = dx/mag, dy/mag, dz/mag

    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)
    ddz = np.gradient(dz, t)

    bx = ty_*ddz - tz_*ddy
    by = tz_*ddx - tx_*ddz
    bz = tx_*ddy - ty_*ddx
    bmag = np.sqrt(bx**2 + by**2 + bz**2) + 1e-10
    bx, by, bz = bx/bmag, by/bmag, bz/bmag

    nx = by*tz_ - bz*ty_
    ny = bz*tx_ - bx*tz_
    nz = bx*ty_ - by*tx_

    theta = np.linspace(0, 2*np.pi, tube_res)
    X = np.zeros((n_pts, tube_res))
    Y = np.zeros((n_pts, tube_res))
    Z = np.zeros((n_pts, tube_res))

    for i in range(n_pts):
        for j in range(tube_res):
            X[i, j] = x[i] + tube_radius * (nx[i]*np.cos(theta[j]) + bx[i]*np.sin(theta[j]))
            Y[i, j] = y[i] + tube_radius * (ny[i]*np.cos(theta[j]) + by[i]*np.sin(theta[j]))
            Z[i, j] = z[i] + tube_radius * (nz[i]*np.cos(theta[j]) + bz[i]*np.sin(theta[j]))

    return X, Y, Z, x, y, z


def klein_bottle_surface(n=200):
    """Klein bottle immersion in 3D (figure-8 parametrization)."""
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, 2*np.pi, n)
    u, v = np.meshgrid(u, v)
    r = 2.0
    x = (r + np.cos(u/2)*np.sin(v) - np.sin(u/2)*np.sin(2*v)) * np.cos(u)
    y = (r + np.cos(u/2)*np.sin(v) - np.sin(u/2)*np.sin(2*v)) * np.sin(u)
    z = np.sin(u/2)*np.sin(v) + np.cos(u/2)*np.sin(2*v)
    return x, y, z


def hopf_fibration_fibers(n_fibers=40, n_pts=500):
    """Fibers of the Hopf fibration S^3 -> S^2 via stereographic projection."""
    fibers = []
    for i in range(n_fibers):
        theta_base = np.pi * (i + 0.5) / n_fibers
        phi_base = 2 * np.pi * (i * 0.618033988)

        t = np.linspace(0, 2*np.pi, n_pts)
        ct = np.cos(theta_base / 2)
        st = np.sin(theta_base / 2)

        x1 = ct * np.cos(t)
        x2 = ct * np.sin(t)
        x3 = st * np.cos(t + phi_base)
        x4 = st * np.sin(t + phi_base)

        denom = 1 - x4 + 1e-10
        px = 0.8 * x1 / denom
        py = 0.8 * x2 / denom
        pz = 0.8 * x3 / denom

        fibers.append((px, py, pz, theta_base / np.pi))
    return fibers


def render_alternative_topologies():
    fig = plt.figure(figsize=(21, 9), dpi=200)

    fig.text(0.5, 0.97,
             "Alternative Topological Confinement Geometries",
             ha='center', fontsize=20, fontweight='bold', color='#ffffff')
    fig.text(0.5, 0.935,
             "Beyond the torus: other surfaces that could confine photonic modes",
             ha='center', fontsize=11, color='#888888')

    # ── 1. Trefoil knot tube ──
    ax1 = setup_ax(fig, (1, 3, 1), elev=25, azim=-40)

    X, Y, Z, cx, cy, cz = trefoil_knot_tube(n_pts=1500, tube_radius=0.22, tube_res=25)
    n_pts_t = X.shape[0]

    # Color the tube with a gradient + pseudo-shading
    facecolors = np.zeros((*X.shape, 4))
    for i in range(n_pts_t):
        frac = i / n_pts_t
        c = cm.cool(0.15 + 0.75 * frac)
        for j in range(X.shape[1]):
            # Simple depth-based shading
            depth = 0.6 + 0.4 * (Z[i, j] - Z.min()) / (Z.max() - Z.min() + 1e-10)
            facecolors[i, j] = [c[0]*depth, c[1]*depth, c[2]*depth, 0.7]

    ax1.plot_surface(X, Y, Z, facecolors=facecolors,
                     edgecolor='none', antialiased=True,
                     rcount=150, ccount=25, zorder=2)

    ax1.plot(cx, cy, cz, color='#ffffff', linewidth=0.4, alpha=0.3, zorder=3)

    ax1.set_xlim(-4, 4); ax1.set_ylim(-4, 4); ax1.set_zlim(-2, 2)
    ax1.set_box_aspect([1, 1, 0.5])

    ax1.text2D(0.5, 0.10, "Trefoil Knot Tube",
               transform=ax1.transAxes, ha='center',
               fontsize=14, fontweight='bold', color='#00ccff')
    ax1.text2D(0.5, 0.04,
               "(2,3) torus knot — simplest non-trivial knot",
               transform=ax1.transAxes, ha='center',
               fontsize=9, color='#888888')
    ax1.text2D(0.5, -0.01,
               "Thickened knot complement as confinement cavity",
               transform=ax1.transAxes, ha='center',
               fontsize=8, color='#555555', style='italic')

    # ── 2. Klein bottle ──
    ax2 = setup_ax(fig, (1, 3, 2), elev=22, azim=-50)

    x_kb, y_kb, z_kb = klein_bottle_surface(n=150)
    norm_z = (z_kb - z_kb.min()) / (z_kb.max() - z_kb.min() + 1e-10)
    fc_kb = cm.magma(norm_z * 0.65 + 0.2)
    fc_kb[:, :, 3] = 0.55

    ax2.plot_surface(x_kb, y_kb, z_kb,
                     facecolors=fc_kb,
                     edgecolor='#2a1030', linewidth=0.03,
                     rcount=80, ccount=80,
                     antialiased=True, zorder=2)

    ax2.set_xlim(-4, 4); ax2.set_ylim(-4, 4); ax2.set_zlim(-2.5, 2.5)
    ax2.set_box_aspect([1, 1, 0.6])

    ax2.text2D(0.5, 0.10, "Klein Bottle",
               transform=ax2.transAxes, ha='center',
               fontsize=14, fontweight='bold', color='#ff66cc')
    ax2.text2D(0.5, 0.04,
               "Non-orientable surface — single-sided confinement",
               transform=ax2.transAxes, ha='center',
               fontsize=9, color='#888888')
    ax2.text2D(0.5, -0.01,
               "Charge chirality from topological twist",
               transform=ax2.transAxes, ha='center',
               fontsize=8, color='#555555', style='italic')

    # ── 3. Hopf fibration ──
    ax3 = setup_ax(fig, (1, 3, 3), elev=25, azim=-35)

    fibers = hopf_fibration_fibers(n_fibers=50, n_pts=500)
    for fx, fy, fz, param in fibers:
        c = cm.twilight(param)
        ax3.plot(fx, fy, fz, color=c, linewidth=0.45, alpha=0.75, zorder=2)

    ax3.set_xlim(-3, 3); ax3.set_ylim(-3, 3); ax3.set_zlim(-3, 3)
    ax3.set_box_aspect([1, 1, 1])

    ax3.text2D(0.5, 0.10, "Hopf Fibration",
               transform=ax3.transAxes, ha='center',
               fontsize=14, fontweight='bold', color='#cc88ff')
    ax3.text2D(0.5, 0.04,
               "S³ → S² fiber bundle — interlocking circles",
               transform=ax3.transAxes, ha='center',
               fontsize=9, color='#888888')
    ax3.text2D(0.5, -0.01,
               "Every fiber links every other fiber exactly once",
               transform=ax3.transAxes, ha='center',
               fontsize=8, color='#555555', style='italic')

    fig.text(0.5, 0.01,
             "Each topology supports distinct photon confinement modes with quantized path-length spectra",
             ha='center', fontsize=8, color='#555555')

    plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.05,
                        wspace=0.02)
    fig.savefig('/home/jmoss/code/physics/alternative_topologies.png',
                dpi=200, bbox_inches='tight',
                facecolor='#0a0a12', edgecolor='none')
    print("  Saved: alternative_topologies.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: All 6 shapes — complete gallery
# ═══════════════════════════════════════════════════════════════════════

def render_gallery():
    fig = plt.figure(figsize=(21, 14), dpi=200)

    fig.text(0.5, 0.985,
             "Topological Photon Confinement — Complete Gallery",
             ha='center', fontsize=20, fontweight='bold', color='#ffffff')

    # Row 1 header
    fig.text(0.5, 0.955,
             "Torus knot candidates reproducing m_μ/m_e = 206.7682843",
             ha='center', fontsize=10, color='#00ccff')

    for i, geom in enumerate(GEOMETRIES):
        ax = setup_ax(fig, (2, 3, i+1), elev=28, azim=-55 + i*10)
        plot_torus(ax, geom)
        p, q = geom["p"], geom["q"]
        ax.text2D(0.5, 0.05,
                  f"({p},{q}) / (1,0)   ρ={geom['rho']:.4f}",
                  transform=ax.transAxes, ha='center',
                  fontsize=10, fontweight='bold', color='#00ccff')
        ax.text2D(0.5, -0.01,
                  f"score = {geom['score']}",
                  transform=ax.transAxes, ha='center',
                  fontsize=8, color='#666666')

    # Row 2 header
    fig.text(0.5, 0.485,
             "Alternative topological confinement surfaces",
             ha='center', fontsize=10, color='#ff66cc')

    # Trefoil
    ax4 = setup_ax(fig, (2, 3, 4), elev=25, azim=-40)
    X, Y, Z, cx, cy, cz = trefoil_knot_tube(n_pts=1200, tube_radius=0.22, tube_res=20)
    n_pts_t = X.shape[0]
    fc = np.zeros((*X.shape, 4))
    for i in range(n_pts_t):
        c = cm.cool(0.15 + 0.75 * i / n_pts_t)
        fc[i, :] = [c[0], c[1], c[2], 0.65]
    ax4.plot_surface(X, Y, Z, facecolors=fc, edgecolor='none',
                     rcount=120, ccount=20, antialiased=True)
    ax4.set_xlim(-4, 4); ax4.set_ylim(-4, 4); ax4.set_zlim(-2, 2)
    ax4.set_box_aspect([1, 1, 0.5])
    ax4.text2D(0.5, 0.05, "Trefoil Knot Tube",
               transform=ax4.transAxes, ha='center',
               fontsize=10, fontweight='bold', color='#00ccff')

    # Klein
    ax5 = setup_ax(fig, (2, 3, 5), elev=22, azim=-50)
    x_kb, y_kb, z_kb = klein_bottle_surface(n=120)
    norm_z = (z_kb - z_kb.min()) / (z_kb.max() - z_kb.min() + 1e-10)
    fc_kb = cm.magma(norm_z * 0.65 + 0.2)
    fc_kb[:, :, 3] = 0.55
    ax5.plot_surface(x_kb, y_kb, z_kb, facecolors=fc_kb,
                     edgecolor='#2a1030', linewidth=0.03,
                     rcount=60, ccount=60, antialiased=True)
    ax5.set_xlim(-4, 4); ax5.set_ylim(-4, 4); ax5.set_zlim(-2.5, 2.5)
    ax5.set_box_aspect([1, 1, 0.6])
    ax5.text2D(0.5, 0.05, "Klein Bottle",
               transform=ax5.transAxes, ha='center',
               fontsize=10, fontweight='bold', color='#ff66cc')

    # Hopf
    ax6 = setup_ax(fig, (2, 3, 6), elev=25, azim=-35)
    fibers = hopf_fibration_fibers(n_fibers=40, n_pts=300)
    for fx, fy, fz, param in fibers:
        c = cm.twilight(param)
        ax6.plot(fx, fy, fz, color=c, linewidth=0.35, alpha=0.7)
    ax6.set_xlim(-3, 3); ax6.set_ylim(-3, 3); ax6.set_zlim(-3, 3)
    ax6.set_box_aspect([1, 1, 1])
    ax6.text2D(0.5, 0.05, "Hopf Fibration",
               transform=ax6.transAxes, ha='center',
               fontsize=10, fontweight='bold', color='#cc88ff')

    plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.02,
                        wspace=0.02, hspace=0.08)
    fig.savefig('/home/jmoss/code/physics/topology_gallery.png',
                dpi=200, bbox_inches='tight',
                facecolor='#0a0a12', edgecolor='none')
    print("  Saved: topology_gallery.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Rendering visualizations...\n")

    print("[1/4] Three torus-knot candidates...")
    render_torus_candidates()

    print("[2/4] Hero view of best candidate...")
    render_hero_view()

    print("[3/4] Alternative topologies...")
    render_alternative_topologies()

    print("[4/4] Complete gallery...")
    render_gallery()

    print("\nDone! 4 images saved in /home/jmoss/code/physics/:")
    print("  electron_torus_candidates.png  — 3 best-fit torus knots")
    print("  electron_hero.png              — detailed #1 candidate")
    print("  alternative_topologies.png     — trefoil, Klein bottle, Hopf fibration")
    print("  topology_gallery.png           — all 6 shapes")
