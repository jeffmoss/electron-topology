#!/usr/bin/env python3
"""
Visualize the toroidal electron geometries from the resonance search.

Produces publication-quality 3D renders:
  1. electron_hero.png          — showpiece of the best (21,10) geometry
  2. electron_torus_candidates.png — three best-fit torus-knot candidates
  3. three_topologies.png       — torus knot vs Berger sphere vs Lens space
  4. topology_gallery.png       — 2x3 gallery of all shapes

Based on Williamson & van der Mark (1997): the electron as a
Compton-wavelength photon confined in a toroidal double-helix.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

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
DISPLAY_RHO = 0.22


# ═══════════════════════════════════════════════════════════════════════
# Geometry primitives
# ═══════════════════════════════════════════════════════════════════════

def torus_surface(R, rho, n_theta=80, n_phi=200):
    """Generate torus mesh. Returns x, y, z, theta_grid, phi_grid."""
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


def electron_loop(rho, R=1.0, n_pts=2000, lift=1.07):
    """(1,0) poloidal loop — the electron ground state."""
    t = np.linspace(0, 2 * np.pi, n_pts)
    r = rho * lift
    x = (R + r * np.cos(t)) * np.ones_like(t)
    y = np.zeros_like(t)
    z = r * np.sin(t)
    return x, y, z


def compute_shading(theta_grid, phi_grid, light_dir=None):
    """Phong-like pseudo-shading from surface normals."""
    if light_dir is None:
        light_dir = np.array([0.35, -0.45, 0.82])
    light_dir = light_dir / np.linalg.norm(light_dir)
    nx = np.cos(theta_grid) * np.cos(phi_grid)
    ny = np.cos(theta_grid) * np.sin(phi_grid)
    nz = np.sin(theta_grid)
    # Diffuse
    diffuse = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
    diffuse = np.clip((diffuse + 1) / 2, 0, 1)
    # Specular highlight
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
    """Convert shade array to RGBA facecolors with a deep blue base."""
    facecolors = np.zeros((*shade.shape, 4))
    for ch in range(3):
        facecolors[:, :, ch] = base_color[ch] * shade
    # Add subtle blue-violet rim lighting
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
    # Glow layer (wider, dimmer)
    if glow:
        for i in range(0, n - seg_size, seg_size):
            j = min(i + seg_size + 1, n)
            frac = i / n
            c = cmap(cmap_range[0] + (cmap_range[1] - cmap_range[0]) * frac)
            ax.plot(x[i:j], y[i:j], z[i:j],
                    color=c, linewidth=lw * glow_lw_mult,
                    alpha=glow_alpha, zorder=zorder - 1, solid_capstyle='round')
    # Main bright curve
    for i in range(0, n - seg_size, seg_size):
        j = min(i + seg_size + 1, n)
        frac = i / n
        c = cmap(cmap_range[0] + (cmap_range[1] - cmap_range[0]) * frac)
        ax.plot(x[i:j], y[i:j], z[i:j],
                color=c, linewidth=lw, alpha=alpha,
                zorder=zorder, solid_capstyle='round')


def plot_torus(ax, geom, show_electron=True, display_rho=DISPLAY_RHO,
               knot_lw=0.9, elec_lw=3.0, surface_alpha=0.32,
               wireframe=True, n_theta=80, n_phi=200):
    """Render torus surface + knot + electron loop on ax."""
    p, q = geom["p"], geom["q"]
    R = 1.0
    rho = display_rho

    # ── Surface ──
    x_t, y_t, z_t, theta_grid, phi_grid = torus_surface(
        R, rho, n_theta=n_theta, n_phi=n_phi)
    shade = compute_shading(theta_grid, phi_grid)
    fc = make_torus_facecolors(shade, alpha=surface_alpha)
    ax.plot_surface(x_t, y_t, z_t, facecolors=fc, edgecolor='none',
                    rcount=n_theta, ccount=n_phi,
                    antialiased=True, zorder=1)

    # ── Wireframe ──
    if wireframe:
        x_w, y_w, z_w, _, _ = torus_surface(R, rho, n_theta=20, n_phi=60)
        ax.plot_wireframe(x_w, y_w, z_w, color='#3a3a7a',
                          linewidth=0.12, alpha=0.18,
                          rcount=20, ccount=60, zorder=2)

    # ── Muon mode: (p,q) torus knot ──
    x_k, y_k, z_k = torus_knot_curve(p, q, rho, R)
    plot_colored_curve(ax, x_k, y_k, z_k, cmap_name='cool',
                       cmap_range=(0.12, 0.88), lw=knot_lw, alpha=0.94,
                       glow=True, glow_lw_mult=3.8, glow_alpha=0.10,
                       seg_size=70, zorder=4)

    # ── Electron mode: gold (1,0) loop ──
    if show_electron:
        x_e, y_e, z_e = electron_loop(rho, R)
        # Glow
        ax.plot(x_e, y_e, z_e, color='#ffd966', linewidth=elec_lw * 3.0,
                alpha=0.10, zorder=4, solid_capstyle='round')
        ax.plot(x_e, y_e, z_e, color='#ffc800', linewidth=elec_lw * 1.6,
                alpha=0.22, zorder=5, solid_capstyle='round')
        # Core
        ax.plot(x_e, y_e, z_e, color='#ffdd44', linewidth=elec_lw,
                alpha=0.95, zorder=6, solid_capstyle='round')

    # ── Axis limits ──
    lim = R + rho + 0.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    zlim = rho * 1.8
    ax.set_zlim(-zlim, zlim)
    ax.set_box_aspect([1, 1, rho * 2.2])


def setup_3d_ax(fig, rect, elev=25, azim=-60):
    """Create a clean dark 3D axes from a rect [l, b, w, h]."""
    ax = fig.add_axes(rect, projection='3d', computed_zorder=False)
    ax.set_facecolor(BG_COLOR)
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    return ax


def setup_ax_subplot(fig, pos, elev=25, azim=-60):
    """Create dark 3D axes from subplot position tuple."""
    ax = fig.add_subplot(*pos, projection='3d', computed_zorder=False)
    ax.set_facecolor(BG_COLOR)
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    return ax


# ═══════════════════════════════════════════════════════════════════════
# Berger Sphere (squashed S^3) rendering
# ═══════════════════════════════════════════════════════════════════════

def berger_sphere_surface(lam=0.55, n_theta=60, n_phi=120):
    """Oblate sphere representing a squashed S^3 (Berger sphere)."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = lam * np.cos(theta)
    return x, y, z, theta, phi


def berger_hopf_fibers(lam=0.55, n_fibers=12, n_pts=600):
    """Hopf fiber circles on the Berger sphere at various latitudes."""
    fibers = []
    latitudes = np.linspace(0.15 * np.pi, 0.85 * np.pi, n_fibers)
    for theta0 in latitudes:
        phi = np.linspace(0, 2 * np.pi, n_pts)
        x = np.sin(theta0) * np.cos(phi)
        y = np.sin(theta0) * np.sin(phi)
        z = lam * np.cos(theta0) * np.ones_like(phi)
        fibers.append((x, y, z, theta0 / np.pi))
    return fibers


def berger_shading(theta_grid, phi_grid, lam=0.55):
    """Phong shading for the Berger sphere."""
    light_dir = np.array([0.4, -0.3, 0.85])
    light_dir /= np.linalg.norm(light_dir)
    # Normals for oblate spheroid
    nx = np.sin(theta_grid) * np.cos(phi_grid)
    ny = np.sin(theta_grid) * np.sin(phi_grid)
    nz = lam * np.cos(theta_grid)
    mag = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-10
    nx, ny, nz = nx / mag, ny / mag, nz / mag
    diffuse = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
    diffuse = np.clip((diffuse + 1) / 2, 0, 1)
    view_dir = np.array([0.0, -0.3, 0.95])
    view_dir /= np.linalg.norm(view_dir)
    halfway = light_dir + view_dir
    halfway /= np.linalg.norm(halfway)
    spec = nx * halfway[0] + ny * halfway[1] + nz * halfway[2]
    spec = np.clip(spec, 0, 1) ** 40
    shade = 0.15 + 0.60 * diffuse + 0.25 * spec
    return shade


def plot_berger_sphere(ax, lam=0.55):
    """Render the Berger sphere with Hopf fibers."""
    x, y, z, theta_grid, phi_grid = berger_sphere_surface(lam)
    shade = berger_shading(theta_grid, phi_grid, lam)

    # Warm gold base color
    base = np.array([0.32, 0.22, 0.06])
    fc = np.zeros((*shade.shape, 4))
    for ch in range(3):
        fc[:, :, ch] = base[ch] * shade
    # Warm rim glow
    rim = np.clip(1.0 - shade, 0, 1) ** 2
    fc[:, :, 0] += 0.12 * rim
    fc[:, :, 1] += 0.06 * rim
    fc[:, :, 2] += 0.02 * rim
    fc[:, :, :3] = np.clip(fc[:, :, :3], 0, 1)
    fc[:, :, 3] = 0.35

    ax.plot_surface(x, y, z, facecolors=fc, edgecolor='none',
                    rcount=60, ccount=120, antialiased=True, zorder=1)

    # Subtle wireframe
    x_w, y_w, z_w, _, _ = berger_sphere_surface(lam, n_theta=16, n_phi=32)
    ax.plot_wireframe(x_w, y_w, z_w, color='#6a5a2a', linewidth=0.12,
                      alpha=0.15, rcount=16, ccount=32, zorder=2)

    # Hopf fibers
    fibers = berger_hopf_fibers(lam, n_fibers=14, n_pts=500)
    amber_cmap = matplotlib.colormaps['YlOrBr']
    for fx, fy, fz, param in fibers:
        c = amber_cmap(0.3 + 0.5 * param)
        # Glow
        ax.plot(fx, fy, fz, color=c, linewidth=3.5, alpha=0.08,
                zorder=3, solid_capstyle='round')
        # Core
        ax.plot(fx, fy, fz, color=c, linewidth=1.2, alpha=0.85,
                zorder=4, solid_capstyle='round')

    # Great circle equator for emphasis
    phi = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(phi), np.sin(phi), np.zeros_like(phi),
            color='#ffbb33', linewidth=2.0, alpha=0.7, zorder=5)
    ax.plot(np.cos(phi), np.sin(phi), np.zeros_like(phi),
            color='#ffbb33', linewidth=5.0, alpha=0.10, zorder=4)

    lim = 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lam * 1.3, lam * 1.3)
    ax.set_box_aspect([1, 1, lam * 1.3])


# ═══════════════════════════════════════════════════════════════════════
# Lens Space L(n,1) rendering
# ═══════════════════════════════════════════════════════════════════════

def lens_space_wedge(n_lens=5, n_pts=60):
    """Fundamental domain of L(n,1) as a cone/wedge in 3D.
    Render a solid cone with angular wedge 2*pi/n."""
    wedge_angle = 2 * np.pi / n_lens
    r = np.linspace(0, 1.0, n_pts)
    theta = np.linspace(0, wedge_angle, n_pts)
    R, Theta = np.meshgrid(r, theta)
    height = 1.2
    # Cone: z tapers linearly with r
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z = height * (1.0 - R)
    return X, Y, Z, Theta, R


def lens_geodesics(n_lens=5, n_geo=6, n_pts=800):
    """Helical geodesics inside the lens space fundamental domain."""
    curves = []
    wedge_angle = 2 * np.pi / n_lens
    for i in range(n_geo):
        t = np.linspace(0, 1.0, n_pts)
        r0 = 0.2 + 0.6 * (i / max(n_geo - 1, 1))
        # Helical trajectory winding around the cone
        n_turns = 2 + i * 0.5
        theta = wedge_angle * 0.1 + (wedge_angle * 0.8) * t
        r = r0 * (0.5 + 0.5 * np.sin(2 * np.pi * n_turns * t))
        height = 1.2
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = height * (1.0 - r) * (0.3 + 0.7 * np.cos(np.pi * t)**2)
        curves.append((x, y, z, i / max(n_geo - 1, 1)))
    return curves


def plot_lens_space(ax, n_lens=5):
    """Render Lens space L(n,1) fundamental domain with geodesics."""
    X, Y, Z, Theta, R = lens_space_wedge(n_lens)

    # Shading: use height and angle for visual interest
    light_dir = np.array([0.3, -0.4, 0.85])
    light_dir /= np.linalg.norm(light_dir)
    shade = 0.3 + 0.5 * Z / 1.2 + 0.2 * np.clip(
        np.cos(Theta) * light_dir[0] + np.sin(Theta) * light_dir[1], 0, 1)
    shade = np.clip(shade, 0, 1)

    # Purple/magenta base
    base = np.array([0.22, 0.06, 0.30])
    fc = np.zeros((*shade.shape, 4))
    for ch in range(3):
        fc[:, :, ch] = base[ch] * shade
    # Magenta rim
    rim = np.clip(1.0 - shade, 0, 1) ** 1.5
    fc[:, :, 0] += 0.10 * rim
    fc[:, :, 1] += 0.02 * rim
    fc[:, :, 2] += 0.08 * rim
    fc[:, :, :3] = np.clip(fc[:, :, :3], 0, 1)
    fc[:, :, 3] = 0.40

    ax.plot_surface(X, Y, Z, facecolors=fc, edgecolor='none',
                    rcount=60, ccount=60, antialiased=True, zorder=1)

    # Draw the bottom face (disk wedge at z=0)
    r_pts = np.linspace(0, 1.0, 30)
    theta_pts = np.linspace(0, 2 * np.pi / n_lens, 30)
    Rb, Tb = np.meshgrid(r_pts, theta_pts)
    Xb = Rb * np.cos(Tb)
    Yb = Rb * np.sin(Tb)
    Zb = np.zeros_like(Xb)
    fc_b = np.full((*Zb.shape, 4), [0.12, 0.04, 0.18, 0.30])
    ax.plot_surface(Xb, Yb, Zb, facecolors=fc_b, edgecolor='none',
                    rcount=30, ccount=30, antialiased=True, zorder=0)

    # Wireframe edges
    wedge_angle = 2 * np.pi / n_lens
    # Edge lines of the wedge
    for theta_e in [0, wedge_angle]:
        r_line = np.linspace(0, 1.0, 100)
        xe = r_line * np.cos(theta_e)
        ye = r_line * np.sin(theta_e)
        ze = 1.2 * (1.0 - r_line)
        ax.plot(xe, ye, ze, color='#8844aa', linewidth=1.0, alpha=0.5, zorder=3)
    # Base arc
    t_arc = np.linspace(0, wedge_angle, 100)
    ax.plot(np.cos(t_arc), np.sin(t_arc), np.zeros(100),
            color='#8844aa', linewidth=1.0, alpha=0.5, zorder=3)

    # Geodesic curves
    curves = lens_geodesics(n_lens, n_geo=7, n_pts=600)
    mag_cmap = matplotlib.colormaps['cool']
    for cx, cy, cz, param in curves:
        c = mag_cmap(0.5 + 0.45 * param)
        # Glow
        ax.plot(cx, cy, cz, color=c, linewidth=3.0, alpha=0.08,
                zorder=4, solid_capstyle='round')
        # Core
        ax.plot(cx, cy, cz, color=c, linewidth=1.0, alpha=0.80,
                zorder=5, solid_capstyle='round')

    lim = 1.15
    ax.set_xlim(-0.3, lim)
    ax.set_ylim(-0.3, lim)
    ax.set_zlim(-0.1, 1.35)
    ax.set_box_aspect([1, 1, 1.0])


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: "The Shape of the Electron" — electron_hero.png
# ═══════════════════════════════════════════════════════════════════════

def render_hero_view():
    fig = plt.figure(figsize=(16, 10), dpi=300)
    geom = GEOMETRIES[0]
    p, q, rho = geom["p"], geom["q"], geom["rho"]

    # ── Large main 3D view ──
    ax1 = setup_3d_ax(fig, [0.0, 0.02, 0.62, 0.88], elev=22, azim=-55)
    plot_torus(ax1, geom, knot_lw=1.1, elec_lw=3.5, surface_alpha=0.30,
               n_theta=80, n_phi=200)

    # ── Title block (top center over main view) ──
    fig.text(0.31, 0.97,
             "The Electron as a Confined Photon",
             ha='center', fontsize=19, fontweight='bold', color='#ffffff')
    fig.text(0.31, 0.943,
             "Williamson & van der Mark (1997)",
             ha='center', fontsize=10, color='#888888')

    # ── Top view (upper right) ──
    ax2 = setup_3d_ax(fig, [0.61, 0.52, 0.38, 0.42], elev=88, azim=0)
    plot_torus(ax2, geom, show_electron=False, knot_lw=0.55,
               surface_alpha=0.25, wireframe=True, n_theta=60, n_phi=150)
    ax2.text2D(0.5, 0.93, "Top view", transform=ax2.transAxes,
               ha='center', fontsize=9, color='#666666')

    # ── Side view (lower right) ──
    ax3 = setup_3d_ax(fig, [0.61, 0.06, 0.38, 0.42], elev=2, azim=0)
    plot_torus(ax3, geom, show_electron=False, knot_lw=0.55,
               surface_alpha=0.25, wireframe=True, n_theta=60, n_phi=150)
    ax3.text2D(0.5, 0.93, "Side view", transform=ax3.transAxes,
               ha='center', fontsize=9, color='#666666')

    # ── Info panel (bottom left) ──
    info_lines = [
        (f"Best-fit geometry:  ({p},{q}) / (1,0)", '#00ddff', 'bold', 11),
        (f"Aspect ratio  rho = {rho:.9f}", '#cccccc', 'normal', 9),
        ("", BG_COLOR, 'normal', 9),
        (f"Muon mode:      {p} poloidal x {q} toroidal windings", '#bbbbbb', 'normal', 9),
        ("Electron mode:  single poloidal loop (1,0)", '#bbbbbb', 'normal', 9),
        ("", BG_COLOR, 'normal', 9),
        (f"Path length ratio = {TARGET_RATIO}", '#ffdd44', 'bold', 10),
        (f"Residual error:  {geom['score']}  (0.05 ppb)", '#cccccc', 'normal', 9),
        ("", BG_COLOR, 'normal', 9),
        ("110 billion evaluations  |  CUDA + Rust", '#777777', 'normal', 8),
        ("GPU: 64-pt Gauss-Legendre  |  CPU: 10,000-pt f64", '#777777', 'normal', 8),
    ]
    y0 = 0.175
    for text, color, weight, size in info_lines:
        fig.text(0.02, y0, text, fontsize=size, color=color,
                 fontweight=weight, fontfamily='monospace')
        y0 -= 0.023

    fig.savefig('/home/jmoss/code/physics/electron_hero.png',
                dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    print("  Saved: electron_hero.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: "Three Candidates" — electron_torus_candidates.png
# ═══════════════════════════════════════════════════════════════════════

def render_torus_candidates():
    fig = plt.figure(figsize=(21, 9), dpi=200)

    fig.text(0.5, 0.97,
             "The Shape of the Electron",
             ha='center', fontsize=22, fontweight='bold', color='#ffffff')
    fig.text(0.5, 0.935,
             "Toroidal Resonance Candidates  --  Williamson & van der Mark (1997)",
             ha='center', fontsize=11, color='#888888')
    fig.text(0.5, 0.907,
             f"Target: m_mu / m_e = {TARGET_RATIO}  |  25 ppb uncertainty",
             ha='center', fontsize=10, color='#666666')

    panel_width = 0.30
    panel_height = 0.72
    x_starts = [0.02, 0.35, 0.68]

    for i, geom in enumerate(GEOMETRIES):
        ax = setup_3d_ax(fig, [x_starts[i], 0.10, panel_width, panel_height],
                         elev=28, azim=-58 + i * 12)
        plot_torus(ax, geom, knot_lw=0.85, elec_lw=2.8, surface_alpha=0.30,
                   n_theta=70, n_phi=180)

        p, q, rho = geom["p"], geom["q"], geom["rho"]

        # Mode label
        ax.text2D(0.5, 0.08,
                  f"({p},{q}) / (1,0)",
                  transform=ax.transAxes, ha='center',
                  fontsize=15, fontweight='bold', color='#00ddff')
        ax.text2D(0.5, 0.02,
                  f"rho = {rho:.9f}    score = {geom['score']}",
                  transform=ax.transAxes, ha='center',
                  fontsize=9, color='#888888')
        ax.text2D(0.5, -0.03,
                  geom['label'],
                  transform=ax.transAxes, ha='center',
                  fontsize=9, color='#666666', style='italic')

        # Rank badge
        rank_colors = ['#00ffcc', '#44aaff', '#8888cc']
        ax.text2D(0.05, 0.92, f"#{geom['rank']}",
                  transform=ax.transAxes, fontsize=14, fontweight='bold',
                  color=rank_colors[i])

    fig.text(0.5, 0.015,
             "Cyan-magenta: muon mode (p,q) torus knot   |   "
             "Gold: electron ground state (1,0) poloidal loop   |   "
             "Torus tube exaggerated for visibility",
             ha='center', fontsize=8, color='#555555')

    fig.savefig('/home/jmoss/code/physics/electron_torus_candidates.png',
                dpi=200, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    print("  Saved: electron_torus_candidates.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: "Three Topologies" — three_topologies.png
# ═══════════════════════════════════════════════════════════════════════

def render_three_topologies():
    fig = plt.figure(figsize=(21, 9), dpi=200)

    fig.text(0.5, 0.97,
             "Topological Photon Confinement  --  Three Geometries",
             ha='center', fontsize=22, fontweight='bold', color='#ffffff')
    fig.text(0.5, 0.935,
             "Closed Riemannian manifolds supporting quantized photon path-length spectra",
             ha='center', fontsize=11, color='#888888')

    panel_width = 0.30
    panel_height = 0.72
    x_starts = [0.02, 0.35, 0.68]

    # ── Panel 1: Best torus knot ──
    ax1 = setup_3d_ax(fig, [x_starts[0], 0.10, panel_width, panel_height],
                       elev=25, azim=-55)
    plot_torus(ax1, GEOMETRIES[0], knot_lw=0.85, elec_lw=2.8,
               surface_alpha=0.30, n_theta=70, n_phi=180)

    ax1.text2D(0.5, 0.10, "Torus Knot  (21,10) / (1,0)",
               transform=ax1.transAxes, ha='center',
               fontsize=13, fontweight='bold', color='#00ddff')
    ax1.text2D(0.5, 0.04,
               f"rho = 0.048614995  |  score = 1.0e-8",
               transform=ax1.transAxes, ha='center',
               fontsize=9, color='#888888')
    ax1.text2D(0.5, -0.02,
               "Best-fit toroidal resonance",
               transform=ax1.transAxes, ha='center',
               fontsize=8, color='#666666', style='italic')

    # ── Panel 2: Berger Sphere ──
    ax2 = setup_3d_ax(fig, [x_starts[1], 0.10, panel_width, panel_height],
                       elev=22, azim=-45)
    plot_berger_sphere(ax2, lam=0.55)

    ax2.text2D(0.5, 0.10, "Berger Sphere  --  Squashed S^3",
               transform=ax2.transAxes, ha='center',
               fontsize=13, fontweight='bold', color='#ffbb33')
    ax2.text2D(0.5, 0.04,
               "lambda = 0.55  |  Hopf fiber spectrum  |  pending",
               transform=ax2.transAxes, ha='center',
               fontsize=9, color='#888888')
    ax2.text2D(0.5, -0.02,
               "Sub-Riemannian geodesics on squashed 3-sphere",
               transform=ax2.transAxes, ha='center',
               fontsize=8, color='#666666', style='italic')

    # ── Panel 3: Lens Space ──
    ax3 = setup_3d_ax(fig, [x_starts[2], 0.10, panel_width, panel_height],
                       elev=28, azim=-35)
    plot_lens_space(ax3, n_lens=5)

    ax3.text2D(0.5, 0.10, "Lens Space  L(n,1)",
               transform=ax3.transAxes, ha='center',
               fontsize=13, fontweight='bold', color='#cc66ff')
    ax3.text2D(0.5, 0.04,
               "n = 5  |  closed geodesic spectrum  |  pending",
               transform=ax3.transAxes, ha='center',
               fontsize=9, color='#888888')
    ax3.text2D(0.5, -0.02,
               "Cyclic quotient of S^3 with helical geodesics",
               transform=ax3.transAxes, ha='center',
               fontsize=8, color='#666666', style='italic')

    fig.text(0.5, 0.015,
             "Each topology supports distinct confinement modes with quantized path-length ratios",
             ha='center', fontsize=8, color='#555555')

    fig.savefig('/home/jmoss/code/physics/three_topologies.png',
                dpi=200, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    print("  Saved: three_topologies.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: Gallery — topology_gallery.png
# ═══════════════════════════════════════════════════════════════════════

def render_gallery():
    fig = plt.figure(figsize=(21, 14), dpi=200)

    fig.text(0.5, 0.985,
             "Topological Photon Confinement  --  Complete Gallery",
             ha='center', fontsize=22, fontweight='bold', color='#ffffff')

    # Row 1: Three torus knots
    fig.text(0.5, 0.960,
             "Torus knot candidates reproducing m_mu/m_e = 206.7682843",
             ha='center', fontsize=10, color='#00ddff')

    col_x = [0.01, 0.34, 0.67]
    pw, ph = 0.31, 0.40

    for i, geom in enumerate(GEOMETRIES):
        ax = setup_3d_ax(fig, [col_x[i], 0.52, pw, ph],
                         elev=28, azim=-55 + i * 10)
        plot_torus(ax, geom, knot_lw=0.7, elec_lw=2.2, surface_alpha=0.28,
                   n_theta=60, n_phi=150)
        p, q = geom["p"], geom["q"]
        ax.text2D(0.5, 0.04,
                  f"({p},{q}) / (1,0)   rho={geom['rho']:.6f}",
                  transform=ax.transAxes, ha='center',
                  fontsize=10, fontweight='bold', color='#00ddff')
        ax.text2D(0.5, -0.02,
                  f"score = {geom['score']}",
                  transform=ax.transAxes, ha='center',
                  fontsize=8, color='#666666')

    # Row 2: Three topologies
    fig.text(0.5, 0.490,
             "Three topological confinement geometries",
             ha='center', fontsize=10, color='#cc88ff')

    # Torus (best) — different angle
    ax4 = setup_3d_ax(fig, [col_x[0], 0.04, pw, ph],
                       elev=45, azim=-30)
    plot_torus(ax4, GEOMETRIES[0], knot_lw=0.7, elec_lw=2.2,
               surface_alpha=0.28, n_theta=60, n_phi=150)
    ax4.text2D(0.5, 0.04, "Torus (21,10) — oblique view",
               transform=ax4.transAxes, ha='center',
               fontsize=10, fontweight='bold', color='#00ddff')

    # Berger sphere
    ax5 = setup_3d_ax(fig, [col_x[1], 0.04, pw, ph],
                       elev=20, azim=-50)
    plot_berger_sphere(ax5, lam=0.55)
    ax5.text2D(0.5, 0.04, "Berger Sphere (squashed S^3)",
               transform=ax5.transAxes, ha='center',
               fontsize=10, fontweight='bold', color='#ffbb33')

    # Lens space
    ax6 = setup_3d_ax(fig, [col_x[2], 0.04, pw, ph],
                       elev=30, azim=-40)
    plot_lens_space(ax6, n_lens=5)
    ax6.text2D(0.5, 0.04, "Lens Space L(5,1)",
               transform=ax6.transAxes, ha='center',
               fontsize=10, fontweight='bold', color='#cc66ff')

    fig.savefig('/home/jmoss/code/physics/topology_gallery.png',
                dpi=200, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    print("  Saved: topology_gallery.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Rendering visualizations...\n")

    print("[1/4] Hero view of best candidate...")
    render_hero_view()

    print("[2/4] Three torus-knot candidates...")
    render_torus_candidates()

    print("[3/4] Three topologies (torus, Berger sphere, Lens space)...")
    render_three_topologies()

    print("[4/4] Complete gallery...")
    render_gallery()

    print("\nDone! 4 images saved in /home/jmoss/code/physics/:")
    print("  electron_hero.png              — showpiece of best (21,10) geometry")
    print("  electron_torus_candidates.png  — 3 best-fit torus knots side by side")
    print("  three_topologies.png           — torus vs Berger sphere vs Lens space")
    print("  topology_gallery.png           — 2x3 gallery of all shapes")
