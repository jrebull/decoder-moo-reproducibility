"""
Two data-driven 'artwork' figures for the report:
  art_dali.png  -- surrealist melting clocks (Dali) = the visa wait backlog.
  art_mural.png -- cubist-muralist Voronoi vortex (Picasso/Rivera/Frida/Dali)
                   = the MOHHO siege/convergence (energy decay cool->hot).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Ellipse, Polygon, Circle
from matplotlib.transforms import Affine2D
from scipy.spatial import Voronoi
from pathlib import Path as FPath

FIG = FPath("../reporte_final/figures"); FIG.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1) DALI -- "La persistencia de la espera"
# ----------------------------------------------------------------------
def melting_clock(ax, cx, cy, r, droop, hands, country, years, tilt=0.0):
    tr = Affine2D().rotate_deg_around(cx, cy, tilt) + ax.transData
    # long surreal shadow
    ax.add_patch(Ellipse((cx+1.7*r, cy-r*0.05), r*4.2, r*0.5,
                 color="#3b2a17", alpha=0.18, zorder=1))
    # drip (melting) hanging from the bottom -- longer when more years waited
    bx = cx
    drip = Path([(cx-r*0.62, cy-r*0.55), (cx-r*0.4, cy-r-droop*0.6),
                 (bx, cy-r-droop), (cx+r*0.4, cy-r-droop*0.6),
                 (cx+r*0.62, cy-r*0.55), (cx-r*0.62, cy-r*0.55)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY])
    ax.add_patch(PathPatch(drip, fc="#ede0bf", ec="#9c7b3e", lw=1.6, zorder=3, transform=tr))
    # face
    ax.add_patch(Ellipse((cx, cy), 2*r, 1.9*r, fc="#f3e9cf", ec="#9c7b3e", lw=2.4, zorder=4, transform=tr))
    ax.add_patch(Ellipse((cx, cy), 2*r, 1.9*r, fc="none", ec="#cbb27a", lw=0.8, zorder=5, transform=tr))
    # ticks
    for k in range(12):
        a = np.deg2rad(k*30)
        x0, y0 = cx+0.86*r*np.sin(a), cy+0.82*r*np.cos(a)
        x1, y1 = cx+0.95*r*np.sin(a), cy+0.90*r*np.cos(a)
        ax.plot([x0, x1], [y0, y1], color="#6b4a2b", lw=1.3, zorder=6, transform=tr)
    # hands
    for ang, ln, w in hands:
        a = np.deg2rad(ang)
        ax.plot([cx, cx+ln*r*np.sin(a)], [cy, cy+ln*r*np.cos(a)],
                color="#3b2a17", lw=w, zorder=7, solid_capstyle="round", transform=tr)
    ax.add_patch(Circle((cx, cy), 0.05*r, fc="#3b2a17", zorder=8, transform=tr))
    # ant trail (Dali motif) along the rim
    for k in range(7):
        a = np.deg2rad(-60+k*12)
        ax.plot(cx+0.99*r*np.sin(a), cy+0.95*r*np.cos(a), marker=(3,0,0),
                ms=3.0, color="#241a0e", zorder=8, transform=tr)
    # labels
    ax.text(cx, cy-r-droop-0.18, f"{country}", ha="center", va="top",
            fontsize=10.5, style="italic", color="#2b1d0e", zorder=9,
            fontfamily="serif")
    ax.text(cx, cy+0.30*r, f"{years}", ha="center", va="center",
            fontsize=15, weight="bold", color="#7a1f12", zorder=9,
            fontfamily="serif", transform=tr)
    ax.text(cx, cy-0.16*r, "años", ha="center", va="center",
            fontsize=6.5, color="#6b4a2b", zorder=9, transform=tr)

def make_dali():
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    # sky gradient (teal -> ochre), Cadaques light
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    sky = np.zeros((256, 1, 3))
    top = np.array([0.74, 0.85, 0.84]); bot = np.array([0.93, 0.82, 0.62])
    for i in range(256):
        t = i/255; sky[i, 0] = top*(1-t)+bot*t
    ax.imshow(sky, extent=[0, 10, 2.5, 6.2], aspect="auto", zorder=0, origin="upper")
    # desert floor
    ax.add_patch(Polygon([[0, 0], [10, 0], [10, 2.7], [0, 2.5]], closed=True,
                 fc="#c8a061", ec="none", zorder=0))
    ax.add_patch(Polygon([[0, 0], [10, 0], [10, 1.0], [0, 1.2]], closed=True,
                 fc="#b8884a", ec="none", alpha=0.6, zorder=0))
    # distant cliffs
    ax.add_patch(Polygon([[7.6, 2.6], [8.5, 3.7], [9.2, 2.6]], fc="#9c7448", ec="none", alpha=0.85, zorder=0))
    ax.add_patch(Polygon([[8.7, 2.6], [9.4, 3.2], [10, 2.6]], fc="#86653e", ec="none", alpha=0.8, zorder=0))
    # the Dali pedestal (block) with a clock draped on its edge
    ax.add_patch(Polygon([[0.3, 1.0], [2.6, 1.0], [2.6, 3.1], [0.3, 3.1]],
                 fc="#7c5a36", ec="#4a3420", lw=1.5, zorder=2))
    ax.add_patch(Polygon([[2.6, 1.0], [3.1, 1.25], [3.1, 3.35], [2.6, 3.1]],
                 fc="#5f452a", ec="#4a3420", lw=1.0, zorder=2))
    # a dead branch
    ax.plot([9.9, 8.2, 7.4], [5.6, 5.2, 4.4], color="#4a3420", lw=3.5, zorder=2, solid_capstyle="round")
    # melting clocks: droop proportional to wait years (more wait = more melted time)
    melting_clock(ax, 1.65, 3.25, 0.72, 0.4+13/14, [(150, 0.55, 3), (40, 0.78, 2)], "India", 13, tilt=-8)
    melting_clock(ax, 4.7, 1.95, 0.78, 0.4+10/14, [(210, 0.5, 3), (-20, 0.8, 2)], "China", 10, tilt=10)
    melting_clock(ax, 7.5, 4.05, 0.6, 0.3+4/14, [(120, 0.5, 2.5), (300, 0.75, 1.8)], "México", 4, tilt=-15)
    melting_clock(ax, 8.9, 1.7, 0.5, 0.3+4/14, [(60, 0.5, 2.3), (250, 0.7, 1.6)], "Filipinas", 4, tilt=6)
    # title
    ax.text(0.18, 6.0, "La persistencia de la espera", fontsize=17, weight="bold",
            color="#2b1d0e", fontfamily="serif", style="italic", va="top")
    ax.text(0.2, 5.55, "homenaje a Dalí — el tiempo que el sistema FIFO derrite sobre cada nacionalidad",
            fontsize=8.2, color="#3b2a17", fontfamily="serif", style="italic", va="top")
    plt.tight_layout(pad=0.3)
    plt.savefig(FIG/"art_dali.png", dpi=170, facecolor="#cdbd9a")
    plt.close()
    print("saved art_dali.png")

# ----------------------------------------------------------------------
# 2) MURAL -- cubist Voronoi vortex (the siege / energy decay)
# ----------------------------------------------------------------------
PALETTE_OUT = ["#1B3B6F", "#1E7D6E", "#2E6E8E", "#3a5a78"]      # cool = exploración
PALETTE_MID = ["#C75B39", "#7B2D8E", "#4E7A3A", "#1E7D6E"]      # transición
PALETTE_IN  = ["#B5121B", "#F2A104", "#E0411F", "#D98324"]      # hot = asedio/captura

def make_mural(seed=7):
    rng = np.random.default_rng(seed)
    cx, cy = 0.5, 0.5
    pts = [[cx, cy]]
    # vortex of seeds: denser toward the center (the prey) -> small cubist shards in the siege
    for ring in range(1, 13):
        rad = (ring/13)**1.35 * 0.62
        n = max(5, int(7*ring))
        for k in range(n):
            a = 2*np.pi*k/n + ring*0.55 + rng.normal(0, 0.05)
            rr = rad + rng.normal(0, 0.012)
            pts.append([cx+rr*np.cos(a), cy+rr*np.sin(a)*0.92])
    pts = np.array(pts)
    # sentinel ring so inner regions are finite
    sent = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2*np.pi, 24, endpoint=False)])*4 + [cx, cy]
    vor = Voronoi(np.vstack([pts, sent]))

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.04, 1.04); ax.axis("off")
    ax.add_patch(Polygon([[-1,-1],[2,-1],[2,2],[-1,2]], fc="#0e0a14", zorder=0))

    for i in range(len(pts)):
        reg = vor.regions[vor.point_region[i]]
        if not reg or -1 in reg:
            continue
        poly = vor.vertices[reg]
        d = np.hypot(pts[i,0]-cx, pts[i,1]-cy)
        if d < 0.18:   pal = PALETTE_IN
        elif d < 0.40: pal = PALETTE_MID
        else:          pal = PALETTE_OUT
        col = pal[rng.integers(len(pal))]
        ax.add_patch(Polygon(poly, closed=True, fc=col, ec="#0e0a14",
                     lw=1.4, joinstyle="miter", zorder=2, alpha=0.96))

    # radiant central eye/sun (Frida/Dalí) = the optimum / leader (rabbit)
    for rr, c in [(0.085, "#F2A104"), (0.055, "#E0411F"), (0.028, "#FCEFC7")]:
        ax.add_patch(Circle((cx, cy), rr, fc=c, ec="#0e0a14", lw=1.2, zorder=6))
    for k in range(24):                          # sun rays
        a = 2*np.pi*k/24
        ax.plot([cx+0.09*np.cos(a), cx+0.135*np.cos(a)],
                [cy+0.09*np.sin(a), cy+0.135*np.sin(a)],
                color="#F2A104", lw=2.0, zorder=5, solid_capstyle="round")
    ax.add_patch(Circle((cx, cy), 0.012, fc="#0e0a14", zorder=7))  # pupil

    # stylized hawks (cubist chevrons) swooping in from the cool exterior
    for (hx, hy, s, rot) in [(0.16,0.82,0.05,-25),(0.85,0.74,0.045,200),
                              (0.80,0.20,0.05,150),(0.18,0.22,0.04,40),(0.5,0.93,0.05,180)]:
        tr = Affine2D().rotate_deg_around(hx, hy, rot) + ax.transData
        wing = Path([(hx-s,hy),(hx-0.3*s,hy+0.5*s),(hx,hy+0.12*s),
                     (hx+0.3*s,hy+0.5*s),(hx+s,hy),(hx,hy-0.18*s),(hx-s,hy)],
                    [Path.MOVETO,Path.CURVE3,Path.CURVE3,Path.CURVE3,Path.CURVE3,Path.LINETO,Path.CLOSEPOLY])
        ax.add_patch(PathPatch(wing, fc="#0e0a14", ec="#FCEFC7", lw=0.8, zorder=6, transform=tr))

    # bold mural frame
    for lw,c in [(10,"#0e0a14"),(4,"#B5121B"),(1.5,"#F2A104")]:
        ax.add_patch(Polygon([[0,0],[1,0],[1,1],[0,1]], closed=True, fill=False, ec=c, lw=lw, zorder=8))

    ax.text(0.5, 1.005, "El asedio", ha="center", va="bottom", fontsize=18, weight="bold",
            color="#FCEFC7", fontfamily="serif", zorder=9)
    ax.text(0.5, -0.035, "mural cúbico de la convergencia MOHHO — del frío de la exploración al fuego del asedio",
            ha="center", va="top", fontsize=8.0, color="#cbb27a", style="italic",
            fontfamily="serif", zorder=9)
    plt.tight_layout(pad=0.4)
    plt.savefig(FIG/"art_mural.png", dpi=170, facecolor="#0e0a14")
    plt.close()
    print("saved art_mural.png")

if __name__ == "__main__":
    make_dali()
    make_mural()
