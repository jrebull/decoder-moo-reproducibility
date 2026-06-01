"""Architecture diagram of Visa Predict AI (clean, no overlaps) -> figures/architecture.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
FIG = Path("../reporte_final/figures"); FIG.mkdir(parents=True, exist_ok=True)

BG="#0e1420"; PANEL="#172033"; INK="#e9eff8"; SUB="#a7b6cd"

def box(ax,x,y,w,h,title,sub,acc):
    ax.add_patch(FancyBboxPatch((x+0.35,y-0.45),w,h, boxstyle="round,pad=0.02,rounding_size=2.0",
                 fc="#05080f", ec="none", zorder=2, mutation_aspect=1))         # sombra
    ax.add_patch(FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.02,rounding_size=2.0",
                 fc=PANEL, ec=acc, lw=2.2, zorder=3, mutation_aspect=1))
    ax.add_patch(FancyBboxPatch((x,y+h-1.5),w,1.5, boxstyle="round,pad=0.02,rounding_size=2.0",
                 fc=acc, ec="none", zorder=4, mutation_aspect=1, alpha=0.16))     # franja título
    ax.text(x+w/2, y+h-0.9, title, ha="center", va="center", color=INK, fontsize=10.6, weight="bold", zorder=5)
    ax.text(x+w/2, y+(h-1.5)/2, sub, ha="center", va="center", color=SUB, fontsize=7.7, zorder=5, linespacing=1.45)

def band(ax,y,h,label,acc):
    ax.add_patch(FancyBboxPatch((2.5,y),95,h, boxstyle="round,pad=0.1,rounding_size=1.4",
                 fc=acc, ec="none", alpha=0.06, zorder=1))
    # pestaña de etiqueta (arriba-izquierda, por ENCIMA de las cajas)
    ax.add_patch(FancyBboxPatch((4.0,y+h-2.7),len(label)*0.92+3.5,2.2,
                 boxstyle="round,pad=0.02,rounding_size=1.0", fc=acc, ec="none", alpha=0.20, zorder=2))
    ax.text(4.9,y+h-1.6,label,ha="left",va="center",color=acc,fontsize=8.8,weight="bold",zorder=3)

def arrow(ax,x1,y1,x2,y2,c,lw=2.4,style="-|>"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=17,
                 lw=lw,color=c,zorder=6,shrinkA=1,shrinkB=1))

fig,ax=plt.subplots(figsize=(9.6,7.0)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
SKY="#46c0f6"; GRN="#39d98a"; AMB="#f5b73d"; PUR="#b08bف" if False else "#b08bf5"; RED="#fb7185"

ax.text(50,96.5,"Arquitectura de Visa Predict AI v2.0",ha="center",color=INK,fontsize=16,weight="bold")
ax.text(50,92.8,"FastAPI  ·  Nuxt 4  ·  ECharts   —   núcleo Python 3.12 desacoplado",ha="center",color=SUB,fontsize=9,style="italic")

# ---- bandas (con espacio para etiqueta arriba) ----
band(ax,77,13.5,"INTERFAZ  ·  cliente",SKY)
band(ax,59,15.0,"API  ·  servidor",GRN)
band(ax,24,32.0,"NÚCLEO DE OPTIMIZACIÓN  ·  Python 3.12",AMB)
band(ax,4,17.0,"DATOS",PUR)

# ---- interfaz ----
box(ax,7,77.5,25,7.6,"Nuxt 4 + TypeScript","10 páginas · ECharts\nformatters · composables",SKY)
box(ax,37.5,77.5,25,7.6,"Simulación en vivo","HawkHunt (canvas)\ntablero · barras vs FIFO",SKY)
box(ax,68,77.5,25,7.6,"Visualización","Pareto 3D · heatmap\nconvergencia · galería",SKY)
# ---- api ----
box(ax,11,59.5,33,8.2,"FastAPI — 11 endpoints REST","/scenarios /summary /pareto\n/allocation /impact /optimize …",GRN)
box(ax,56,59.5,33,8.2,"WebSocket  /ws/simulation","transmite cada iteración\ndel algoritmo en vivo",GRN)
# ---- núcleo: fila 1 ----
box(ax,7,41.5,25,9.0,"problem · decoder","f1, f2, f3 · SPV\ngreedy (factibilidad)",AMB)
box(ax,37.5,41.5,25,9.0,"MOHHO","6 operadores · energía E(t)\narchivo + crowding",AMB)
box(ax,68,41.5,25,9.0,"NSGA-II","SBX + mut. polinomial\n(benchmark)",AMB)
# ---- núcleo: fila 2 ----
box(ax,7,28.5,25,9.0,"FIFO baseline","permutación por\nantigüedad",AMB)
box(ax,37.5,28.5,25,9.0,"experimento","30 corridas · semillas\nHV · IGD · spacing",AMB)
box(ax,68,28.5,25,9.0,"config","105 grupos · topes\nE0~U(-1,1) · LB/UB",AMB)
# ---- datos ----
box(ax,11,5.5,34,9.0,"Resultados precomputados","30 corridas · 406 soluciones Pareto\nconvergencia · sensibilidad",PUR)
box(ax,55,5.5,34,9.0,"Datos del problema","21 países × 5 categorías = 105\nbacklog USCIS · Visa Bulletin",PUR)

# ---- flujos ----
arrow(ax,27,77.5,27,67.9,GRN); arrow(ax,27,67.9,27,77.5,GRN)
ax.text(24.3,72.7,"REST",color=GRN,fontsize=7.3,ha="right",rotation=90,va="center")
arrow(ax,72.5,77.3,72.5,67.9,RED,lw=2.9,style="<|-|>")
ax.text(74.0,72.7,"stream",color=RED,fontsize=7.3,ha="left",rotation=90,va="center")
arrow(ax,28,59.5,28,50.7,AMB); arrow(ax,72,59.5,52,50.7,AMB)
arrow(ax,27,28.5,27,14.7,PUR); arrow(ax,50,28.5,55,14.7,PUR); arrow(ax,40,14.7,40,28.5,PUR)

# marco
ax.add_patch(FancyBboxPatch((1.6,2.5),96.8,90.5,boxstyle="round,pad=0.1,rounding_size=2",
             fc="none",ec="#2a3650",lw=1.5,zorder=0))
plt.tight_layout(pad=0.3)
plt.savefig(FIG/"architecture.png",dpi=300,facecolor=BG)
print("saved architecture.png (300 dpi)")
