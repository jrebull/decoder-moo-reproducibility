"""Generalization figure from second_instance.json + base (stats_test/controls)."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R='app/data/results/'
plt.rcParams.update({'font.family':'serif','font.size':9,'savefig.bbox':'tight'})
d=json.load(open(R+'second_instance.json'))['instances']
st=json.load(open(R+'stats_test.json')); ctl=json.load(open(R+'controls.json'))
base={'mohho':float(np.mean(st['mohho_hv'])),
      'random':ctl['random_restart']['stats']['mean'],
      'nsga':float(np.mean(st['nsga2_hv']))}
labels=['Base']+[f'P{r["instance"]}' for r in d]
moh=[base['mohho']]+[r['mohho_hv_mean'] for r in d]
rnd=[base['random']]+[r['random_hv_mean'] for r in d]
nsg=[base['nsga']]+[r['nsga2_hv_mean'] for r in d]
relR=[100*rnd[i]/moh[i] for i in range(len(moh))]; relN=[100*nsg[i]/moh[i] for i in range(len(moh))]
x=np.arange(len(labels)); w=0.26
fig,ax=plt.subplots(figsize=(6.2,3.3))
b1=ax.bar(x-w,[100]*len(x),w,color='#2E86DE',alpha=0.88,label='MOHHO',edgecolor='k',linewidth=0.3)
b2=ax.bar(x,relR,w,color='#9AA3AF',alpha=0.88,label='Random restart',edgecolor='k',linewidth=0.3)
b3=ax.bar(x+w,relN,w,color='#E67E22',alpha=0.88,label='NSGA-II',edgecolor='k',linewidth=0.3)
ax.axhline(100,color='#2E86DE',lw=0.8,ls=':')
ymin=min(min(relR),min(relN))-2
ax.set_ylim(ymin,101.5)
ax.set_ylabel('Hypervolume, % of MOHHO')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel('Instance (Base + five perturbed-demand instances P1--P5)')
ax.legend(handles=[b1,b2,b3],loc='lower center',bbox_to_anchor=(0.5,1.01),ncol=3,fontsize=8.5,frameon=False,columnspacing=2.2,handlelength=1.4)
ax.grid(axis='y',alpha=0.25)
fig.savefig('../figures/generalization.pdf'); fig.savefig('../figures/generalization.png',dpi=200)
print('generalization fig:',[f'{v:.1f}' for v in relR],'random%',[f'{v:.1f}' for v in relN],'nsga%')
