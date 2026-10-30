from pathlib import Path
import argparse,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('figure',choices=['S3','S7']);a=p.parse_args();root=R/('Fig_'+a.figure);out=root/'reproduced';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':['Aptos','DejaVu Sans'],'font.size':11,'xtick.direction':'in','ytick.direction':'in'})
if a.figure=='S3':
 fig,axs=plt.subplots(1,3,figsize=(10,3));rows=[]
 for ax,name in zip(axs,['low','medium','high']):
  ref=pd.read_csv(root/'data/curves'/(name+'__h0p5000.csv'));value=np.interp(.5,ref.strain,ref['SEA(J/kg)'])
  for token,h in [('4p0000',4),('2p0000',2),('1p0000',1),('0p5000',.5)]:
   df=pd.read_csv(root/'data/curves'/(name+'__h'+token+'.csv'));sea=np.interp(.5,df.strain,df['SEA(J/kg)']);ax.plot(df.strain,df['SEA(J/kg)'],label=str(h)+' mm')
   rows.append(dict(structure=name,mesh_mm=h,end_strain=df.strain.iloc[-1],SEA_05=sea,error_pct=100*abs(sea/value-1)))
  ax.set_xlabel('Strain');ax.set_ylabel('SEA (J/kg)');ax.legend(frameon=False)
 pd.DataFrame(rows).to_csv(out/'mesh_endpoint_comparison.csv',index=False)
else:
 points=pd.read_csv(root/'data/figure_source_points.csv');summary=pd.read_csv(root/'data/five_fraction_comparison.csv');fig,axs=plt.subplots(1,3,figsize=(10,3))
 for ax,key,col in zip(axs,['work','mass','SEA'],['#2F5597','#C55A11','#1E5A29']):
  for xi,g in points.groupby('xi'):ax.scatter(np.full(len(g),xi),g[key+'_ratio_OFF'],s=10,alpha=.3,color=col)
  ax.errorbar(summary.xi,summary[key+'_ratio_mean'],yerr=summary[key+'_ratio_SD'].fillna(0),marker='o',color=col,capsize=3)
  ax.set_xlabel('Topology fraction');ax.set_ylabel(key+' / OFF');ax.set_xticks([0,.25,.5,.75,1]);ax.axhline(1,color='gray',linestyle='--',linewidth=.7)
 summary.to_csv(out/'five_fraction_comparison.csv',index=False)
for ax in axs:ax.tick_params(top=False,right=False);ax.minorticks_off();ax.grid(False)
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=300);fig.savefig(out/'comparison.pdf');plt.close(fig)
