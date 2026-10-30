from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parent
curves=pd.read_csv(ROOT/'curve_plot_data.csv')
points=pd.read_csv(ROOT/'scatter_plot_data.csv')
plt.rcParams.update({'font.family':'Aptos','font.size':12,'axes.linewidth':1,
                     'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,2,figsize=(8.8,6.6),layout='constrained')
for row,group in enumerate(['low','middle']):
    left,right=axes[row]
    part=points[points.group.eq(group)]
    assert len(part)==5
    for sample in part.itertuples():
        c=curves[curves.sample_id.eq(sample.sample_id)].sort_values('strain')
        left.plot(c.strain,c.FEA_SEA,color=sample.color,lw=1.6)
        left.plot(c.strain,c.GNN_SEA,color=sample.color,lw=1.6,ls='--')
        right.scatter(sample.FEA_SEA_last_actual,sample.GNN_SEA_at_0p5,color=sample.color,s=45)
    left.set(xlim=(0,.5),ylim=(0,None),xlabel='Strain',ylabel='SEA (J/kg)')
    left.legend(handles=[Line2D([],[],color='#4D4D4D',label='FEA'),Line2D([],[],color='#4D4D4D',ls='--',label='GNN')],frameon=False)
    values=np.r_[part.FEA_SEA_last_actual,part.GNN_SEA_at_0p5]
    margin=max(3,np.ptp(values)*.15); lo=values.min()-margin; hi=values.max()+margin
    right.plot([lo,hi],[lo,hi],color='#9E9E9E',lw=1)
    right.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='FEA SEA (J/kg)',ylabel='GNN SEA (J/kg)')
    right.set_aspect('equal',adjustable='box')
for i,ax in enumerate(axes.flat):
    ax.tick_params(direction='in',top=False,right=False)
    ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.text(-.15,1.04,'('+chr(97+i)+')',transform=ax.transAxes,fontsize=14)
for ext in ['png','pdf']:
    fig.savefig(ROOT/f'expanded_numerical_validation.{ext}',dpi=400)
plt.close(fig)
