from pathlib import Path
from copy import deepcopy
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

OUT = Path(__file__).resolve().parent
families = ['Random-6', 'Single-loop', 'Random-12']
labels = ['Random (6 segments)', 'Single-loop', 'Random (12 segments)']
colors = ['#2F5597', '#C55A11', '#1E5A29']
data = pd.read_csv(OUT / 'endpoint_predictions.csv')
metrics = []
for family in families:
    rows = data[data.family.eq(family)]
    y, p = rows.true_SEA_50.to_numpy(), rows.pred_SEA_50.to_numpy()
    assert np.isfinite(y).all() and np.isfinite(p).all() and np.all(y != 0)
    metrics.append(dict(family=family, n=len(y), MAE=mean_absolute_error(y,p),
                        RMSE=float(np.sqrt(mean_squared_error(y,p))), R2=r2_score(y,p),
                        MAPE=float(np.mean(np.abs((p-y)/y))*100), bias=float(np.mean(p-y))))
pd.DataFrame(metrics).to_csv(OUT / 'metrics.csv', index=False, encoding='utf-8-sig')
plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'axes.linewidth':1.1,'svg.fonttype':'none','pdf.fonttype':42})
fig, axes = plt.subplots(1,3,figsize=(11.2,3.8))
for i,(ax,family,label,color,metric) in enumerate(zip(axes,families,labels,colors,metrics)):
    rows=data[data.family.eq(family)]
    y,p=rows.true_SEA_50.to_numpy(),rows.pred_SEA_50.to_numpy()
    low=max(0.,float(min(y.min(),p.min()))-25.)
    high=float(max(y.max(),p.max()))+25.
    ax.plot([low,high],[low,high],color='#4D4D4D',lw=1.1)
    ax.scatter(y,p,s=24,color=color,alpha=.65,edgecolors='none')
    ax.set(xlim=(low,high),ylim=(low,high),xlabel='FEA SEA (J/kg)',ylabel='Predicted SEA (J/kg)',title=label)
    ax.text(.05,.94,rf'$R^2 = {metric["R2"]:.3f}$',transform=ax.transAxes,va='top',fontsize=15)
    ax.text(-.12,1.12,f'({chr(97+i)})',transform=ax.transAxes,fontsize=15)
    ax.tick_params(direction='in',top=True,right=True,length=4)
    ax.minorticks_off()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
fig.subplots_adjust(left=.075,right=.985,bottom=.19,top=.80,wspace=.42)
for suffix in ['png','pdf','svg']:
    fig.savefig(OUT/f'cross_family_parity.{suffix}',dpi=400,facecolor='white')
plt.close(fig)
