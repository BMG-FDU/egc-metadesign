from pathlib import Path
import csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

R=Path(__file__).resolve().parent
MODE=json.loads((R/'config.json').read_text(encoding='utf-8'))['figure']
C=['#2F5597','#C55A11','#1E5A29','#6A51A3','#238A8D']
LABELS=['P1','P2','P3','P4','P5']
OUT=R/'output'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,
 'xtick.labelsize':7.1,'ytick.labelsize':7.1,'axes.linewidth':.7,
 'xtick.direction':'in','ytick.direction':'in','pdf.fonttype':42,'ps.fonttype':42,
 'mathtext.fontset':'stix','font.weight':'normal'})

def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def base(name):return read(R/'data/original_panels'/name)
models={r['label']:r for r in read(R/'data/models.csv')}
audit=[]
def curve(p,kind):
 if kind in ['nominal','solid','contact']:
  suffix={'nominal':'nominal','solid':'no_contact','contact':'contact'}[kind]
  path=R/'data/raw_baseline'/f'{p}_{suffix}_curve.csv'
 elif kind=='matched0475':path=R/'data/new0475'/f'{p}_section0475_curve.csv'
 else:path=R/'data/new0475'/f'{p}_section0475_curve.csv'
 rows=read(path)
 u=np.array([float(r['U2_mm']) for r in rows]);force=np.array([float(r['RF2_N']) for r in rows])
 strain=np.array([float(r['strain']) for r in rows])
 if np.min(np.diff(strain)) < -1e-7:raise ValueError(f'Nonmonotonic loading: {path}')
 work=np.r_[0,np.cumsum(np.diff(u)*(force[1:]+force[:-1])/2)]/1000
 volume=float(models[p]['solid_volume_mm3' if kind in ['solid','contact','matched0475'] else 'nominal_beam_volume_mm3'])
 mass=volume*1.15e-6
 audit.append(dict(sample=p,model=kind,file=str(path.relative_to(R)),end_strain=strain[-1],mass_kg=mass,work_J=work[-1]))
 return strain,work,work/mass,force
def compare(p,b,s):
 end=min(.5,b[0][-1],s[0][-1]);x=np.linspace(0,end,501)
 bw=np.interp(x,b[0],b[1]);sw=np.interp(x,s[0],s[1])
 bs=np.interp(x,b[0],b[2]);ss=np.interp(x,s[0],s[2])
 return dict(sample=p,common_strain=end,beam_work_J=bw[-1],solid_work_J=sw[-1],
  beam_SEA_J_kg=bs[-1],solid_SEA_J_kg=ss[-1],work_difference_pct=(bw[-1]/sw[-1]-1)*100,
  SEA_difference_pct=(bs[-1]/ss[-1]-1)*100,
  work_curve_NRMSE_pct=100*np.sqrt(np.mean((bw-sw)**2))/np.ptp(sw),
  SEA_curve_NRMSE_pct=100*np.sqrt(np.mean((bs-ss)**2))/np.ptp(ss))
def finish(fig,axs,name):
 for i,ax in enumerate(np.asarray(axs).ravel()):
  ax.text(-.13,1.035,f'({chr(97+i)})',transform=ax.transAxes,fontsize=8.5,va='bottom',fontweight='normal')
  ax.grid(False);ax.minorticks_off();ax.tick_params(direction='in',length=2.4,width=.65)
  if ax.axison:
   if not isinstance(ax.xaxis.get_major_locator(),matplotlib.ticker.FixedLocator):ax.xaxis.set_major_locator(MaxNLocator(4))
   ax.yaxis.set_major_locator(MaxNLocator(4))
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=600,bbox_inches='tight')
 plt.close(fig)
def s4():
 fig,axs=plt.subplots(2,3,figsize=(7.35,4.6));fig.subplots_adjust(wspace=.42,hspace=.5,top=.84)
 edges=base('FigS4a_geometry_edges.csv');ax=axs[0,0]
 for i,p in enumerate(LABELS):
  dx=(i%2)*37 if i<4 else 18.5;dy=(2-i//2)*27
  for row in edges:
   if row['sample']!=p:continue
   ax.plot(np.array([float(row['x1_mm']),float(row['x2_mm'])])*.2+dx,
    np.array([float(row['y1_mm']),float(row['y2_mm'])])*.2+dy,color=C[i],lw=.3)
  ax.text(dx,dy+21,p,fontsize=7.1)
 ax.set_aspect('equal',adjustable='datalim');ax.axis('off');write('FigS4a_geometry_edges.csv',edges)
 metrics=[];points=[];curves=[];ax=axs[0,1]
 for i,p in enumerate(LABELS):
  b=curve(p,'nominal');s=curve(p,'solid');m=curve(p,'matched0475')
  for kind,data,ls in [('nominal',b,'-'),('solid',s,'--')]:
   end=min(.5,data[0][-1]);x=np.linspace(0,end,501);y=np.interp(x,data[0],data[2])
   ax.plot(x,y,color=C[i],ls=ls,lw=1.1)
   curves += [dict(sample=p,model=kind,strain=a,SEA_J_kg=z) for a,z in zip(x,y)]
  for kind,data in [('nominal',b),('matched0475',m)]:
   row=compare(p,data,s);row['model']=kind;metrics.append(row);points.append(row)
 handles=[Line2D([],[],color=c,label=p) for c,p in zip(C,LABELS)]+[Line2D([],[],color='k',label='B31'),Line2D([],[],color='k',ls='--',label='C3D8R')]
 ax.legend(handles=handles,ncol=2,frameon=False,fontsize=7.1);ax.set(xlabel='Engineering strain',ylabel='Tensile loading work (J/kg)',xlim=(0,.5));ax.set_xticks(np.linspace(0,.5,6))
 write('FigS4b_loading_curves.csv',curves);write('comparison_metrics.csv',metrics);write('FigS4c_terminal_comparison.csv',points)
 ax=axs[0,2];vals=[]
 for kind,col,mark,title in [('nominal',C[0],'o',r'Nominal B31, $\nu_s=0$'),('matched0475',C[2],'s',r'Matched B31, $\nu_s=0.475$')]:
  rr=[r for r in points if r['model']==kind];x=[r['solid_SEA_J_kg'] for r in rr];y=[r['beam_SEA_J_kg'] for r in rr];vals+=x+y
  ax.scatter(x,y,s=18,color=col,marker=mark,label=title)
 lim=(min(vals)*.9,max(vals)*1.08);ax.plot(lim,lim,color='#9E9E9E',lw=1);ax.set(xlim=lim,ylim=lim,xlabel='C3D8R loading work (J/kg)',ylabel='B31 loading work (J/kg)');ax.legend(frameon=False,fontsize=7.1,loc='lower left',bbox_to_anchor=(0,1.01))
 rows=[]
 for p in LABELS:
  on=curve(p,'contact');off=curve(p,'solid');v=compare(p,on,off)
  rows.append(dict(sample=p,common_strain=v['common_strain'],contact_change_pct=v['work_difference_pct']))
 ax=axs[1,0];y=[r['contact_change_pct'] for r in rows];ax.bar(range(5),y,color=C,width=.48);ax.scatter(range(5),y,c=C,s=12);ax.axhline(0,color='#9E9E9E',lw=.8);ax.set_xticks(range(5),LABELS);lim=max(.01,max(abs(v) for v in y)*1.2);ax.set(ylim=(-lim,lim),ylabel='Contact-induced change (%)');write('FigS4d_contact_change.csv',rows)
 energy=base('FigS4e_f_energy_ratios.csv');write('FigS4e_f_energy_ratios.csv',energy)
 for ax,key,title in [(axs[1,1],'kinetic_internal_pct','Kinetic / internal energy (%)'),(axs[1,2],'artificial_internal_pct','Artificial / internal energy (%)')]:
  for i,p in enumerate(LABELS):
   rr=[r for r in energy if r['sample']==p and r[key]!=''];ax.plot([float(r['normalised_time']) for r in rr],[float(r[key]) for r in rr],color=C[i],lw=1.1)
  ax.set(xlim=(.2,1),xlabel='Normalised loading time',ylabel=title)
 finish(fig,axs,'FigS4_dynamic_beam_solid_contact')
 return metrics
def s5():
 fig,axs=plt.subplots(1,2,figsize=(7.35,3));fig.subplots_adjust(wspace=.37,top=.79,bottom=.22)
 rows=base('FigS5a_absolute_loading_work.csv');rows=[r for r in rows if r['variant']!='section049'];out=[];metrics=[]
 variants=['baseline','section0475','top_rotation_fixed','bottom_rotation_free','lateral_free']
 for p in LABELS[:3]:
  b=curve(p,'nominal');n=curve(p,'nominal0475');m=compare(p,n,b);m['model']='nominal0475_vs_baseline';metrics.append(m)
  out += [dict(sample=p,variant='baseline',SEA_J_kg=m['solid_SEA_J_kg']),dict(sample=p,variant='section0475',SEA_J_kg=m['beam_SEA_J_kg'])]
  out += [r for r in rows if r['sample']==p and r['variant']!='baseline']
 for p in ['P1','P3','P2']:
  y=[float(next(r['SEA_J_kg'] for r in out if r['sample']==p and r['variant']==v)) for v in variants]
  axs[0].plot(range(5),y,'o-',color=C[LABELS.index(p)],lw=1.1,ms=3.5,label=p)
 axs[0].set_ylabel('Tensile loading work (J/kg)');axs[0].set_xticks(range(5),['Baseline','Section\n'+r'$\nu_s=0.475$','Top\nrotation\nfixed','Bottom\nrotation\nfree','Lateral\nrelease']);axs[0].legend(frameon=False,fontsize=7.1)
 write('FigS5a_absolute_loading_work.csv',out);write('section0475_comparison_metrics.csv',metrics)
 local=base('FigS5b_local_strain.csv');write('FigS5b_local_strain.csv',local)
 for variant,col,offset,label in [('nominal',C[0],-.1,'Beam'),('contact',C[1],.1,'Solid')]:
  rr=[next(r for r in local if r['sample']==p and r['variant']==variant) for p in LABELS];xx=np.arange(5)+offset
  axs[1].scatter(xx,[float(r['peak_engineering_strain']) for r in rr],s=16,color=col,label=label+' maximum')
  axs[1].scatter(xx,[float(r['final_p99_engineering_strain']) for r in rr],s=16,facecolors='none',edgecolors=col,label=label+' 99th percentile')
 axs[1].axhline(1.522504,color='#9E9E9E',ls='--',lw=1);axs[1].set_xticks(range(5),LABELS);axs[1].set_ylabel('Local engineering strain');axs[1].legend(frameon=False,fontsize=7.1,ncol=2,loc='lower left',bbox_to_anchor=(0,1.01))
 finish(fig,axs,'FigS5_dynamic_sensitivity_local_strain');return metrics

if __name__=='__main__':
 metrics=s4() if MODE=='S4' else s5()
 write('raw_curve_audit.csv',audit)
 (OUT/'初步核查报告.md').write_text('# '+MODE+'：0.475重画\n\n采用原始RF–U积分计算响应。\n\n图c表示单位质量拉伸加载功；work_curve_NRMSE与SEA_curve_NRMSE分别报告加载功和单位质量加载功曲线误差。\n\n'+json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
 print('Output:',OUT)
