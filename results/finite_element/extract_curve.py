from pathlib import Path
import argparse,csv,numpy as np
from odbAccess import openOdb
p=argparse.ArgumentParser();p.add_argument('odb');p.add_argument('--height',type=float,required=True);p.add_argument('--mass-kg',type=float,required=True);p.add_argument('--output',required=True);a=p.parse_args()
odb=openOdb(str(Path(a.odb).resolve()),readOnly=True)
try:
 step=list(odb.steps.values())[-1];ins=list(odb.rootAssembly.instances.values())[0]
 top=odb.rootAssembly.nodeSets['TOP'] if 'TOP' in odb.rootAssembly.nodeSets else ins.nodeSets['TOP']
 rows=[];work=0.;previous=None
 for fr in step.frames:
  if 'U' not in fr.fieldOutputs or 'RF' not in fr.fieldOutputs: continue
  def values(key):
   result=[]
   for v in fr.fieldOutputs[key].getSubset(region=top).values:
    try: result.append(v.data[1])
    except Exception: result.append(v.dataDouble[1])
   return result
  u=float(np.mean(values('U')));force=float(np.sum(values('RF')))
  if previous is not None:work+=(force+previous[1])*.5*(u-previous[0])*.001
  rows.append(dict(time_s=fr.frameValue,strain=u/a.height,U2_mm=u,RF2_N=force,loading_work_J=work,SEA_J_kg=work/a.mass_kg));previous=(u,force)
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
finally:odb.close()
