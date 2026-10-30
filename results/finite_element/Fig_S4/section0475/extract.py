from pathlib import Path
import sys,csv,json,traceback
import numpy as np
from odbAccess import openOdb
ROOT=Path(sys.argv[-1])
def write(p,rows):
    if not rows:return
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def arr(field):
    values=[]
    for v in field.values:
        try:values.append(v.data)
        except Exception:values.append(v.dataDouble)
    return np.array(values,dtype=float)
saved=ROOT/'results/extraction.json'
summ=json.loads(saved.read_text()) if saved.exists() else []
done={r['job'] for r in summ if r.get('completed')}
for meta in csv.DictReader((ROOT/'results/manifest.csv').open(encoding='utf-8-sig')):
    name=meta['job'];dest=ROOT/'results'/(name+'_curve_to50.csv')
    if name in done:continue
    odbfile=Path(meta['odb_path']);odbfile=odbfile if odbfile.is_absolute() else ROOT/odbfile
    if not odbfile.exists() or odbfile.with_suffix('.lck').exists():continue
    sta=odbfile.with_suffix('.sta')
    if not sta.exists():continue
    summ=[r for r in summ if r['job']!=name]
    try:
        odb=openOdb(meta['odb_path'],readOnly=True)
        step=list(odb.steps.values())[-1];asm=odb.rootAssembly
        top=asm.nodeSets['TOP'] if 'TOP' in asm.nodeSets else list(asm.instances.values())[0].nodeSets['TOP']
        rows=[];work=0.
        for frame in step.frames:
            if 'U' not in frame.fieldOutputs or 'RF' not in frame.fieldOutputs:continue
            u=float(arr(frame.fieldOutputs['U'].getSubset(region=top))[:,1].mean())
            force=float(arr(frame.fieldOutputs['RF'].getSubset(region=top))[:,1].sum())
            if not np.isfinite(u) or not np.isfinite(force):break
            if rows:work+=(u-rows[-1]['U2_mm'])*(force+rows[-1]['RF2_N'])/2
            rows.append(dict(time_s=frame.frameValue,strain=u/float(meta['height_mm']),U2_mm=u,RF2_N=force,
                work_N_mm=work,SEA_J_kg=work/(1000*float(meta['mass_kg']))))
        sta=Path(meta['odb_path']).with_suffix('.sta')
        completed=sta.exists() and 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in sta.read_text(errors='replace')
        complete=completed and abs(rows[-1]['strain']-.5)<=.001
        write(ROOT/'results'/(name+'_curve.csv'),rows)
        if complete:write(dest,rows)
        hist={}
        for region in step.historyRegions.values():
            for k,h in region.historyOutputs.items():
                if k.startswith('ALL') or k=='ETOTAL':hist[k]=np.array(h.data,dtype=float)
        energies=[]
        if 'ALLIE' in hist:
            for t,ie in hist['ALLIE']:
                r=dict(time_s=float(t));r.update({k:float(np.interp(t,a[:,0],a[:,1])) for k,a in hist.items()})
                r['KE_IE']=r.get('ALLKE',0)/ie if ie>1e-9 else None;energies.append(r)
        write(ROOT/'results'/(name+'_energy.csv'),energies)
        valid=[r['KE_IE'] for r in energies if r['time_s']>=.01 and r['KE_IE'] is not None and np.isfinite(r['KE_IE'])]
        summ.append(dict(meta,completed=complete,solver_completed=completed,final_strain=rows[-1]['strain'],
             SEA_J_kg=rows[-1]['SEA_J_kg'],work_J=rows[-1]['work_N_mm']/1000,
             KE_IE_late_mean=float(np.mean(valid)) if valid else None,KE_IE_final=energies[-1]['KE_IE'] if energies else None))
        odb.close();print(name,complete,rows[-1]['SEA_J_kg'],flush=True)
    except Exception as exc:
        summ.append(dict(meta,error=str(exc)));traceback.print_exc()
    (ROOT/'results/extraction.json').write_text(json.dumps(summ,indent=2))
