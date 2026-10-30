"""Abaqus Python: unscaled curves, energies, contact and local-strain diagnostics."""
from pathlib import Path
import json,csv,math,traceback
import numpy as np
import time
from odbAccess import openOdb
ROOT=Path(__file__).resolve().parent
def data(v):
    try:return v.data
    except Exception:return v.dataDouble
def arrays(field):
    try:
        blocks=[np.asarray(b.data,dtype=float) for b in field.bulkDataBlocks]
        if blocks:return np.concatenate(blocks,axis=0)
    except Exception:pass
    return np.array([data(v) for v in field.values],dtype=float)
def local_le(field,beam):
    a=arrays(field);labels=list(field.componentLabels)
    if not len(a):return np.array([])
    if beam:
        le=a[:,labels.index('LE11')];return le[np.isfinite(le)]
    if a.ndim==1:return a
    a=a[np.all(np.isfinite(a),axis=1)]
    if not len(a):return np.array([])
    ten=np.zeros((len(a),3,3))
    for key,i,j in [('LE11',0,0),('LE22',1,1),('LE33',2,2),('LE12',0,1),('LE13',0,2),('LE23',1,2)]:
        if key in labels:
            ten[:,i,j]=a[:,labels.index(key)]*(.5 if i!=j else 1.)
            ten[:,j,i]=ten[:,i,j]
    return np.linalg.eigvalsh(ten)[:,-1]
def writecsv(path,rows):
    if not rows:return
    with open(str(path),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def extract(meta):
    name=meta['job'];path=ROOT/'jobs'/(name+'.odb')
    odb=openOdb(str(path),readOnly=True)
    step=list(odb.steps.values())[-1];ins=list(odb.rootAssembly.instances.values())[0]
    top=odb.rootAssembly.nodeSets['TOP'] if 'TOP' in odb.rootAssembly.nodeSets else None
    if top is None:top=ins.nodeSets['TOP']
    beam=meta['element']=='B31';rows=[];work=0.;peak=-float('inf');final=np.array([])
    contact_pressure=0.;contact_force=0.;contact_keys=set()
    for fr in step.frames:
        if 'U' not in fr.fieldOutputs or 'RF' not in fr.fieldOutputs:continue
        uu=arrays(fr.fieldOutputs['U'].getSubset(region=top));rf=arrays(fr.fieldOutputs['RF'].getSubset(region=top))
        if not np.all(np.isfinite(uu)) or not np.all(np.isfinite(rf)):continue
        u=float(np.mean(uu[:,1]));force=float(np.sum(rf[:,1]))
        if rows:work+=.5*(force+rows[-1]['RF2_N'])*(u-rows[-1]['U2_mm'])
        le=local_le(fr.fieldOutputs['LE'],beam) if 'LE' in fr.fieldOutputs else np.array([])
        if len(le):peak=max(peak,float(np.max(le)));final=le
        for key,field in fr.fieldOutputs.items():
            if any(x in key.upper() for x in ['CPRESS','CNORMF','CSTRESS','CFORCE']):
                contact_keys.add(key);aa=arrays(field)
                if len(aa):
                    val=float(np.max(np.abs(aa)))
                    if 'CPRESS' in key.upper() or 'CSTRESS' in key.upper():contact_pressure=max(contact_pressure,val)
                    else:contact_force=max(contact_force,val)
        rows.append(dict(time_s=float(fr.frameValue),strain=u/float(meta['height_mm']),U2_mm=u,RF2_N=force,
                         work_N_mm=work,SEA_J_kg=work/(1000*float(meta['mass_kg'])),
                         max_local_engineering_strain=float(math.expm1(np.max(le))) if len(le) and np.max(le)<30 else None))
    writecsv(ROOT/'results'/(name+'_curve.csv'),rows)
    hist={}
    for reg in step.historyRegions.values():
        for key,out in reg.historyOutputs.items():
            if key.startswith('ALL') or key=='ETOTAL':hist[key]=np.array(out.data,dtype=float)
    energ=[]
    if 'ALLIE' in hist:
        tt=hist['ALLIE'][:,0];ie=hist['ALLIE'][:,1]
        for i,t in enumerate(tt):
            rr={'time_s':float(t)}
            for k,v in hist.items():rr[k]=float(np.interp(t,v[:,0],v[:,1]))
            rr['KE_IE']=rr.get('ALLKE',0)/ie[i] if ie[i]>1e-9 else None
            rr['AE_IE']=rr.get('ALLAE',0)/ie[i] if ie[i]>1e-9 else None
            energ.append(rr)
    writecsv(ROOT/'results'/(name+'_energy.csv'),energ)
    last=rows[-1] if rows else {}
    sta=ROOT/'jobs'/(name+'.sta')
    solver_complete=sta.exists() and 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in sta.read_text(errors='replace')
    # Explicit displacement integration can leave a small endpoint drift.
    # Accept successful target-time runs within 0.001 absolute strain; never
    # extrapolate an unfinished response to 0.5.
    full=solver_complete and abs(last.get('strain',0)-.5)<=.001 and last.get('time_s',0)>=float(meta['time_s'])*.9999
    sea50=None
    if full:
        clipped=[r for r in rows if r['strain']<.5]
        after=next((r for r in rows if r['strain']>=.5),None)
        if clipped and after is not None:
            before=clipped[-1];alpha=(.5-before['strain'])/(after['strain']-before['strain'])
            endpoint={k:(before[k]+alpha*(after[k]-before[k]) if isinstance(before[k],(int,float)) and isinstance(after[k],(int,float)) else None) for k in before}
            endpoint['strain']=.5;endpoint['U2_mm']=.5*float(meta['height_mm'])
            endpoint['work_N_mm']=before['work_N_mm']+.5*(before['RF2_N']+endpoint['RF2_N'])*(endpoint['U2_mm']-before['U2_mm'])
            endpoint['SEA_J_kg']=endpoint['work_N_mm']/(1000*float(meta['mass_kg']))
            clipped.append(endpoint);sea50=endpoint['SEA_J_kg']
        else:sea50=last['SEA_J_kg'];clipped=rows
        writecsv(ROOT/'results'/(name+'_curve_to50.csv'),clipped)
    result=dict(meta,complete=bool(full),final_strain=last.get('strain'),SEA_50_J_kg=sea50,
                SEA_evaluation_strain=min(.5,last.get('strain',0)) if full else None,target_strain_tolerance=.001,
                local_strain_convention='Abaqus engineering shear; maximum principal invariant checked',
                global_planar_beam_constraint='ALL_NODES,3,5,0.' in (ROOT/'jobs'/(name+'.inp')).read_text(),
                solver_precision='double' if name.endswith('_double') or '_diagnostic' in name else 'single',
                loading_work_to_last_frame_J_kg=last.get('SEA_J_kg'),peak_local_engineering_strain=math.expm1(peak) if peak<30 else None,
                final_local_strain_p99=math.expm1(float(np.percentile(final,99))) if len(final) else None,
                final_fraction_above_material_max=float(np.mean(final>math.log1p(1.522504))) if len(final) else None,
                contact_field_keys=sorted(contact_keys),maximum_contact_pressure_MPa=contact_pressure,maximum_contact_force_N=contact_force)
    for var in ['KE_IE','AE_IE']:
        valid=[r[var] for r in energ if r['time_s']>=.2*float(meta['time_s']) and r[var] is not None]
        result[var+'_late_mean']=float(np.mean(valid)) if valid else None
        result[var+'_late_max']=float(max(valid)) if valid else None
        result[var+'_final']=energ[-1].get(var) if energ else None
    odb.close();return result
def main():
    manifest=list(csv.DictReader(open(str(ROOT/'results/job_manifest.csv'),encoding='utf-8-sig')));results=[]
    for row in manifest:
        lock=ROOT/'jobs'/(row['job']+'.lck')
        if lock.exists():
            print('Waiting for active job before extraction:',row['job'],flush=True)
            deadline=time.time()+7200
            while lock.exists() and time.time()<deadline:time.sleep(5)
            if lock.exists():results.append(dict(row,error='Still running after extraction wait'));continue
        if not (ROOT/'jobs'/(row['job']+'.odb')).exists():results.append(dict(row,error='ODB not produced'));continue
        try:
            result=extract(row);results.append(result)
            print(row['job'],result['complete'],result['SEA_50_J_kg'],flush=True)
        except Exception as exc:results.append(dict(row,error=str(exc)));traceback.print_exc()
        (ROOT/'results/diagnostics.json').write_text(json.dumps(results,indent=2))
    (ROOT/'results/diagnostics.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
