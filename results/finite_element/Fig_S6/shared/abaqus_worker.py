from __future__ import print_function
import os, sys, json, math, csv, traceback, builtins
from pathlib import Path
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path(sys.argv[-2]).resolve().parents[1] / 'shared'
sys.path.insert(0,str(ROOT/'sources'))
import simulation_source as sim
from abaqus import mdb
from odbAccess import openOdb
import numpy as np

def write_csv(path, rows):
    with open(str(path),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def field(frame, name, count):
    if name not in frame.fieldOutputs:
        if frame.frameValue == 0: return {}, []
        raise RuntimeError('Missing required field '+name)
    output=frame.fieldOutputs[name]
    groups={}
    for v in output.values:
        label = v.nodeLabel if name in ('U','RF') else v.elementLabel
        data=np.atleast_1d(v.data).astype(float)
        groups.setdefault(label,[]).append(data)
    if name=='ELSE' and any(len(v)!=1 for v in groups.values()):
        raise RuntimeError('ELSE must have one whole-element value per element')
    result={k:np.mean(v,axis=0) for k,v in groups.items()}
    if len(result)!=count: raise RuntimeError('Incomplete field coverage '+name)
    return result,list(output.componentLabels)

def run(config, sample):
    cfg=json.loads(Path(config).read_text(encoding='utf-8'))
    meta=json.loads((ROOT/'samples'/(sample+'.json')).read_text(encoding='utf-8'))
    task=cfg['task']; job='R22_'+task+'_'+sample
    if Path(job+'.odb').exists():
        suffix=1
        while Path(job+'_retry%02d.odb' % suffix).exists(): suffix+=1
        job=job+'_retry%02d' % suffix
    sim.NUM_CPUS=cfg['cpus'];sim.NUM_GPUS=0;sim.MEMORY_PCT=cfg['memory_percent'];sim.BASE_DIR=os.getcwd()
    sim.import_model(str(ROOT/'samples'/(sample+'.inp')),'Model')
    sim.define_material_and_section('Model');sim.assemble_and_step('Model')
    mdl=mdb.models['Model']
    if task=='loadpath':
        instance=list(mdl.rootAssembly.instances.values())[0]
        element_set=mdl.rootAssembly.Set(name='LOADPATH_ALL_ELEMS',elements=instance.elements)
        mdl.HistoryOutputRequest(name='H-SECTION-MOMENT',createStepName='Step-1',variables=('SM1','SM2','SM3'),region=element_set,numIntervals=100)
    required={'deformation':('U','RF'),'loadpath':('U','RF','SF'),'energy':('U','RF','ELSE')}[task]
    mdl.fieldOutputRequests['F-Output-1'].setValues(variables=required,numIntervals=100)
    mdl.historyOutputRequests['H-ENERGY'].setValues(variables=('ALLIE','ALLKE','ALLSE','ALLAE','ALLVD','ALLWK','ETOTAL'),numIntervals=100)
    if not sim.submit_job('Model',job): raise RuntimeError('Abaqus job did not complete')
    odb=openOdb(job+'.odb',readOnly=True)
    try:
        step=odb.steps['Step-1']; nodes=np.asarray(meta['nodes']);edges=meta['edges'];ne=len(edges)
        top={i+1 for i,p in enumerate(nodes) if abs(p[1]-100)<1e-6}
        hist={};moment_histories={}
        for region in step.historyRegions.values():
            for key,output in region.historyOutputs.items():
                if key in ('ALLIE','ALLKE','ALLSE','ALLAE','ALLVD','ALLWK','ETOTAL'):
                    hist[key]=list(output.data)
            if task=='loadpath':
                point=getattr(region,'point',None)
                element=getattr(point,'element',None) if point else None
                if element is not None:
                    section_data={}
                    for name in ('SM1','SM2','SM3'):
                        if name in region.historyOutputs:
                            section_data[name]=list(region.historyOutputs[name].data)
                    if section_data:
                        moment_histories.setdefault(element.label,[]).append(section_data)
        if task=='loadpath':
            missing=[e['element'] for e in edges if e['element'] not in moment_histories]
            if missing: raise RuntimeError('Missing section moment histories for elements '+str(missing[:10]))
        for key in ('ALLIE','ALLKE','ALLSE'):
            if key not in hist: raise RuntimeError('Missing history '+key)
        rows=[]; snapshots=[]; displacement=[];force=[];energy_errors=[];components={}
        snapshot_indices={min(range(len(step.frames)),key=lambda j:abs(step.frames[j].frameValue/sim.EXPLICIT_TIME_PERIOD-t)) for t in (.2871407,.5,1.)}
        mass_kg=builtins.sum([e['length_mm'] for e in edges])*sim.PROFILE_A*sim.PROFILE_B*sim.MATERIAL_DENSITY*1000
        for fi,frame in enumerate(step.frames):
            ud,_=field(frame,'U',len(nodes));rf,_=field(frame,'RF',len(nodes))
            if not ud or not rf: raise RuntimeError('Missing nodal output')
            u=float(np.mean([ud[n][1] for n in top]));f=float(builtins.sum([rf[n][1] for n in top]))
            displacement.append(u);force.append(f)
            row=dict(sample=sample,family=meta['family'],time=frame.frameValue,strain=u/100.,displacement_mm=u,force_N=f,mass_kg=mass_kg)
            extra={}
            if task=='loadpath':
                sf,labels=field(frame,'SF',ne)
                force_idx=[i for i,name in enumerate(labels) if name.startswith('SF')]
                components={'SF':labels,'SM':['SM1','SM2','SM3']}
                if sf and 'SF1' not in labels: raise RuntimeError('Cannot identify axial component SF1: '+str(labels))
                for e in edges:
                    k=e['element'];values=sf[k] if sf else np.zeros(0)
                    moment_norm=0.
                    if fi in snapshot_indices:
                        moment_components=[]
                        for name in ('SM1','SM2','SM3'):
                            values_at_points=[]
                            for history in moment_histories[k]:
                                data=history.get(name)
                                if data:
                                    arr=np.asarray(data)
                                    values_at_points.append(float(np.interp(frame.frameValue,arr[:,0],arr[:,1])))
                            if not values_at_points: raise RuntimeError('Missing %s history for element %s' % (name,k))
                            moment_components.append(float(np.mean(values_at_points)))
                        moment_norm=float(np.linalg.norm(moment_components))
                    extra[k]={'axial_force_N':float(values[labels.index('SF1')]) if sf else 0.,
                               "section_force_norm_N":float(np.linalg.norm(values[force_idx])) if sf else 0.,
                               "section_moment_norm_Nmm":moment_norm}
                weights=np.array([abs(extra[e['element']]['axial_force_N'])*e['length_mm'] for e in edges])
                row['axial_load_participation']=float(weights.sum()**2/(ne*np.dot(weights,weights))) if np.dot(weights,weights)>0 else 0.
                row['internal_axial_load_fraction']=float(builtins.sum([weights[i] for i,e in enumerate(edges) if e['category']=='internal'])/weights.sum()) if weights.sum()>0 else 0.
            elif task=='energy':
                energy,_=field(frame,'ELSE',ne)
                for e in edges: extra[e['element']]={'elastic_energy_Nmm':float(energy[e['element']][0]) if energy else 0.}
                total=builtins.sum([v['elastic_energy_Nmm'] for v in extra.values()])
                ref=float(np.interp(frame.frameValue,*np.asarray(hist['ALLSE']).T))
                if abs(ref)>1e-7: energy_errors.append(abs(total-ref)/abs(ref))
                row['elastic_energy_Nmm']=total
                row['internal_energy_fraction']=builtins.sum([extra[e['element']]['elastic_energy_Nmm'] for e in edges if e['category']=='internal'])/total if total>0 else 0.
                energies=np.array([extra[e['element']]['elastic_energy_Nmm'] for e in edges])
                row['energy_participation']=float(total**2/(ne*np.dot(energies,energies))) if np.dot(energies,energies)>0 else 0.
            measures=[]
            for e in edges:
                k=e['element'];a=e['n1'];b=e['n2'];p=nodes[a-1];q=nodes[b-1]
                dv=q-p;deformed=dv+ud[b]-ud[a]
                angle=math.atan2(deformed[1],deformed[0])-math.atan2(dv[1],dv[0])
                angle=(angle+math.pi)%(2*math.pi)-math.pi
                measures.append((e['length_mm'],abs(angle)*180/math.pi,abs(np.linalg.norm(deformed)/e['length_mm']-1)))
                if fi in snapshot_indices:
                    value=dict(sample=sample,family=meta['family'],strain=u/100.,element=k,category=e['category'],length_mm=e['length_mm'],
                               x0=p[0]+ud[a][0],y0=p[1]+ud[a][1],x1=q[0]+ud[b][0],y1=q[1]+ud[b][1],
                               rotation_deg=abs(angle)*180/math.pi,chord_extension=np.linalg.norm(deformed)/e['length_mm']-1)
                    value.update(extra.get(k,{}));snapshots.append(value)
            if task=='deformation':
                lengths,angles,extensions=np.asarray(measures).T
                row['rotation_deg']=float(np.average(angles,weights=lengths))
                row['absolute_chord_extension']=float(np.average(extensions,weights=lengths))
                row['extension_p90']=float(np.percentile(extensions,90))
            rows.append(row)
        if abs(rows[-1]['strain']-.5)>.005: raise RuntimeError('Target strain not reached')
        fsign=1 if np.trapz(force,displacement)>=0 else -1
        work=np.r_[0,np.cumsum((np.array(force[:-1])+np.array(force[1:]))*.5*np.diff(displacement))]*fsign*.001
        for row,w in zip(rows,work): row['loading_work_J']=float(w);row['SEA_J_kg']=float(w/mass_kg)
        ratios=[]
        for t,ke in hist['ALLKE']:
            ie=float(np.interp(t,*np.asarray(hist['ALLIE']).T))
            strain=float(np.interp(t,[r['time'] for r in rows],[r['strain'] for r in rows]))
            if strain>=.05 and abs(ie)>1e-7:ratios.append(abs(ke/ie))
        if not ratios: raise RuntimeError('Insufficient dynamic energy histories')
        check=dict(completed=True,target_strain=rows[-1]['strain'],mass_kg=mass_kg,SEA_J_kg=rows[-1]['SEA_J_kg'],
                   max_KE_IE_after_005=max(ratios),median_KE_IE_after_005=float(np.median(ratios)),
                   dynamic_review_required=max(ratios)>.1,field_components=components,
                   max_ELSE_ALLSE_relative_error=max(energy_errors) if energy_errors else None)
        if task=='energy' and (not energy_errors or max(energy_errors)>.05): raise RuntimeError('Element/global elastic energy mismatch >5%')
        write_csv(Path('curves.csv'),rows);write_csv(Path('element_snapshots.csv'),snapshots)
        Path('history.json').write_text(json.dumps(hist,indent=2))
        Path('status.json').write_text(json.dumps(check,indent=2))
    finally: odb.close()

if __name__=='__main__':
    try: run(sys.argv[-2],sys.argv[-1])
    except Exception:
        Path('failure.txt').write_text(traceback.format_exc());traceback.print_exc();sys.exit(1)
