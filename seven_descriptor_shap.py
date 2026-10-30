from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

FEATURE_NAMES=['N_nodes','E_edges','Mean_Edge_Len','Std_Edge_Len','Sparsity',
               'Orientation_fraction_0_60deg','Orientation_fraction_60_120deg']
LABELS=['Node count','Edge count','Mean edge length','Edge-length SD','Sparsity',
        'Orientation fraction 0–60°','Orientation fraction 60–120°']

def extract_seven_descriptors(json_path, inside_hexagon):

    with open(json_path,encoding='utf-8') as f:obj=json.load(f)
    nodes=obj.get('nodes',obj.get('vertices',[])); links=obj.get('links',obj.get('edges',[]))
    vertices=[]; seen=set()
    for node in nodes:
        label=node.get('label') if isinstance(node,dict) else None
        if label in ['V1','V2','V3','V4','V5','V6'] and label not in seen:
            vertices.append(node); seen.add(label)
        if len(vertices)==6:break
    if len(vertices)!=6:return None
    def pos(node):
        if isinstance(node,dict):
            if node.get('pos') is not None and len(node['pos'])>=2:return float(node['pos'][0]),float(node['pos'][1])
            if 'x' in node and 'y' in node:return float(node['x']),float(node['y'])
            nid=node.get('id')
            if isinstance(nid,str) and ',' in nid:
                a,b=nid.split(',',1); return float(a),float(b)
        return 0.,0.
    vp=[pos(n) for n in vertices]; center=tuple(np.mean(vp,axis=0))
    polygon=[pos(n) for n in sorted(vertices,key=lambda n:math.atan2(pos(n)[1]-center[1],pos(n)[0]-center[0]))]
    selected=[]; positions={}
    for node in nodes:
        p=pos(node)
        if inside_hexagon(p,polygon,center):
            nid=node.get('id',node.get('name',len(positions))) if isinstance(node,dict) else len(positions)
            selected.append(node); positions[str(nid)]=np.array(p,dtype=float)
    edges=[]; unique=set()
    for edge in links:
        if not isinstance(edge,dict):continue
        u=str(edge.get('source',edge.get('u'))); v=str(edge.get('target',edge.get('v')))
        if u in positions and v in positions and u!=v:
            key=frozenset([u,v])
            if key not in unique:unique.add(key); edges.append((u,v))
    lengths=[]; directions=np.zeros(3,dtype=int)
    for u,v in edges:
        dx,dy=positions[v]-positions[u]; lengths.append(float(np.hypot(dx,dy)))
        angle=float(np.degrees(np.arctan2(dy,dx)))%180.
        directions[min(int(angle//60.),2)]+=1
    n=len(selected); e=len(edges)
    mean=float(np.mean(lengths)) if e else 0.
    sd=float(np.std(lengths,ddof=1)) if e>1 else 0.
    sparsity=float(np.clip(1-2*e/(n*(n-1)),0,1)) if n>1 else 1.
    fractions=directions/max(e,1)
    return np.array([n,e,mean,sd,sparsity,fractions[0],fractions[1]],dtype=float)

def analyze_seven_descriptors(X,y,outdir,*,sample_ids=None,seed=42,
                              fidelity_fraction=.2,predictor_path=None,make_plots=True,
                              saved_splits=None):
    X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float).reshape(-1)
    if X.ndim!=2 or X.shape[1]!=7 or len(X)!=len(y):raise ValueError('Expected N-by-7 descriptors with N endpoint targets')
    if len(y)<20:raise ValueError('At least 20 samples required for fitting/validation/holdout')
    if not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('Nonfinite descriptors/targets')
    if not 0<fidelity_fraction<.5:raise ValueError('Holdout fraction must be between 0 and .5')
    ids=list(sample_ids) if sample_ids is not None else list(range(len(y)))
    if len(ids)!=len(y) or len(set(ids))!=len(ids):raise ValueError('Sample IDs must be unique and match descriptor rows')
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(X,columns=FEATURE_NAMES)
    keys=df.round(12).astype(str).agg('|'.join,axis=1)
    groups=pd.factorize(keys,sort=True)[0]; all_groups=np.unique(groups)
    if saved_splits is None:
        fitting_groups,held_groups=train_test_split(all_groups,test_size=fidelity_fraction,random_state=seed)
        fitting_groups,val_groups=train_test_split(fitting_groups,test_size=.2,random_state=seed+1)
        train=np.flatnonzero(np.isin(groups,fitting_groups)); val=np.flatnonzero(np.isin(groups,val_groups))
        held=np.flatnonzero(np.isin(groups,held_groups))
    else:
        saved_splits=np.asarray(saved_splits,dtype=str)
        if len(saved_splits)!=len(y) or set(saved_splits)!={'fit','validation','final_holdout'}:
            raise ValueError('Saved splits must contain fit, validation and final_holdout for every row')
        if pd.DataFrame({'group':groups,'split':saved_splits}).groupby('group')['split'].nunique().max()>1:
            raise ValueError('Identical descriptor groups occur in multiple saved splits')
        train=np.flatnonzero(saved_splits=='fit'); val=np.flatnonzero(saved_splits=='validation')
        held=np.flatnonzero(saved_splits=='final_holdout')
    if min(len(train),len(val),len(held))<2:raise ValueError('Too few independent descriptor groups')
    df.insert(0,'graph_id',ids); df['GNN_SEA_J_per_kg']=y; df['descriptor_group']=groups
    df['proxy_split']='fit'; df.loc[val,'proxy_split']='validation'; df.loc[held,'proxy_split']='final_holdout'
    df.to_csv(out/'SHAP_seven_descriptors_input_table.csv',index=False,encoding='utf-8-sig')
    metric_rows=[]; all_predictions=[]; parameters=[]; fitted={}
    for label,width in [('Original_5',5),('Original_5_plus_orientation',7)]:
        scaler=StandardScaler().fit(X[train,:width])
        model=LinearRegression().fit(scaler.transform(X[train,:width]),y[train])
        beta=model.coef_/scaler.scale_
        for split,idx in [('fit',train),('validation',val),('final_holdout',held)]:
            p=model.predict(scaler.transform(X[idx,:width]))
            metric_rows.append(dict(proxy=label,split=split,n=len(idx),R2=float(r2_score(y[idx],p)),
               MAE_J_per_kg=float(mean_absolute_error(y[idx],p)),RMSE_J_per_kg=float(np.sqrt(mean_squared_error(y[idx],p)))))
            all_predictions.extend(dict(proxy=label,split=split,graph_id=ids[i],GNN_SEA_J_per_kg=float(y[i]),
                proxy_prediction_J_per_kg=float(v)) for i,v in zip(idx,p))
        parameters.extend(dict(proxy=label,feature=FEATURE_NAMES[j],raw_coefficient=float(beta[j]),
             fit_mean=float(scaler.mean_[j]),fit_scale=float(scaler.scale_[j]),
             expected_value_J_per_kg=float(model.intercept_)) for j in range(width))
        fitted[label]=(scaler,model)
    pd.DataFrame(metric_rows).to_csv(out/'SHAP_proxy_fidelity_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(all_predictions).to_csv(out/'SHAP_proxy_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(parameters).to_csv(out/'SHAP_proxy_parameters.csv',index=False,encoding='utf-8-sig')
    scaler,model=fitted['Original_5_plus_orientation']
    beta=model.coef_/scaler.scale_; phi=(X[held]-scaler.mean_)*beta
    expected=float(model.intercept_); prediction=model.predict(scaler.transform(X[held]))
    residual=float(np.max(np.abs(phi.sum(1)+expected-prediction)))
    if not np.allclose(phi.sum(1)+expected,prediction,atol=1e-8,rtol=1e-10):raise RuntimeError('Linear SHAP additivity failed')
    values=df.iloc[held].copy(); values['proxy_prediction_J_per_kg']=prediction
    values['SHAP_expected_J_per_kg']=expected
    for j,name in enumerate(FEATURE_NAMES):values['SHAP_'+name]=phi[:,j]
    values.to_csv(out/'SHAP_seven_descriptors_values.csv',index=False,encoding='utf-8-sig')
    importance=np.abs(phi).mean(0); order=np.argsort(importance)[::-1]
    pd.DataFrame(dict(feature=FEATURE_NAMES,mean_abs_SHAP_J_per_kg=importance,
       raw_coefficient=beta)).sort_values('mean_abs_SHAP_J_per_kg',ascending=False).to_csv(
       out/'SHAP_seven_descriptors_importance.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(X[train],columns=FEATURE_NAMES).corr().to_csv(out/'SHAP_seven_descriptors_fit_correlation.csv',encoding='utf-8-sig')
    plot_rows=[]
    for rank,j in enumerate(order):
        low,high=np.percentile(X[held,j],[2,98]); colors=np.clip((X[held,j]-low)/max(high-low,1e-12),0,1)
        jitter=np.random.default_rng(seed+j).uniform(-.22,.22,len(held))
        for k,i in enumerate(held):plot_rows.append(dict(graph_id=ids[i],feature=FEATURE_NAMES[j],
             SHAP_J_per_kg=float(phi[k,j]),jittered_y=float(rank+jitter[k]),color_normalized_0_1=float(colors[k])))
    pd.DataFrame(plot_rows).to_csv(out/'SHAP_seven_descriptors_plot_data.csv',index=False,encoding='utf-8-sig')
    protocol=dict(method='Exact interventional SHAP for seven-descriptor linear proxy',seed=seed,
      orientation_reference='120–180 degrees; undirected angle modulo 180 relative to global +x',
      cell='First V1–V6 polygon; not necessarily center cell',fit_n=len(train),validation_n=len(val),holdout_n=len(held),
      scaling='Descriptors only, fitted on fitting groups; targets remain J/kg',
      interpretation='Proxy-level associations, not direct GNN attributions or mechanical causality',
      expected_value_J_per_kg=expected,max_additivity_error_J_per_kg=residual,
      predictor_sha256=hashlib.sha256(Path(predictor_path).read_bytes()).hexdigest() if predictor_path else None)
    (out/'SHAP_seven_descriptors_protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    if make_plots:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap,Normalize
        from matplotlib.ticker import MaxNLocator
        cmap=LinearSegmentedColormap.from_list('paper_blue_orange',['#2F5597','#D9D9D9','#C55A11'])
        def finish(fig,ax,stem):
            ax.grid(False); ax.tick_params(direction='in'); ax.minorticks_off()
            ax.xaxis.set_major_locator(MaxNLocator(5))
            for spine in ax.spines.values():spine.set_linewidth(.95)
            fig.savefig(out/(stem+'.png'),dpi=600,bbox_inches='tight')
            fig.savefig(out/(stem+'.pdf'),bbox_inches='tight'); plt.close(fig)
        fig,ax=plt.subplots(figsize=(3.6,2.8))
        frame=pd.DataFrame(plot_rows)
        for j in order:
            part=frame[frame.feature==FEATURE_NAMES[j]]
            ax.scatter(part.SHAP_J_per_kg,part.jittered_y,c=part.color_normalized_0_1,
                  cmap=cmap,vmin=0,vmax=1,s=6,alpha=.7,linewidths=0)
        ax.set_yticks(range(7),[LABELS[j] for j in order]); ax.invert_yaxis()
        ax.axvline(0,c='#9E9E9E',lw=.7,ls='--'); ax.set_xlabel('SHAP value (J/kg)')
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=cmap),ax=ax,pad=.02)
        cb.set_ticks([0,1],labels=['Low','High']); cb.set_label('Feature value')
        finish(fig,ax,'SHAP_seven_descriptors_beeswarm')
        fig,ax=plt.subplots(figsize=(3.3,2.5)); ax.barh(range(7),importance[order],color='#2F5597')
        ax.set_yticks(range(7),[LABELS[j] for j in order]); ax.invert_yaxis(); ax.set_xlabel('Mean |SHAP| (J/kg)')
        finish(fig,ax,'SHAP_seven_descriptors_importance')
        fig,ax=plt.subplots(figsize=(3.1,2.8)); ax.scatter(y[held],prediction,s=9,color='#C55A11',alpha=.6)
        low=min(y[held].min(),prediction.min()); high=max(y[held].max(),prediction.max())
        ax.plot([low,high],[low,high],c='#4D4D4D',lw=1,ls='--')
        ax.set(xlabel='GNN prediction (J/kg)',ylabel='Proxy prediction (J/kg)')
        finish(fig,ax,'SHAP_seven_descriptors_fidelity')
    return dict(metrics=metric_rows,holdout=held,fit=train,validation=val,phi=phi,
                prediction=prediction,expected=expected,protocol=protocol)
