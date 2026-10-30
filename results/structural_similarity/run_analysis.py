import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

# Bound BLAS threads before importing numerical libraries.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from network_similarity import representations, ExactKernelIndex

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
OUT = ROOT/'outputs'
OUT.mkdir(exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    package = ROOT.parents[1]
    manifest = pd.read_csv(package/'splits/data_split_manifest.csv').sort_values('dataset_index')
    assert manifest.graph_id.is_unique and len(manifest) == 5899
    train_idx, test_idx = train_test_split(list(range(len(manifest))), test_size=.15, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=.15/.85, random_state=42)
    for name, indices in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]:
        assert set(manifest.loc[manifest.split == name, 'dataset_index']) == set(indices)
    for column in ['graph_path', 'label_path']:
        manifest[column] = manifest[column].map(lambda value: str(package/value))
    endpoints = pd.read_csv(ROOT/'nearest_neighbor_885.csv')[['test_sample','FEA_SEA','GNN_SEA','absolute_error']]
    assert set(endpoints.test_sample) == set(manifest.loc[manifest.split == 'test','graph_id'])
    return manifest, endpoints


def plot(table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family':'Aptos','font.size':12,'font.weight':'normal',
                         'axes.linewidth':1.1,'xtick.direction':'in','ytick.direction':'in',
                         'xtick.minor.visible':False,'ytick.minor.visible':False,'pdf.fonttype':42,'ps.fonttype':42})
    ordered = table.sort_values(['structural_similarity','test_sample']).copy()
    ordered['group'] = np.repeat([1,2,3], 295)
    summary = ordered.groupby('group').agg(sample_count=('test_sample','size'),
          similarity_min=('structural_similarity','min'),similarity_max=('structural_similarity','max'),
          MAE=('absolute_error','mean')).reset_index()
    summary.to_csv(OUT/'three_equal_count_groups.csv', index=False, encoding='utf-8-sig')
    ordered.to_csv(OUT/'panel_b_plot_data.csv', index=False, encoding='utf-8-sig')
    fig, axes = plt.subplots(1,2,figsize=(9,3.6))
    ax = axes[0]
    counts, edges, _ = ax.hist(table.structural_similarity,bins=np.linspace(0,1,21),color='#2F5597',edgecolor='white',linewidth=.6)
    pd.DataFrame({'bin_left':edges[:-1],'bin_right':edges[1:],'count':counts.astype(int)}).to_csv(OUT/'panel_a_histogram_data.csv',index=False)
    ax.set(xlabel='Nearest-neighbor similarity',ylabel='Number of test samples',xlim=(0,1))
    ax.yaxis.set_major_locator(MaxNLocator(5,integer=True))
    ax = axes[1]
    colors = ['#2F5597','#C55A11','#1E5A29']
    for row, color in zip(summary.itertuples(), colors):
        subset = ordered[ordered.group == row.group]
        ax.scatter(subset.structural_similarity,subset.absolute_error,s=13,color=color,alpha=.45,edgecolors='none',rasterized=True)
        ax.plot([row.similarity_min,row.similarity_max],[row.MAE,row.MAE],color=color,lw=1.7)
        ax.text(.03,.95-(row.group-1)*.09,f'Group {row.group} MAE = {row.MAE:.2f}',color=color,transform=ax.transAxes,va='top',fontsize=10)
    ax.set(xlabel='Nearest-neighbor similarity',ylabel=r'Absolute error (J kg$^{-1}$)',xlim=(0,1))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    for ax, label in zip(axes,['(a)','(b)']):
        ax.text(-.13,1.04,label,transform=ax.transAxes,fontsize=15,fontweight='normal')
        ax.set_xticks(np.linspace(0,1,6)); ax.tick_params(top=False,right=False,length=4,width=1)
        ax.grid(False); ax.minorticks_off()
    fig.tight_layout(w_pad=2.2)
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'nearest_neighbor_error_analysis.{ext}',dpi=600,bbox_inches='tight')
    plt.close(fig)


def run(manifest, endpoints, sensitivity):
    cache = OUT/'representations.json'
    signature = sha(ROOT.parents[1]/'splits/data_split_manifest.csv') + sha(ROOT/'network_similarity.py')
    run_signature = signature + sha(ROOT/'config.json') + sha(ROOT/'nearest_neighbor_885.csv')
    guard = OUT/'configuration_signature.txt'
    if guard.exists():
        assert guard.read_text() == run_signature, 'Configuration changed; move aside old outputs before rerunning.'
    else:
        guard.write_text(run_signature)
    if cache.exists():
        stored = json.loads(cache.read_text())
        assert stored['signature'] == signature, 'Representation cache stale; remove outputs/representations.json and rerun.'
        reps = stored['representations']
    else:
        reps = {}
        audits = []
        used = manifest[manifest.split.isin(['train','test'])]
        for i,r in enumerate(used.itertuples()):
            assert sha(r.graph_path) == r.graph_sha256, 'Dataset changed: '+r.graph_id
            reps[r.graph_id] = representations(json.loads(Path(r.graph_path).read_text()))
            audits.append({'graph_id':r.graph_id,'split':r.split,**reps[r.graph_id]['audit']})
            if i % 250 == 0:
                print(f'Extracted {i}/{len(used)} networks',flush=True)
        cache.write_text(json.dumps({'signature':signature,'representations':reps}),encoding='utf-8')
        pd.DataFrame(audits).to_csv(OUT/'extraction_audit.csv',index=False,encoding='utf-8-sig')
    train_ids = sorted(manifest.loc[manifest.split=='train','graph_id'])
    test_ids = sorted(manifest.loc[manifest.split=='test','graph_id'])
    endpoint_index = endpoints.set_index('test_sample')
    tolerances = [CFG['coordinate_tolerance']]
    if sensitivity:
        tolerances += [t for t in CFG['sensitivity_tolerances'] if t not in tolerances]
    summaries = []
    for tolerance in tolerances:
        index = ExactKernelIndex([reps[x] for x in train_ids],tolerance,CFG['angle_tolerance_degrees'])
        partial = OUT/f'checkpoint_tolerance_{tolerance:.3f}.csv'
        rows = pd.read_csv(partial).to_dict('records') if partial.exists() else []
        completed = {r['test_sample'] for r in rows}
        assert completed.issubset(test_ids) and len(completed)==len(rows)
        start = time.time()
        for sample in test_ids:
            if sample in completed:
                continue
            scores = index.similarities(reps[sample])
            k = int(np.argmax(scores))
            ep = endpoint_index.loc[sample]
            rows.append({'test_sample':sample,'nearest_training_sample':train_ids[k],
                         'structural_similarity':float(scores[k]),'FEA_SEA':float(ep.FEA_SEA),
                         'GNN_SEA':float(ep.GNN_SEA),'absolute_error':float(ep.absolute_error)})
            if len(rows)%10==0 or len(rows)==885:
                temporary = partial.with_suffix('.tmp')
                pd.DataFrame(rows).to_csv(temporary,index=False,encoding='utf-8-sig')
                temporary.replace(partial)
                print(f'tolerance={tolerance}: {len(rows)}/885, elapsed={time.time()-start:.1f}s',flush=True)
        table = pd.DataFrame(rows).sort_values('test_sample')
        assert len(table)==885 and table.test_sample.is_unique
        summaries.append({'tolerance':tolerance,'mean_similarity':table.structural_similarity.mean(),
                          'median_similarity':table.structural_similarity.median(),
                          'overall_endpoint_MAE':table.absolute_error.mean()})
        ordered = table.sort_values(['structural_similarity','test_sample'])
        for i,g in enumerate(np.array_split(ordered,3),1):
            summaries[-1][f'group_{i}_MAE'] = g.absolute_error.mean()
        if tolerance == CFG['coordinate_tolerance']:
            table.to_csv(OUT/'nearest_neighbor_885.csv',index=False,encoding='utf-8-sig')
            table.rename(columns=dict(zip(table.columns,['测试样本','最近邻训练样本','结构相似度','FEA SEA','GNN SEA','绝对误差']))).to_csv(OUT/'最近邻结果_885行.csv',index=False,encoding='utf-8-sig')
            plot(table)
    pd.DataFrame(summaries).to_csv(OUT/'tolerance_sensitivity_summary.csv',index=False,encoding='utf-8-sig')
    (OUT/'run_complete.json').write_text(json.dumps({'test_count':885,'training_count':4129,
       'neighbors_compared_per_test':4129,'configuration':CFG,'source_signature':signature},indent=2),encoding='utf-8')
    print('Completed. Primary table has exactly 885 rows and 6 columns.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--sensitivity',action='store_true')
    args = parser.parse_args()
    manifest, endpoints = prepare()
    if not args.preflight:
        run(manifest,endpoints,args.sensitivity)
