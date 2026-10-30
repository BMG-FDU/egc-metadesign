from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent

def summarize(frame, group_col, true_col, pred_col):
    rows = []
    for group, part in frame.groupby(group_col, sort=False):
        y, p = part[true_col].to_numpy(), part[pred_col].to_numpy()
        delta = p-y
        rows.append(dict(group=group, n=len(y), MAE_J_kg=np.abs(delta).mean(),
                         RMSE_J_kg=np.sqrt(np.mean(delta**2)), R2=r2_score(y,p),
                         MAPE_percent=100*np.mean(np.abs(delta/y))))
    return pd.DataFrame(rows)

for name, count in [('cross_family',300),('expanded_numerical',73)]:
    folder=ROOT/name
    manifest=pd.read_csv(folder/'sample_manifest.csv')
    assert len(manifest)==count
    for item in manifest.itertuples():
        assert hashlib.sha256((folder/item.graph).read_bytes()).hexdigest()==item.graph_sha256
    if name=='cross_family':
        data=pd.read_csv(folder/'endpoint_predictions.csv')
        result=summarize(data,'family','true_SEA_50','pred_SEA_50')
        reference=pd.read_csv(folder/'metrics.csv').set_index('family')
        for row in result.itertuples():
            assert np.isclose(row.MAE_J_kg,reference.loc[row.group,'MAE'])
            assert np.isclose(row.R2,reference.loc[row.group,'R2'])
    else:
        data=pd.read_csv(folder/'endpoint_results.csv')
        result=summarize(data,'group','FEA_SEA_last_actual','GNN_SEA_at_0p5')
        assert dict(zip(result.group,result.n))=={'low':37,'middle':36}
        assert np.allclose(result.set_index('group').loc[['low','middle'],'MAPE_percent'],[3.710448,3.457800],atol=1e-5)
    result.to_csv(folder/'summary.csv',index=False,encoding='utf-8-sig')
    print(name+'\n'+result.to_string(index=False))
