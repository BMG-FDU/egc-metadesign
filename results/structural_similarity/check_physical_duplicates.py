import argparse, collections, hashlib, json
from pathlib import Path
import pandas as pd

def canonical_signature(path, digits=6):
    with open(path, "r", encoding="utf-8") as f: obj = json.load(f)
    nodes = obj.get("nodes", []); links = obj.get("links", []) or obj.get("edges", [])
    coords=[]; id_to_i={}
    for i,n in enumerate(nodes):
        pos=n.get("pos")
        if not pos or len(pos)<2:
            pos=[float(x) for x in str(n.get("id", "0,0")).split(",")[:2]]
        coords.append((float(pos[0]),float(pos[1]),str(n.get("cell_type", ""))))
        id_to_i[str(n.get("id",i))]=i
    cx=sum(x for x,y,t in coords)/len(coords); cy=sum(y for x,y,t in coords)/len(coords)
    keys=[(round(x-cx,digits),round(y-cy,digits),t) for x,y,t in coords]
    edges=[]
    for e in links:
        u=id_to_i.get(str(e.get("source"))); v=id_to_i.get(str(e.get("target")))
        if u is not None and v is not None: edges.append(tuple(sorted((keys[u],keys[v]))))
    payload=(tuple(sorted(keys)),tuple(sorted(edges)))
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = pd.read_csv(root/'splits/data_split_manifest.csv')
    report = {'split_counts': manifest.split.value_counts().to_dict()}
    exact = {}
    for column, path_column in [('graph_sha256','graph_path'),('label_sha256','label_path')]:
        buckets = collections.defaultdict(list)
        for row in manifest.to_dict('records'):
            digest = hashlib.sha256((root/row[path_column]).read_bytes()).hexdigest()
            assert digest == row[column], row['graph_id']
            buckets[digest].append({'split':row['split'],'graph_id':row['graph_id']})
        exact[column] = [v for v in buckets.values() if len({x['split'] for x in v}) > 1]
    report['exact_hash_cross_split_groups'] = exact
    canonical = {}
    for digits in (6,5):
        buckets = collections.defaultdict(list)
        for row in manifest.to_dict('records'):
            signature = canonical_signature(root/row['graph_path'], digits)
            buckets[signature].append({'split':row['split'],'graph_id':row['graph_id']})
        canonical[str(digits)] = [v for v in buckets.values() if len({x['split'] for x in v}) > 1]
    report['translation_centered_geometry_connectivity_signature_cross_split_groups'] = canonical
    report['canonical_scope'] = 'Centered coordinates, cell_type and undirected connectivity; node IDs, node order, edge order and translation ignored; rounding at 1e-6 and 1e-5.'
    out = args.out or Path(__file__).with_name('duplicate_check_result.json')
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
