from pathlib import Path
from collections import Counter
import argparse
import csv
import json
from route_algorithms import (load_graph, analyze_graph, get_layout,
    algo_basic_find_path, algo_momentum_find_path, algo_lookahead_find_path,
    turning_angle_deg)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=Path, default=here/'graph'/'pattern8_single_5x3_0.json')
    parser.add_argument('--output', type=Path, default=here/'output')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    nodes, edges = load_graph(args.graph)
    status, odd, degree, adj = analyze_graph(nodes, edges)
    if status not in ('eulerian_circuit', 'eulerian_trail'):
        raise ValueError(f'Input graph is not traversable: {status}')
    pos = get_layout(nodes)
    if set(degree)-set(pos):
        raise ValueError('Missing coordinates for active nodes')
    start = odd[0] if odd else next(iter(degree))
    paths = {
        'Hierholzer': algo_basic_find_path(edges, adj, start),
        'Momentum': algo_momentum_find_path(edges, adj, start, pos),
        'Look-ahead': algo_lookahead_find_path(edges, adj, start, pos, weight=0.5),
    }
    expected = Counter(tuple(sorted((u,v))) for _,u,v,_ in edges)
    summary = []
    turns = []
    for method, path in paths.items():
        actual = Counter(tuple(sorted((u,v))) for u,v in zip(path,path[1:]))
        if actual != expected:
            raise AssertionError(f'{method}: edge coverage mismatch')
        angles = []
        for k in range(1,len(path)-1):
            angle = turning_angle_deg(pos[path[k-1]],pos[path[k]],pos[path[k+1]])
            if angle is not None:
                angles.append(angle)
                turns.append([method,k,path[k],angle])
        summary.append([method,len(edges),len(odd),path[0]==path[-1],len(angles),
                        sum(angles),sum(a>150 for a in angles),
                        sum(a>150 for a in angles)/len(angles) if angles else 0])
    with (args.output/'route_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f)
        w.writerow(['method','edge_count','odd_degree_nodes','closed','turn_count',
                    'total_turn_deg','turns_above_150_deg','fraction_above_150_deg'])
        w.writerows(summary)
    with (args.output/'turning_angles.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['method','path_index','node','turn_deg']); w.writerows(turns)
    (args.output/'paths.json').write_text(json.dumps(paths,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
