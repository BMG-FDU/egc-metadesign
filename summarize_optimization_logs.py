import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    rows = []
    for run in manifest['runs']:
        folder = root / run['path']
        for item in run['files']:
            path = folder / item['name']
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'File hash mismatch: {path}')
        best, first, count = -math.inf, None, 0
        with (folder / 'evaluations.csv').open(encoding='utf-8-sig', newline='') as f:
            for count, record in enumerate(csv.DictReader(f), 1):
                if int(record['call_index']) != count or int(record['seed']) != run['seed']:
                    raise ValueError(f'Call index or seed mismatch: {folder}, row {count}')
                sea = float(record['sea_raw'])
                if not math.isfinite(sea):
                    raise ValueError(f'Non-finite SEA: {folder}, row {count}')
                best = max(best, sea)
                if not math.isclose(best, float(record['best_so_far']), abs_tol=1e-5):
                    raise ValueError(f'Historical-best mismatch: {folder}, row {count}')
                if first is None and sea > 750:
                    first = count
        if count != manifest['budget']:
            raise ValueError(f'Incomplete evaluation budget: {folder}: {count}')
        tail = []
        with (folder / 'episode_ends.csv').open(encoding='utf-8-sig', newline='') as f:
            for record in csv.DictReader(f):
                end = int(record.get('end_call_index', record.get('gnn_calls')))
                value = float(record.get('episode_best_sea_raw', record.get('performance')))
                if 0.9 * manifest['budget'] < end <= manifest['budget']:
                    tail.append(value)
        if not tail:
            raise ValueError(f'No final-window episode records: {folder}')
        rows.append(dict(method=run['method'], seed=run['seed'], gnn_calls=count,
                         first_call_sea_gt_750=first, final_best_sea=best,
                         tail_episode_best_mean=statistics.mean(tail),
                         tail_episode_count=len(tail), model_sha256=run['model_sha256']))
    output = root / 'summary'
    output.mkdir(exist_ok=True)
    write_csv(output / 'per_run_summary.csv', rows)
    aggregate = []
    for method in dict.fromkeys(row['method'] for row in rows):
        group = [row for row in rows if row['method'] == method]
        result = dict(method=method, seed_count=len(group),
                      threshold_reached_count=sum(row['first_call_sea_gt_750'] is not None for row in group))
        for metric in ('first_call_sea_gt_750', 'final_best_sea', 'tail_episode_best_mean'):
            values = [row[metric] for row in group if row[metric] is not None]
            result[metric + '_mean'] = statistics.mean(values) if values else ''
            result[metric + '_sd'] = statistics.stdev(values) if len(values) > 1 else ''
        aggregate.append(result)
    write_csv(output / 'aggregate_summary.csv', aggregate)
    print(f'Verified {len(rows)} runs; {sum(row["gnn_calls"] for row in rows):,} evaluations.')
    print(f'Summaries: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'results' / 'optimization_logs')
    summarize(parser.parse_args().log_dir)
