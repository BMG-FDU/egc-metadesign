import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader

from GNN import (GINCurveRegressorWithDenom, curve_value_at_strain,
                 parse_networkx_json, read_csv_new)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(reference, prediction):
    reference = np.asarray(reference, dtype=float).ravel()
    prediction = np.asarray(prediction, dtype=float).ravel()
    error = prediction - reference
    return {'MAE_J_kg': float(np.mean(np.abs(error))),
            'RMSE_J_kg': float(np.sqrt(np.mean(error ** 2))),
            'R2': float(r2_score(reference, prediction))}


def curve_checks(values, tolerance):
    differences = np.diff(values, axis=1)
    return {
        'zero_strain_abs_mean_J_kg': float(np.mean(np.abs(values[:, 0]))),
        'zero_strain_abs_max_J_kg': float(np.max(np.abs(values[:, 0]))),
        'curves_with_nonzero_start': int(np.sum(np.abs(values[:, 0]) > tolerance)),
        'curves_with_negative_values': int(np.sum(np.any(values < -tolerance, axis=1))),
        'negative_points': int(np.sum(values < -tolerance)),
        'minimum_SEA_J_kg': float(values.min()),
        'curves_with_local_decreases': int(np.sum(np.any(differences < -tolerance, axis=1))),
        'decreasing_intervals': int(np.sum(differences < -tolerance)),
        'largest_local_drop_J_kg': float(max(0.0, -differences.min()))}


def main():
    parser = argparse.ArgumentParser(description='Evaluate a saved GNN without training or resplitting.')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--checkpoint', type=Path, default=Path('results/best_model.pth'))
    parser.add_argument('--manifest', type=Path, default=Path('splits/data_split_manifest.csv'))
    parser.add_argument('--outdir', type=Path, default=Path('outputs/original_full_evaluation'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Absolute tolerance in J/kg for curve checks.')
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda path: path.resolve() if path.is_absolute() else (root / path).resolve()
    checkpoint, manifest, outdir = map(resolve, (args.checkpoint, args.manifest, args.outdir))
    if args.batch_size < 1 or not np.isfinite(args.tolerance) or args.tolerance < 0:
        raise ValueError('Batch size must be positive and tolerance finite and nonnegative')
    split = pd.read_csv(manifest)
    required = {'split', 'graph_id', 'graph_path', 'label_path'}
    if not required.issubset(split.columns):
        raise ValueError('Manifest is missing required columns: ' + str(sorted(required - set(split.columns))))
    if split['graph_id'].duplicated().any():
        raise ValueError('Manifest contains duplicate graph IDs')
    test = split.loc[split['split'].eq('test')].copy().reset_index(drop=True)
    if len(test) < 2:
        raise ValueError('Fixed manifest must contain at least two test samples')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    decoder_in = int(state['head_curve.project.weight'].shape[1])
    if decoder_in not in (512, 516):
        raise ValueError('Expected a CNN checkpoint with decoder input width 512 or 516')
    use_global_feat = decoder_in == 516
    model = GINCurveRegressorWithDenom(
        in_channels=4, out_dim=100, denom_channels=int(state['head_denom.4.bias'].shape[0]),
        hidden=256, num_layers=8, dropout=0.2, num_global_features=4,
        use_global_feat=use_global_feat, pooling_type='dual', use_denom_head=True,
        use_edge_feat=True, decoder_type='cnn', seq_len=100, out_channels_curve=1)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    grid = np.linspace(0.0, 0.5, 100)
    graphs, references = [], []
    for row in test.itertuples(index=False):
        graph_path, label_path = resolve(Path(row.graph_path)), resolve(Path(row.label_path))
        for path, column in ((graph_path, 'graph_sha256'), (label_path, 'label_sha256')):
            expected = getattr(row, column, None)
            if isinstance(expected, str) and expected and sha256(path).lower() != expected.lower():
                raise ValueError('Manifest hash mismatch: ' + str(path))
        graph = parse_networkx_json(str(graph_path))
        if graph.graph_id != row.graph_id:
            raise ValueError('Graph ID mismatch: ' + str(graph_path))
        strain, sea, *_ = read_csv_new(str(label_path))
        if sea is None:
            raise ValueError('Missing SEA column: ' + str(label_path))
        strain, sea = np.asarray(strain, dtype=float), np.asarray(sea, dtype=float)
        if len(strain) < 2 or not np.isfinite(strain).all() or not np.isfinite(sea).all():
            raise ValueError('Invalid reference curve: ' + str(label_path))
        order = np.argsort(strain, kind='stable')
        strain, sea = strain[order], sea[order]
        strain, first = np.unique(strain, return_index=True)
        sea = sea[first]
        if strain[0] > 1e-8 or strain[-1] < 0.5 - 1e-8:
            raise ValueError('Reference curve does not cover strain 0 to 0.5: ' + str(label_path))
        graphs.append(graph)
        references.append(np.interp(grid, strain, sea))
    predictions = []
    with torch.inference_mode():
        for batch in DataLoader(graphs, batch_size=args.batch_size, shuffle=False):
            batch = batch.to(device)
            normalized, log_denom = model(batch)
            real = normalized.reshape(batch.num_graphs, 1, 100) * torch.exp(log_denom[:, :1]).unsqueeze(-1)
            predictions.append(real[:, 0, :].cpu().numpy())
    reference, prediction = np.stack(references), np.concatenate(predictions, axis=0)
    if not np.isfinite(prediction).all():
        raise ValueError('Model produced nonfinite predictions')
    true_end = curve_value_at_strain(reference, grid)
    pred_end = curve_value_at_strain(prediction, grid)
    endpoint_metrics = metrics(true_end, pred_end)
    nonzero = np.abs(true_end) > 1e-12
    endpoint_metrics['MARE_percent'] = float(np.mean(np.abs((pred_end[nonzero] - true_end[nonzero]) / true_end[nonzero])) * 100) if nonzero.any() else None
    endpoint_metrics['MARE_sample_count'] = int(nonzero.sum())
    summary = {
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
        'manifest': str(manifest), 'manifest_sha256': sha256(manifest),
        'split_counts': {str(key): int(value) for key, value in split['split'].value_counts().items()},
        'test_samples': len(test), 'model': 'Supplied GNN predictor', 'use_global_feat': use_global_feat,
        'strain_interval': [0.0, 0.5], 'grid_points': 100, 'device': str(device),
        'curve_metric_definition': 'All test samples and 100 strain points pooled, in J/kg',
        'full_curve': metrics(reference, prediction), 'endpoint_strain_0_5': endpoint_metrics,
        'check_tolerance_J_kg': args.tolerance,
        'prediction_checks': curve_checks(prediction, args.tolerance),
        'reference_checks': curve_checks(reference, args.tolerance)}
    rows = pd.DataFrame({
        'graph_id': test['graph_id'], 'FEA_SEA_at_0_5_J_kg': true_end,
        'GNN_SEA_at_0_5_J_kg': pred_end, 'endpoint_absolute_error_J_kg': np.abs(pred_end - true_end),
        'curve_MAE_J_kg': np.mean(np.abs(prediction - reference), axis=1),
        'curve_RMSE_J_kg': np.sqrt(np.mean((prediction - reference) ** 2, axis=1)),
        'predicted_SEA_at_zero_J_kg': prediction[:, 0],
        'negative_points': np.sum(prediction < -args.tolerance, axis=1),
        'decreasing_intervals': np.sum(np.diff(prediction, axis=1) < -args.tolerance, axis=1)})
    outdir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(outdir / 'per_sample_metrics.csv', index=False)
    pd.DataFrame({
        'graph_id': np.repeat(test['graph_id'].to_numpy(), len(grid)),
        'strain': np.tile(grid, len(test)), 'FEA_SEA_J_kg': reference.ravel(),
        'GNN_SEA_J_kg': prediction.ravel()}).to_csv(outdir / 'test_curves.csv', index=False)
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=False))
    print('Saved evaluation outputs to', outdir)


if __name__ == '__main__':
    main()
