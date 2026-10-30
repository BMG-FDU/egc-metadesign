import matplotlib
matplotlib.use('Agg')

import os
import argparse
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
import hashlib
import sys
import subprocess
import networkx as nx

import train_ppo

from lattice_env import LatticeDesignEnv, EnvConfig
from lattice_state import LatticeState
from hexagonal_generator import node_link_data

from tqdm.auto import tqdm

NUM_POINTS_PER_EDGE = EnvConfig.NUM_POINTS_PER_EDGE
LOOP_ACTIONS_TOTAL = EnvConfig.STAGE1_ACTIONS
LINK_ACTIONS_TOTAL = EnvConfig.LINK_ACTIONS_TOTAL
NODE_ACTIONS_TOTAL = EnvConfig.NODE_ACTIONS_TOTAL
PPO_TOTAL_ACTIONS = LatticeDesignEnv.TOTAL_ACTIONS
LINK_EDGE_PAIRS = LatticeDesignEnv.LINK_EDGE_PAIRS

assert NUM_POINTS_PER_EDGE == 10
assert LOOP_ACTIONS_TOTAL == 2
assert LINK_ACTIONS_TOTAL == 30
assert NODE_ACTIONS_TOTAL == 120
assert PPO_TOTAL_ACTIONS == 152
assert len(LINK_EDGE_PAIRS) == 15

COLORS = {
    'primary': "#7C95E1",
    'secondary': "#B0C7FB",
    'tertiary': "#D0D1F9",
    'light': "#F2E6F1",
    'accent': "#EED6EC",
    'text': "#333333",
    'gray': "#999999",
    'highlight': "#D04040"
}

METHOD_COLORS = {
    'PPO_NoCurriculum': COLORS['gray'],
    'PPO_LoopLinkNode': COLORS['primary'],
    'PPO_LoopNodeLink': COLORS['tertiary'],
    'Greedy': COLORS['secondary'],
    'Hill Climbing': "#9BB0E0",
    'Random': COLORS['accent']
}


def mean_last_fraction(values, frac: float = 0.1) -> float:
    if values is None or len(values) == 0:
        return 0.0
    frac = float(frac)
    frac = max(0.0, min(1.0, frac))
    k = int(np.ceil(len(values) * frac))
    k = max(1, k)
    return float(np.mean(values[-k:]))


def _get_gnn_evaluation_history(result):

    events = result.get('gnn_evaluation_history', []) or []
    if not events:
        return None
    frame = pd.DataFrame(events)
    required = {'gnn_call', 'best_so_far_real'}
    if not required.issubset(frame.columns):
        raise ValueError(
            'gnn_evaluation_history must contain gnn_call and best_so_far_real.'
        )
    frame['gnn_call'] = pd.to_numeric(frame['gnn_call'], errors='coerce')
    frame['best_so_far_real'] = pd.to_numeric(
        frame['best_so_far_real'], errors='coerce'
    )
    return frame.dropna(subset=['gnn_call', 'best_so_far_real']).sort_values('gnn_call')


def _budget_window_stats(result, common_budget=None, frac: float = 0.1):


    budget = int(common_budget if common_budget is not None else
                 result.get('gnn_budget', result.get('gnn_calls', 0)))
    lower = int(np.floor((1.0 - float(frac)) * budget))
    perf = np.asarray(result.get('episode_perfs', []), dtype=float)
    calls = np.asarray(result.get('gnn_calls_history', []), dtype=float)
    if len(perf) != len(calls):
        raise ValueError('Episode-best values and actual terminal call counts must align')
    mask = (calls > lower) & (calls <= budget) & np.isfinite(perf)
    vals = perf[mask]
    if not len(vals):
        return float('nan'), float('nan'), 0, 'episode_best_at_episode_end'
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else float('nan')), int(len(vals)), 'episode_best_at_episode_end'


def _first_threshold_call(result, threshold, common_budget):

    events = _get_gnn_evaluation_history(result)
    if events is None:
        return None, 'unavailable_episode_end_only'
    events = events[(events.gnn_call >= 1) & (events.gnn_call <= common_budget)]
    calls = events.gnn_call.to_numpy(dtype=int)
    if not np.array_equal(calls, np.arange(1, common_budget + 1)):
        raise ValueError('Exact N750 requires a complete ordered trace of calls 1..B')
    hits = events.loc[events.best_so_far_real > threshold, 'gnn_call']
    return (int(hits.iloc[0]) if len(hits) else None), 'per_gnn_evaluation'


def std_last_fraction(values, frac: float = 0.1) -> float:
    if values is None or len(values) == 0:
        return 0.0
    frac = float(frac)
    frac = max(0.0, min(1.0, frac))
    k = int(np.ceil(len(values) * frac))
    k = max(1, k)
    return float(np.std(values[-k:]))


def extend_curve_to_max(x_values, y_values, max_x):
    if len(x_values) == 0:
        return np.array([0, max_x]), np.array([0, 0])
    x_arr = np.array(x_values)
    y_arr = np.array(y_values)
    if x_arr[-1] < max_x:
        x_arr = np.append(x_arr, max_x)
        y_arr = np.append(y_arr, y_arr[-1])
    return x_arr, y_arr


def _trapezoid(y_values, x_values) -> float:
    if hasattr(np, 'trapezoid'):
        return float(
            np.trapezoid(y_values, x_values)
        )
    return float(np.trapz(y_values, x_values))


def _prepare_best_so_far_curve(
    result,
    common_budget: int,
):
    events = _get_gnn_evaluation_history(result)
    if events is not None:
        curve = events[
            (events['gnn_call'] >= 0)
            & (events['gnn_call'] <= common_budget)
        ].copy()
        curve['global_best'] = curve['best_so_far_real'].cummax()
        x = curve['gnn_call'].to_numpy(dtype=float)
        y = curve['global_best'].to_numpy(dtype=float)
        if len(x) and x[0] > 0:
            x = np.insert(x, 0, 0.0)
            y = np.insert(y, 0, 0.0)
        if len(x) and x[-1] < common_budget:
            x = np.append(x, float(common_budget))
            y = np.append(y, y[-1])
        if len(x):
            return x, y

    history = result.get(
        'global_best_history',
        [],
    )
    gnn_history = result.get(
        'gnn_calls_history',
        [],
    )

    if history is None or len(history) == 0:
        return (
            np.array([0.0, float(common_budget)]),
            np.array([0.0, 0.0]),
        )

    if gnn_history is None or len(gnn_history) != len(history):
        raise ValueError(
            f"{result.get('method', 'method')}: GNN-call history is missing or "
            "not aligned with best-so-far observations; refusing to synthesize "
            "a call axis."
        )

    curve = pd.DataFrame({
        'gnn_calls': pd.to_numeric(
            pd.Series(gnn_history),
            errors='coerce',
        ),
        'global_best': pd.to_numeric(
            pd.Series(history),
            errors='coerce',
        ),
    }).dropna()

    curve = curve[
        (curve['gnn_calls'] >= 0)
        & (curve['gnn_calls'] <= common_budget)
    ].sort_values('gnn_calls')

    if curve.empty:
        return (
            np.array([0.0, float(common_budget)]),
            np.array([0.0, 0.0]),
        )

    curve = curve.groupby(
        'gnn_calls',
        as_index=False,
    )['global_best'].max()
    curve['global_best'] = (
        curve['global_best'].cummax()
    )

    x = curve[
        'gnn_calls'
    ].to_numpy(dtype=float)
    y = curve[
        'global_best'
    ].to_numpy(dtype=float)

    if x[0] > 0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 0.0)

    if x[-1] < common_budget:
        x = np.append(
            x,
            float(common_budget),
        )
        y = np.append(
            y,
            y[-1],
        )

    prepared = pd.DataFrame({
        'x': x,
        'y': y,
    })
    prepared = (
        prepared.groupby(
            'x',
            as_index=False,
        )['y']
        .max()
        .sort_values('x')
    )
    prepared['y'] = prepared['y'].cummax()

    return (
        prepared['x'].to_numpy(dtype=float),
        prepared['y'].to_numpy(dtype=float),
    )


def calculate_spi_metrics(results):
    if not results:
        return {}, 0.0, 0, 0.0


    common_budget = min(int(r.get('gnn_calls', 0)) for r in results)
    if common_budget < 2:
        raise ValueError('At least two actual GNN evaluations per method are required')
    reference_best = max(float(_prepare_best_so_far_curve(r, common_budget)[1][-1])
                         for r in results)
    reference_best = max(reference_best, np.finfo(float).eps)
    high_performance_threshold = 750.0

    metrics_map = {}

    for result in results:
        method = result['method']
        perfs = (
            result.get('episode_perfs', [])
            or []
        )

        avg_last10, std_last10, n_budget_observations, metric_resolution = (
            _budget_window_stats(result, common_budget=common_budget, frac=0.1)
        )

        x, y = _prepare_best_so_far_curve(
            result,
            common_budget=common_budget,
        )

        auc = _trapezoid(y, x)
        quality_score = float(
            auc
            / (
                float(common_budget)
                * float(reference_best)
            )
        )
        quality_score = max(
            0.0,
            quality_score,
        )

        calls_to_750, threshold_resolution = _first_threshold_call(
            result, high_performance_threshold, common_budget)
        speed_score = (float(np.log(common_budget / max(1, calls_to_750)) /
                             np.log(common_budget)) if calls_to_750 is not None else 0.0)
        speed_score = float(np.clip(speed_score, 0.0, 1.0))

        if avg_last10 > 0:
            coefficient_of_variation = (
                std_last10 / avg_last10
            )
            consistency_score = float(
                1.0
                / (
                    1.0
                    + coefficient_of_variation
                )
            )
        else:
            consistency_score = 0.0

        spi = float(
            (
                quality_score
                * speed_score
                * consistency_score
            ) ** (1.0 / 3.0)
        )

        metrics_map[method] = {
            'best_performance': float(
                result.get(
                    'best_performance',
                    0.0,
                )
            ),
            'avg_performance_last10': float(
                avg_last10
            ),
            'n_final_budget_observations': int(n_budget_observations),
            'final_budget_metric_resolution': metric_resolution,
            'threshold_resolution': threshold_resolution,
            'threshold_reached': calls_to_750 is not None,
            'common_budget_calls': int(common_budget),
            'high_performance_threshold': float(high_performance_threshold),
            'std_performance': float(
                std_last10
            ),
            'efficiency_nAUC': float(
                quality_score
            ),
            'calls_to_750': (
                calls_to_750
            ),
            'entry_speed_score': float(
                speed_score
            ),
            'consistency_score': float(
                consistency_score
            ),
            'Search Performance Index': float(
                spi
            ),
        }

    return (
        metrics_map,
        float(reference_best),
        int(common_budget),
        float(high_performance_threshold),
    )


def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj


def save_best_structure_json(lattice_state, filepath: str, performance: float, method: str, gnn_calls: int):
    if lattice_state is None:
        return

    G = lattice_state.to_networkx_graph()
    data = node_link_data(G)

    for node in data['nodes']:
        x, y = node['id']
        node['pos'] = [float(x), float(y)]
        node['id'] = f"{x},{y}"

    for link in data['links']:
        sx, sy = link['source']
        tx, ty = link['target']
        link['source'] = f"{sx},{sy}"
        link['target'] = f"{tx},{ty}"

    loop1 = lattice_state.loop1 if hasattr(lattice_state, 'loop1') else lattice_state.center_loop1
    loop2 = lattice_state.loop2 if hasattr(lattice_state, 'loop2') else lattice_state.center_loop2
    loop1_converted = [(int(e), int(p)) for e, p in loop1] if loop1 else []
    loop2_converted = [(int(e), int(p)) for e, p in loop2] if loop2 else []

    data['metadata'] = {
        'performance': float(performance),
        'method': str(method),
        'gnn_calls': int(gnn_calls),
        'loop1': loop1_converted,
        'loop2': loop2_converted,
    }

    data = convert_to_json_serializable(data)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  💾 JSON saved: {filepath}")


def visualize_best_structure(lattice_state, filepath: str, performance: float, method: str, gnn_calls: int):
    if lattice_state is None:
        return

    G = lattice_state.to_networkx_graph()

    fig, ax = plt.subplots(figsize=(10, 10))
    pos = {n: n for n in G.nodes()}

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.8, width=2.0, edge_color=COLORS['primary'])
    if len(pos) > 0:
        xs, ys = zip(*pos.keys())
        ax.scatter(xs, ys, s=2, c=COLORS['accent'], linewidths=0.8, zorder=3)

    ax.set_aspect('equal')
    title = f"{method}\nPerformance: {performance:.6f} | GNN Calls: {gnn_calls:,}"
    ax.set_title(title, fontsize=14, color=COLORS['text'], fontfamily='serif')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"  🖼️  PNG saved: {filepath}")


def save_all_best_structures(results, save_dir):
    structures_dir = os.path.join(save_dir, 'best_structures')
    os.makedirs(structures_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("📐 Saving Best Structures for Each Method")
    print("=" * 60)

    for r in results:
        method = r['method']
        method_clean = method.replace(' ', '_').replace('/', '_')

        if r.get('best_lattice_state') is not None:
            print(f"\n🔹 {method}:")
            json_path = os.path.join(structures_dir, f'{method_clean}_best.json')
            save_best_structure_json(
                r['best_lattice_state'], json_path, r['best_performance'], method, r.get('gnn_calls', 0)
            )

            png_path = os.path.join(structures_dir, f'{method_clean}_best.png')
            visualize_best_structure(
                r['best_lattice_state'], png_path, r['best_performance'], method, r.get('gnn_calls', 0)
            )
        else:
            print(f"\n⚠️  {method}: No lattice state saved")

    plot_structures_comparison(results, structures_dir)
    print("\n" + "=" * 60)


def plot_structures_comparison(results, save_dir):
    valid_results = [r for r in results if r.get('best_lattice_state') is not None]
    if len(valid_results) == 0:
        return

    n_plots = len(valid_results)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, r in enumerate(valid_results):
        ax = axes[idx]
        method = r['method']
        lattice_state = r['best_lattice_state']

        G = lattice_state.to_networkx_graph()
        pos = {n: n for n in G.nodes()}

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.8, width=1.5,
                               edge_color=METHOD_COLORS.get(method, COLORS['gray']))
        if len(pos) > 0:
            xs, ys = zip(*pos.keys())
            ax.scatter(xs, ys, s=1, c=COLORS['accent'], zorder=3)

        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        title = f"{method}\n{r['best_performance']:.6f}"
        title_color = METHOD_COLORS.get(method, COLORS['text'])
        ax.set_title(title, fontsize=11, fontweight='bold', color=title_color)

    for idx in range(len(valid_results), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    comparison_path = os.path.join(save_dir, 'all_methods_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n📊 Structure comparison saved: {comparison_path}")


def plot_comparison(results, save_dir, compare_mode='all'):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.color'] = COLORS['text']
    plt.rcParams['axes.labelcolor'] = COLORS['text']
    plt.rcParams['xtick.color'] = COLORS['text']
    plt.rcParams['ytick.color'] = COLORS['text']

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    max_gnn_calls = max(r.get('gnn_calls', 0) for r in results)

    ax1 = axes[0, 0]
    for r in results:
        method = r['method']
        perfs = r['episode_perfs']
        gnn_history = r.get('gnn_calls_history', [])
        if len(gnn_history) != len(perfs):
            raise ValueError(f"{method}: missing GNN-call history for episode observations.")

        c = METHOD_COLORS.get(method, COLORS['gray'])
        if len(perfs) > 10:
            ax1.plot(gnn_history, perfs, '.', alpha=0.2, markersize=2, color=c)

        if len(perfs) >= 20:
            window = min(50, len(perfs) // 5)
            smoothed = np.convolve(perfs, np.ones(window) / window, mode='valid')
            x_smooth = gnn_history[window - 1:]
            x_extended, y_extended = extend_curve_to_max(x_smooth, smoothed, max_gnn_calls)
            ax1.plot(x_extended, y_extended, '-', color=c, linewidth=2, label=method)
        elif len(perfs) >= 3:
            x_extended, y_extended = extend_curve_to_max(gnn_history, perfs, max_gnn_calls)
            ax1.plot(x_extended, y_extended, '-', color=c, linewidth=2, label=method)
            ax1.plot(gnn_history, perfs, 'o', color=c, markersize=6, alpha=0.8)
        else:
            ax1.plot(gnn_history, perfs, 'o-', color=c, linewidth=2, markersize=8, label=method)

    ax1.set_xlabel('Cumulative GNN Calls', fontsize=11)
    ax1.set_ylabel('Performance', fontsize=11)
    ax1.set_title('Episode Performance', fontsize=12, color=COLORS['text'])
    ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax1.set_xlim(0, max_gnn_calls * 1.02)
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)

    ax2 = axes[0, 1]
    for r in results:
        method = r['method']
        history = r['global_best_history']
        gnn_history = r.get('gnn_calls_history', [])
        if len(gnn_history) != len(history):
            raise ValueError(f"{method}: missing GNN-call history for best-so-far observations.")

        c = METHOD_COLORS.get(method, COLORS['gray'])
        x_extended, y_extended = extend_curve_to_max(gnn_history, history, max_gnn_calls)
        ax2.plot(x_extended, y_extended, '-', color=c, linewidth=2.5, label=method)

    ax2.set_xlabel('Cumulative GNN Calls', fontsize=11)
    ax2.set_ylabel('Global Best Performance', fontsize=11)
    ax2.set_title('Global Best Evolution', fontsize=12, color=COLORS['text'])
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax2.set_xlim(0, max_gnn_calls * 1.02)
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)

    ax3 = axes[1, 0]
    methods = [r['method'] for r in results]
    best_perfs = [r['best_performance'] for r in results]
    bar_colors = [METHOD_COLORS.get(m, COLORS['gray']) for m in methods]
    bars = ax3.bar(range(len(methods)), best_perfs, color=bar_colors, alpha=0.9)
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax3.set_ylabel('Best Performance', fontsize=11)
    ax3.set_title('Final Best Performance', fontsize=12, color=COLORS['text'])
    ax3.spines['top'].set_visible(True)
    ax3.spines['right'].set_visible(True)
    for bar, perf in zip(bars, best_perfs):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{perf:.5f}', ha='center', va='bottom', fontsize=9, color=COLORS['text'])

    ax4 = axes[1, 1]
    spi_map, _, _, _ = calculate_spi_metrics(
        results
    )
    spi_values = [
        spi_map.get(
            method,
            {},
        ).get(
            'Search Performance Index',
            0.0,
        )
        for method in methods
    ]
    bars = ax4.bar(
        range(len(methods)),
        spi_values,
        color=bar_colors,
        alpha=0.9,
    )
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(
        [
            method.replace('_', '\n')
            for method in methods
        ],
        fontsize=9,
    )
    ax4.set_ylabel(
        'Search Performance Index (SPI)',
        fontsize=11,
    )
    ax4.set_title(
        'Composite Search Performance',
        fontsize=12,
        color=COLORS['text'],
    )
    ax4.set_ylim(
        0,
        max(
            1.0,
            max(
                spi_values,
                default=0.0,
            ) * 1.15,
        ),
    )
    ax4.spines['top'].set_visible(True)
    ax4.spines['right'].set_visible(True)
    for bar, spi_value in zip(
        bars,
        spi_values,
    ):
        ax4.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f'{spi_value:.4f}',
            ha='center',
            va='bottom',
            fontsize=9,
            color=COLORS['text'],
        )

    plt.tight_layout()
    filename = 'comparison.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n📊 Comparison plot saved: {os.path.join(save_dir, filename)}")


def apply_decoded_action(
    state: LatticeState,
    action_type: str,
    loop_id: int,
    params,
) -> bool:

    if action_type == 'loop':
        return bool(state.apply_loop_action(loop_id))

    if action_type == 'link':
        edge_a, edge_b = params
        return bool(
            state.apply_link_action(
                loop_id,
                edge_a,
                edge_b,
            )
        )

    if action_type == 'node':
        side_id, point_idx = params
        return bool(
            state.apply_node_action(
                loop_id,
                side_id,
                point_idx,
            )
        )

    return False


def _make_gnn_pbar(method: str, gnn_budget: int, leave: bool = True):
    return tqdm(
        total=int(gnn_budget),
        desc=f"{method} | GNN Calls",
        unit="calls",
        dynamic_ncols=True,
        leave=leave,
        smoothing=0.05
    )

def _update_gnn_pbar(env: LatticeDesignEnv, pbar, last_calls: int) -> int:
    calls = int(env.get_gnn_forward_calls()) if hasattr(env, 'get_gnn_forward_calls') else 0
    calls = min(calls, int(env.gnn_budget) if env.gnn_budget is not None else calls)
    delta = calls - int(last_calls)
    if delta > 0:
        pbar.update(delta)
        last_calls = calls
    return int(last_calls)

def _close_gnn_pbar(env: LatticeDesignEnv, pbar, last_calls: int):
    if pbar is None:
        return
    calls = int(env.get_gnn_forward_calls()) if hasattr(env, 'get_gnn_forward_calls') else int(last_calls)
    if env.gnn_budget is not None:
        calls = min(calls, int(env.gnn_budget))
    if calls > int(last_calls):
        pbar.update(calls - int(last_calls))
    pbar.close()


def _set_method_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def run_greedy(args, env_kwargs, gnn_budget: int):
    _set_method_seed(args.seed)
    print(
        "\nRunning: Greedy Search "
        "(Exhaustive 152 Actions per Step + Strict Forward Budget)"
    )

    env = LatticeDesignEnv(
        gnn_model_path=env_kwargs['gnn_model_path'],
        edge_length=env_kwargs['edge_length'],
        num_points_per_edge=env_kwargs['num_points_per_edge'],
        max_steps=env_kwargs['max_steps'],
        curriculum_enabled=False,
        curriculum_stage=3,
        use_potential_shaping=True,
        action_order='loop_link_node',
        gnn_budget=gnn_budget,
        verbose=False
    )
    env.reset_gnn_counter(budget=gnn_budget, clear_cache=False)
    env.run_seed = int(args.seed)

    max_steps = env.effective_max_steps
    num_actions = PPO_TOTAL_ACTIONS

    episode_perfs = []
    global_best_history = []
    gnn_calls_history = []
    global_best = 0.0
    best_lattice_state = None
    best_gnn_calls = 0
    start_time = time.time()

    pbar = _make_gnn_pbar("Greedy", gnn_budget, leave=True)
    last_calls = 0

    try:
        while not env.gnn_budget_exhausted():
            env.reset()
            state = env.lattice_state.clone()

            cur_perf = env.evaluate_lattice_state(state)[2]
            last_calls = _update_gnn_pbar(env, pbar, last_calls)

            best_perf_ep = cur_perf
            best_state_ep = state.clone()
            best_gnn_calls_ep = env.get_gnn_forward_calls()

            for step_index in range(max_steps):
                env.current_step = step_index + 1
                if env.gnn_budget_exhausted():
                    break

                best_action_perf = -float('inf')
                best_next_state = None


                for action in range(num_actions):
                    if env.gnn_budget_exhausted():
                        break

                    a_type, loop_id, params = env._decode_action(int(action))
                    if a_type == 'invalid':
                        continue

                    cand = state.clone()
                    ok = apply_decoded_action(cand, a_type, loop_id, params)
                    if not ok:
                        continue

                    env.last_action = int(action)
                    perf_real = env.evaluate_lattice_state(cand)[2]
                    last_calls = _update_gnn_pbar(env, pbar, last_calls)

                    if perf_real > best_action_perf:
                        best_action_perf = perf_real
                        best_next_state = cand

                if best_next_state is None:
                    break

                state = best_next_state

                if best_action_perf > best_perf_ep:
                    best_perf_ep = best_action_perf
                    best_state_ep = state.clone()
                    best_gnn_calls_ep = env.get_gnn_forward_calls()

            episode_perfs.append(best_perf_ep)

            if best_perf_ep > global_best:
                global_best = best_perf_ep
                best_lattice_state = best_state_ep.clone()
                best_gnn_calls = best_gnn_calls_ep

            global_best_history.append(global_best)
            gnn_calls_history.append(env.get_gnn_forward_calls())
            last_calls = _update_gnn_pbar(env, pbar, last_calls)

    finally:
        _close_gnn_pbar(env, pbar, last_calls)

    return {
        'method': 'Greedy',
        'episode_perfs': episode_perfs,
        'global_best_history': global_best_history,
        'best_performance': float(env.gnn_best_so_far_real),
        'num_episodes': len(episode_perfs),
        'elapsed_time': time.time() - start_time,
        'gnn_calls': env.get_gnn_forward_calls(),
        'gnn_budget': gnn_budget,
        'seed': int(args.seed),
        'gnn_calls_history': gnn_calls_history,
        'gnn_evaluation_history': list(env.gnn_evaluation_history),
        'best_lattice_state': env.gnn_best_state.clone() if env.gnn_best_state is not None else None,
        'best_gnn_calls': int(env.gnn_best_call),
        'predictor_sha256': env.predictor_sha256,
        'predictor_use_global_feat': env.predictor_use_global_feat
    }


def run_hill_climbing(args, env_kwargs, gnn_budget: int):
    _set_method_seed(args.seed)
    print("\nRunning: Hill Climbing (Fair + Strict Forward Budget, FULL PPO ACTION SPACE)")

    env = LatticeDesignEnv(
        gnn_model_path=env_kwargs['gnn_model_path'],
        edge_length=env_kwargs['edge_length'],
        num_points_per_edge=env_kwargs['num_points_per_edge'],
        max_steps=env_kwargs['max_steps'],
        curriculum_enabled=False,
        curriculum_stage=3,
        use_potential_shaping=True,
        action_order='loop_link_node',
        gnn_budget=gnn_budget,
        verbose=False
    )
    env.reset_gnn_counter(budget=gnn_budget, clear_cache=False)
    env.run_seed = int(args.seed)

    max_steps = env.effective_max_steps
    num_actions = PPO_TOTAL_ACTIONS

    episode_perfs = []
    global_best_history = []
    gnn_calls_history = []
    global_best = 0.0
    best_lattice_state = None
    best_gnn_calls = 0
    start_time = time.time()

    rng = np.random.default_rng(args.seed)

    pbar = _make_gnn_pbar("Hill Climbing", gnn_budget, leave=True)
    last_calls = 0

    try:
        while not env.gnn_budget_exhausted():
            env.reset()
            state = env.lattice_state.clone()
            cur_perf = env.evaluate_lattice_state(state)[2]
            last_calls = _update_gnn_pbar(env, pbar, last_calls)

            best_perf_ep = cur_perf
            best_state_ep = state.clone()
            best_gnn_calls_ep = env.get_gnn_forward_calls()

            for step_index in range(max_steps):
                env.current_step = step_index + 1
                if env.gnn_budget_exhausted():
                    break

                action = int(rng.integers(0, num_actions))
                a_type, loop_id, params = env._decode_action(int(action))
                if a_type == 'invalid':
                    continue

                cand = state.clone()
                ok = apply_decoded_action(cand, a_type, loop_id, params)
                if not ok:
                    continue

                env.last_action = int(action)
                perf_real = env.evaluate_lattice_state(cand)[2]
                last_calls = _update_gnn_pbar(env, pbar, last_calls)

                if perf_real >= cur_perf:
                    state = cand
                    cur_perf = perf_real

                if perf_real > best_perf_ep:
                    best_perf_ep = perf_real
                    best_state_ep = cand.clone()
                    best_gnn_calls_ep = env.get_gnn_forward_calls()

            episode_perfs.append(best_perf_ep)

            if best_perf_ep > global_best:
                global_best = best_perf_ep
                best_lattice_state = best_state_ep.clone()
                best_gnn_calls = best_gnn_calls_ep

            global_best_history.append(global_best)
            gnn_calls_history.append(env.get_gnn_forward_calls())
            last_calls = _update_gnn_pbar(env, pbar, last_calls)

    finally:
        _close_gnn_pbar(env, pbar, last_calls)

    return {
        'method': 'Hill Climbing',
        'episode_perfs': episode_perfs,
        'global_best_history': global_best_history,
        'best_performance': float(env.gnn_best_so_far_real),
        'num_episodes': len(episode_perfs),
        'elapsed_time': time.time() - start_time,
        'gnn_calls': env.get_gnn_forward_calls(),
        'gnn_budget': gnn_budget,
        'seed': int(args.seed),
        'gnn_calls_history': gnn_calls_history,
        'gnn_evaluation_history': list(env.gnn_evaluation_history),
        'best_lattice_state': env.gnn_best_state.clone() if env.gnn_best_state is not None else None,
        'best_gnn_calls': int(env.gnn_best_call),
        'predictor_sha256': env.predictor_sha256,
        'predictor_use_global_feat': env.predictor_use_global_feat
    }


def run_random(args, env_kwargs, gnn_budget: int):
    _set_method_seed(args.seed)
    print("\nRunning: Random Search (Strict Forward Budget, FULL PPO ACTION SPACE)")

    env = LatticeDesignEnv(
        gnn_model_path=env_kwargs['gnn_model_path'],
        edge_length=env_kwargs['edge_length'],
        num_points_per_edge=env_kwargs['num_points_per_edge'],
        max_steps=env_kwargs['max_steps'],
        curriculum_enabled=False,
        curriculum_stage=3,
        use_potential_shaping=True,
        action_order='loop_link_node',
        gnn_budget=gnn_budget,
        verbose=False
    )
    env.reset_gnn_counter(budget=gnn_budget, clear_cache=False)
    env.run_seed = int(args.seed)

    max_steps = env.effective_max_steps
    num_actions = PPO_TOTAL_ACTIONS

    episode_perfs = []
    global_best_history = []
    gnn_calls_history = []
    global_best = 0.0
    best_lattice_state = None
    best_gnn_calls = 0
    start_time = time.time()

    rng = np.random.default_rng(args.seed)

    pbar = _make_gnn_pbar("Random", gnn_budget, leave=True)
    last_calls = 0

    try:
        while not env.gnn_budget_exhausted():
            env.reset()
            state = env.lattice_state.clone()

            init_perf = env.evaluate_lattice_state(state)[2]
            last_calls = _update_gnn_pbar(env, pbar, last_calls)

            best_perf_ep = init_perf
            best_state_ep = state.clone()
            best_gnn_calls_ep = env.get_gnn_forward_calls()

            for step_index in range(max_steps):
                env.current_step = step_index + 1
                if env.gnn_budget_exhausted():
                    break

                action = int(rng.integers(0, num_actions))
                a_type, loop_id, params = env._decode_action(int(action))
                if a_type == 'invalid':
                    continue

                cand = state.clone()
                ok = apply_decoded_action(cand, a_type, loop_id, params)
                if not ok:
                    continue

                env.last_action = int(action)
                perf_real = env.evaluate_lattice_state(cand)[2]
                last_calls = _update_gnn_pbar(env, pbar, last_calls)

                state = cand

                if perf_real > best_perf_ep:
                    best_perf_ep = perf_real
                    best_state_ep = cand.clone()
                    best_gnn_calls_ep = env.get_gnn_forward_calls()

            episode_perfs.append(best_perf_ep)

            if best_perf_ep > global_best:
                global_best = best_perf_ep
                best_lattice_state = best_state_ep.clone()
                best_gnn_calls = best_gnn_calls_ep

            global_best_history.append(global_best)
            gnn_calls_history.append(env.get_gnn_forward_calls())
            last_calls = _update_gnn_pbar(env, pbar, last_calls)

    finally:
        _close_gnn_pbar(env, pbar, last_calls)

    return {
        'method': 'Random',
        'episode_perfs': episode_perfs,
        'global_best_history': global_best_history,
        'best_performance': float(env.gnn_best_so_far_real),
        'num_episodes': len(episode_perfs),
        'elapsed_time': time.time() - start_time,
        'gnn_calls': env.get_gnn_forward_calls(),
        'gnn_budget': gnn_budget,
        'seed': int(args.seed),
        'gnn_calls_history': gnn_calls_history,
        'gnn_evaluation_history': list(env.gnn_evaluation_history),
        'best_lattice_state': env.gnn_best_state.clone() if env.gnn_best_state is not None else None,
        'best_gnn_calls': int(env.gnn_best_call),
        'predictor_sha256': env.predictor_sha256,
        'predictor_use_global_feat': env.predictor_use_global_feat
    }


def save_comparison_csv(results, save_dir):
    csv_dir = os.path.join(
        save_dir,
        'csv_data',
    )
    os.makedirs(
        csv_dir,
        exist_ok=True,
    )

    for result in results:
        method = result[
            'method'
        ].replace(' ', '_')
        gnn_history = result.get('gnn_calls_history', [])
        if len(gnn_history) != len(result['episode_perfs']):
            raise ValueError('Cannot export synthetic or misaligned GNN-call coordinates')
        df = pd.DataFrame({
            'gnn_calls': gnn_history[
                :len(result['episode_perfs'])
            ],
            'performance': result[
                'episode_perfs'
            ],
            'global_best': result[
                'global_best_history'
            ],
        })
        df.to_csv(
            os.path.join(
                csv_dir,
                f'{method}_data.csv',
            ),
            index=False,
        )
        events = _get_gnn_evaluation_history(result)
        if events is not None:
            events = events.assign(method=result['method'], seed=result.get('seed'),
                                   predictor_sha256=result.get('predictor_sha256'))
            events.to_csv(os.path.join(csv_dir, f'{method}_per_gnn_call.csv'), index=False)

    (
        spi_map,
        reference_best,
        common_budget,
        threshold_750,
    ) = calculate_spi_metrics(results)

    summary_data = []
    for result in results:
        metrics = spi_map[result['method']]
        summary_data.append({
            'method': result['method'],
            'best_performance': metrics[
                'best_performance'
            ],
            'avg_performance_last10': metrics[
                'avg_performance_last10'
            ],
            'n_final_budget_observations': metrics[
                'n_final_budget_observations'
            ],
            'final_budget_metric_resolution': metrics[
                'final_budget_metric_resolution'
            ],
            'threshold_resolution': metrics['threshold_resolution'],
            'threshold_reached': metrics['threshold_reached'],
            'common_budget_calls': int(common_budget),
            'high_performance_threshold_J_kg': float(threshold_750),
            'predictor_sha256': result.get('predictor_sha256'),
            'std_performance': metrics[
                'std_performance'
            ],
            'efficiency_nAUC': metrics[
                'efficiency_nAUC'
            ],
            'calls_to_750': metrics[
                'calls_to_750'
            ],
            'Search Performance Index': metrics[
                'Search Performance Index'
            ],
        })

    summary_df = pd.DataFrame(
        summary_data
    )
    summary_df = summary_df.sort_values(
        'Search Performance Index',
        ascending=False,
    )
    summary_df.to_csv(
        os.path.join(
            csv_dir,
            'summary.csv',
        ),
        index=False,
    )

    metadata = {
        'common_gnn_budget': int(
            common_budget
        ),
        'common_reference_best': float(
            reference_best
        ),
        'high_performance_threshold': float(
            threshold_750
        ),
        'total_actions_per_method': int(
            PPO_TOTAL_ACTIONS
        ),
        'loop_actions': int(
            LOOP_ACTIONS_TOTAL
        ),
        'link_actions': int(
            LINK_ACTIONS_TOTAL
        ),
        'node_actions': int(
            NODE_ACTIONS_TOTAL
        ),
        'num_points_per_edge': int(NUM_POINTS_PER_EDGE),
        'random_seed': int(results[0].get('seed', 42)) if results else None,
        'final_budget_mean_definition': 'arithmetic mean of episode-best raw SEA for episode-end actual GNN-call counts in (0.9B, B]',
        'threshold_definition': 'first logged GNN forward-call index with best-so-far raw SEA strictly greater than 750 J/kg',
        'gnn_calls_definition': 'actual successful predictor forwards; initialization and tested candidates included; cache hits excluded',
        'not_reached': 'null; never imputed as reaching at B',
        'predictor_sha256': results[0].get('predictor_sha256') if results else None,
        'greedy_protocol': 'exhaustive_all_152_actions_per_step',
        'greedy_actions_evaluated_per_step': int(PPO_TOTAL_ACTIONS),
    }
    with open(
        os.path.join(
            csv_dir,
            'spi_metadata.json',
        ),
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_summary_table(results):
    (
        spi_map,
        reference_best,
        common_budget,
        threshold_750,
    ) = calculate_spi_metrics(results)

    print("\\n" + "=" * 168)
    print(
        "COMPARISON SUMMARY "
        "(Strict GNN Budget; Unified 152-Action Space)"
    )
    print("=" * 168)
    print(
        f"{'Method':<25} "
        f"{'Best':>12} "
        f"{'Avg(last10%)':>14} "
        f"{'Std(last10%)':>14} "
        f"{'nAUC':>10} "
        f"{'Calls@750':>12} "
        f"{'SPI':>10} "
        f"{'Episodes':>10} "
        f"{'GNN Calls':>12}"
    )
    print(
        f"Reference best = {reference_best:.6f}; "
        f"common high-performance threshold = {threshold_750:.6f}; "
        f"common budget = {common_budget:,}"
    )
    print("-" * 168)

    sorted_results = sorted(
        results,
        key=lambda result: spi_map[
            result['method']
        ]['Search Performance Index'],
        reverse=True,
    )

    for index, result in enumerate(
        sorted_results
    ):
        marker = '🏆' if index == 0 else '  '
        metrics = spi_map[result['method']]
        calls_90 = metrics[
            'calls_to_750'
        ]
        calls_text = (
            f"{calls_90:,}"
            if calls_90 is not None
            else "Not reached"
        )

        print(
            f"{marker}{result['method']:<23} "
            f"{metrics['best_performance']:>12.6f} "
            f"{metrics['avg_performance_last10']:>14.6f} "
            f"{metrics['std_performance']:>14.6f} "
            f"{metrics['efficiency_nAUC']:>10.6f} "
            f"{calls_text:>12} "
            f"{metrics['Search Performance Index']:>10.6f} "
            f"{result['num_episodes']:>10,} "
            f"{result.get('gnn_calls', 0):>12,}"
        )

    print("=" * 168)


def run_multi_seed_comparison(args):
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Seeds must be distinct')
    root = os.path.join(args.save_dir, 'multiseed_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, 'run_config.json'), 'w', encoding='utf-8') as stream:
        json.dump(vars(args), stream, indent=2)
    frames = []
    for seed in args.seeds:
        seed_dir = os.path.join(root, f'seed_{seed}')
        command = [sys.executable, os.path.abspath(__file__), '--gnn_model', os.path.abspath(args.gnn_model),
                   '--gnn_budget', str(args.gnn_budget), '--max_steps', str(args.max_steps),
                   '--seed', str(seed), '--save_dir', seed_dir,
                   '--methods', *args.methods, '--ppo_variants', *args.ppo_variants]
        if args.expected_model_sha256:
            command.extend(['--expected_model_sha256', args.expected_model_sha256])
        subprocess.run(command, check=True)
        from pathlib import Path
        summaries = list(Path(seed_dir).glob('compare_*/csv_data/summary.csv'))
        if len(summaries) != 1:
            raise RuntimeError('Expected exactly one completed comparison per seed')
        frame = pd.read_csv(summaries[0])
        if not (frame.threshold_resolution == 'per_gnn_evaluation').all():
            raise RuntimeError('Multi-seed N750 requires exact per-evaluation records')
        frames.append(frame.assign(seed=int(seed)))
    joined = pd.concat(frames, ignore_index=True)
    joined.to_csv(os.path.join(root, 'per_seed_comparison_metrics.csv'), index=False)
    rows = []
    for method, frame in joined.groupby('method', sort=False):
        reached = frame.calls_to_750.dropna()
        rows.append({'method': method, 'n_seeds': len(frame), 'n_reached': len(reached),
                     'n_not_reached': len(frame) - len(reached),
                     'N750_mean_reached': reached.mean(), 'N750_SD_reached': reached.std(ddof=1),
                     'N750_median_reached': reached.median(),
                     'best_SEA_mean': frame.best_performance.mean(),
                     'best_SEA_SD': frame.best_performance.std(ddof=1),
                     'final_window_episode_best_mean': frame.avg_performance_last10.mean(),
                     'final_window_episode_best_SD': frame.avg_performance_last10.std(ddof=1)})
    pd.DataFrame(rows).to_csv(os.path.join(root, 'cross_seed_summary.csv'), index=False)
    print(f'Multi-seed comparison completed: {root}')


def main():
    parser = argparse.ArgumentParser(description='Compare optimization methods with STRICT GNN forwards budget')
    parser.add_argument('--gnn_budget', type=int, default=100000, help='Total GNN forward budget (strict)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used independently by each method')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='Independent seeds, e.g. 0 1 2')
    parser.add_argument('--expected_model_sha256', default=None, help='Optional mandatory checkpoint identity')
    parser.add_argument('--max_steps', type=int, default=60, help='Maximum steps per episode')
    parser.add_argument('--gnn_model', type=str, default='results/best_model.pth', help='GNN model path')
    parser.add_argument('--methods', type=str, nargs='+', default=['all'],
                        choices=['ppo', 'greedy', 'hill', 'random', 'all'])
    parser.add_argument('--ppo_variants', type=str, nargs='+',
                        default=['loop_link_node'],
                        choices=['no_curriculum', 'loop_link_node', 'loop_node_link', 'all'])
    parser.add_argument('--save_dir', type=str, default='comparison_results', help='Output directory')
    args = parser.parse_args()

    if args.expected_model_sha256:
        with open(args.gnn_model, 'rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest().upper()
        if digest != args.expected_model_sha256.upper():
            raise ValueError(f"Wrong GNN predictor: {digest}")
    else:
        with open(args.gnn_model, 'rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest().upper()
    if args.seeds is not None:
        run_multi_seed_comparison(args)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_dir = os.path.join(args.save_dir, f'compare_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'run_config.json'), 'w', encoding='utf-8') as stream:
        json.dump(dict(vars(args), predictor_sha256=digest,
                       num_points_per_edge=NUM_POINTS_PER_EDGE, total_actions=PPO_TOTAL_ACTIONS,
                       normalization_denominator=EnvConfig.MAX_ENERGY,
                       normalized_SEA_capped=EnvConfig.NORMALIZED_SEA_CAP,
                       stage_thresholds=[train_ppo.Config.STAGE1_THRESHOLD,
                                         train_ppo.Config.STAGE2_THRESHOLD,
                                         train_ppo.Config.STAGE3_THRESHOLD]), stream, indent=2)

    print("\n" + "=" * 70)
    print("🔬 Algorithm Comparison - X-axis: Cumulative GNN Calls (STRICT)")
    print("=" * 70)
    print(f"GNN Budget: {args.gnn_budget:,}")
    print(
        "Unified action space: "
        f"{PPO_TOTAL_ACTIONS} = "
        f"{LOOP_ACTIONS_TOTAL} LOOP + "
        f"{LINK_ACTIONS_TOTAL} LINK + "
        f"{NODE_ACTIONS_TOTAL} NODE"
    )
    print(
        f"Internal points per edge: {NUM_POINTS_PER_EDGE}"
    )
    print(f"Random seed: {args.seed}")
    print(f"PPO Variants: {args.ppo_variants}")
    print(f"Output: {save_dir}")
    print("=" * 70)

    env_kwargs = {
        'gnn_model_path': args.gnn_model,
        'max_steps': args.max_steps,
        'edge_length': 1.0,
        'num_points_per_edge': NUM_POINTS_PER_EDGE,
    }

    if 'all' in args.methods:
        args.methods = ['ppo', 'greedy', 'hill', 'random']
    if 'all' in args.ppo_variants:
        args.ppo_variants = ['no_curriculum', 'loop_link_node', 'loop_node_link']

    results = []

    if 'ppo' in args.methods:
        if 'no_curriculum' in args.ppo_variants:
            results.append(train_ppo.run_ppo_compare(
                gnn_budget=args.gnn_budget,
                gnn_model_path=args.gnn_model,
                use_curriculum=False,
                action_order='loop_link_node',
                max_steps=args.max_steps,
                device='cpu',
                verbose_curriculum=0,
                show_gnn_progress=True,
                pbar_leave=True,
                seed=args.seed
            ))

        if 'loop_link_node' in args.ppo_variants:
            results.append(train_ppo.run_ppo_compare(
                gnn_budget=args.gnn_budget,
                gnn_model_path=args.gnn_model,
                use_curriculum=True,
                action_order='loop_link_node',
                max_steps=args.max_steps,
                device='cpu',
                verbose_curriculum=1,
                show_gnn_progress=True,
                pbar_leave=True,
                seed=args.seed
            ))

        if 'loop_node_link' in args.ppo_variants:
            results.append(train_ppo.run_ppo_compare(
                gnn_budget=args.gnn_budget,
                gnn_model_path=args.gnn_model,
                use_curriculum=True,
                action_order='loop_node_link',
                max_steps=args.max_steps,
                device='cpu',
                verbose_curriculum=1,
                show_gnn_progress=True,
                pbar_leave=True,
                seed=args.seed
            ))

    runners = {
        'greedy': run_greedy,
        'hill': run_hill_climbing,
        'random': run_random
    }
    for m in args.methods:
        if m != 'ppo' and m in runners:
            results.append(runners[m](args, env_kwargs, args.gnn_budget))

    if any(int(r['gnn_calls']) != args.gnn_budget for r in results):
        raise RuntimeError('A method failed to complete the shared actual GNN-forward budget')
    results_for_json = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != 'best_lattice_state'}
        results_for_json.append(r_copy)

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump([{k: v for k, v in r.items()
                    if k not in ['episode_perfs', 'global_best_history', 'gnn_calls_history', 'gnn_evaluation_history']}
                   for r in results_for_json], f, indent=2)

    save_comparison_csv(results, save_dir)

    if len(results) > 1:
        plot_comparison(results, save_dir)

    save_all_best_structures(results, save_dir)
    print_summary_table(results)

    print(f"\n✅ All results saved to: {save_dir}")


if __name__ == "__main__":
    main()
