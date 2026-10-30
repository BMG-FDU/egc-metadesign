import os
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from stable_baselines3.common.callbacks import BaseCallback

from lattice_state import LatticeState
from hexagonal_generator import node_link_data

PAPER_COLORS = {
    'primary': "#7C95E1",
    'secondary': "#B0C7FB",
    'tertiary': "#D0D1F9",
    'light': "#F2E6F1",
    'accent': "#EED6EC",
    'text': "#333333",
    'highlight': "#D04040"
}


class BestStructureTracker(BaseCallback):
    def __init__(
        self,
        save_dir: str,
        save_freq: int = 100,
        milestone_freq: int = 1000,
        top_k: int = 10,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.milestone_freq = int(milestone_freq)
        self.top_k = int(top_k)

        self.best_performance = 0.0
        self.best_lattice_state = None
        self.best_episode = 0

        self.episode_count = 0
        self.performance_history = []
        self.best_history = []
        self.milestones_saved = set()

        self.episode_snapshots = []

        self.top_structures = []

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'structures'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'structures', 'top10'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'milestones'), exist_ok=True)

    def _get_base_env(self):
        env = self.training_env.envs[0]
        while hasattr(env, 'env'):
            env = env.env
        return env

    def _safe_float(self, x, default=0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _state_key_from_loops(self, loop1, loop2):
        try:
            return (tuple((int(e), int(p)) for e, p in loop1),
                    tuple((int(e), int(p)) for e, p in loop2))
        except Exception:
            return (tuple(loop1), tuple(loop2))

    def _state_key(self, lattice_state: LatticeState):
        loop1 = lattice_state.loop1 if hasattr(lattice_state, 'loop1') else lattice_state.center_loop1
        loop2 = lattice_state.loop2 if hasattr(lattice_state, 'loop2') else lattice_state.center_loop2
        return self._state_key_from_loops(loop1, loop2)

    def _build_state_from_loops(self, base_env, loop1, loop2) -> LatticeState:
        edge_length = getattr(base_env, 'edge_length', 1.0)
        num_points_per_edge = getattr(base_env, 'num_points_per_edge', 10)
        st = LatticeState(
            edge_length=edge_length,
            num_points_per_edge=num_points_per_edge,
            initialize_random=False,
        )
        st.loop1 = copy.deepcopy(loop1)
        st.loop2 = copy.deepcopy(loop2)
        st.center_loop1 = st.loop1
        st.center_loop2 = st.loop2
        return st

    def _evaluate_state_if_possible(self, base_env, lattice_state) -> float:
        if lattice_state is None:
            return 0.0
        if hasattr(base_env, 'evaluate_lattice_state'):
            try:
                _, _, perf_real = base_env.evaluate_lattice_state(lattice_state)
                return self._safe_float(perf_real, 0.0)
            except Exception:
                return 0.0
        return 0.0

    def _get_window_best_snapshot(self, milestone_episode: int):
        if milestone_episode <= 0:
            return None
        start_ep = max(1, int(milestone_episode) - self.milestone_freq + 1)
        end_ep = int(milestone_episode)
        items = [x for x in self.episode_snapshots if start_ep <= int(x['episode']) <= end_ep]
        if not items:
            return None
        return max(items, key=lambda x: float(x['perf']))

    def _maybe_update_topk(self, lattice_state: LatticeState, perf: float, episode: int) -> bool:
        if lattice_state is None:
            return False

        perf = float(perf)
        key = self._state_key(lattice_state)
        changed = False

        for item in self.top_structures:
            if item['key'] == key:
                if perf > item['perf']:
                    item['perf'] = perf
                    item['episode'] = int(episode)
                    item['state'] = lattice_state.clone()
                    changed = True
                return changed

        if len(self.top_structures) < self.top_k:
            self.top_structures.append({
                'perf': perf,
                'episode': int(episode),
                'state': lattice_state.clone(),
                'key': key
            })
            changed = True
        else:
            worst = min(self.top_structures, key=lambda x: x['perf'])
            if perf > worst['perf']:
                self.top_structures.remove(worst)
                self.top_structures.append({
                    'perf': perf,
                    'episode': int(episode),
                    'state': lattice_state.clone(),
                    'key': key
                })
                changed = True

        if changed:
            self.top_structures.sort(key=lambda x: x['perf'], reverse=True)
        return changed

    def _save_topk_structures(self):
        top_dir = os.path.join(self.save_dir, 'structures', 'top10')
        os.makedirs(top_dir, exist_ok=True)

        summary = []
        for idx, item in enumerate(self.top_structures, start=1):
            st = item['state']
            perf = float(item['perf'])
            ep = int(item['episode'])
            json_path = os.path.join(top_dir, f"top_{idx:02d}.json")
            png_path = os.path.join(top_dir, f"top_{idx:02d}.png")

            self._save_structure_json(st, json_path, perf, ep, extra_meta={'rank': idx})
            self._visualize_structure(st, png_path, perf, ep, title_prefix=f"Top {idx:02d}")

            summary.append({'rank': idx, 'episode': ep, 'performance': perf, 'key': str(item['key'])})

        for rank in range(len(self.top_structures) + 1, self.top_k + 1):
            for ext in ['json', 'png']:
                p = os.path.join(top_dir, f"top_{rank:02d}.{ext}")
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        with open(os.path.join(top_dir, "top10_summary.json"), "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        if self.top_structures:
            best_item = self.top_structures[0]
            self.best_performance = float(best_item['perf'])
            self.best_episode = int(best_item['episode'])
            self.best_lattice_state = best_item['state'].clone()
            self._save_best_structure()

    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            self.episode_count += 1
            episode = self.episode_count
            base_env = self._get_base_env()

            current_lattice_state = None
            best_loop1 = info.get('best_loop1', None)
            best_loop2 = info.get('best_loop2', None)

            if best_loop1 is not None and best_loop2 is not None:
                try:
                    current_lattice_state = self._build_state_from_loops(base_env, best_loop1, best_loop2)
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] Failed to build state from info loops: {e}")
                    current_lattice_state = None

            if current_lattice_state is None:
                if hasattr(base_env, 'best_state_in_episode') and base_env.best_state_in_episode is not None:
                    current_lattice_state = base_env.best_state_in_episode.clone()
                elif hasattr(base_env, 'lattice_state') and base_env.lattice_state is not None:
                    current_lattice_state = base_env.lattice_state.clone()

            current_perf = self._safe_float(info.get('best_performance', 0.0), 0.0)

            perf_eval = self._evaluate_state_if_possible(base_env, current_lattice_state)
            if perf_eval > 0:
                current_perf = perf_eval

            self.performance_history.append(current_perf)

            if current_lattice_state is not None:
                self.episode_snapshots.append({
                    'episode': int(episode),
                    'perf': float(current_perf),
                    'state': current_lattice_state.clone(),
                })

            topk_changed = False
            if current_lattice_state is not None:
                topk_changed = self._maybe_update_topk(current_lattice_state, current_perf, episode)

            if topk_changed:
                new_best = self.top_structures[0]['perf'] if self.top_structures else 0.0
                if len(self.best_history) == 0 or float(new_best) > float(self.best_history[-1][1]):
                    self.best_history.append((episode, float(new_best)))

                if self.verbose:
                    top1 = self.top_structures[0]
                    print(f"\n🏆 Top-K 更新! Episode {episode}, "
                          f"Top1 Perf: {float(top1['perf']):.6f} (Ep {int(top1['episode'])})")

                self._save_topk_structures()

            if self.milestone_freq and (episode % self.milestone_freq == 0):
                if episode not in self.milestones_saved:
                    best_item = self._get_window_best_snapshot(episode)
                    if best_item is None and current_lattice_state is not None:
                        best_item = {
                            'episode': int(episode),
                            'perf': float(current_perf),
                            'state': current_lattice_state.clone()
                        }

                    if best_item is not None:
                        self._save_milestone_structure(
                            lattice_state=best_item['state'],
                            episode=episode,
                            performance=float(best_item['perf']),
                            name=f"ep{episode}",
                            window_start=max(1, episode - self.milestone_freq + 1),
                            window_end=episode,
                            source_episode=int(best_item['episode'])
                        )
                        self.milestones_saved.add(episode)

            if self.save_freq and (episode % self.save_freq == 0):
                self._save_checkpoint()

        return True

    def _save_structure_json(self, lattice_state, filepath: str, performance: float, episode: int, extra_meta: dict = None):
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
        loop1_conv = [(int(e), int(p)) for e, p in loop1] if loop1 else []
        loop2_conv = [(int(e), int(p)) for e, p in loop2] if loop2 else []

        meta = {
            'performance': float(performance),
            'episode': int(episode),
            'loop1': loop1_conv,
            'loop2': loop2_conv,
        }
        if extra_meta:
            meta.update({k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in extra_meta.items()})

        data['metadata'] = meta

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _visualize_structure(self, lattice_state, filepath: str,
                            performance: float, episode: int, title_prefix: str = ""):
        if lattice_state is None:
            return

        G = lattice_state.to_networkx_graph()
        plt.figure(figsize=(10, 10))
        pos = {n: n for n in G.nodes()}

        nx.draw_networkx_edges(G, pos, alpha=0.8, width=2.0, edge_color=PAPER_COLORS['primary'])

        if len(pos) > 0:
            xs, ys = zip(*pos.keys())
            plt.scatter(xs, ys, s=2, c=PAPER_COLORS['accent'], linewidths=0.8, zorder=3)

        plt.axis('equal')
        if title_prefix:
            title = f"{title_prefix}\nPerformance: {performance:.6f}, Episode: {episode}"
        else:
            title = f"Performance: {performance:.6f}, Episode: {episode}"

        plt.title(title, fontsize=14, color=PAPER_COLORS['text'], fontfamily='serif')
        plt.xticks([])
        plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    def _save_best_structure(self):
        if self.best_lattice_state is None:
            return
        json_path = os.path.join(self.save_dir, 'structures', 'best_structure.json')
        self._save_structure_json(self.best_lattice_state, json_path, self.best_performance, self.best_episode)

        png_path = os.path.join(self.save_dir, 'structures', 'best_structure.png')
        self._visualize_structure(self.best_lattice_state, png_path, self.best_performance, self.best_episode, title_prefix="Best Structure")

    def _save_milestone_structure(self, lattice_state, episode: int, performance: float, name: str,
                                  window_start: int, window_end: int, source_episode: int = None):
        if lattice_state is None:
            return

        json_path = os.path.join(self.save_dir, 'milestones', f'structure_{name}.json')
        extra_meta = {
            'milestone_episode': int(episode),
            'window_start': int(window_start),
            'window_end': int(window_end),
            'source_episode': int(source_episode if source_episode is not None else episode)
        }
        self._save_structure_json(lattice_state, json_path, performance, episode, extra_meta=extra_meta)

        png_path = os.path.join(self.save_dir, 'milestones', f'structure_{name}.png')
        title_prefix = f"Milestone Ep {episode}\nWindow [{window_start}, {window_end}]"
        if source_episode is not None:
            title_prefix += f"\nSource Ep {int(source_episode)}"
        self._visualize_structure(lattice_state, png_path, performance, episode, title_prefix=title_prefix)

    def _save_checkpoint(self):
        checkpoint = {
            'episode': int(self.episode_count),
            'best_performance': float(self.best_performance),
            'best_episode': int(self.best_episode),
            'best_history': [(int(ep), float(p)) for ep, p in self.best_history],
            'milestones_saved': sorted([int(x) for x in self.milestones_saved]),
            'recent_performances': [float(x) for x in self.performance_history[-100:]],
            'top_k': int(self.top_k),
            'top_structures_summary': [
                {'rank': i + 1, 'episode': int(it['episode']), 'performance': float(it['perf'])}
                for i, it in enumerate(self.top_structures)
            ]
        }
        path = os.path.join(self.save_dir, 'tracker_checkpoint.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    def _on_training_end(self):

        base_env = self._get_base_env()
        evaluated_state = getattr(base_env, 'gnn_best_state', None)
        evaluated_best = float(getattr(base_env, 'gnn_best_so_far_real', 0.0))
        if evaluated_state is not None and evaluated_best > self.best_performance:
            self._maybe_update_topk(evaluated_state, evaluated_best,
                                    int(getattr(base_env, 'current_episode', self.episode_count)))
        if self.top_structures:
            self._save_topk_structures()
        else:
            self._save_best_structure()
        self._save_checkpoint()
        self._plot_optimization_curve()
        self._plot_milestone_comparison()

    def _plot_optimization_curve(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax1 = axes[0]
        ax1.plot(self.performance_history, '.', color=PAPER_COLORS['secondary'], alpha=0.4, markersize=4, markeredgewidth=0)

        if len(self.performance_history) > 20:
            window = min(50, len(self.performance_history) // 5)
            smoothed = np.convolve(self.performance_history, np.ones(window) / window, mode='valid')
            ax1.plot(range(window - 1, len(self.performance_history)), smoothed, '-', color=PAPER_COLORS['primary'], lw=2, label='Moving Avg')

        if self.best_history:
            best_eps, best_perfs = zip(*self.best_history)
            ax1.scatter(best_eps, best_perfs, c=PAPER_COLORS['accent'], s=80, marker='*', zorder=5, label='Top-1 Updates')

        ax1.set_xlabel('Episode', fontsize=12, color=PAPER_COLORS['text'])
        ax1.set_ylabel('Performance', fontsize=12, color=PAPER_COLORS['text'])
        ax1.set_title('Optimization Progress', fontsize=14, color=PAPER_COLORS['text'])
        ax1.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.9)
        ax1.spines['top'].set_visible(True)
        ax1.spines['right'].set_visible(True)

        ax2 = axes[1]
        if self.best_history:
            episodes, performances = zip(*self.best_history)
            ax2.step(episodes, performances, '-', color=PAPER_COLORS['primary'], lw=2, where='post')
            ax2.scatter(episodes, performances, c=PAPER_COLORS['accent'], s=2, marker='o', zorder=5)
            ax2.fill_between(episodes, performances, min(performances), step='post', color=PAPER_COLORS['secondary'], alpha=0.2)

        ax2.set_xlabel('Episode', fontsize=12, color=PAPER_COLORS['text'])
        ax2.set_ylabel('Best Performance', fontsize=12, color=PAPER_COLORS['text'])
        ax2.set_title('Top-1 Best Evolution', fontsize=14, color=PAPER_COLORS['text'])
        ax2.spines['top'].set_visible(True)
        ax2.spines['right'].set_visible(True)

        plt.tight_layout()
        curve_path = os.path.join(self.save_dir, 'optimization_curve.png')
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 优化曲线已保存 (Paper Style): {curve_path}")

    def _plot_milestone_comparison(self):
        milestones_dir = os.path.join(self.save_dir, 'milestones')
        if not os.path.exists(milestones_dir):
            return

        milestone_files = sorted([
            os.path.join(milestones_dir, f)
            for f in os.listdir(milestones_dir)
            if f.endswith('.json')
        ])

        best_json = os.path.join(self.save_dir, 'structures', 'best_structure.json')
        if os.path.exists(best_json):
            milestone_files.append(best_json)

        if len(milestone_files) < 2:
            return

        if len(milestone_files) > 6:
            indices = [0] + list(np.linspace(1, len(milestone_files) - 2, 4, dtype=int)) + [len(milestone_files) - 1]
            milestone_files = [milestone_files[i] for i in sorted(set(indices))]

        n_plots = len(milestone_files)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, json_path in enumerate(milestone_files):
            ax = axes[idx]
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                G = nx.Graph()
                for node in data['nodes']:
                    pos = node.get('pos')
                    if pos:
                        G.add_node((float(pos[0]), float(pos[1])))
                for link in data['links']:
                    src, tgt = link['source'], link['target']
                    if isinstance(src, str):
                        src = tuple(map(float, src.split(',')))
                    if isinstance(tgt, str):
                        tgt = tuple(map(float, tgt.split(',')))
                    if src in G.nodes() and tgt in G.nodes():
                        G.add_edge(src, tgt)

                pos = {n: n for n in G.nodes()}
                nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.8, width=1, edge_color=PAPER_COLORS['primary'])
                if len(pos) > 0:
                    xs, ys = zip(*pos.keys())
                    ax.scatter(xs, ys, s=1, c=PAPER_COLORS['accent'], zorder=3)

                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

                metadata = data.get('metadata', {})
                perf = metadata.get('performance', 0)
                episode = metadata.get('episode', 0)
                basename = os.path.basename(json_path)

                if 'best' in basename:
                    title = f"Best (Ep {episode})\n{perf:.4f}"
                    ax.set_title(title, fontsize=11, fontweight='bold', color=PAPER_COLORS['primary'])
                else:
                    milestone_ep = metadata.get('milestone_episode', episode)
                    window_start = metadata.get('window_start', None)
                    window_end = metadata.get('window_end', None)
                    source_ep = metadata.get('source_episode', episode)

                    if window_start is not None and window_end is not None:
                        title = f"Milestone Ep {milestone_ep}\nWindow [{window_start}, {window_end}]\nBest: {perf:.4f}"
                    else:
                        title = f"Milestone Ep {milestone_ep}\nBest: {perf:.4f}"

                    if int(source_ep) != int(milestone_ep):
                        title += f"\n(Source Ep {source_ep})"
                    ax.set_title(title, fontsize=11, color=PAPER_COLORS['text'])

            except Exception:
                ax.axis('off')

        for idx in range(len(milestone_files), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        comparison_path = os.path.join(self.save_dir, 'milestone_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 里程碑对比图已保存 (Paper Style): {comparison_path}")


def visualize_json_structure(json_path: str, output_path: str = None, show: bool = False, figsize=(10, 10)):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    G = nx.Graph()
    for node in data['nodes']:
        pos = node.get('pos')
        if pos:
            G.add_node((float(pos[0]), float(pos[1])))
    for link in data['links']:
        src, tgt = link['source'], link['target']
        if isinstance(src, str):
            src = tuple(map(float, src.split(',')))
        if isinstance(tgt, str):
            tgt = tuple(map(float, tgt.split(',')))
        if src in G.nodes() and tgt in G.nodes():
            G.add_edge(src, tgt)

    metadata = data.get('metadata', {})
    performance = metadata.get('performance', 'N/A')

    plt.figure(figsize=figsize)
    pos = {n: n for n in G.nodes()}
    nx.draw_networkx_edges(G, pos, alpha=0.8, width=1.0, edge_color=PAPER_COLORS['primary'])
    if len(pos) > 0:
        xs, ys = zip(*pos.keys())
        plt.scatter(xs, ys, s=30, c=PAPER_COLORS['accent'],
                    edgecolors=PAPER_COLORS['primary'], linewidths=0.8, zorder=3)

    plt.axis('equal')
    title = 'Lattice Structure'
    if performance != 'N/A':
        if isinstance(performance, float):
            title += f'\nPerformance: {performance:.6f}'
        else:
            title += f'\nPerformance: {performance}'

    plt.title(title, fontsize=14, color=PAPER_COLORS['text'], fontfamily='serif')
    plt.xticks([])
    plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    if output_path is None:
        output_path = json_path.replace('.json', '.png')
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    if show:
        plt.show()
    else:
        plt.close()
    return G


if __name__ == "__main__":
    print("Testing BestStructureTracker (Paper Style + Top-10 + VecEnv-safe + Sliding-Window Milestone Best)...")
