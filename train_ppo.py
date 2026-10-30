import matplotlib
matplotlib.use('Agg')

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import random
import time
import copy

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import torch
import torch.nn as nn

from lattice_env import LatticeDesignEnv, EnvConfig
from lattice_state import LatticeState
from best_structure_tracker import BestStructureTracker

from tqdm.auto import tqdm


COLORS = {
    'primary': "#7C95E1",
    'secondary': "#B0C7FB",
    'tertiary': "#D0D1F9",
    'bg_line': "#F2E6F1",
    'accent': "#EED6EC",
    'text': "#333333",
    'success': "#5CA05C",
    'warning': "#FFA500"
}


class Config:
    GNN_MODEL_PATH = 'results/best_model.pth'
    SAVE_DIR = 'ppo_results'
    SEED = 42
    TOTAL_TIMESTEPS = 150000
    MAX_STEPS_PER_EPISODE = 60

    USE_CURRICULUM = True
    ACTION_ORDER = 'loop_link_node'

    STAGE1_THRESHOLD = 0.5525
    STAGE2_THRESHOLD = 0.595
    STAGE3_THRESHOLD = 0.7225
    MIN_EPISODES_PER_STAGE = 400
    EVAL_WINDOW = 50
    STABILITY_THRESHOLD = 0.07
    COOLDOWN_EPISODES = 60
    CONSECUTIVE_SUCCESS_REQUIRED = 3

    LEARNING_RATE = 3e-4
    GAMMA = 0.98
    ENT_COEF = 0.05
    VF_COEF = 1.0
    N_STEPS = 2048
    BATCH_SIZE = 512
    N_EPOCHS = 15
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    MAX_GRAD_NORM = 0.5
    USE_LR_SCHEDULE = True

    USE_POTENTIAL_SHAPING = True
    PI_ARCH = [512, 512, 256]
    VF_ARCH = [512, 512, 256]

    SAVE_CHECKPOINTS = True
    SAVE_FREQ = 30000

    NUM_POINTS_PER_EDGE = EnvConfig.NUM_POINTS_PER_EDGE
    EDGE_LENGTH = 1.0


def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj


def cosine_schedule(initial_value: float, final_value: float = 0.0):
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * (1 + np.cos(np.pi * (1 - progress_remaining))) / 2
    return func


def get_action_masks(env) -> np.ndarray:
    return env.action_masks()


class GNNBudgetStopCallback(BaseCallback):
    def __init__(self, budget: int, verbose: int = 0):
        super().__init__(verbose)
        self.budget = int(budget)

    def _get_base_env(self):
        env = self.training_env.envs[0]
        while hasattr(env, 'env'):
            env = env.env
        return env

    def _on_step(self) -> bool:
        base_env = self._get_base_env()
        if hasattr(base_env, 'get_gnn_forward_calls'):
            if base_env.get_gnn_forward_calls() >= self.budget:
                if self.verbose:
                    print(f"[STOP] GNN forward budget reached: {base_env.get_gnn_forward_calls()}/{self.budget}")
                return False
        return True


class GNNProgressBarCallback(BaseCallback):
    def __init__(self, total_budget: int, desc: str = "PPO", leave: bool = True, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.total_budget = int(total_budget)
        self.desc = str(desc)
        self.leave = bool(leave)
        self._pbar = None
        self._last_calls = 0

    def _get_base_env(self):
        env = self.training_env.envs[0]
        while hasattr(env, 'env'):
            env = env.env
        return env

    def _on_training_start(self) -> None:
        self._last_calls = 0
        self._pbar = tqdm(
            total=self.total_budget,
            desc=f"{self.desc} | GNN Calls",
            unit="calls",
            dynamic_ncols=True,
            leave=self.leave,
            smoothing=0.05
        )

    def _on_step(self) -> bool:
        if self._pbar is None:
            return True
        base_env = self._get_base_env()
        calls = int(base_env.get_gnn_forward_calls()) if hasattr(base_env, 'get_gnn_forward_calls') else 0
        calls = min(calls, self.total_budget)
        delta = calls - self._last_calls
        if delta > 0:
            self._pbar.update(delta)
            self._last_calls = calls
        return True

    def _on_training_end(self) -> None:
        if self._pbar is None:
            return
        base_env = self._get_base_env()
        calls = int(base_env.get_gnn_forward_calls()) if hasattr(base_env, 'get_gnn_forward_calls') else self._last_calls
        calls = min(calls, self.total_budget)
        if calls > self._last_calls:
            self._pbar.update(calls - self._last_calls)
            self._last_calls = calls
        self._pbar.close()
        self._pbar = None


def _extract_best_loops_from_info(info: dict):
    if not isinstance(info, dict):
        return None, None
    loop1 = info.get('best_loop1', None)
    loop2 = info.get('best_loop2', None)
    return loop1, loop2


def _build_lattice_state_from_loops(base_env, loop1, loop2):
    edge_length = getattr(base_env, 'edge_length', 1.0)
    npp = getattr(base_env, 'num_points_per_edge', 10)

    st = LatticeState(
        edge_length=edge_length,
        num_points_per_edge=npp,
        initialize_random=False,
    )
    st.loop1 = copy.deepcopy(loop1)
    st.loop2 = copy.deepcopy(loop2)
    st.center_loop1 = st.loop1
    st.center_loop2 = st.loop2
    return st


class ComparePerformanceTracker(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.episode_best = []
        self.global_best = 0.0
        self.global_best_history = []
        self.gnn_calls_history = []
        self.curriculum_stage_history = []
        self.available_actions_history = []
        self.effective_steps_history = []

        self.best_lattice_state = None
        self.best_gnn_calls = 0

    def _get_base_env(self):
        env = self.training_env.envs[0]
        while hasattr(env, 'env'):
            env = env.env
        return env

    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            perf_real = float(info.get('best_performance', 0.0))
            self.episode_best.append(perf_real)

            base_env = self._get_base_env()

            calls = info.get('gnn_forward_calls', base_env.get_gnn_forward_calls())
            calls = int(calls)
            self.gnn_calls_history.append(calls)
            self.curriculum_stage_history.append(int(info.get('curriculum_stage', 3)))
            self.available_actions_history.append(int(info.get('available_actions', 0)))
            self.effective_steps_history.append(int(info.get('effective_max_steps', 0)))

            if perf_real > self.global_best:
                self.global_best = perf_real
                self.best_gnn_calls = calls

                loop1, loop2 = _extract_best_loops_from_info(info)
                if loop1 is not None and loop2 is not None:
                    try:
                        self.best_lattice_state = _build_lattice_state_from_loops(base_env, loop1, loop2)
                    except Exception:
                        self.best_lattice_state = None
                else:
                    if hasattr(base_env, 'best_state_in_episode') and base_env.best_state_in_episode is not None:
                        self.best_lattice_state = base_env.best_state_in_episode.clone()
                    elif hasattr(base_env, 'lattice_state') and base_env.lattice_state is not None:
                        self.best_lattice_state = base_env.lattice_state.clone()

            self.global_best_history.append(self.global_best)

        return True


class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        stage_thresholds,
        min_episodes_per_stage: int,
        eval_window: int,
        stability_threshold: float,
        cooldown_episodes: int,
        consecutive_required: int = 3,
        verbose: int = 1
    ):
        super().__init__(verbose=verbose)
        self.stage_thresholds = list(stage_thresholds)
        self.min_episodes = int(min_episodes_per_stage)
        self.eval_window = int(eval_window)
        self.stability_threshold = float(stability_threshold)
        self.cooldown = int(cooldown_episodes)
        self.consecutive_required = int(consecutive_required)

        self.current_stage = 1
        self.episodes_in_stage = 0
        self.total_episodes = 0
        self.stage_performances = []
        self.cooldown_remaining = 0
        self.consecutive_success_count = 0


        self.last_effective_steps = 10
        self.last_available_actions = 2

    def _get_base_env(self):
        env = self.training_env.envs[0]
        while hasattr(env, 'env'):
            env = env.env
        return env

    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            base_env = self._get_base_env()

            perf_norm = info.get('best_performance_normalized', 0.0)
            if not perf_norm:
                perf_real = float(info.get('best_performance', 0.0))
                max_energy = float(getattr(base_env, 'MAX_ENERGY', 1.0))
                perf_norm = (perf_real / max_energy) if (max_energy > 0 and perf_real > 0) else 0.0

            self.stage_performances.append(float(perf_norm))
            self.episodes_in_stage += 1
            self.total_episodes += 1

            if hasattr(base_env, 'set_episodes_in_stage'):
                try:
                    base_env.set_episodes_in_stage(self.episodes_in_stage)
                except Exception:
                    pass

            cur_steps = info.get('effective_max_steps', self.last_effective_steps)
            cur_actions = info.get('available_actions', self.last_available_actions)

            if cur_steps != self.last_effective_steps and self.verbose:
                print(f"[Curriculum] Steps updated: {self.last_effective_steps} → {cur_steps} (Stage {self.current_stage})")
            self.last_effective_steps = cur_steps

            if cur_actions != self.last_available_actions and self.verbose:
                print(f"[Curriculum] Actions unlocked: {self.last_available_actions} → {cur_actions} (Stage {self.current_stage})")
            self.last_available_actions = cur_actions

            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1

            if self.current_stage < 3:
                if self.cooldown_remaining == 0 and self.episodes_in_stage >= self.min_episodes:
                    self._check_transition(base_env)

            if self.verbose and (self.episodes_in_stage % 50 == 0):
                self._print_progress()

        return True

    def _check_transition(self, base_env):
        if len(self.stage_performances) < self.eval_window:
            return
        recent = self.stage_performances[-self.eval_window:]
        avg, std = float(np.mean(recent)), float(np.std(recent))
        threshold = float(self.stage_thresholds[self.current_stage - 1])

        ok = (avg >= threshold and std <= self.stability_threshold)
        if ok:
            self.consecutive_success_count += 1
            if self.verbose and self.consecutive_success_count < self.consecutive_required:
                print(f"[Curriculum] ✔ Consecutive success: {self.consecutive_success_count}/{self.consecutive_required} "
                      f"(avg={avg:.4f}, std={std:.4f}, thr={threshold:.4f})")
        else:
            if self.consecutive_success_count > 0 and self.verbose:
                print(f"[Curriculum] ✗ Consecutive reset (avg={avg:.4f}, std={std:.4f}, thr={threshold:.4f})")
            self.consecutive_success_count = 0

        if self.consecutive_success_count >= self.consecutive_required:
            self._advance_stage(base_env, avg, std)
            self.consecutive_success_count = 0

    def _advance_stage(self, base_env, avg, std):
        old = self.current_stage
        self.current_stage = min(self.current_stage + 1, 3)

        self.episodes_in_stage = 0
        self.stage_performances = []
        self.cooldown_remaining = self.cooldown

        if hasattr(base_env, 'set_curriculum_stage'):
            try:
                base_env.set_curriculum_stage(self.current_stage)
            except Exception:
                pass

        try:
            self.last_effective_steps = int(getattr(base_env, 'effective_max_steps', self.last_effective_steps))
            self.last_available_actions = int(getattr(base_env, 'available_actions', self.last_available_actions))
        except Exception:
            pass

        if self.verbose:
            print("\n" + "-" * 70)
            print(f"🎓 CURRICULUM ADVANCE: Stage {old} → {self.current_stage} (Episode {self.total_episodes})")
            print(f"   Recent avg/std: {avg:.4f} ± {std:.4f}")
            print(f"   Now steps={self.last_effective_steps}, actions={self.last_available_actions} (will unlock gradually)")
            print("-" * 70 + "\n")

    def _print_progress(self):
        if not self.stage_performances:
            return
        recent = self.stage_performances[-self.eval_window:] if len(self.stage_performances) >= self.eval_window else self.stage_performances
        print(f"[Curriculum] Stage {self.current_stage} | ep_in_stage={self.episodes_in_stage}/{self.min_episodes} "
              f"| perf={float(np.mean(recent)):.4f} ± {float(np.std(recent)):.4f} "
              f"| actions={self.last_available_actions}")


class Logger(BaseCallback):
    def __init__(self, check_freq=512, save_dir='logs', verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.episode_performances = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_steps = []
        self.episode_actions = []
        self.best_ever = 0.0
        self.entropies = []
        self.evs = []
        self.curriculum_stages = []

    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            perf = info.get('best_performance', 0)
            self.episode_performances.append(perf)

            ep_info = info.get('episode', {})
            self.episode_rewards.append(ep_info.get('r', 0))
            self.episode_lengths.append(ep_info.get('l', 1))
            self.episode_steps.append(info.get('effective_max_steps', 50))
            self.episode_actions.append(info.get('available_actions', LatticeDesignEnv.TOTAL_ACTIONS))

            if perf > self.best_ever:
                self.best_ever = perf

            self.curriculum_stages.append(info.get('curriculum_stage', 3))

        if self.n_calls % self.check_freq == 0:
            self._collect_metrics()
            if self.verbose and len(self.episode_performances) > 0:
                self._print_diagnostics()

        return True

    def _collect_metrics(self):
        if hasattr(self.model, 'logger') and self.model.logger:
            logs = self.model.logger.name_to_value
            ev = logs.get('train/explained_variance')
            if ev is not None:
                self.evs.append(ev)
            ent = logs.get('train/entropy_loss')
            if ent is not None:
                self.entropies.append(-ent)

    def _print_diagnostics(self):
        recent_perf = self.episode_performances[-30:]
        recent_actions = self.episode_actions[-1] if self.episode_actions else 0
        print(f"\n{'='*70}")
        print(f"📊 Episode {len(self.episode_performances)} | "
              f"Perf: {np.mean(recent_perf):.6f} | "
              f"Best: {self.best_ever:.6f} | "
              f"Actions: {recent_actions}")
        print(f"{'='*70}")

    def _on_training_end(self):
        self._save_data()
        self._save_csv_data()
        self._plot_curves()

    def _save_data(self):
        data = convert_to_json_serializable({
            'episode_performances': self.episode_performances,
            'episode_rewards': self.episode_rewards,
            'episode_actions': self.episode_actions,
            'best_ever': self.best_ever,
        })
        with open(os.path.join(self.save_dir, 'training_data.json'), 'w') as f:
            json.dump(data, f, indent=2)

    def _save_csv_data(self):
        csv_dir = os.path.join(self.save_dir, 'csv_data')
        os.makedirs(csv_dir, exist_ok=True)

        if self.episode_performances:
            df = pd.DataFrame({
                'episode': range(1, len(self.episode_performances) + 1),
                'performance': self.episode_performances,
                'reward': self.episode_rewards if self.episode_rewards else [0] * len(self.episode_performances),
                'actions': self.episode_actions if self.episode_actions else [LatticeDesignEnv.TOTAL_ACTIONS] * len(self.episode_performances),
                'stage': self.curriculum_stages if self.curriculum_stages else [3] * len(self.episode_performances)
            })
            df.to_csv(os.path.join(csv_dir, 'episode_data.csv'), index=False)

        print(f"📊 CSV数据已保存到: {csv_dir}/")

    def _plot_curves(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        n_episodes = len(self.episode_performances)
        episodes = range(n_episodes)

        ax = axes[0, 0]
        ax.plot(episodes, self.episode_performances, '.', alpha=0.15, markersize=2, color=COLORS['secondary'])
        if n_episodes > 30:
            window = min(50, n_episodes // 10)
            smoothed = np.convolve(self.episode_performances, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, n_episodes), smoothed, '-', lw=2, color=COLORS['primary'], label='Smoothed')
        ax.set_title('Episode Performance', fontsize=12, color=COLORS['text'])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy Absorption (J)')
        ax.legend(loc='lower right', fontsize=8)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        ax = axes[0, 1]
        if self.episode_rewards:
            ax.plot(episodes, self.episode_rewards, '.', alpha=0.15, markersize=2, color=COLORS['secondary'])
            if n_episodes > 30:
                window = min(50, n_episodes // 10)
                smoothed = np.convolve(self.episode_rewards, np.ones(window) / window, mode='valid')
                ax.plot(range(window - 1, n_episodes), smoothed, '-', lw=2, color=COLORS['primary'])
        ax.set_title('Episode Rewards', fontsize=12, color=COLORS['text'])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        ax = axes[0, 2]
        if self.entropies:
            ax.plot(range(len(self.entropies)), self.entropies, lw=1.5, color=COLORS['primary'])
            ax.legend(fontsize=7, loc='upper right')
        ax.set_title('Policy Entropy', fontsize=12, color=COLORS['text'])
        ax.set_xlabel('Update')
        ax.set_ylabel('Entropy')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        ax = axes[1, 0]
        if self.evs:
            ax.plot(range(len(self.evs)), self.evs, lw=1.5, color=COLORS['primary'])
            ax.axhline(y=0.8, color=COLORS['success'], ls='--', alpha=0.7, label='Good (0.8)')
            ax.axhline(y=0.5, color=COLORS['warning'], ls='--', alpha=0.7, label='OK (0.5)')
            ax.legend(fontsize=8)
        ax.set_title('Explained Variance', fontsize=12, color=COLORS['text'])
        ax.set_xlabel('Update')
        ax.set_ylabel('Explained Variance')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        ax = axes[1, 1]
        if self.curriculum_stages and self.episode_actions:
            color1 = COLORS['primary']
            ax.plot(episodes, self.curriculum_stages, '-', lw=2, color=color1, label='Stage')
            ax.set_ylabel('Curriculum Stage', color=COLORS['text'])
            ax.set_yticks([1, 2, 3])
            ax.tick_params(axis='y', labelcolor=COLORS['text'])

            lines1, labels1 = ax.get_legend_handles_labels()
            ax.legend(lines1, labels1, loc='center right', fontsize=8)

        ax.set_title('Curriculum Progress & Action Unlocking', fontsize=12, color=COLORS['text'])
        ax.set_xlabel('Episode')
        ax.spines['top'].set_visible(True)

        ax = axes[1, 2]
        if self.episode_rewards and self.episode_lengths:
            rps = [r / max(1, l) for r, l in zip(self.episode_rewards, self.episode_lengths)]
            ax.plot(episodes, rps, '.', alpha=0.15, markersize=2, color=COLORS['secondary'])
            if n_episodes > 30:
                window = min(50, n_episodes // 10)
                smoothed = np.convolve(rps, np.ones(window) / window, mode='valid')
                ax.plot(range(window - 1, n_episodes), smoothed, '-', lw=2, color=COLORS['primary'])
        ax.set_title('Reward per Step', fontsize=12, color=COLORS['text'])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward / Step')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=200)
        plt.close()
        print(f"📈 训练曲线已保存: {self.save_dir}/training_curves.png")


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f'model_{self.n_calls}.pt')
            torch.save({'policy': self.model.policy.state_dict()}, path)
        return True


def train_with_structure_tracking():
    config = Config()
    print(f"\n{'='*70}")
    print("PPO Training - 动作顺序配置")
    print(f"{'='*70}")
    print(f"Curriculum: {config.USE_CURRICULUM}")
    print(f"Action Order: {config.ACTION_ORDER}")
    print(f"Timesteps: {config.TOTAL_TIMESTEPS}")
    print(f"NUM_POINTS_PER_EDGE: {config.NUM_POINTS_PER_EDGE}")
    print(f"{'='*70}")

    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.SAVE_DIR, f'run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump({k: v for k, v in vars(Config).items() if not k.startswith('_')}, f, indent=2)

    base_env = LatticeDesignEnv(
        gnn_model_path=config.GNN_MODEL_PATH,
        edge_length=config.EDGE_LENGTH,
        num_points_per_edge=config.NUM_POINTS_PER_EDGE,
        max_steps=config.MAX_STEPS_PER_EPISODE,
        curriculum_enabled=config.USE_CURRICULUM,
        curriculum_stage=1 if config.USE_CURRICULUM else 3,
        use_potential_shaping=config.USE_POTENTIAL_SHAPING,
        action_order=config.ACTION_ORDER,
        verbose=True
    )
    env = ActionMasker(base_env, get_action_masks)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    lr = cosine_schedule(config.LEARNING_RATE, config.LEARNING_RATE * 0.1) if config.USE_LR_SCHEDULE else config.LEARNING_RATE

    model = MaskablePPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=lr,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        policy_kwargs=dict(
            net_arch=dict(pi=config.PI_ARCH, vf=config.VF_ARCH),
            activation_fn=nn.Tanh
        ),
        verbose=0,
        device='cpu',
        tensorboard_log=os.path.join(log_dir, 'tensorboard'),
        seed=config.SEED
    )

    logger = Logger(check_freq=max(512, config.N_STEPS // 4), save_dir=log_dir)
    structure_tracker = BestStructureTracker(save_dir=log_dir, save_freq=100, verbose=1)
    callbacks = [logger, structure_tracker]

    if config.USE_CURRICULUM:
        callbacks.append(CurriculumCallback(
            stage_thresholds=[config.STAGE1_THRESHOLD, config.STAGE2_THRESHOLD, config.STAGE3_THRESHOLD],
            min_episodes_per_stage=config.MIN_EPISODES_PER_STAGE,
            eval_window=config.EVAL_WINDOW,
            stability_threshold=config.STABILITY_THRESHOLD,
            cooldown_episodes=config.COOLDOWN_EPISODES,
            consecutive_required=config.CONSECUTIVE_SUCCESS_REQUIRED
        ))

    if config.SAVE_CHECKPOINTS:
        callbacks.append(CheckpointCallback(
            save_freq=max(config.SAVE_FREQ, config.N_STEPS * 3),
            save_path=os.path.join(log_dir, 'checkpoints')
        ))

    try:
        model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted! Saving...")
        structure_tracker._on_training_end()

    torch.save({
        'policy': model.policy.state_dict(),
        'best_performance': structure_tracker.best_performance
    }, os.path.join(log_dir, 'final_policy.pt'))

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"Best Performance: {structure_tracker.best_performance:.6f}")
    print(f"Output: {log_dir}")
    print(f"{'='*70}")

    return structure_tracker.best_performance, log_dir


def run_ppo_compare(
    *,
    gnn_budget: int,
    gnn_model_path: str = None,
    use_curriculum: bool = True,
    action_order: str = None,
    max_steps: int = None,
    device: str = 'cpu',
    verbose_curriculum: int = 1,
    show_gnn_progress: bool = True,
    pbar_leave: bool = True,
    seed: int = None,
    action_curriculum: bool = None,
    step_curriculum: bool = None,
) -> dict:
    cfg = Config()
    if seed is None:
        seed = int(cfg.SEED)
    seed = int(seed)
    if action_curriculum is None:
        action_curriculum = bool(use_curriculum)
    if step_curriculum is None:
        step_curriculum = bool(use_curriculum)
    any_curriculum = bool(action_curriculum or step_curriculum)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if gnn_model_path is None:
        gnn_model_path = cfg.GNN_MODEL_PATH
    if action_order is None:
        action_order = cfg.ACTION_ORDER
    if max_steps is None:
        max_steps = cfg.MAX_STEPS_PER_EPISODE

    method_name = (
        "PPO_NoCurriculum" if not use_curriculum
        else ("PPO_LoopLinkAnyPair" if action_order == 'loop_link_node' else "PPO_LoopNodeLink")
    )

    base_env = LatticeDesignEnv(
        gnn_model_path=gnn_model_path,
        edge_length=cfg.EDGE_LENGTH,
        num_points_per_edge=cfg.NUM_POINTS_PER_EDGE,
        max_steps=max_steps,
        curriculum_enabled=any_curriculum,
        curriculum_stage=1 if any_curriculum else 3,
        action_curriculum_enabled=action_curriculum,
        step_curriculum_enabled=step_curriculum,
        use_potential_shaping=cfg.USE_POTENTIAL_SHAPING,
        action_order=action_order,
        gnn_budget=int(gnn_budget),
        verbose=False
    )
    base_env.reset_gnn_counter(budget=int(gnn_budget), clear_cache=False)
    base_env.run_seed = seed

    env = ActionMasker(base_env, get_action_masks)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    lr = cosine_schedule(cfg.LEARNING_RATE, cfg.LEARNING_RATE * 0.1) if cfg.USE_LR_SCHEDULE else cfg.LEARNING_RATE

    model = MaskablePPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=lr,
        n_steps=cfg.N_STEPS,
        batch_size=cfg.BATCH_SIZE,
        n_epochs=cfg.N_EPOCHS,
        gamma=cfg.GAMMA,
        gae_lambda=cfg.GAE_LAMBDA,
        clip_range=cfg.CLIP_RANGE,
        ent_coef=cfg.ENT_COEF,
        vf_coef=cfg.VF_COEF,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        policy_kwargs=dict(net_arch=dict(pi=cfg.PI_ARCH, vf=cfg.VF_ARCH), activation_fn=nn.Tanh),
        verbose=0,
        device=device,
        seed=seed
    )

    tracker = ComparePerformanceTracker(verbose=0)

    stop_cb = GNNBudgetStopCallback(int(gnn_budget), verbose=0)

    pbar_cb = None
    if show_gnn_progress:
        pbar_cb = GNNProgressBarCallback(
            total_budget=int(gnn_budget),
            desc=method_name,
            leave=pbar_leave,
            verbose=0
        )

    callbacks = [tracker]

    if any_curriculum:
        callbacks.append(CurriculumCallback(
            stage_thresholds=[cfg.STAGE1_THRESHOLD, cfg.STAGE2_THRESHOLD, cfg.STAGE3_THRESHOLD],
            min_episodes_per_stage=cfg.MIN_EPISODES_PER_STAGE,
            eval_window=cfg.EVAL_WINDOW,
            stability_threshold=cfg.STABILITY_THRESHOLD,
            cooldown_episodes=cfg.COOLDOWN_EPISODES,
            consecutive_required=cfg.CONSECUTIVE_SUCCESS_REQUIRED,
            verbose=verbose_curriculum
        ))

    if pbar_cb is not None:
        callbacks.append(pbar_cb)

    callbacks.append(stop_cb)

    total_timesteps = int(gnn_budget * max_steps * 2)

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    elapsed = time.time() - t0

    actual_gnn_calls = base_env.get_gnn_forward_calls()

    return {
        'method': method_name,
        'episode_perfs': tracker.episode_best,
        'global_best_history': tracker.global_best_history,
        'best_performance': float(base_env.gnn_best_so_far_real),
        'num_episodes': len(tracker.episode_best),
        'elapsed_time': elapsed,
        'gnn_calls': int(actual_gnn_calls),
        'gnn_budget': int(gnn_budget),
        'gnn_calls_history': tracker.gnn_calls_history,
        'gnn_evaluation_history': list(base_env.gnn_evaluation_history),
        'curriculum_stage_history': tracker.curriculum_stage_history,
        'available_actions_history': tracker.available_actions_history,
        'effective_steps_history': tracker.effective_steps_history,
        'use_curriculum': any_curriculum,
        'action_curriculum': bool(action_curriculum),
        'step_curriculum': bool(step_curriculum),
        'action_order': action_order,
        'seed': seed,
        'best_lattice_state': base_env.gnn_best_state.clone() if base_env.gnn_best_state is not None else None,
        'best_gnn_calls': int(base_env.gnn_best_call),
        'predictor_sha256': base_env.predictor_sha256,
        'predictor_use_global_feat': base_env.predictor_use_global_feat
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--curriculum', action='store_true', default=None)
    parser.add_argument('--no_curriculum', action='store_false', dest='curriculum')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--action_order', type=str, default=None,
                        choices=['loop_link_node', 'loop_node_link'])
    args = parser.parse_args()

    if args.timesteps:
        Config.TOTAL_TIMESTEPS = args.timesteps
    if args.curriculum is not None:
        Config.USE_CURRICULUM = args.curriculum
    if args.save_dir:
        Config.SAVE_DIR = args.save_dir
    if args.action_order:
        Config.ACTION_ORDER = args.action_order


if __name__ == "__main__":
    parse_args()
    train_with_structure_tracking()
