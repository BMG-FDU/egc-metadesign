#!/usr/bin/env python3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import json
import os
import copy
from typing import Tuple, Dict, Any, Optional
from collections import deque

from lattice_state import LatticeState
from GNN import parse_networkx_json, GINCurveRegressorWithDenom


class EnvConfig:
    MAX_ENERGY = 850
    EDGE_LENGTH = 1.0
    NUM_POINTS_PER_EDGE = 10
    MAX_STEPS = 60
    USE_POTENTIAL_SHAPING = True
    CURRICULUM_ENABLED = True
    INITIAL_STAGE = 1

    EXTRA_DIM = 32

    STAGE1_BASE_STEPS = 10
    STAGE1_MAX_STEPS = 10
    STAGE2_BASE_STEPS = 10
    STAGE2_MAX_STEPS = 35
    STAGE3_BASE_STEPS = 35
    STAGE3_MAX_STEPS = 60
    STEPS_INCREASE_INTERVAL = 15

    ACTION_UNLOCK_EPISODES = 80
    STAGE1_ACTIONS = 2

    LINK_ACTIONS_PER_LOOP = 15
    LINK_ACTIONS_TOTAL = 30
    NODE_POSITIONS_PER_EDGE = 10
    NODE_ACTIONS_PER_LOOP = 6 * NODE_POSITIONS_PER_EDGE  
    NODE_ACTIONS_TOTAL = 2 * NODE_ACTIONS_PER_LOOP      

    STAGE2_ACTIONS_LOOP_LINK_NODE = 32  
    STAGE2_ACTIONS_LOOP_NODE_LINK = 122  
    STAGE3_ACTIONS = 152                


class RewardConfig:
    DELTA_MULTIPLIER_LOOP = 20.0
    DELTA_MULTIPLIER_OTHER = 12.0
    PERF_REWARD_COEF = 0.3
    POTENTIAL_COEF = 3.0
    EXPLORATION_REWARD = 0.3
    INVALID_ACTION_PENALTY = -0.5
    UNAVAILABLE_ACTION_PENALTY = -1.0
    LOOP_BONUS_THRESHOLD = 0.02
    LOOP_BONUS_MULTIPLIER = 10.0
    HIGH_PERF_THRESHOLD = 0.80
    HIGH_PERF_MULTIPLIER = 3.0
    REWARD_CLIP_MIN = -3.0
    REWARD_CLIP_MAX = 5.0


class LatticeDesignEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    MAX_ENERGY = EnvConfig.MAX_ENERGY
    LINK_EDGE_PAIRS = tuple((i, j) for i in range(6) for j in range(i + 1, 6))
    TOTAL_ACTIONS = EnvConfig.STAGE3_ACTIONS

    def __init__(
        self,
        gnn_model_path: str = 'results/best_model.pth',
        edge_length: float = EnvConfig.EDGE_LENGTH,
        num_points_per_edge: int = EnvConfig.NUM_POINTS_PER_EDGE,
        max_steps: int = EnvConfig.MAX_STEPS,
        curriculum_enabled: bool = EnvConfig.CURRICULUM_ENABLED,
        curriculum_stage: int = EnvConfig.INITIAL_STAGE,
        use_potential_shaping: bool = EnvConfig.USE_POTENTIAL_SHAPING,
        action_order: str = 'loop_link_node',  
        gnn_budget: Optional[int] = None,      
        verbose: bool = False
    ):
        super().__init__()

        self.edge_length = float(edge_length)
        self.num_points_per_edge = int(num_points_per_edge)
        if self.num_points_per_edge != EnvConfig.NODE_POSITIONS_PER_EDGE:
            raise ValueError(
                'This published configuration uses exactly 10 internal '
                'points per edge, giving 152 actions. '
                f'Received num_points_per_edge={self.num_points_per_edge}.'
            )
        self.max_steps = max_steps
        self.use_potential_shaping = use_potential_shaping
        self.verbose = verbose

        self.action_order = action_order
        if action_order not in ['loop_link_node', 'loop_node_link']:
            raise ValueError(
                f"Invalid action_order: {action_order}. Must be 'loop_link_node' or 'loop_node_link'"
            )

        self.curriculum_enabled = curriculum_enabled
        self.curriculum_stage = curriculum_stage
        self.episodes_in_stage = 0
        self._prev_stage_actions = EnvConfig.STAGE1_ACTIONS
        self._setup_curriculum()

        self.action_space = spaces.Discrete(self.TOTAL_ACTIONS)
        self.curve_dim = 100
        self.extra_dim = EnvConfig.EXTRA_DIM
        state_dim = self.curve_dim + self.extra_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnn_model = self._load_gnn_model(gnn_model_path)

        self.lattice_state: Optional[LatticeState] = None
        self.current_step = 0
        self.prev_curve = None
        self.prev_performance = 0.0            
        self.prev_performance_real = 0.0        
        self.best_performance = 0.0             
        self.best_performance_real = 0.0
        self.best_state_in_episode: Optional[LatticeState] = None
        self.best_ever = 0.0                    
        self.best_ever_real = 0.0
        self.performance_history = deque(maxlen=50)

        self.visited_states = set()
        self.episode_visited_states = set()

        self._cache: Dict[int, Tuple[np.ndarray, float, float]] = {}
        self.gnn_forward_calls = 0
        self.gnn_cache_hits = 0
        self.gnn_cache_misses = 0
        self.gnn_budget = gnn_budget

        if self.verbose:
            print(f"[LatticeEnv] action_order={self.action_order}, stage={self.curriculum_stage}, "
                  f"actions={self.available_actions}, num_points_per_edge={self.num_points_per_edge}")
            if self.gnn_budget is not None:
                print(f"[LatticeEnv] GNN forward budget enabled: {self.gnn_budget}")

    def _setup_curriculum(self):
        if self.action_order == 'loop_link_node':
            stage2_actions = EnvConfig.STAGE2_ACTIONS_LOOP_LINK_NODE  
        else:
            stage2_actions = EnvConfig.STAGE2_ACTIONS_LOOP_NODE_LINK  

        if self.curriculum_enabled:
            if self.curriculum_stage == 1:
                self.available_action_types = ['loop']
                self.effective_max_steps = EnvConfig.STAGE1_MAX_STEPS
                self.available_actions = EnvConfig.STAGE1_ACTIONS
                self._target_actions = EnvConfig.STAGE1_ACTIONS

            elif self.curriculum_stage == 2:
                self.available_action_types = ['loop', 'link'] if self.action_order == 'loop_link_node' else ['loop', 'node']
                self.effective_max_steps = self._calculate_gradual_steps(
                    EnvConfig.STAGE2_BASE_STEPS, EnvConfig.STAGE2_MAX_STEPS
                )
                self._target_actions = stage2_actions
                self.available_actions = self._calculate_gradual_actions(self._prev_stage_actions, stage2_actions)

            else:
                self.available_action_types = ['loop', 'link', 'node'] if self.action_order == 'loop_link_node' else ['loop', 'node', 'link']
                self.effective_max_steps = self._calculate_gradual_steps(
                    EnvConfig.STAGE3_BASE_STEPS, EnvConfig.STAGE3_MAX_STEPS
                )
                self._target_actions = EnvConfig.STAGE3_ACTIONS
                self.available_actions = self._calculate_gradual_actions(self._prev_stage_actions, EnvConfig.STAGE3_ACTIONS)
        else:
            self.available_actions = EnvConfig.STAGE3_ACTIONS
            self.effective_max_steps = self.max_steps
            self.available_action_types = ['loop', 'link', 'node'] if self.action_order == 'loop_link_node' else ['loop', 'node', 'link']
            self._target_actions = EnvConfig.STAGE3_ACTIONS

    def _calculate_gradual_steps(self, base_steps: int, max_steps: int) -> int:
        additional_steps = self.episodes_in_stage // EnvConfig.STEPS_INCREASE_INTERVAL
        return min(base_steps + additional_steps, max_steps)

    def _calculate_gradual_actions(self, start_actions: int, target_actions: int) -> int:
        if self.episodes_in_stage >= EnvConfig.ACTION_UNLOCK_EPISODES:
            return target_actions
        progress = self.episodes_in_stage / EnvConfig.ACTION_UNLOCK_EPISODES
        smooth_progress = progress * progress * (3 - 2 * progress)  # smoothstep
        unlocked = int(start_actions + (target_actions - start_actions) * smooth_progress)
        return max(start_actions, min(unlocked, target_actions))

    def set_episodes_in_stage(self, episodes: int):
        old_steps = self.effective_max_steps
        old_actions = self.available_actions
        self.episodes_in_stage = episodes

        stage2_actions = EnvConfig.STAGE2_ACTIONS_LOOP_LINK_NODE if self.action_order == 'loop_link_node' \
            else EnvConfig.STAGE2_ACTIONS_LOOP_NODE_LINK

        if self.curriculum_enabled:
            if self.curriculum_stage == 2:
                self.effective_max_steps = self._calculate_gradual_steps(
                    EnvConfig.STAGE2_BASE_STEPS, EnvConfig.STAGE2_MAX_STEPS
                )
                self.available_actions = self._calculate_gradual_actions(self._prev_stage_actions, stage2_actions)
            elif self.curriculum_stage == 3:
                self.effective_max_steps = self._calculate_gradual_steps(
                    EnvConfig.STAGE3_BASE_STEPS, EnvConfig.STAGE3_MAX_STEPS
                )
                self.available_actions = self._calculate_gradual_actions(self._prev_stage_actions, EnvConfig.STAGE3_ACTIONS)

        if self.verbose:
            if old_steps != self.effective_max_steps:
                print(f"Steps: {old_steps} → {self.effective_max_steps}")
            if old_actions != self.available_actions:
                print(f"Actions: {old_actions} → {self.available_actions}")

    def set_curriculum_stage(self, stage: int):
        old_stage = self.curriculum_stage
        if old_stage != stage:
            self._prev_stage_actions = self.available_actions
            self.episodes_in_stage = 0
        self.curriculum_stage = min(stage, 3)
        self._setup_curriculum()
        if self.verbose and old_stage != self.curriculum_stage:
            print(f"\n[LatticeEnv] Stage {old_stage} → {self.curriculum_stage}")
            print(f"   动作空间: {self._prev_stage_actions} → {self._target_actions} (渐进解锁)")
            print(f"   初始步数: {self.effective_max_steps}")

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.TOTAL_ACTIONS, dtype=bool)
        mask[:self.available_actions] = True
        return mask

    def valid_action_mask(self) -> np.ndarray:
        return self.action_masks()

    def get_available_actions(self) -> list:
        return list(range(self.available_actions))

    def reset_gnn_counter(self, budget: Optional[int] = None, clear_cache: bool = False):
        self.gnn_forward_calls = 0
        self.gnn_cache_hits = 0
        self.gnn_cache_misses = 0
        if budget is not None:
            self.gnn_budget = int(budget)
        if clear_cache:
            self._cache = {}

    def get_gnn_forward_calls(self) -> int:
        return int(self.gnn_forward_calls)

    def gnn_budget_exhausted(self) -> bool:
        return (self.gnn_budget is not None) and (self.gnn_forward_calls >= self.gnn_budget)

    def _load_gnn_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GNN model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        denom_channels = 1
        if 'head_denom.4.bias' in checkpoint:
            denom_channels = checkpoint['head_denom.4.bias'].shape[0]

        model = GINCurveRegressorWithDenom(
            in_channels=4, out_dim=100, denom_channels=denom_channels,
            hidden=256, num_layers=8, dropout=0.2, num_global_features=4,
            use_global_feat=False, pooling_type='dual', use_denom_head=True,
            use_edge_feat=True, decoder_type='cnn', seq_len=100, out_channels_curve=1
        ).to(self.device)

        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def _graph_to_pyg_data_from_state(self, lattice_state: LatticeState):
        G = lattice_state.to_networkx_graph()
        import tempfile
        from networkx.readwrite import json_graph

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_json = f.name

        data = json_graph.node_link_data(G)
        for node in data['nodes']:
            x, y = node['id']
            node['pos'] = [x, y]
            node['id'] = f"{x},{y}"
        for link in data['links']:
            sx, sy = link['source']
            tx, ty = link['target']
            link['source'] = f"{sx},{sy}"
            link['target'] = f"{tx},{ty}"

        with open(temp_json, 'w') as f:
            json.dump(data, f)

        try:
            graph_data = parse_networkx_json(temp_json)
            graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
            graph_data = graph_data.to(self.device)
        finally:
            if os.path.exists(temp_json):
                os.remove(temp_json)

        return graph_data

    def _state_hash_from_state(self, lattice_state: LatticeState) -> int:
        return hash((tuple(lattice_state.center_loop1), tuple(lattice_state.center_loop2)))

    def evaluate_lattice_state(self, lattice_state: LatticeState) -> Tuple[np.ndarray, float, float]:
        state_hash = self._state_hash_from_state(lattice_state)

        if state_hash in self._cache:
            self.gnn_cache_hits += 1
            return self._cache[state_hash]

        self.gnn_cache_misses += 1

        if self.gnn_budget_exhausted():
            return np.zeros(self.curve_dim), 0.0, 0.0

        try:
            graph_data = self._graph_to_pyg_data_from_state(lattice_state)
            with torch.no_grad():
                pred_norm, pred_denom_log = self.gnn_model(graph_data)
                curve = pred_norm[0].cpu().numpy()
                pred_denom = torch.exp(pred_denom_log)
                pred_real = pred_norm * pred_denom[:, :1]
                performance_real = pred_real.max().item()
                performance_normalized = float(np.clip(performance_real / self.MAX_ENERGY, 0, 1))

            self.gnn_forward_calls += 1
            self._cache[state_hash] = (curve, performance_normalized, performance_real)
            return self._cache[state_hash]
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Evaluation failed: {e}")
            return np.zeros(self.curve_dim), 0.0, 0.0

    def _evaluate(self) -> Tuple[np.ndarray, float, float]:
        return self.evaluate_lattice_state(self.lattice_state)

    def _get_state(self) -> np.ndarray:
        curve, perf, _ = self._evaluate()

        extra = []

        extra.extend([
            perf, self.best_performance, self.best_ever,
            perf - self.prev_performance, self.best_ever - perf
        ])

        progress = self.current_step / max(1, self.effective_max_steps)
        action_progress = self.available_actions / self.TOTAL_ACTIONS
        extra.extend([
            progress,
            float(self.curriculum_stage) / 3.0,
            action_progress,
            1.0 if perf > self.best_performance else 0.0
        ])

        hist_list = list(self.performance_history)
        extra.extend([
            float(np.mean(hist_list) if hist_list else perf),
            float(np.std(hist_list) if len(hist_list) > 1 else 0.0)
        ])

        for loop in [self.lattice_state.center_loop1, self.lattice_state.center_loop2]:
            edge_order = [e_idx / 5.0 for e_idx, _ in loop]
            extra.extend(edge_order)

            denom = max(1, (self.num_points_per_edge - 1))
            point_positions = [p_idx / denom for _, p_idx in loop]
            extra.extend([
                float(np.mean(point_positions)),
                float(np.std(point_positions)),
                float(np.min(point_positions)),
                float(np.max(point_positions)),
            ])

        if len(extra) > self.extra_dim:
            raise ValueError(f"extra feature length {len(extra)} exceeds extra_dim={self.extra_dim}. "
                             f"Please increase EnvConfig.EXTRA_DIM.")
        while len(extra) < self.extra_dim:
            extra.append(0.0)

        extra = np.asarray(extra, dtype=np.float32)
        return np.concatenate([curve.astype(np.float32), extra]).astype(np.float32)

    def _decode_action(self, action: int) -> Tuple[str, int, Any]:
        """Decode one of the fixed 152 actions."""
        action = int(action)
        if not 0 <= action < self.TOTAL_ACTIONS:
            return ('invalid', 0, ())

        node_actions_per_loop = EnvConfig.NODE_ACTIONS_PER_LOOP  
        node_positions = EnvConfig.NODE_POSITIONS_PER_EDGE       

        if self.action_order == 'loop_link_node':
            if action < EnvConfig.STAGE1_ACTIONS:
                return ('loop', action + 1, ())

            if action < EnvConfig.STAGE2_ACTIONS_LOOP_LINK_NODE:
                local_action = action - EnvConfig.STAGE1_ACTIONS
                loop_id = (
                    local_action // EnvConfig.LINK_ACTIONS_PER_LOOP
                ) + 1
                pair_idx = (
                    local_action % EnvConfig.LINK_ACTIONS_PER_LOOP
                )
                edge_a, edge_b = self.LINK_EDGE_PAIRS[pair_idx]
                return ('link', loop_id, (edge_a, edge_b))

            local_action = (
                action - EnvConfig.STAGE2_ACTIONS_LOOP_LINK_NODE
            )
            loop_id = (local_action // node_actions_per_loop) + 1
            remainder = local_action % node_actions_per_loop
            side_id = (remainder // node_positions) + 1
            point_idx = remainder % node_positions
            return ('node', loop_id, (side_id, point_idx))

        # loop_node_link
        if action < EnvConfig.STAGE1_ACTIONS:
            return ('loop', action + 1, ())

        if action < EnvConfig.STAGE2_ACTIONS_LOOP_NODE_LINK:
            local_action = action - EnvConfig.STAGE1_ACTIONS
            loop_id = (local_action // node_actions_per_loop) + 1
            remainder = local_action % node_actions_per_loop
            side_id = (remainder // node_positions) + 1
            point_idx = remainder % node_positions
            return ('node', loop_id, (side_id, point_idx))

        local_action = (
            action - EnvConfig.STAGE2_ACTIONS_LOOP_NODE_LINK
        )
        loop_id = (
            local_action // EnvConfig.LINK_ACTIONS_PER_LOOP
        ) + 1
        pair_idx = local_action % EnvConfig.LINK_ACTIONS_PER_LOOP
        edge_a, edge_b = self.LINK_EDGE_PAIRS[pair_idx]
        return ('link', loop_id, (edge_a, edge_b))

    def _is_action_available(self, action: int) -> bool:
        return action < self.available_actions

    def _compute_potential(self, performance: float) -> float:
        return performance * RewardConfig.POTENTIAL_COEF

    def _compute_reward(
        self,
        current_perf: float,
        prev_perf: float,
        prev_best: float,
        action_success: bool,
        is_new_state: bool,
        is_loop_action: bool = False,
        action_unavailable: bool = False
    ) -> Tuple[float, Dict]:
        cfg = RewardConfig
        reward_breakdown: Dict[str, float] = {}
        base_reward = 0.0

        if action_unavailable:
            base_reward = cfg.UNAVAILABLE_ACTION_PENALTY
            reward_breakdown['unavailable_penalty'] = cfg.UNAVAILABLE_ACTION_PENALTY
            reward_breakdown['total'] = float(base_reward)
            return float(base_reward), reward_breakdown

        # ---------- episode-best delta ----------
        current_delta = float(current_perf - prev_perf)
        new_best = float(max(prev_best, current_perf))
        best_delta = float(new_best - prev_best)  # >= 0

        multiplier = cfg.DELTA_MULTIPLIER_LOOP if is_loop_action else cfg.DELTA_MULTIPLIER_OTHER
        delta_reward = best_delta * multiplier
        base_reward += delta_reward

        reward_breakdown['delta'] = float(delta_reward)
        reward_breakdown['delta_best_raw'] = float(best_delta)
        reward_breakdown['delta_current_raw'] = float(current_delta)

        perf_reward = float(current_perf) * cfg.PERF_REWARD_COEF
        base_reward += perf_reward
        reward_breakdown['perf_reward'] = float(perf_reward)

        if self.use_potential_shaping:
            shaping = self._compute_potential(new_best) - self._compute_potential(prev_best)
            base_reward += float(shaping)
            reward_breakdown['shaping'] = float(shaping)

        if is_new_state:
            base_reward += cfg.EXPLORATION_REWARD
            reward_breakdown['exploration'] = float(cfg.EXPLORATION_REWARD)
        else:
            reward_breakdown['exploration'] = 0.0

        if not action_success:
            base_reward += cfg.INVALID_ACTION_PENALTY
            reward_breakdown['invalid_penalty'] = float(cfg.INVALID_ACTION_PENALTY)
        else:
            reward_breakdown['invalid_penalty'] = 0.0

        if is_loop_action and best_delta > cfg.LOOP_BONUS_THRESHOLD:
            loop_bonus = best_delta * cfg.LOOP_BONUS_MULTIPLIER
            base_reward += float(loop_bonus)
            reward_breakdown['loop_bonus'] = float(loop_bonus)
        else:
            reward_breakdown['loop_bonus'] = 0.0

        prev_high = max(0.0, float(prev_best) - cfg.HIGH_PERF_THRESHOLD)
        new_high = max(0.0, float(new_best) - cfg.HIGH_PERF_THRESHOLD)
        high_inc = new_high - prev_high  # >= 0
        if high_inc > 0:
            high_perf = high_inc * cfg.HIGH_PERF_MULTIPLIER
            base_reward += float(high_perf)
            reward_breakdown['high_perf'] = float(high_perf)
        else:
            reward_breakdown['high_perf'] = 0.0

        final_reward = float(np.clip(base_reward, cfg.REWARD_CLIP_MIN, cfg.REWARD_CLIP_MAX))
        reward_breakdown['total'] = float(final_reward)
        return float(final_reward), reward_breakdown

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
            import random
            random.seed(seed)

        self.lattice_state = LatticeState(
            edge_length=self.edge_length,
            num_points_per_edge=self.num_points_per_edge
        )

        self.current_step = 0
        self.prev_curve, self.prev_performance, self.prev_performance_real = self._evaluate()
        self.best_performance = self.prev_performance
        self.best_performance_real = self.prev_performance_real
        self.best_state_in_episode = self.lattice_state.clone()
        self.performance_history.clear()
        self.performance_history.append(self.prev_performance)

        self.episode_visited_states = set()
        initial_hash = self._state_hash_from_state(self.lattice_state)
        self.episode_visited_states.add(initial_hash)
        self.visited_states.add(initial_hash)

        if self.prev_performance > self.best_ever:
            self.best_ever = self.prev_performance
            self.best_ever_real = self.prev_performance_real

        return self._get_state(), {
            'initial_performance': self.prev_performance_real,
            'initial_performance_normalized': self.prev_performance,
            'curriculum_stage': self.curriculum_stage,
            'available_actions': self.available_actions,
            'target_actions': self._target_actions,
            'effective_max_steps': self.effective_max_steps,
            'action_mask': self.action_masks(),
            'gnn_forward_calls': self.get_gnn_forward_calls(),
            'gnn_cache_hits': int(self.gnn_cache_hits),
            'gnn_cache_misses': int(self.gnn_cache_misses),
            'gnn_budget': int(self.gnn_budget) if self.gnn_budget is not None else None
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        prev_perf = self.prev_performance
        prev_best = self.best_performance  

        action_unavailable = not self._is_action_available(action)
        if action_unavailable:
            reward, reward_breakdown = self._compute_reward(
                current_perf=prev_perf,
                prev_perf=prev_perf,
                prev_best=prev_best,
                action_success=False,
                is_new_state=False,
                is_loop_action=False,
                action_unavailable=True
            )
            terminated = False
            truncated = self.current_step >= self.effective_max_steps
            truncated = truncated or self.gnn_budget_exhausted()

            best_loop1 = copy.deepcopy(self.best_state_in_episode.loop1) if self.best_state_in_episode else None
            best_loop2 = copy.deepcopy(self.best_state_in_episode.loop2) if self.best_state_in_episode else None
            final_loop1 = copy.deepcopy(self.lattice_state.loop1) if self.lattice_state else None
            final_loop2 = copy.deepcopy(self.lattice_state.loop2) if self.lattice_state else None

            return self._get_state(), reward, terminated, truncated, {
                'performance': self.prev_performance_real,
                'best_performance': self.best_performance_real,
                'action_success': False,
                'action_unavailable': True,
                'curriculum_stage': self.curriculum_stage,
                'available_actions': self.available_actions,
                'effective_max_steps': self.effective_max_steps,
                'action_mask': self.action_masks(),
                'reward_breakdown': reward_breakdown,
                'gnn_forward_calls': self.get_gnn_forward_calls(),
                'gnn_budget_exhausted': self.gnn_budget_exhausted(),
                'best_loop1': best_loop1,
                'best_loop2': best_loop2,
                'final_loop1': final_loop1,
                'final_loop2': final_loop2,
            }

        action_type, loop_id, params = self._decode_action(action)
        saved_lattice = self.lattice_state.clone()

        if action_type == 'loop':
            success = self.lattice_state.apply_loop_action(loop_id)
        elif action_type == 'node':
            success = self.lattice_state.apply_node_action(loop_id, params[0], params[1])
        elif action_type == 'link':
            success = self.lattice_state.apply_link_action(loop_id, params[0], params[1])
        else:
            success = False

        is_loop_action = (action_type == 'loop')

        if success:
            self.prev_curve, current_perf, current_perf_real = self._evaluate()
            state_hash = self._state_hash_from_state(self.lattice_state)
            is_new_state = state_hash not in self.episode_visited_states
            if is_new_state:
                self.episode_visited_states.add(state_hash)
                self.visited_states.add(state_hash)
        else:
            self.lattice_state = saved_lattice
            current_perf = prev_perf
            current_perf_real = self.prev_performance_real
            is_new_state = False

        reward, reward_breakdown = self._compute_reward(
            current_perf=current_perf,
            prev_perf=prev_perf,
            prev_best=prev_best,
            action_success=success,
            is_new_state=is_new_state,
            is_loop_action=is_loop_action,
            action_unavailable=False
        )

        if current_perf > self.best_performance:
            self.best_performance = current_perf
            self.best_performance_real = current_perf_real
            self.best_state_in_episode = self.lattice_state.clone()

        if current_perf > self.best_ever:
            self.best_ever = current_perf
            self.best_ever_real = current_perf_real

        self.prev_performance = current_perf
        self.prev_performance_real = current_perf_real
        self.performance_history.append(current_perf)

        terminated = False
        truncated = self.current_step >= self.effective_max_steps
        truncated = truncated or self.gnn_budget_exhausted()

        best_loop1 = copy.deepcopy(self.best_state_in_episode.loop1) if self.best_state_in_episode else None
        best_loop2 = copy.deepcopy(self.best_state_in_episode.loop2) if self.best_state_in_episode else None
        final_loop1 = copy.deepcopy(self.lattice_state.loop1) if self.lattice_state else None
        final_loop2 = copy.deepcopy(self.lattice_state.loop2) if self.lattice_state else None

        return self._get_state(), reward, terminated, truncated, {
            'performance': current_perf_real,
            'best_performance': self.best_performance_real,
            'best_ever': self.best_ever_real,
            'performance_normalized': current_perf,
            'best_performance_normalized': self.best_performance,
            'action_success': success,
            'action_unavailable': False,
            'is_new_state': is_new_state,
            'curriculum_stage': self.curriculum_stage,
            'available_actions': self.available_actions,
            'effective_max_steps': self.effective_max_steps,
            'unique_states': len(self.episode_visited_states),
            'action_mask': self.action_masks(),
            'reward_breakdown': reward_breakdown,
            'gnn_forward_calls': self.get_gnn_forward_calls(),
            'gnn_cache_hits': int(self.gnn_cache_hits),
            'gnn_cache_misses': int(self.gnn_cache_misses),
            'gnn_budget_exhausted': self.gnn_budget_exhausted(),
            'best_loop1': best_loop1,
            'best_loop2': best_loop2,
            'final_loop1': final_loop1,
            'final_loop2': final_loop2,
        }

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step {self.current_step}/{self.effective_max_steps}: "
                  f"perf={self.prev_performance_real:.6f}, "
                  f"stage={self.curriculum_stage}, "
                  f"actions={self.available_actions}/{self._target_actions}, "
                  f"gnn_forward={self.gnn_forward_calls}")