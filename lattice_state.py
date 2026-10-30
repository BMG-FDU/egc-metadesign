import random
import copy
import math
import networkx as nx
import numpy as np
from typing import List, Tuple
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hexagonal_generator import TiledHexGraph, HexagonalGraph

def _euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def merge_close_nodes_graph_local(G: nx.Graph, tol: float = 1e-6, ndigits: int = 8) -> nx.Graph:
    nodes = list(G.nodes())
    merged_coords = []
    mapping = {}
    for n in nodes:
        matched = False
        for rep in merged_coords:
            if _euclidean(n, rep) <= tol:
                mapping[n] = rep
                matched = True
                break
        if not matched:
            rep = (round(n[0], ndigits), round(n[1], ndigits))
            merged_coords.append(rep)
            mapping[n] = rep

    G2 = nx.Graph()
    for orig in nodes:
        rep = mapping[orig]
        attrs = G.nodes[orig].copy() if G.nodes[orig] else {}
        if rep not in G2:
            G2.add_node(rep, **attrs)
        else:
            existing = G2.nodes[rep]
            for k, v in attrs.items():
                if k not in existing:
                    existing[k] = v

    for u, v, ed in G.edges(data=True):
        ru = mapping.get(u, (round(u[0], ndigits), round(u[1], ndigits)))
        rv = mapping.get(v, (round(v[0], ndigits), round(v[1], ndigits)))
        if ru == rv:
            continue
        if not G2.has_edge(ru, rv):
            G2.add_edge(ru, rv, **(ed.copy() if ed else {}))

    return G2

def _proj_param_on_segment(a, b, p):
    ax, ay = a; bx, by = b; px, py = p
    vx, vy = bx - ax, by - ay
    denom = vx*vx + vy*vy
    if denom == 0:
        return 0.0
    t = ((px - ax) * vx + (py - ay) * vy) / denom
    return t

def _distance_point_to_segment(a, b, p):
    t = _proj_param_on_segment(a, b, p)
    if t < 0:
        return _euclidean(p, a)
    if t > 1:
        return _euclidean(p, b)
    proj = (a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]))
    return _euclidean(p, proj)

def break_long_edges_with_intermediate_nodes_local(G: nx.Graph, tol: float = 1e-5) -> nx.Graph:
    nodes = list(G.nodes())
    edges = list(G.edges())
    G2 = nx.Graph()
    for n in nodes:
        G2.add_node(n, **(G.nodes[n].copy() if G.nodes[n] else {}))

    for u, v in edges:
        intermediates = []
        for w in nodes:
            if w == u or w == v:
                continue
            dist = _distance_point_to_segment(u, v, w)
            if dist <= tol:
                t = _proj_param_on_segment(u, v, w)
                if 0.0 <= t <= 1.0:
                    intermediates.append((t, w))
        if intermediates:
            intermediates.sort(key=lambda x: x[0])
            seq = [u] + [w for t, w in intermediates] + [v]
            for i in range(len(seq)-1):
                a, b = seq[i], seq[i+1]
                if a != b and not G2.has_edge(a, b):
                    ed = G.get_edge_data(a, b) or G.get_edge_data(u, v) or {}
                    G2.add_edge(a, b, **(ed.copy() if ed else {}))
        else:
            if not G2.has_edge(u, v):
                G2.add_edge(u, v, **(G.get_edge_data(u, v) or {}).copy())

    return G2

class LatticeState:
    def __init__(self, edge_length=1.0, num_points_per_edge=10, ndigits=8,
                 num_cols=5, num_rows=3, num_loops=2,
                 initialize_random=True):
        self.edge_length = edge_length
        self.num_points_per_edge = num_points_per_edge
        self.ndigits = ndigits
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.num_loops = num_loops

        self.tiled_gen = TiledHexGraph(
            num_cols=num_cols,
            num_rows=num_rows,
            edge_length=edge_length,
            num_points_per_edge=num_points_per_edge,
            ndigits=ndigits
        )

        self.hex_template = HexagonalGraph(
            edge_length=edge_length,
            num_points_per_edge=num_points_per_edge,
            num_loops=num_loops,
            ndigits=ndigits
        )

        self.loop1 = []
        self.loop2 = []

        self.center_loop1 = self.loop1
        self.center_loop2 = self.loop2

        if initialize_random:
            self.initialize_random_loops()

    def initialize_random_loops(self):
        self.loop1 = self._generate_random_loop_indices()
        self.loop2 = self._generate_random_loop_indices()
        self.center_loop1 = self.loop1
        self.center_loop2 = self.loop2

    def _generate_random_loop_indices(self) -> List[Tuple[int, int]]:
        edge_order = list(range(6))
        random.shuffle(edge_order)
        loop = []
        for edge_idx in edge_order:
            if self.num_points_per_edge > 0:
                point_idx = random.randint(0, self.num_points_per_edge - 1)
            else:
                point_idx = 0
            loop.append((edge_idx, point_idx))
        return loop

    def apply_node_action(self, loop_id: int, side_id: int, point_idx: int) -> bool:
        loop = self.loop1 if loop_id == 1 else self.loop2
        edge_idx = side_id - 1
        if edge_idx < 0 or edge_idx >= 6:
            return False
        if point_idx < 0 or point_idx >= self.num_points_per_edge:
            return False

        target_pos = None
        for i, (e_idx, _) in enumerate(loop):
            if e_idx == edge_idx:
                target_pos = i
                break
        if target_pos is None:
            return False

        loop[target_pos] = (edge_idx, int(point_idx))
        return True

    def apply_link_action(self, loop_id: int, edge_a: int, edge_b: int = None) -> bool:
        loop = self.loop1 if loop_id == 1 else self.loop2

        if edge_b is None:
            if isinstance(edge_a, (tuple, list)) and len(edge_a) == 2:
                edge_a, edge_b = int(edge_a[0]), int(edge_a[1])
            else:
                return False

        edge_a = int(edge_a)
        edge_b = int(edge_b)

        if edge_a == edge_b:
            return False
        if not (0 <= edge_a < 6 and 0 <= edge_b < 6):
            return False

        pos_a = None
        pos_b = None
        for i, (e_idx, _) in enumerate(loop):
            if e_idx == edge_a:
                pos_a = i
            elif e_idx == edge_b:
                pos_b = i

        if pos_a is None or pos_b is None:
            return False

        loop[pos_a], loop[pos_b] = loop[pos_b], loop[pos_a]
        return True

    def apply_loop_action(self, loop_id: int) -> bool:
        new_loop = self._generate_random_loop_indices()
        if loop_id == 1:
            self.loop1 = new_loop
            self.center_loop1 = self.loop1
        else:
            self.loop2 = new_loop
            self.center_loop2 = self.loop2
        return True

    def to_networkx_graph(self, merge_tol: float = 1e-6, break_tol: float = 1e-5) -> nx.Graph:
        try:
            HexagonalGraph._cached_patterns.clear()
        except Exception:
            pass

        unified_hex = UnifiedHexagonalGraph(
            edge_length=self.edge_length,
            num_points_per_edge=self.num_points_per_edge,
            num_loops=self.num_loops,
            ndigits=self.ndigits,
            loop1_indices=self.loop1,
            loop2_indices=self.loop2
        )
        unified_hex.build_graph()

        G_global = nx.Graph()
        L = self.edge_length
        cell_id = 0
        for i in range(self.num_cols):
            x = i * 1.5 * L
            if ((i + 1) % 2) == 1:
                n_cells = self.num_rows
                y_start = -0.5 * math.sqrt(3) * L
            else:
                n_cells = self.num_rows - 1
                y_start = 0.0

            for j in range(n_cells):
                y = y_start + j * math.sqrt(3) * L

                G_local = unified_hex.graph

                for (node_x, node_y), data in G_local.nodes(data=True):
                    gx = round(node_x + x, self.ndigits)
                    gy = round(node_y + y, self.ndigits)
                    new_pt = (gx, gy)
                    if new_pt not in G_global:
                        G_global.add_node(new_pt, **(data.copy() if data else {}))

                for u, v, edata in G_local.edges(data=True):
                    u2 = (round(u[0] + x, self.ndigits), round(u[1] + y, self.ndigits))
                    v2 = (round(v[0] + x, self.ndigits), round(v[1] + y, self.ndigits))
                    if u2 != v2 and not G_global.has_edge(u2, v2):
                        G_global.add_edge(u2, v2, **(edata.copy() if edata else {}))

                cell_id += 1

        try:
            from hexagonal_generator import merge_close_nodes_graph, break_long_edges_with_intermediate_nodes
            G_global = merge_close_nodes_graph(G_global, tol=merge_tol, ndigits=self.ndigits)
            G_global = break_long_edges_with_intermediate_nodes(G_global, tol=break_tol)
        except Exception as e:
            print(f"[INFO] 使用本地合并/断边实现 (原因: {e})")
            try:
                G_global = merge_close_nodes_graph_local(G_global, tol=merge_tol, ndigits=self.ndigits)
                G_global = break_long_edges_with_intermediate_nodes_local(G_global, tol=break_tol)
            except Exception as e2:
                print(f"[WARN] 本地合并/断边实现失败: {e2}")
                traceback.print_exc()

        return G_global

    def get_state_vector(self) -> List[float]:
        G = self.to_networkx_graph()
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G) if num_nodes > 1 else 0
        edge_lengths = []
        for u, v in G.edges():
            dx, dy = u[0] - v[0], u[1] - v[1]
            edge_lengths.append(math.sqrt(dx**2 + dy**2))
        avg_edge_len = np.mean(edge_lengths) if edge_lengths else 0
        degrees = [d for n, d in G.degree()]
        std_deg = np.std(degrees) if degrees else 0
        loop1_points = [p_idx for e_idx, p_idx in self.loop1]
        loop2_points = [p_idx for e_idx, p_idx in self.loop2]
        loop1_var = np.var(loop1_points) if loop1_points else 0
        loop2_var = np.var(loop2_points) if loop2_points else 0
        return [num_nodes, num_edges, density, avg_edge_len, std_deg, loop1_var, loop2_var]

    def clone(self):
        cloned = LatticeState(
            edge_length=self.edge_length,
            num_points_per_edge=self.num_points_per_edge,
            ndigits=self.ndigits,
            num_cols=self.num_cols,
            num_rows=self.num_rows,
            num_loops=self.num_loops,
            initialize_random=False
        )
        cloned.loop1 = copy.deepcopy(self.loop1)
        cloned.loop2 = copy.deepcopy(self.loop2)
        cloned.center_loop1 = cloned.loop1
        cloned.center_loop2 = cloned.loop2
        return cloned

    def __repr__(self):
        return f"LatticeState(Unified {self.num_cols}x{self.num_rows}, L1={self.loop1}, L2={self.loop2})"


class UnifiedHexagonalGraph(HexagonalGraph):
    def __init__(self, edge_length, num_points_per_edge, num_loops, ndigits,
                 loop1_indices, loop2_indices):
        super().__init__(edge_length, num_points_per_edge, num_loops, ndigits)
        self.loop1_indices = loop1_indices
        self.loop2_indices = loop2_indices

    def build_graph(self):
        self.graph.clear()
        for i, v in enumerate(self.vertices):
            self.graph.add_node(v, label=f"V{i+1}")

        edge_points = []
        for i in range(6):
            p0 = self.vertices[i]
            p1 = self.vertices[(i+1) % 6]
            pts = self._generate_points_on_edge(p0, p1)
            edge_points.append([p0] + pts + [p1])

        path_edges = []

        loop1_nodes = []
        for edge_idx, point_idx in self.loop1_indices:
            if edge_points[edge_idx]:
                internal_pts = edge_points[edge_idx][1:-1]
                if internal_pts and point_idx < len(internal_pts):
                    loop1_nodes.append(internal_pts[point_idx])

        for i in range(len(loop1_nodes) - 1):
            path_edges.append((loop1_nodes[i], loop1_nodes[i+1]))

        loop2_nodes = []
        for edge_idx, point_idx in self.loop2_indices:
            if edge_points[edge_idx]:
                internal_pts = edge_points[edge_idx][1:-1]
                if internal_pts and point_idx < len(internal_pts):
                    loop2_nodes.append(internal_pts[point_idx])

        for i in range(len(loop2_nodes) - 1):
            path_edges.append((loop2_nodes[i], loop2_nodes[i+1]))

        if loop1_nodes and loop2_nodes:
            path_edges.append((loop1_nodes[0], loop2_nodes[-1]))
            path_edges.append((loop2_nodes[0], loop1_nodes[-1]))

        for u, v in path_edges:
            if u not in self.graph:
                self.graph.add_node(u)
            if v not in self.graph:
                self.graph.add_node(v)
            if not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v)
                self.internal_edges.append((u, v))

        self._find_and_add_intersections()
        self._replace_internal_edges_with_intersection_segments()

        for i in range(6):
            potential_points = edge_points[i]
            active_points = [p for p in potential_points if p in self.graph]
            if self.num_loops <= 0:
                active_points = [potential_points[0], potential_points[-1]]
            for k in range(len(active_points) - 1):
                u, v = active_points[k], active_points[k+1]
                if not self.graph.has_edge(u, v):
                    self.graph.add_edge(u, v)

        return self.graph


if __name__ == "__main__":
    print("测试统一单元版 LatticeState（含合并/断边）...")
    print("="*60)

    state = LatticeState()
    print(f"初始状态: {state}")
    print(f"Loop1: {state.loop1}")
    print(f"Loop2: {state.loop2}")

    G = state.to_networkx_graph()
    print(f"\n生成的图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    comps = list(nx.connected_components(G))
    print(f"连通分量数量: {len(comps)}; 各分量大小: {[len(c) for c in comps]}")

    print("\n测试动作...")

    success = state.apply_node_action(1, 1, 7)
    print(f"NODE动作 (loop1, side1, point7): {'成功' if success else '失败'}")

    success = state.apply_link_action(1, 0, 3)
    print(f"LINK动作 (loop1, edge0, edge3): {'成功' if success else '失败'}")

    old_loop1 = state.loop1.copy()
    success = state.apply_loop_action(1)
    print(f"LOOP动作 (loop1): {'成功' if success else '失败'}")
    print(f"  Loop1变化: {old_loop1} -> {state.loop1}")

    cloned = state.clone()
    print(f"\n克隆状态: {cloned}")

    state.apply_loop_action(2)
    print(f"原状态Loop2: {state.loop2}")
    print(f"克隆状态Loop2: {cloned.loop2}")
    print(f"克隆独立性: {'正确' if state.loop2 != cloned.loop2 else '错误'}")

    try:
        import matplotlib.pyplot as plt

        G = state.to_networkx_graph()
        pos = {n: n for n in G.nodes()}

        plt.figure(figsize=(12, 8))
        nx.draw_networkx_edges(G, pos, alpha=0.7, width=0.8)
        nx.draw_networkx_nodes(G, pos, node_size=15, node_color='steelblue')
        plt.axis('equal')
        plt.title('Unified Lattice (All cells use same topology) - merged & broken')
        plt.savefig('unified_lattice_test.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✅ 可视化保存: unified_lattice_test.png")
    except Exception as e:
        print(f"\n⚠️ 可视化跳过: {e}")

    print("\n" + "="*60)
    print("✅ 统一单元版 LatticeState 测试完成!")
