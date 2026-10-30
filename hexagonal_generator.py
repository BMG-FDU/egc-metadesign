import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
import math
from itertools import combinations
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph


def node_link_data(graph):
    try:
        return json_graph.node_link_data(graph, edges="links")
    except TypeError:
        return json_graph.node_link_data(graph, link="links")


def _round_pt(p, ndigits=8):
    return (round(float(p[0]), ndigits), round(float(p[1]), ndigits))

def dist_sq(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def point_on_segment(p, a, b, tol=1e-5):


    if not (min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol and
            min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol):
        return False

    cross_prod = (p[1] - a[1]) * (b[0] - a[0]) - (p[0] - a[0]) * (b[1] - a[1])
    if abs(cross_prod) > tol:
        return False
    return True

def merge_close_nodes_graph(G, tol=1e-6, ndigits=8):
    nodes = list(G.nodes())
    n = len(nodes)
    rep = {}


    sorted_nodes = sorted(nodes, key=lambda x: (x[0], x[1]))

    for i in range(n):
        curr = sorted_nodes[i]
        if curr in rep: continue
        rep[curr] = curr


        for j in range(i+1, n):
            target = sorted_nodes[j]
            if target[0] - curr[0] > tol: break

            if target in rep: continue

            if np.linalg.norm(np.array(curr) - np.array(target)) < tol:
                rep[target] = curr

    newpos = {}
    for orig, r in rep.items():
        px, py = float(r[0]), float(r[1])
        newpos[orig] = (round(px, ndigits), round(py, ndigits))

    newG = nx.Graph()
    for orig, data in G.nodes(data=True):
        coord = newpos[orig]
        if coord not in newG:
            newG.add_node(coord, **data)
        else:

            for k, v in data.items():
                if k not in newG.nodes[coord]:
                    newG.nodes[coord][k] = v

    for u, v, ed in G.edges(data=True):
        u2 = newpos[u]
        v2 = newpos[v]
        if u2 != v2:
            if not newG.has_edge(u2, v2):
                newG.add_edge(u2, v2, **ed)

    return newG

def break_long_edges_with_intermediate_nodes(G, tol=1e-5):


    nodes = list(G.nodes())
    edges = list(G.edges(data=True))

    edges_to_remove = []
    edges_to_add = []


    for u, v, data in edges:

        points_on_edge = []
        for n in nodes:
            if n == u or n == v: continue
            if point_on_segment(n, u, v, tol):
                points_on_edge.append(n)

        if points_on_edge:

            edges_to_remove.append((u, v))


            points_on_edge.sort(key=lambda p: (p[0]-u[0])**2 + (p[1]-u[1])**2)


            chain = [u] + points_on_edge + [v]
            for i in range(len(chain) - 1):
                p_start = chain[i]
                p_end = chain[i+1]

                edges_to_add.append((p_start, p_end, data))


    G.remove_edges_from(edges_to_remove)
    for u, v, data in edges_to_add:

        if not G.has_edge(u, v):
            G.add_edge(u, v, **data)

    return G


def merge_nodes(polygons, tol=1e-9, ndigits=8):
    nodes = []
    index_map = {}
    polys_idx = []
    if tol <= 0: prec = 12
    else: prec = max(0, int(abs(math.log10(tol)))) if tol < 1 else 0

    for poly in polygons:
        poly_idx = []
        for pt in poly:
            x, y = float(pt[0]), float(pt[1])
            key = (round(x, prec), round(y, prec))
            if key in index_map:
                poly_idx.append(index_map[key])
            else:
                idx = len(nodes)
                nodes.append([x, y])
                index_map[key] = idx
                poly_idx.append(idx)
        polys_idx.append(poly_idx)
    return nodes, polys_idx

def build_links_from_polygons(polygons_idx):
    pairs = set()
    for poly in polygons_idx:
        n = len(poly)
        if n < 2: continue
        for k in range(n):
            a, b = poly[k], poly[(k+1) % n]
            pairs.add((min(a,b)+1, max(a,b)+1))
    return pairs

def process_json_to_inp(json_path, output_dir, target_length=100.0):
    with open(json_path, 'r') as f:
        graph = json.load(f)

    nodes_coords = []
    links_pairs = set()
    original_id_to_idx = {}

    if 'nodes' in graph and isinstance(graph['nodes'], list):
        idx = 1
        for nd in graph['nodes']:
            pos = nd.get('pos') or nd.get('id')
            if isinstance(pos, str):
                try: pos = [float(v) for v in pos.split(',')]
                except: pos = None
            if isinstance(nd.get('id'), list): pos = nd['id']

            if pos:
                nodes_coords.append([float(pos[0]), float(pos[1])])
                nid = nd.get('id')
                key = str(nid) if nid is not None else str(idx)
                original_id_to_idx[key] = idx
                idx += 1

    if 'links' in graph:
        for lk in graph['links']:
            s, t = lk.get('source'), lk.get('target')
            s_idx = original_id_to_idx.get(str(s))
            t_idx = original_id_to_idx.get(str(t))
            if s_idx and t_idx:
                links_pairs.add((min(s_idx,t_idx), max(s_idx,t_idx)))

    if not nodes_coords:
        print(f"[WARN] No nodes found for INP conversion in {json_path}")
        return

    ys = [p[1] for p in nodes_coords]
    xs = [p[0] for p in nodes_coords]
    height = max(ys) - min(ys)
    scale = float(target_length) / height if height > 0 else 1.0

    out_nodes = []
    for i, (x, y) in enumerate(nodes_coords, 1):
        out_nodes.append((i, (x - min(xs))*scale, (y - min(ys))*scale, 0.0))

    base = os.path.splitext(os.path.basename(json_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    inp_path = os.path.join(output_dir, f"{base}.inp")

    with open(inp_path, 'w') as f:
        f.write("*Heading\n")
        f.write(f"** Converted from JSON: {os.path.basename(json_path)}\n")
        f.write("*Node\n")
        for nid, x, y, z in out_nodes:
            f.write(f" {nid}, {x:.6f}, {y:.6f}, {z:.6f}\n")
        f.write("*Element, type=B31\n")
        for eid, (n1, n2) in enumerate(sorted(links_pairs), 1):
            f.write(f" {eid}, {n1}, {n2}\n")
        f.write("** End of input\n")
    print(f"[INFO] INP created: {inp_path}")
    return inp_path


class HexagonalGraph:
    _cached_patterns = {}

    def __init__(self, edge_length, num_points_per_edge, num_loops=2, ndigits=8, pattern_name=None):
        self.edge_length = float(edge_length)
        self.num_points_per_edge = int(num_points_per_edge)
        self.num_loops = int(num_loops)
        self.graph = nx.Graph()
        self.ndigits = int(ndigits)
        self.vertices = self._generate_hexagon_vertices()
        self.internal_edges = []
        self.min_distance_tol = 1e-6
        self.pattern_name = pattern_name or "single"

    def _generate_hexagon_vertices(self):
        angle_step = 360.0 / 6.0
        verts = []
        for i in range(6):
            theta = math.radians(angle_step * i)
            x = self.edge_length * math.cos(theta)
            y = self.edge_length * math.sin(theta)
            verts.append(_round_pt((x, y), self.ndigits))
        return verts

    def _generate_points_on_edge(self, p0, p1):
        pts = []
        n = self.num_points_per_edge
        for i in range(1, n + 1):
            t = i / (n + 1)
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            pts.append(_round_pt((x, y), self.ndigits))
        return pts

    @staticmethod
    def _line_intersection(line1, line2):
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if abs(denom) < 1e-14: return None
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (_round_pt((x, y)))
        return None

    def _point_on_line_segment(self, point, a, b, tol=None):
        if tol is None: tol = self.min_distance_tol
        return point_on_segment(point, a, b, tol)

    def _find_and_add_intersections(self):
        intersections = set()
        for (e1, e2) in combinations(self.internal_edges, 2):
            pt = self._line_intersection(e1, e2)
            if pt is not None:
                too_close = False
                for existing in self.graph.nodes():
                    if np.linalg.norm(np.array(existing) - np.array(pt)) < self.min_distance_tol:
                        too_close = True; break
                if not too_close: intersections.add(pt)
        for pt in intersections:
            if pt not in self.graph:
                self.graph.add_node(pt, label=f"I_{pt[0]}_{pt[1]}")

    def _replace_internal_edges_with_intersection_segments(self):
        intersection_nodes = [n for n, attr in self.graph.nodes(data=True) if str(attr.get('label','')).startswith('I_')]
        edges_to_remove = []
        edges_to_add = []
        for (u, v) in list(self.internal_edges):
            pts_on = [u, v]
            for inter in intersection_nodes:
                if self._point_on_line_segment(inter, u, v):
                    pts_on.append(inter)
            pts_on = list({p for p in pts_on})
            pts_on.sort(key=lambda p: np.linalg.norm(np.array(p) - np.array(u)))
            edges_to_remove.append((u, v))
            for i in range(len(pts_on)-1):
                p1, p2 = pts_on[i], pts_on[i+1]
                if np.linalg.norm(np.array(p1)-np.array(p2)) > self.min_distance_tol:
                    edges_to_add.append((p1, p2))
        for u, v in edges_to_remove:
            if self.graph.has_edge(u, v): self.graph.remove_edge(u, v)
        for u, v in edges_to_add:
            if not self.graph.has_edge(u, v): self.graph.add_edge(u, v)

    def build_graph(self):
        key = self.pattern_name or "single"
        cached = HexagonalGraph._cached_patterns.get(key)
        if cached is not None:
            nodes_list, edges_list, cached_internal_edges = cached
            self.graph.clear()
            for (x, y), data in nodes_list:
                self.graph.add_node((x, y), **(data.copy() if data else {}))
            for u, v, ed in edges_list:
                self.graph.add_edge(u, v, **(ed.copy() if ed else {}))
            self.internal_edges = list(cached_internal_edges)
            return self.graph

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
        existing_edges_set = set()

        def generate_loop_sequence():


            edge_order = list(range(6))
            random.shuffle(edge_order)
            sequence_nodes = []
            for edge_idx in edge_order:
                pts_on_edge = edge_points[edge_idx]
                internal_pts = pts_on_edge[1:-1]
                if internal_pts:
                    chosen = random.choice(internal_pts)
                else:

                    chosen = random.choice([pts_on_edge[0], pts_on_edge[-1]])
                sequence_nodes.append(chosen)
            return sequence_nodes

        if self.num_loops <= 0:
            pass
        elif self.num_loops == 2:
            loop1 = generate_loop_sequence()
            loop2 = generate_loop_sequence()
            for i in range(len(loop1)-1): path_edges.append((loop1[i], loop1[i+1]))
            for i in range(len(loop2)-1): path_edges.append((loop2[i], loop2[i+1]))
            path_edges.append((loop1[0], loop2[-1]))
            path_edges.append((loop2[0], loop1[-1]))
        else:

            for _ in range(self.num_loops):
                lp = generate_loop_sequence()
                for i in range(len(lp)-1): path_edges.append((lp[i], lp[i+1]))
                path_edges.append((lp[-1], lp[0]))

        for u, v in path_edges:
            if np.linalg.norm(np.array(u)-np.array(v)) < self.min_distance_tol: continue
            if u not in self.graph: self.graph.add_node(u)
            if v not in self.graph: self.graph.add_node(v)
            edge_key = tuple(sorted((u, v)))
            if edge_key not in existing_edges_set:
                if not self.graph.has_edge(u, v):
                    self.graph.add_edge(u, v)
                    self.internal_edges.append((u, v))
                    existing_edges_set.add(edge_key)

        self._find_and_add_intersections()
        self._replace_internal_edges_with_intersection_segments()


        for i in range(6):
            potential_points = edge_points[i]
            active_points = [p for p in potential_points if p in self.graph]
            if self.num_loops <= 0:
                active_points = [potential_points[0], potential_points[-1]]
            for k in range(len(active_points) - 1):
                u, v = active_points[k], active_points[k+1]
                if not self.graph.has_edge(u, v): self.graph.add_edge(u, v)

        nodes_list = [(n, d.copy()) for n, d in self.graph.nodes(data=True)]
        edges_list = [(u, v, ed.copy()) for u, v, ed in self.graph.edges(data=True)]
        HexagonalGraph._cached_patterns[key] = (nodes_list, edges_list, list(self.internal_edges))
        return self.graph


class TiledHexGraph:
    def __init__(self, num_cols, num_rows, edge_length, num_points_per_edge, desired_base_pairs=0, ndigits=8):
        self.num_cols = int(num_cols)
        self.num_rows = int(num_rows)
        self.edge_length = float(edge_length)
        self.num_points_per_edge = int(num_points_per_edge)
        self.ndigits = int(ndigits)
        self.global_graph = nx.Graph()
        self.min_distance_tol = 1e-6
        self.centers = []

    def _compute_centers_single(self):
        L = self.edge_length
        centers = []
        for i in range(self.num_cols):
            x = i * 1.5 * L
            if ((i + 1) % 2) == 1:
                for j in range(self.num_rows):
                    y = (j * math.sqrt(3) * L) - 0.5 * math.sqrt(3) * L
                    centers.append((x, y, 'single'))
            else:
                for j in range(self.num_rows - 1):
                    y = j * math.sqrt(3) * L
                    centers.append((x, y, 'single'))
        return centers

    def _compute_centers_mixed(self):
        L = self.edge_length
        centers = []
        for i in range(self.num_cols):
            x = i * 1.5 * L
            if (i % 2) == 1:
                for j in range(self.num_rows):
                    y = (j * math.sqrt(3) * L) - 0.5 * math.sqrt(3) * L
                    t = ['A', 'B', 'B'][j % 3]
                    centers.append((x, y, t))
            else:
                for j in range(self.num_rows - 1):
                    y = j * math.sqrt(3) * L
                    t = ['B', 'A', 'B'][j % 3]
                    centers.append((x, y, t))
        return centers

    def _compute_centers_row_alternating(self):
        L = self.edge_length
        centers = []
        for i in range(self.num_cols):
            x = i * 1.5 * L
            if (i % 2) == 1:
                n_cells = self.num_rows
                y_start = -0.5 * math.sqrt(3) * L
            else:
                n_cells = self.num_rows - 1
                y_start = 0.0
            for j in range(n_cells):
                y = y_start + j * math.sqrt(3) * L
                typ = 'A' if (j % 2) == 0 else 'B'
                centers.append((x, y, typ))
        return centers

    def _compute_centers_col_alternate(self):
        L = self.edge_length
        centers = []
        for i in range(self.num_cols):
            x = i * 1.5 * L
            is_odd_col = (i % 2) == 1
            typ = 'A' if is_odd_col else 'B'
            if is_odd_col:
                for j in range(self.num_rows):
                    y = (j * math.sqrt(3) * L) - 0.5 * math.sqrt(3) * L
                    centers.append((x, y, typ))
            else:
                for j in range(self.num_rows - 1):
                    y = j * math.sqrt(3) * L
                    centers.append((x, y, typ))
        return centers

    def build(self, mode='single', pattern_a_params=None, pattern_b_params=None):
        if mode in ('cell_alternating', 'alternating'): mode = 'row_alternating'

        if mode == 'mixed': centers = self._compute_centers_mixed()
        elif mode == 'row_alternating': centers = self._compute_centers_row_alternating()
        elif mode == 'col_alternating': centers = self._compute_centers_col_alternate()
        else: centers = self._compute_centers_single()

        self.centers = centers[:]
        self.global_graph.clear()


        patterns = {}
        if mode in ('mixed', 'row_alternating', 'col_alternating'):
            patterns['A'] = HexagonalGraph(**pattern_a_params, ndigits=self.ndigits, pattern_name='A')
            patterns['B'] = HexagonalGraph(**pattern_b_params, ndigits=self.ndigits, pattern_name='B')
            patterns['A'].build_graph()
            patterns['B'].build_graph()
        else:
            default_loops = pattern_a_params.get('num_loops', 2) if pattern_a_params else 2
            patterns['single'] = HexagonalGraph(self.edge_length, self.num_points_per_edge, num_loops=default_loops, ndigits=self.ndigits, pattern_name='single')
            patterns['single'].build_graph()

        internal_count = 0
        if 'single' in patterns:
            cached = HexagonalGraph._cached_patterns.get('single')
            if cached: internal_count = len(cached[2])
        elif 'A' in patterns:
            cached = HexagonalGraph._cached_patterns.get('A')
            if cached: internal_count = len(cached[2])


        for (cx, cy, typ) in centers:
            G_local = patterns[typ].graph
            for (x, y), data in G_local.nodes(data=True):
                gx, gy = round(x + cx, self.ndigits), round(y + cy, self.ndigits)
                new_pt = (gx, gy)
                if new_pt not in self.global_graph:
                    attrs = data.copy() if data else {}
                    attrs['cell_type'] = typ
                    self.global_graph.add_node(new_pt, **attrs)
                else:
                    if data:
                        for k, v in data.items():
                            if k not in self.global_graph.nodes[new_pt]:
                                self.global_graph.nodes[new_pt][k] = v

            for u, v in G_local.edges():
                u2 = (round(u[0]+cx, self.ndigits), round(u[1]+cy, self.ndigits))
                v2 = (round(v[0]+cx, self.ndigits), round(v[1]+cy, self.ndigits))
                if u2 == v2: continue
                if not self.global_graph.has_edge(u2, v2):
                    self.global_graph.add_edge(u2, v2)


        self.global_graph = merge_close_nodes_graph(self.global_graph, tol=self.min_distance_tol, ndigits=self.ndigits)


        self.global_graph = break_long_edges_with_intermediate_nodes(self.global_graph, tol=1e-5)

        return self.global_graph, internal_count

    def visualize(self, figsize=(8,8), node_size=12, show=False, save_path=None, title_suffix=""):
        if not self.global_graph.nodes(): return
        pos = {n: n for n in self.global_graph.nodes()}
        colors = []
        for n in self.global_graph.nodes():
            ct = self.global_graph.nodes[n].get('cell_type')
            colors.append('tab:red' if ct=='A' else ('tab:blue' if ct=='B' else '0.5'))

        plt.figure(figsize=figsize)
        if isinstance(self.global_graph, nx.MultiGraph):
            base = nx.Graph()
            base.add_nodes_from(self.global_graph.nodes(data=True))
            base.add_edges_from((u, v) for u, v, _k, d in self.global_graph.edges(keys=True, data=True) if not d.get("auxiliary", False))
            aux = nx.Graph()
            aux.add_edges_from((u, v) for u, v, _k, d in self.global_graph.edges(keys=True, data=True) if d.get("auxiliary", False))
            nx.draw_networkx_edges(base, pos, alpha=0.65, width=0.65, edge_color="0.2")
            if aux.number_of_edges():
                nx.draw_networkx_edges(aux, pos, alpha=0.95, width=1.0, edge_color="tab:orange", style="dashed")
        else:
            nx.draw_networkx_edges(self.global_graph, pos, alpha=0.6, width=0.6)
        xs, ys = zip(*pos.keys())
        plt.scatter(xs, ys, s=node_size, c=colors, edgecolors='k', linewidths=0.2, zorder=3)
        plt.axis('equal')
        plt.title(f"Tiled hex graph {title_suffix}".strip())
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"[INFO] Plot saved: {save_path}")
        if show: plt.show()
        plt.close()

    def save_to_json(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data = node_link_data(self.global_graph)
        for node in data['nodes']:
            x, y = node['id']
            node['pos'] = [x, y]
            node['id'] = f"{x},{y}"
        for link in data['links']:
            sx, sy = link['source']
            tx, ty = link['target']
            link['source'] = f"{sx},{sy}"
            link['target'] = f"{tx},{ty}"
        with open(filename, 'w') as f: json.dump(data, f, indent=2)


def _edge_key(u, v):
    return tuple(sorted((u, v)))


def _point_on_closed_segment(point, a, b, tol=1e-6):
    ax, ay = a; bx, by = b; px, py = point
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length <= tol:
        return math.dist(point, a) <= tol
    t = ((px - ax) * dx + (py - ay) * dy) / (length * length)
    cross = abs((px - ax) * dy - (py - ay) * dx) / length
    return -tol <= t <= 1.0 + tol and cross <= tol


def _copy_to_multigraph(physical_graph):
    route_graph = nx.MultiGraph()
    route_graph.graph.update(copy.deepcopy(physical_graph.graph))
    route_graph.add_nodes_from((n, copy.deepcopy(d)) for n, d in physical_graph.nodes(data=True))
    for u, v, data in physical_graph.edges(data=True):
        route_graph.add_edge(u, v, **copy.deepcopy(data), auxiliary=False)
    return route_graph


def _add_cpp_auxiliary_edges(physical_graph, route_graph):
    odd_nodes = sorted((n for n, degree in physical_graph.degree() if degree % 2), key=lambda p: (p[0], p[1]))
    if not odd_nodes:
        return [], 0
    metric = lambda u, v, data: math.dist(u, v)
    complete = nx.Graph()
    complete.add_nodes_from(odd_nodes)
    paths = {}
    for i, source in enumerate(odd_nodes):
        distances, source_paths = nx.single_source_dijkstra(physical_graph, source, weight=metric)
        paths[source] = source_paths
        for target in odd_nodes[i + 1:]:
            if target not in distances:
                raise ValueError("The physical lattice is disconnected; a single closed Euler route cannot cover it.")
            complete.add_edge(source, target, weight=float(distances[target]))
    matching = nx.min_weight_matching(complete, weight="weight")
    if len(matching) * 2 != len(odd_nodes):
        raise RuntimeError("Could not pair every odd-degree vertex.")
    pairs = sorted((_edge_key(u, v) for u, v in matching), key=lambda pair: pair)
    added = 0
    for pair_id, (u, v) in enumerate(pairs):
        source, target = (u, v) if u in paths else (v, u)
        path = paths[source][target]
        for step, (a, b) in enumerate(zip(path, path[1:])):
            route_graph.add_edge(a, b, **copy.deepcopy(physical_graph.edges[a, b]),
                                 auxiliary=True, augmentation_pair_id=pair_id,
                                 augmentation_step=step)
            added += 1
    return pairs, added


def _template_matches_odd_set(physical_graph, template_path, edge_length, tol=1e-6):
    data = json.loads(Path(template_path).read_text(encoding="utf-8"))
    expected = {tuple(round(float(x) * edge_length, 8) for x in point)
                for segment in data["segments"] for point in segment}
    actual = {tuple(round(float(x), 8) for x in node)
              for node, degree in physical_graph.degree() if degree % 2}
    return len(expected) == len(actual) and all(
        any(math.dist(point, candidate) <= tol for candidate in actual)
        for point in expected
    )


def _add_boundary_template(physical_graph, route_graph, template_path, edge_length):
    data = json.loads(Path(template_path).read_text(encoding="utf-8"))
    scale = float(edge_length)
    segments = [([(float(a[0]) * scale, float(a[1]) * scale),
                  (float(b[0]) * scale, float(b[1]) * scale)])
                for a, b in data["segments"]]
    added = 0
    for u, v, edge_data in physical_graph.edges(data=True):
        for segment_id, (a, b) in enumerate(segments):
            if _point_on_closed_segment(u, a, b) and _point_on_closed_segment(v, a, b):
                route_graph.add_edge(u, v, **copy.deepcopy(edge_data), auxiliary=True,
                                     augmentation_template_segment=segment_id)
                added += 1
                break
    return added


def eulerize_graph(physical_graph, method="auto", *, mode="single", num_cols=5,
                   num_rows=3, edge_length=1.0, template_path=None):


    if not physical_graph or physical_graph.number_of_edges() == 0:
        raise ValueError("Cannot create an Eulerian route from an empty lattice.")
    if not nx.is_connected(physical_graph):
        raise ValueError("The physical lattice is disconnected; one closed Euler route is not possible.")
    eligible_template = (mode == "single" and int(num_cols) == 5 and int(num_rows) == 3)
    if template_path is None and method in ("auto", "template"):
        template_path = Path(__file__).resolve().with_name("boundary_template_5x3.json")
    template_matches = bool(method in ("auto", "template") and template_path and Path(template_path).is_file() and
                             _template_matches_odd_set(physical_graph, template_path, edge_length))
    chosen = method
    if chosen == "auto":
        chosen = "template" if eligible_template and template_matches else (
            "cpp_fallback_template_odd_set_mismatch" if eligible_template else "cpp")
    if chosen == "template" and not eligible_template:
        raise ValueError("The fixed template is validated only for single mode with 5 columns x 3 rows.")
    if chosen not in ("template", "cpp", "cpp_fallback_template_odd_set_mismatch"):
        raise ValueError(f"Unknown Eulerization method: {method}")
    if chosen == "cpp_fallback_template_odd_set_mismatch":
        chosen_for_graph = chosen
        algorithm = "cpp"
    else:
        chosen_for_graph = chosen
        algorithm = chosen

    route_graph = _copy_to_multigraph(physical_graph)
    odd_before = sum(degree % 2 for _, degree in physical_graph.degree())
    pair_data = []
    if algorithm == "template":
        if not template_path or not Path(template_path).is_file():
            raise FileNotFoundError(f"Boundary template not found: {template_path}")
        added = _add_boundary_template(physical_graph, route_graph, template_path, edge_length)
        if not nx.is_eulerian(route_graph):
            if method == "auto":
                route_graph = _copy_to_multigraph(physical_graph)
                pair_data, added = _add_cpp_auxiliary_edges(physical_graph, route_graph)
                chosen_for_graph = "cpp_fallback"
            else:
                raise ValueError("The fixed template did not Eulerize this generated graph; no output was written.")
    else:
        pair_data, added = _add_cpp_auxiliary_edges(physical_graph, route_graph)
    if not nx.is_eulerian(route_graph):
        raise RuntimeError("Eulerization verification failed; no output was written.")
    route_graph.graph.update({
        "purpose": "path",
        "eulerization_method": chosen_for_graph,
        "odd_vertices_before_augmentation": int(odd_before),
        "auxiliary_edge_instances": int(added),
        "paired_odd_vertices": [[list(u), list(v)] for u, v in pair_data],
        "is_eulerian": True,
    })
    return route_graph


def run_self_test(template_path):
    random.seed(1); np.random.seed(1)
    HexagonalGraph._cached_patterns = {}
    tiled = TiledHexGraph(5, 3, 1.0, 10, ndigits=8)
    physical, _ = tiled.build(mode="single", pattern_a_params={
        "edge_length": 1.0, "num_points_per_edge": 10, "num_loops": 2})
    route = eulerize_graph(physical, "auto", mode="single", num_cols=5,
                           num_rows=3, edge_length=1.0, template_path=template_path)
    assert isinstance(physical, nx.Graph) and not isinstance(physical, nx.MultiGraph)
    assert isinstance(route, nx.MultiGraph) and nx.is_eulerian(route)
    assert set(physical.nodes) == set(route.nodes)
    assert {_edge_key(u, v) for u, v in physical.edges} == {_edge_key(u, v) for u, v in route.edges(keys=False)}
    assert all(route.number_of_edges(u, v) >= 1 for u, v in physical.edges)
    print(f"Self-test passed: physical edges={physical.number_of_edges()}, route instances={route.number_of_edges()}, "
          f"auxiliary={route.graph['auxiliary_edge_instances']}, method={route.graph['eulerization_method']}")


def _load_configuration(config_path):
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate the same hexagonal lattice in physical or Eulerian path mode.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--boundary-template", type=Path)
    parser.add_argument("--purpose", choices=("mechanics", "path"), help="mechanics=single physical edges; path=Eulerian multigraph.")
    parser.add_argument("--eulerization", choices=("auto", "template", "cpp"), help="Path mode only. auto uses the validated 5x3 template when possible, otherwise weighted CPP.")
    parser.add_argument("--cols", type=int, help="Number of array columns.")
    parser.add_argument("--rows", type=int, help="Number of array rows.")
    parser.add_argument("--variants", type=int, help="Number of independent variants to generate.")
    parser.add_argument("--edge-length", type=float, help="Unit-cell edge length in model coordinates.")
    parser.add_argument("--points-per-edge", type=int, help="Candidate points per unit-cell edge.")
    parser.add_argument("--num-loops", type=int, help="Number of internal loops in each unit cell.")
    parser.add_argument("--target-length", type=float, help="Height used for optional INP scaling (mm).")
    parser.add_argument("--mode", choices=("single", "mixed", "row_alternating", "col_alternating", "random"), help="Array cell-pattern selection.")
    parser.add_argument("--seed", type=int, help="Base random seed; variant i uses seed+i.")
    parser.add_argument("--output-dir", help="Output directory for graph JSON and plots.")
    parser.add_argument("--no-visualize", action="store_true", help="Do not save PNG previews.")
    parser.add_argument("--show", action="store_true", help="Display previews interactively.")
    parser.add_argument("--convert-to-inp", action="store_true", help="Write INP for mechanics mode (overrides config).")
    parser.add_argument("--no-inp", action="store_true", help="Disable INP conversion.")
    parser.add_argument("--self-test", action="store_true", help="Run a small topology/Euler verification and exit.")
    args = parser.parse_args(argv)
    cfg = {
        "purpose": "mechanics", "eulerization": "auto",
        "num_cols": 5, "num_rows": 3, "num_variants": 1,
        "edge_length": 1.0, "num_points_per_edge": 10, "num_loops": 2,
        "target_length_mm": 100.0, "selection_mode": "single", "seed": 1,
        "output_dir": "graph", "visualize": True, "show": False,
        "convert_to_inp": False, "figsize": [10, 10], "node_size": 12,
    }
    if args.config is not None:
        cfg.update(_load_configuration(args.config))
    def setting(cli_value, key):
        return cli_value if cli_value is not None else cfg.get(key)

    purpose = setting(args.purpose, "purpose")
    eulerization = setting(args.eulerization, "eulerization")
    num_cols = int(setting(args.cols, "num_cols")); num_rows = int(setting(args.rows, "num_rows"))
    num_variants = int(setting(args.variants, "num_variants")); edge_length = float(setting(args.edge_length, "edge_length"))
    points_per_edge = int(setting(args.points_per_edge, "num_points_per_edge")); num_loops = int(setting(args.num_loops, "num_loops"))
    target_length = float(setting(args.target_length, "target_length_mm")); selection_mode = setting(args.mode, "selection_mode")
    seed = int(setting(args.seed, "seed")); output_dir = Path(setting(args.output_dir, "output_dir"))
    visualize = bool(cfg.get("visualize", True)) and not args.no_visualize
    show = bool(cfg.get("show", False)) or args.show
    convert_to_inp = bool(cfg.get("convert_to_inp", False)) or args.convert_to_inp
    if args.no_inp: convert_to_inp = False
    template_path = None
    if purpose == "path" or args.self_test:
        template_path = args.boundary_template or Path(cfg.get("boundary_template", "boundary_template_5x3.json"))
        if not template_path.is_absolute():
            template_path = Path(__file__).resolve().parent / template_path

    if args.self_test:
        run_self_test(template_path)
        return
    if purpose not in ("mechanics", "path"):
        parser.error("config purpose must be 'mechanics' or 'path'")
    if purpose == "path" and eulerization not in ("auto", "template", "cpp"):
        parser.error("config eulerization must be 'auto', 'template', or 'cpp'")
    if min(num_cols, num_rows, num_variants, points_per_edge) <= 0 or edge_length <= 0:
        parser.error("cols, rows, variants, points-per-edge, and edge-length must be positive")
    if selection_mode not in ("single", "mixed", "row_alternating", "col_alternating", "random"):
        parser.error(f"unsupported selection mode in config: {selection_mode}")
    if convert_to_inp and purpose == "path":
        print("[WARN] INP export is skipped in path mode because the INP converter intentionally collapses parallel edge instances.")
        convert_to_inp = False

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    if visualize: plot_dir.mkdir(parents=True, exist_ok=True)
    concrete_modes = ("single", "mixed", "row_alternating", "col_alternating")
    pattern_a = {"edge_length": edge_length, "num_points_per_edge": points_per_edge, "num_loops": num_loops}
    pattern_b = dict(pattern_a)

    for idx in range(1, num_variants + 1):
        run_seed = seed + idx - 1
        random.seed(run_seed); np.random.seed(run_seed)
        HexagonalGraph._cached_patterns = {}
        tiled = TiledHexGraph(num_cols, num_rows, edge_length, points_per_edge, ndigits=8)
        mode = random.choice(concrete_modes) if selection_mode == "random" else selection_mode
        physical_graph, internal_count = tiled.build(mode=mode, pattern_a_params=pattern_a, pattern_b_params=pattern_b)
        if purpose == "mechanics":
            tiled.global_graph = physical_graph
            tiled.global_graph.graph.update({"purpose": "mechanics", "is_eulerian": bool(nx.is_eulerian(physical_graph))})
        else:
            tiled.global_graph = eulerize_graph(physical_graph, eulerization, mode=mode,
                num_cols=num_cols, num_rows=num_rows, edge_length=edge_length, template_path=template_path)
        task_tag = "physical" if purpose == "mechanics" else "eulerian"
        stem = f"pattern{idx:03d}_{mode}_{num_cols}x{num_rows}_{task_tag}_seed{run_seed}_loops{internal_count}"
        json_path = output_dir / f"{stem}.json"
        tiled.save_to_json(str(json_path))
        print(f"[INFO] Saved graph: {json_path}")
        if visualize:
            tiled.visualize(figsize=tuple(cfg.get("figsize", [10, 10])), node_size=int(cfg.get("node_size", 12)),
                            show=show, save_path=str(plot_dir / f"{stem}.png"), title_suffix=f"{purpose}; seed={run_seed}")
        if convert_to_inp:
            process_json_to_inp(str(json_path), str(output_dir), target_length=target_length)
        graph = tiled.global_graph
        odd_count = sum(degree % 2 for _, degree in graph.degree())
        aux_count = graph.graph.get("auxiliary_edge_instances", 0)
        method_name = graph.graph.get("eulerization_method", "none")
        print(f"[INFO] purpose={purpose}, mode={mode}, nodes={graph.number_of_nodes()}, "
              f"edge_instances={graph.number_of_edges()}, odd_vertices={odd_count}, auxiliary_instances={aux_count}, "
              f"eulerian={nx.is_eulerian(graph)}, method={method_name}")

    print("[INFO] Generation complete.")


if __name__ == "__main__":
    main()
