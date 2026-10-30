from collections import defaultdict, deque
import json
import math
import numpy as np


def load_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = {}

    if isinstance(data, dict) and "nodes" in data:
        for n in data["nodes"]:
            nid = str(n.get("id", n.get("name", n.get("label", ""))))
            if nid:
                nodes[nid] = n

    raw_edges = []

    if isinstance(data, dict):
        edges_source = data.get("edges", data.get("links", []))
    elif isinstance(data, list):
        edges_source = data
    else:
        edges_source = []

    for i, e in enumerate(edges_source):
        u = v = None

        if isinstance(e, dict):
            u = str(e.get("source", e.get("from", e.get("u"))))
            v = str(e.get("target", e.get("to", e.get("v"))))
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            u, v = str(e[0]), str(e[1])

        if u is not None and v is not None:
            raw_edges.append((i, u, v, e))
            nodes.setdefault(u, {"id": u})
            nodes.setdefault(v, {"id": v})

    return nodes, raw_edges


def analyze_graph(nodes, edges):
    deg = defaultdict(int)
    adj = defaultdict(list)

    for eid, u, v, _ in edges:
        if u == v:
            deg[u] += 2
            adj[u].append((u, eid))
            adj[u].append((u, eid))
        else:
            deg[u] += 1
            deg[v] += 1
            adj[u].append((v, eid))
            adj[v].append((u, eid))

    non_zero = [n for n, d in deg.items() if d > 0]

    if not non_zero:
        return "empty", [], deg, adj

    start = non_zero[0]
    seen = {start}
    q = deque([start])

    while q:
        cur = q.popleft()
        for nb, _ in adj[cur]:
            if nb not in seen:
                seen.add(nb)
                q.append(nb)

    if not all(n in seen for n in non_zero):
        return "disconnected", [], deg, adj

    odd_nodes = [n for n, d in deg.items() if d % 2 == 1]

    if len(odd_nodes) > 2:
        return "too_many_odd", odd_nodes, deg, adj

    if len(odd_nodes) == 0:
        return "eulerian_circuit", odd_nodes, deg, adj

    return "eulerian_trail", odd_nodes, deg, adj


def get_layout(nodes):
    raw_pos = {}

    for nid, n in nodes.items():
        x = y = None

        for k in ["x", "X", "px", "cx"]:
            if k in n:
                x = float(n[k])
                break

        for k in ["y", "Y", "py", "cy"]:
            if k in n:
                y = float(n[k])
                break

        if x is None and "pos" in n:
            try:
                x, y = float(n["pos"][0]), float(n["pos"][1])
            except Exception:
                pass

        if x is None and isinstance(n.get("id", None), str):
            try:
                parts = n["id"].split(",")
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
            except Exception:
                pass

        if x is not None and y is not None:
            raw_pos[nid] = (x, y)

    if not raw_pos:
        N = len(nodes)
        for i, nid in enumerate(nodes):
            th = 2 * math.pi * i / max(N, 1)
            raw_pos[nid] = (math.cos(th), math.sin(th))

    xs = [p[0] for p in raw_pos.values()]
    ys = [p[1] for p in raw_pos.values()]

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    w = maxx - minx
    h = maxy - miny
    s = 1.0 / max(w, h) if max(w, h) > 0 else 1.0

    final = {}

    for nid, (x, y) in raw_pos.items():
        nx = (x - minx) * s
        ny = (y - miny) * s
        final[nid] = (nx, ny)

    return final


def vec_score(p1, p2, p3):
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)

    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)

    if m1 == 0 or m2 == 0:
        return -2

    return float(np.dot(v1, v2) / (m1 * m2))


def algo_basic_find_path(edges, adj, start_node):
    adj_copy = {k: list(v) for k, v in adj.items()}
    used = set()
    stack = [start_node]
    path = []

    while stack:
        v = stack[-1]

        while adj_copy[v] and adj_copy[v][-1][1] in used:
            adj_copy[v].pop()

        if adj_copy[v]:
            nb, eid = adj_copy[v].pop()

            if eid not in used:
                used.add(eid)
                stack.append(nb)
        else:
            path.append(stack.pop())

    return path[::-1]


def algo_momentum_find_path(edges, adj, start_node, pos):
    graph = defaultdict(list)

    for eid, u, v, _ in edges:
        graph[u].append((v, eid))
        if u != v:
            graph[v].append((u, eid))

    stack = [start_node]
    path = []
    used = set()

    while stack:
        u = stack[-1]
        cands = [x for x in graph[u] if x[1] not in used]

        if cands:
            best_idx = 0

            if len(stack) >= 2 and u in pos:
                prev = stack[-2]

                if prev in pos:
                    pp, pc = pos[prev], pos[u]
                    best_s = -999

                    for i, (nxt, _) in enumerate(cands):
                        if nxt in pos:
                            s = vec_score(pp, pc, pos[nxt])

                            if s > best_s:
                                best_s = s
                                best_idx = i

            vn, eid = cands[best_idx]
            used.add(eid)
            stack.append(vn)

        else:
            path.append(stack.pop())

    return path[::-1]


def algo_lookahead_find_path(edges, adj, start_node, pos, weight=0.5):
    graph = defaultdict(list)

    for eid, u, v, _ in edges:
        graph[u].append((v, eid))
        if u != v:
            graph[v].append((u, eid))

    stack = [start_node]
    path = []
    used = set()

    while stack:
        u = stack[-1]
        cands = [x for x in graph[u] if x[1] not in used]

        if cands:
            best_idx = 0

            if len(stack) >= 2 and u in pos:
                prev = stack[-2]

                if prev in pos:
                    pp, pc = pos[prev], pos[u]
                    best_total = -9999

                    for i, (vn, eid_n) in enumerate(cands):
                        if vn not in pos:
                            continue

                        pn = pos[vn]
                        s1 = vec_score(pp, pc, pn)

                        subs = [x for x in graph[vn] if x[1] != eid_n and x[1] not in used]

                        if not subs:
                            s2 = 0.0
                        else:
                            s2 = max(
                                (vec_score(pc, pn, pos[w]) for w, _ in subs if w in pos),
                                default=-2.0,
                            )

                        total = s1 + weight * s2

                        if total > best_total:
                            best_total = total
                            best_idx = i

            vf, eid_f = cands[best_idx]
            used.add(eid_f)
            stack.append(vf)

        else:
            path.append(stack.pop())

    return path[::-1]


def turning_angle_deg(p_prev, p_curr, p_next):
    """
    Turning angle at the current vertex.

    0°   -> nearly straight continuation
    180° -> strong reversal / very sharp large turn

    This matches the earlier Fig.2b discussion, where very large angles
    (e.g. >150°) are treated as high-risk sharp turns.
    """
    v_in = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]], dtype=float)
    v_out = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]], dtype=float)

    nin = np.linalg.norm(v_in)
    nout = np.linalg.norm(v_out)
    if nin < 1e-12 or nout < 1e-12:
        return None

    cosang = np.dot(v_in, v_out) / (nin * nout)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = float(np.degrees(np.arccos(cosang)))
    return ang
