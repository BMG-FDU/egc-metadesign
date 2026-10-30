import json
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

EPS = 2e-6


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def clip_segment(p, q, polygon):
    lo, hi = 0.0, 1.0
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        e = b - a
        c, d = cross(e, p - a), cross(e, q - p)
        if abs(d) < 1e-14:
            if c < -EPS:
                return None
        elif d > 0:
            lo = max(lo, -c / d)
        else:
            hi = min(hi, -c / d)
        if lo > hi + 1e-10:
            return None
    p1, q1 = p + lo * (q - p), p + hi * (q - p)
    return (p1, q1) if np.linalg.norm(q1 - p1) > EPS else None


def representations(graph):
    nodes = graph['nodes']
    corners = []
    for i in range(1, 7):
        matches = [n for n in nodes if n.get('label') == 'V' + str(i)]
        if not matches:
            raise ValueError('Missing V1--V6 labels; cannot identify common cell reliably.')
        corners.append(matches[0]['pos'])
    polygon = np.asarray(corners, dtype=float)
    centre = polygon.mean(axis=0)
    height = np.ptp(polygon[:, 1])
    if height <= 0:
        raise ValueError('Invalid cell height')
    polygon = (polygon - centre) / height
    if sum(cross(a, b) for a, b in zip(polygon, np.roll(polygon, -1, axis=0))) < 0:
        polygon = polygon[::-1]
    coords = {str(n['id']): (np.asarray(n['pos'], dtype=float) - centre) / height for n in nodes}
    segments, seen, removed = [], set(), 0
    for edge in graph.get('links', graph.get('edges', [])):
        part = clip_segment(coords[str(edge['source'])], coords[str(edge['target'])], polygon)
        if part is None:
            continue
        p, q = part
        is_border = any(abs(cross(b-a, p-a)) / np.linalg.norm(b-a) < EPS and
                        abs(cross(b-a, q-a)) / np.linalg.norm(b-a) < EPS
                        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)))
        if is_border:
            removed += 1
            continue
        p, q = tuple(np.round(p, 8)), tuple(np.round(q, 8))
        key = tuple(sorted((p, q)))
        if key not in seen and p != q:
            seen.add(key)
            segments.append((np.array(p), np.array(q)))
    if not segments:
        raise ValueError('No internal members after removing common framework')

    points = np.vstack(segments)
    parent = np.arange(len(points))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in cKDTree(points).query_pairs(EPS):
        parent[find(b)] = find(a)
    clusters = {}
    for i in range(len(points)):
        clusters.setdefault(find(i), []).append(i)
    snapped = points.copy()
    for indices in clusters.values():
        snapped[indices] = points[indices].mean(axis=0)
    segments, seen = [], set()
    for p, q in snapped.reshape(-1, 2, 2):
        key = tuple(sorted((tuple(p),tuple(q))))
        if key not in seen and np.linalg.norm(q-p) > EPS:
            segments.append((p,q)); seen.add(key)

    # Actual incidence is retained separately from the merged straight-member geometry.
    adjacency = {}
    for p, q in segments:
        adjacency.setdefault(tuple(p), set()).add(tuple(q))
        adjacency.setdefault(tuple(q), set()).add(tuple(p))
    junctions = []
    for p, neighbors in adjacency.items():
        p = np.array(p)
        vectors = [np.array(q) - p for q in neighbors]
        vectors = [v / np.linalg.norm(v) for v in vectors]
        if len(vectors) == 2 and np.dot(vectors[0], vectors[1]) < -1 + 5e-7:
            continue
        angles = np.array([math.atan2(v[1], v[0]) for v in vectors])
        centres = np.arange(12) * 2 * np.pi / 12
        hist = np.exp(8 * (np.cos(angles[:, None] - centres[None, :]) - 1)).sum(axis=0)
        hist /= hist.sum()
        border = float(any(abs(cross(b-a, p-a)) / np.linalg.norm(b-a) < EPS
                           for a, b in zip(polygon, np.roll(polygon, -1, axis=0))))
        junctions.append([*p, len(neighbors), *hist, border])

    groups = []
    for p, q in sorted(segments, key=lambda pair: -np.linalg.norm(pair[1]-pair[0])):
        direction = (q-p) / np.linalg.norm(q-p)
        if direction[0] < -EPS or (abs(direction[0]) < EPS and direction[1] < 0):
            direction = -direction
        normal = np.array([-direction[1], direction[0]])
        offset = np.dot(normal, p)
        candidates = [g for g in groups if abs(cross(g[0], direction)) < .01 and
                      max(abs(cross(g[0], p)-g[1]),abs(cross(g[0], q)-g[1])) < 2e-6]
        group = min(candidates, key=lambda g: max(abs(cross(g[0], p)-g[1]),abs(cross(g[0], q)-g[1]))) if candidates else None
        if group is None:
            group = [direction, offset, []]
            groups.append(group)
        group[2].append(sorted((np.dot(group[0], p), np.dot(group[0], q))))
    members = []
    for direction, offset, intervals in groups:
        merged = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1] + 5e-6:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        normal = np.array([-direction[1], direction[0]])
        for lo, hi in merged:
            middle = direction * ((lo + hi) / 2) + normal * offset
            theta = math.atan2(direction[1], direction[0])
            members.append([*middle, hi-lo, math.cos(2*theta), math.sin(2*theta)])
    return {'members': members, 'junctions': junctions,
            'audit': {'internal_pieces': len(segments), 'merged_members': len(members),
                      'junctions': len(junctions), 'framework_pieces_removed': removed,
                      'cell_height_source': float(height)}}


def feature_arrays(rep, tolerance, angle_degrees=15):
    m = np.array(rep['members'], dtype=float)
    j = np.array(rep['junctions'], dtype=float)
    angle_scale = 2 * math.radians(angle_degrees)
    mscale = np.array([tolerance, tolerance, 2*tolerance, angle_scale, angle_scale])
    jscale = np.array([tolerance, tolerance, 1.0, *([0.25]*12), 0.5])
    mw = m[:, 2] / m[:, 2].sum()
    jw = np.full(len(j), 1 / len(j))
    return [(m / mscale, mw), (j / jscale, jw)]


def self_kernel(x, w):
    return float(w @ np.exp(-0.5 * cdist(x, x, 'sqeuclidean')) @ w)


class ExactKernelIndex:
    def __init__(self, reps, tolerance, angle_degrees=15):
        items = [feature_arrays(r, tolerance, angle_degrees) for r in reps]
        self.indices = []
        for component in range(2):
            xx, ww, rows, norms = [], [], [], []
            for idx, item in enumerate(items):
                x, w = item[component]
                xx.append(x); ww.extend(w); rows.extend([idx]*len(w))
                norms.append(math.sqrt(self_kernel(x, w)))
            xx = np.vstack(xx)
            aggregation = csr_matrix((ww, (rows, np.arange(len(ww)))), shape=(len(reps), len(ww)))
            self.indices.append((xx, aggregation, np.array(norms)))
        self.tolerance, self.angle = tolerance, angle_degrees

    def similarities(self, rep):
        scores = []
        for (x, w), (train, aggregation, norms) in zip(feature_arrays(rep, self.tolerance, self.angle), self.indices):
            sums = np.empty(len(train))
            for start in range(0, len(train), 10000):
                end = start + 10000
                sums[start:end] = w @ np.exp(-0.5 * cdist(x, train[start:end], 'sqeuclidean'))
            scores.append(np.clip((aggregation @ sums) / (norms * math.sqrt(self_kernel(x, w))), 0, 1))
        return np.sqrt(scores[0]*scores[1])
