import os, sys, json, glob, argparse, math, time
import random
import itertools
import copy
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.collections import LineCollection


try:
    from torch_geometric.loader import DataLoader
except Exception:
    from torch_geometric.data import DataLoader
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, global_max_pool


PAPER_BLUE = "#2F5597"
PAPER_BLUE_DARK = "#1F3864"
PAPER_BLUE_LIGHT = "#4E6EA7"
PAPER_BLUE_LIGHTER = "#8FAADC"
PAPER_BLUE_PALE = "#D9E2F3"

PAPER_GREEN = "#1E5A29"
PAPER_GREEN_LIGHT = "#70AD47"
PAPER_GREEN_PALE = "#DDEFE3"

PAPER_ORANGE = "#C55A11"
PAPER_ORANGE_LIGHT = "#E28A3D"
PAPER_ORANGE_PALE = "#FCE4D6"

PAPER_RED = "#C00000"
PAPER_RED_LIGHT = "#E06666"
PAPER_RED_PALE = "#F4CCCC"

DARK_TEXT = "#222222"
GRAY_LINE = "#7F7F7F"
LIGHT_GRAY = "#D9D9D9"
GRAY_SKELETON = "#666666"
WHITE = "#FFFFFF"


def register_aptos_for_matplotlib():
    font_dirs = [
        Path(r"C:\Windows\Fonts"),
        Path.home() / r"AppData\Local\Microsoft\Windows\Fonts",
    ]
    for font_dir in font_dirs:
        if not font_dir.exists():
            continue
        for pat in ("Aptos*.ttf", "Aptos*.otf"):
            for fp in font_dir.glob(pat):
                try:
                    mpl.font_manager.fontManager.addfont(str(fp))
                except Exception:
                    pass


def set_mpl_style():
    register_aptos_for_matplotlib()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Aptos", "Arial", "DejaVu Sans", "Liberation Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,

        "figure.dpi": 220,
        "savefig.dpi": 600,

        "axes.grid": False,
        "axes.linewidth": 0.95,
        "axes.edgecolor": DARK_TEXT,
        "axes.labelcolor": DARK_TEXT,

        "axes.titlesize": 9.5,
        "axes.titleweight": "normal",
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,

        "xtick.color": DARK_TEXT,
        "ytick.color": DARK_TEXT,
        "text.color": DARK_TEXT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.85,
        "ytick.major.width": 0.85,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,

        "legend.frameon": False,

        "lines.solid_joinstyle": "miter",
        "lines.solid_capstyle": "round",
        "lines.dash_joinstyle": "miter",
        "lines.dash_capstyle": "butt",

        "path.simplify": False,
        "path.simplify_threshold": 0.0,
        "agg.path.chunksize": 0,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axes(ax, full_box=True):
    ax.grid(False)
    for side in ["left", "bottom", "right", "top"]:
        ax.spines[side].set_visible(bool(full_box))
        ax.spines[side].set_linewidth(0.90)
        ax.spines[side].set_color(DARK_TEXT)
    ax.tick_params(axis="both", which="major", direction="out",
                   length=3.2, width=0.85, colors=DARK_TEXT, pad=2)


def sanitize_name(s):
    out = str(s)
    for b in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ', '(', ')']:
        out = out.replace(b, "_")
    return out


def smooth_array(y, weight=0.72):
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    out = np.empty_like(y)
    last = y[0]
    for i, v in enumerate(y):
        if not np.isfinite(v):
            v = last
        last = last * weight + (1.0 - weight) * v
        out[i] = last
    return out


def normalize_01(values, fallback=0.55):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return np.full_like(values, fallback, dtype=float)
    return (values - vmin) / (vmax - vmin)


def make_node_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list(
        "paper_node_blue",
        [PAPER_BLUE_PALE, PAPER_BLUE_LIGHTER, PAPER_BLUE_LIGHT, PAPER_BLUE]
    )


def make_edge_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list(
        "paper_edge_red",
        [PAPER_RED_PALE, PAPER_RED_LIGHT, PAPER_RED]
    )


def save_png(fig, path, dpi=600, transparent=False, pad=0.025):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(path),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=pad,
        transparent=bool(transparent),
        facecolor="none" if transparent else "white",
    )
    plt.close(fig)


set_mpl_style()


TUNING_GRID = {
    'lr': [1e-3, 1e-4, 5e-5, 1e-5],
    'hidden': [256],
    'batch_size': [8, 4],
    'pooling_type': ['dual'],
    'use_global_feat': [1],
    'use_denom_head': [1],
    'use_edge_feat': [1],
    'decoder_type': ['cnn', 'mlp'],
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

try:
    from torch_geometric.nn import MLP
except Exception:
    class MLP(nn.Module):
        def __init__(self, channels, batch_norm=True):
            super().__init__()
            layers = []
            for i in range(len(channels)-1):
                layers.append(nn.Linear(channels[i], channels[i+1]))
                if i < len(channels)-2:
                    if batch_norm:
                        layers.append(nn.BatchNorm1d(channels[i+1]))
                    layers.append(nn.ReLU())
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

def parse_networkx_json(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    base = os.path.splitext(os.path.basename(path))[0]
    nodes = obj.get('nodes', [])
    links = obj.get('links', []) or obj.get('edges', [])
    idx_map = {}
    coords = []
    for i, n in enumerate(nodes):
        nid = n.get('id', None)
        pos = n.get('pos', None)
        if pos and len(pos) >= 2:
            try:
                x = float(pos[0]); y = float(pos[1])
            except Exception:
                x, y = 0.0, 0.0
        else:
            if nid and isinstance(nid, str) and ',' in nid:
                try:
                    px, py = nid.split(',', 1); x, y = float(px), float(py)
                except Exception:
                    x, y = 0.0, 0.0
            else:
                x, y = 0.0, 0.0
        coords.append((x, y))
        idx_map[str(nid) if nid is not None else str(i)] = i
    src, dst = [], []
    def resolve(val):
        if val is None: return None
        if isinstance(val, int): return val
        s = str(val)
        if s in idx_map: return idx_map[s]
        try: return int(s)
        except Exception: return None
    for e in links:
        if not isinstance(e, dict): continue
        u, v = resolve(e.get('source')), resolve(e.get('target'))
        if u is not None and v is not None:
            src.extend([u, v])
            dst.extend([v, u])
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    N = len(coords)
    edge_attr_list = []
    if N > 0 and len(src) > 0:
        for u, v in zip(src, dst):
            if u < N and v < N:
                dx = coords[u][0] - coords[v][0]
                dy = coords[u][1] - coords[v][1]
                dist = math.hypot(dx, dy)
                safe_dist = dist if dist > 1e-6 else 1.0
                nx = dx / safe_dist
                ny = dy / safe_dist
                edge_attr_list.append([dist, nx, ny])
            else:
                edge_attr_list.append([0.0, 0.0, 0.0])
    if edge_attr_list:
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
    degs, mean_len = [0] * N, [0.0] * N
    for idx_e, (u, v) in enumerate(zip(src, dst)):
        if u < N and v < N:
            degs[u] += 1
            d = edge_attr_list[idx_e][0] if idx_e < len(edge_attr_list) else 0.0
            mean_len[u] += d
    for i in range(N):
        if degs[i] > 0:
            mean_len[i] /= degs[i]
    degs_arr = np.array(degs)
    mean_global_len = np.mean(mean_len) if N > 0 else 0.0
    std_deg = np.std(degs_arr) if N > 0 else 0.0
    num_edges = len(src) / 2.0
    density = (2.0 * num_edges) / (N * (N - 1)) if N > 1 else 0.0
    coords_arr = np.array(coords, dtype=np.float32) if coords else np.zeros((0, 2), dtype=np.float32)
    if coords_arr.size > 0:
        coords_arr -= coords_arr.mean(axis=0, keepdims=True)
    deg_arr = np.array(degs, dtype=np.float32).reshape(-1, 1) if N > 0 else np.zeros((0, 1), dtype=np.float32)
    mean_len_arr = np.array(mean_len, dtype=np.float32).reshape(-1, 1) if N > 0 else np.zeros((0, 1), dtype=np.float32)
    x_feat = np.hstack([coords_arr, deg_arr, mean_len_arr]) if N > 0 else np.zeros((0, 4), dtype=np.float32)
    global_features_list = [float(N), float(density), float(mean_global_len), float(std_deg)]
    data = Data(
        x=torch.tensor(x_feat, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=N,
        graph_id=base
    )
    data.global_feat = torch.tensor(global_features_list, dtype=torch.float)
    return data


def is_inside_hexagon(p, poly_coords, center_coord, tol=1e-4):
    for i in range(len(poly_coords)):
        p1 = poly_coords[i]
        p2 = poly_coords[(i + 1) % len(poly_coords)]
        ex, ey = p2[0] - p1[0], p2[1] - p1[1]
        cx_vec, cy_vec = center_coord[0] - p1[0], center_coord[1] - p1[1]
        vx, vy = p[0] - p1[0], p[1] - p1[1]
        cross_center = ex * cy_vec - ey * cx_vec
        cross_p = ex * vy - ey * vx
        if cross_center > 0 and cross_p < -tol:
            return False
        elif cross_center < 0 and cross_p > tol:
            return False
    return True


def read_csv_new(path):
    df = pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')
    cols_orig = list(df.columns)
    cols = [c.strip().lower() for c in cols_orig]
    col_map = {c: i for i, c in enumerate(cols)}
    s = df.iloc[:, col_map.get('strain', 0)].astype(float).to_numpy() if 'strain' in col_map or len(df.columns) > 0 else None
    def get_col(candidates):
        for cand in candidates:
            if cand in col_map:
                return df.iloc[:, col_map[cand]].astype(float).to_numpy(), cols_orig[col_map[cand]]
        for i, c in enumerate(cols):
            if any(sub in c for sub in candidates):
                return df.iloc[:, i].astype(float).to_numpy(), cols_orig[i]
        return None, candidates[0]
    en_abs, en_abs_name = get_col(['sea', 'SEA(J/kg)', 'specific-energy-absorption'])
    return s, en_abs, None, None, {}, en_abs_name, None, None

def sample_curve_to_grid(x, y, grid):
    if x is None or y is None or len(x) == 0:
        return np.zeros_like(grid)
    grid_clamped = np.clip(grid, np.min(x), np.max(x))
    return np.interp(grid_clamped, x, y)

def compute_absolute_curve_metrics(true_curves, pred_curves, eps=1e-12):
    true_arr = np.asarray(true_curves, dtype=float)
    pred_arr = np.asarray(pred_curves, dtype=float)
    if true_arr.size == 0 or pred_arr.size == 0:
        return {
            'mae_abs': float('nan'),
            'rmse_abs': float('nan'),
            'end_rel_err_pct': float('nan'),
        }
    diff = pred_arr - true_arr
    mae_abs = float(np.mean(np.abs(diff)))
    rmse_abs = float(np.sqrt(np.mean(diff ** 2)))
    true_end = true_arr[..., -1].reshape(-1)
    pred_end = pred_arr[..., -1].reshape(-1)
    rel_err = np.abs(pred_end - true_end) / np.maximum(np.abs(true_end), eps)
    end_rel_err_pct = float(np.mean(rel_err) * 100.0)
    return {
        'mae_abs': mae_abs,
        'rmse_abs': rmse_abs,
        'end_rel_err_pct': end_rel_err_pct,
    }

class GraphCurveDataset(PyGDataset):
    def __init__(self, graphs_dir, data_dir, strain_grid):
        super().__init__()
        self.graph_files = sorted(glob.glob(os.path.join(graphs_dir, '*.json')))
        self.data_dir = data_dir
        self.strain_grid = np.array(strain_grid, dtype=float)
        self.L = len(self.strain_grid)
        self.graph_files = [
            p for p in self.graph_files
            if (os.path.exists(os.path.join(self.data_dir, f"{os.path.splitext(os.path.basename(p))[0]}.csv")) or
                os.path.exists(os.path.join(self.data_dir, f"{os.path.splitext(os.path.basename(p))[0]}_Y.csv")))
        ]
    def len(self):
        return len(self.graph_files)
    def get_raw_curve_by_gid(self, gid):
        csvp = os.path.join(self.data_dir, f"{gid}.csv")
        if not os.path.exists(csvp):
            csvp = os.path.join(self.data_dir, f"{gid}_Y.csv")
        return read_csv_new(csvp) if os.path.exists(csvp) else (None,) * 8
    def get(self, idx):
        path = self.graph_files[idx]
        data = parse_networkx_json(path)
        s, en_abs, _, _, _, en_abs_name, _, _ = self.get_raw_curve_by_gid(data.graph_id)
        curves_list = []
        curves_list.append(sample_curve_to_grid(s, en_abs, self.strain_grid))
        curves = np.vstack(curves_list)
        data.y_raw = torch.tensor(curves.ravel(), dtype=torch.float)
        data._orig_colnames = {'SEA': en_abs_name}
        return data

class CNNDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_seq_len, out_channels=1):
        super().__init__()
        self.L = out_seq_len
        self.out_channels = out_channels
        self.l_start = math.ceil(out_seq_len / 4)
        self.c_start = hidden_dim
        self.project = nn.Linear(in_dim, self.c_start * self.l_start)
        self.up1 = nn.ConvTranspose1d(self.c_start, self.c_start // 2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(self.c_start // 2)
        self.up2 = nn.ConvTranspose1d(self.c_start // 2, self.c_start // 4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(self.c_start // 4)
        self.final_conv = nn.Conv1d(self.c_start // 4, out_channels, kernel_size=3, padding=1)
    def forward(self, x):
        bs = x.shape[0]
        x = self.project(x)
        x = x.view(bs, self.c_start, self.l_start)
        x = F.relu(self.bn1(self.up1(x)))
        x = F.relu(self.bn2(self.up2(x)))
        if x.shape[2] != self.L:
            x = F.interpolate(x, size=self.L, mode='linear', align_corners=False)
        x = self.final_conv(x)
        return x.view(bs, -1)

class GINCurveRegressorWithDenom(nn.Module):
    def __init__(self, in_channels, out_dim, denom_channels=3, hidden=128, num_layers=3, dropout=0.0,
                 num_global_features=2, use_global_feat=True, pooling_type='dual',
                 use_denom_head=True, use_edge_feat=False, decoder_type='mlp',
                 seq_len=100, out_channels_curve=1):
        super().__init__()
        self.use_global_feat = use_global_feat
        self.pooling_type = pooling_type
        self.use_denom_head = use_denom_head
        self.use_edge_feat = use_edge_feat
        self.decoder_type = decoder_type
        self.node_encoder = nn.Linear(in_channels, hidden)
        if self.use_edge_feat:
            self.edge_encoder = nn.Linear(3, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([hidden, hidden, hidden])
            if self.use_edge_feat:
                self.convs.append(GINEConv(mlp))
            else:
                self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        pool_dim = hidden * 2 if pooling_type == 'dual' else hidden
        gf_dim = num_global_features if use_global_feat else 0
        head_in_dim = pool_dim + gf_dim
        if self.decoder_type == 'cnn':
            self.head_curve = CNNDecoder(
                in_dim=head_in_dim, hidden_dim=hidden,
                out_seq_len=seq_len, out_channels=out_channels_curve
            )
        else:
            self.head_curve = nn.Sequential(
                nn.Linear(head_in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, max(hidden // 2, 16)), nn.ReLU(),
                nn.Linear(max(hidden // 2, 16), out_dim)
            )
        if self.use_denom_head:
            self.head_denom = nn.Sequential(
                nn.Linear(head_in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, max(hidden // 4, 8)), nn.ReLU(),
                nn.Linear(max(hidden // 4, 8), denom_channels)
            )
        else:
            self.head_denom = None
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        x = self.node_encoder(x)
        if self.use_edge_feat and edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = None
        for conv, bn in zip(self.convs, self.bns):
            if self.use_edge_feat:
                if edge_embedding is not None:
                    x = F.relu(bn(conv(x, edge_index, edge_attr=edge_embedding)))
                else:
                    x = F.relu(bn(conv(x, edge_index)))
            else:
                x = F.relu(bn(conv(x, edge_index)))
        if self.pooling_type == 'dual':
            x_pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        else:
            x_pooled = global_mean_pool(x, batch)
        if self.use_global_feat:
            gf = data.global_feat.view(data.num_graphs, -1)
            h = torch.cat([x_pooled, gf], dim=1)
        else:
            h = x_pooled
        pred_curve = self.head_curve(h)
        if self.use_denom_head:
            pred_denom_log = self.head_denom(h)
        else:
            pred_denom_log = torch.zeros(h.size(0), 1, device=h.device)
        return pred_curve, pred_denom_log

class GraphScalarWrapper(nn.Module):
    def __init__(self, base_model, L, target_index, use_denom_head=True):
        super().__init__()
        self.base_model = base_model
        self.L = int(L)
        self.target_index = int(target_index)
        self.use_denom_head = bool(use_denom_head)
    def forward(self, x, edge_index, **kwargs):
        edge_attr = kwargs.get('edge_attr', None)
        batch = kwargs.get('batch', None)
        global_feat = kwargs.get('global_feat', None)
        data = Data(x=x, edge_index=edge_index)
        if edge_attr is not None:
            data.edge_attr = edge_attr
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        data.batch = batch
        data.num_graphs = int(batch.max().item()) + 1
        if global_feat is not None:
            data.global_feat = global_feat.view(-1) if global_feat.dim() == 2 and global_feat.size(0) == 1 and data.num_graphs == 1 else global_feat
        else:
            default_gf_dim = getattr(self.base_model, 'num_global_features', 4)
            data.global_feat = x.new_zeros((default_gf_dim,), dtype=torch.float)
        pred_curve_norm, pred_denom_log = self.base_model(data)
        B = data.num_graphs
        pred_curve_norm = pred_curve_norm.view(B, 1, self.L)
        if self.use_denom_head:
            denom_pred = torch.exp(pred_denom_log).view(B, -1)
            denom_curve = denom_pred[:, 0].view(B, 1, 1)
        else:
            denom_curve = pred_curve_norm.new_ones((B, 1, 1))
        pred_curve = pred_curve_norm * denom_curve
        idx = max(0, min(self.L - 1, self.target_index))
        y = pred_curve[:, 0, idx]
        return y.view(B, 1)

def run_gnnexplainer_on_test(model, wrapped_ds, ds, test_idx, strain_grid, args, device):
    explain_dir = os.path.join(args.outdir, "gnnexplainer")
    os.makedirs(explain_dir, exist_ok=True)

    target_strain = float(args.explain_target_strain)
    target_index = int(np.argmin(np.abs(strain_grid - target_strain)))
    print(f"\n--- GNNExplainer: target strain={target_strain:.4f}, closest grid={strain_grid[target_index]:.4f} (idx={target_index}) ---")

    scalar_model = GraphScalarWrapper(
        base_model=model,
        L=args.L,
        target_index=target_index,
        use_denom_head=bool(args.use_denom_head),
    ).to(device)
    scalar_model.eval()

    explainer = None
    api = None
    try:
        from torch_geometric.explain import Explainer
        from torch_geometric.explain.algorithm import GNNExplainer
        from torch_geometric.explain.config import ModelConfig
        explainer = Explainer(
            model=scalar_model,
            algorithm=GNNExplainer(epochs=int(args.explainer_epochs)),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=ModelConfig(mode='regression', task_level='graph', return_type='raw')
        )
        api = "new"
        print("Using torch_geometric.explain Explainer API.")
    except Exception:
        explainer = None

    if explainer is None:
        try:
            from torch_geometric.nn.models import GNNExplainer as LegacyGNNExplainer
            explainer = LegacyGNNExplainer(scalar_model, epochs=int(args.explainer_epochs))
            api = "legacy"
            print("Using legacy torch_geometric.nn.models.GNNExplainer API.")
        except Exception as e:
            print("GNNExplainer import failed.", e)
            return

    def _to_numpy_mask(mask):
        if mask is None:
            return None
        arr = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
        return np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def _node_importance_from_mask(mask, n_nodes):
        arr = _to_numpy_mask(mask)
        if arr is None or arr.size == 0:
            return np.zeros(n_nodes, dtype=float)
        score = np.abs(arr) if arr.ndim == 1 else np.mean(np.abs(arr), axis=1)
        score = np.asarray(score, dtype=float).reshape(-1)
        if len(score) < n_nodes:
            score = np.pad(score, (0, n_nodes - len(score)), mode="constant")
        elif len(score) > n_nodes:
            score = score[:n_nodes]
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    def _undirected_edges(edge_index_np, edge_mask_np, n_nodes, aggregate="max"):
        if edge_index_np is None or edge_index_np.ndim != 2 or edge_index_np.shape[1] == 0:
            return [], np.array([], dtype=float)
        if edge_mask_np is None:
            edge_mask_np = np.ones(edge_index_np.shape[1], dtype=float)
        edge_mask_np = np.asarray(edge_mask_np, dtype=float).reshape(-1)
        if len(edge_mask_np) < edge_index_np.shape[1]:
            edge_mask_np = np.pad(edge_mask_np, (0, edge_index_np.shape[1] - len(edge_mask_np)), mode="constant")
        elif len(edge_mask_np) > edge_index_np.shape[1]:
            edge_mask_np = edge_mask_np[:edge_index_np.shape[1]]

        bucket = {}
        for k in range(edge_index_np.shape[1]):
            u, v = int(edge_index_np[0, k]), int(edge_index_np[1, k])
            if u < 0 or v < 0 or u >= n_nodes or v >= n_nodes or u == v:
                continue
            a, b = (u, v) if u <= v else (v, u)
            w = abs(float(edge_mask_np[k])) if np.isfinite(edge_mask_np[k]) else 0.0
            bucket.setdefault((a, b), []).append(w)

        edges, weights = [], []
        for e, vals in bucket.items():
            edges.append(e)
            weights.append(float(np.mean(vals) if aggregate == "mean" else np.max(vals)))
        return edges, np.asarray(weights, dtype=float)

    def _segments_from_edges(coords, edges):
        segments = []
        for u, v in edges:
            if u < len(coords) and v < len(coords):
                segments.append([(coords[u, 0], coords[u, 1]), (coords[v, 0], coords[v, 1])])
        return segments

    def _set_equal_limits(ax, coords, pad_frac=0.06):
        coords = np.asarray(coords, dtype=float)
        xmin, ymin = np.min(coords, axis=0)
        xmax, ymax = np.max(coords, axis=0)
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        span = max(xmax - xmin, ymax - ymin)
        if span <= 1e-12:
            span = 1.0
        span *= (1.0 + 2.0 * pad_frac)
        ax.set_xlim(cx - span / 2.0, cx + span / 2.0)
        ax.set_ylim(cy - span / 2.0, cy + span / 2.0)

    def _draw_node_map(coords, segments, node_score, out_path):
        fig, ax = plt.subplots(figsize=(2.45, 2.45))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.set_aspect("equal", adjustable="box")

        if segments:
            lc_bg = LineCollection(
                segments,
                colors=[mpl.colors.to_rgba(GRAY_SKELETON, 0.24)],
                linewidths=0.45,
                zorder=0,
                capstyle="round",
                joinstyle="miter",
            )
            ax.add_collection(lc_bg)

        node_norm = normalize_01(node_score, fallback=0.55)
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=8.0 + 4.0 * node_norm,
            c=node_score,
            cmap=make_node_cmap(),
            edgecolors=WHITE,
            linewidths=0.25,
            alpha=0.96,
            zorder=2,
        )

        _set_equal_limits(ax, coords)
        ax.axis("off")
        save_png(fig, out_path, dpi=600, transparent=True, pad=0.02)

    def _draw_edge_map(coords, segments, edge_weights, out_path):
        fig, ax = plt.subplots(figsize=(2.45, 2.45))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.set_aspect("equal", adjustable="box")

        if segments and edge_weights.size:
            edge_norm = normalize_01(edge_weights, fallback=0.55)
            edge_cmap = make_edge_cmap()
            edge_colors = edge_cmap(0.20 + 0.78 * edge_norm)
            edge_colors = np.asarray(edge_colors, dtype=float)
            edge_colors[:, 3] = 0.30 + 0.55 * edge_norm
            edge_lws = 0.28 + (2.10 - 0.28) * edge_norm
            lc = LineCollection(
                segments,
                colors=edge_colors,
                linewidths=edge_lws,
                zorder=1,
                capstyle="round",
                joinstyle="miter",
            )
            ax.add_collection(lc)

        _set_equal_limits(ax, coords)
        ax.axis("off")
        save_png(fig, out_path, dpi=600, transparent=True, pad=0.02)

    def _draw_combined_map(coords, segments, node_score, edge_weights, out_path):
        fig, ax = plt.subplots(figsize=(2.45, 2.45))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.set_aspect("equal", adjustable="box")

        if segments and edge_weights.size:
            edge_norm = normalize_01(edge_weights, fallback=0.55)
            edge_cmap = make_edge_cmap()
            edge_colors = edge_cmap(0.18 + 0.80 * edge_norm)
            edge_colors = np.asarray(edge_colors, dtype=float)
            edge_colors[:, 3] = 0.24 + 0.46 * edge_norm
            edge_lws = 0.24 + 1.65 * edge_norm
            lc = LineCollection(
                segments,
                colors=edge_colors,
                linewidths=edge_lws,
                zorder=1,
                capstyle="round",
                joinstyle="miter",
            )
            ax.add_collection(lc)

        node_norm = normalize_01(node_score, fallback=0.55)
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=7.0 + 5.0 * node_norm,
            c=node_score,
            cmap=make_node_cmap(),
            edgecolors=WHITE,
            linewidths=0.22,
            alpha=0.96,
            zorder=3,
        )

        _set_equal_limits(ax, coords)
        ax.axis("off")
        save_png(fig, out_path, dpi=600, transparent=True, pad=0.02)

    n_total = len(test_idx)
    n = min(int(args.explain_num_graphs), n_total)
    chosen = random.sample(range(n_total), n)

    review_paths = []
    summary_rows = []

    for j, local_id in enumerate(chosen):
        gid = ds.get(test_idx[local_id]).graph_id
        data = wrapped_ds[test_idx[local_id]].to(device)
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        global_feat = data.global_feat.view(1, -1) if hasattr(data, "global_feat") else None

        with torch.no_grad():
            pred_scalar = float(scalar_model(x, edge_index, edge_attr=edge_attr, batch=batch, global_feat=global_feat).item())

        if api == "new":
            explanation = explainer(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, global_feat=global_feat)
            node_mask, edge_mask = explanation.node_mask, explanation.edge_mask
        else:
            try:
                node_mask, edge_mask = explainer.explain_graph(x, edge_index, edge_attr=edge_attr)
            except TypeError:
                node_mask, edge_mask = explainer.explain_graph(x, edge_index)

        edge_mask_np = _to_numpy_mask(edge_mask)
        edge_index_np = edge_index.detach().cpu().numpy()
        x_np = data.x.detach().cpu().numpy()

        if x_np.ndim < 2 or x_np.shape[1] < 2:
            continue

        coords = x_np[:, :2].astype(float)
        n_nodes = coords.shape[0]
        node_score = _node_importance_from_mask(node_mask, n_nodes)
        und_edges, edge_weights = _undirected_edges(edge_index_np, edge_mask_np, n_nodes, aggregate="max")
        segments = _segments_from_edges(coords, und_edges)

        np.savez(
            os.path.join(explain_dir, f"{gid}_explain.npz"),
            gid=gid,
            pred_scalar=pred_scalar,
            target_strain=float(strain_grid[target_index]),
            target_index=int(target_index),
            node_mask=node_score,
            edge_mask=edge_mask_np,
            edge_index=edge_index_np,
            x=x_np,
            edge_attr=(edge_attr.detach().cpu().numpy() if edge_attr is not None else None),
        )

        base = os.path.join(explain_dir, f"{gid}")
        node_path = f"{base}_node_importance.png"
        edge_path = f"{base}_edge_importance.png"
        comb_path = f"{base}_combined_importance.png"

        _draw_node_map(coords, segments, node_score, node_path)
        _draw_edge_map(coords, segments, edge_weights, edge_path)
        _draw_combined_map(coords, segments, node_score, edge_weights, comb_path)

        review_paths.append(comb_path)
        summary_rows.append({
            "gid": gid,
            "pred_SEA_J_per_kg": pred_scalar,
            "target_strain": float(strain_grid[target_index]),
            "n_nodes": int(n_nodes),
            "n_edges_undirected": int(len(und_edges)),
            "node_importance_min": float(np.min(node_score)) if node_score.size else 0.0,
            "node_importance_max": float(np.max(node_score)) if node_score.size else 0.0,
            "edge_importance_min": float(np.min(edge_weights)) if edge_weights.size else 0.0,
            "edge_importance_max": float(np.max(edge_weights)) if edge_weights.size else 0.0,
        })

        print(f"[{j+1}/{n}] Explained {gid}: pred SEA (J/kg) @ {strain_grid[target_index]:.3f} = {pred_scalar:.6f}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(explain_dir, "gnnexplainer_paper_palette_summary.csv"), index=False)

    if review_paths:
        import matplotlib.image as mpimg
        m = len(review_paths)
        cols = min(3, m)
        rows = int(math.ceil(m / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.0 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, pth in zip(axes.ravel(), review_paths):
            try:
                ax.imshow(mpimg.imread(pth))
            except Exception:
                pass
            ax.axis("off")
        fig.tight_layout(pad=0.03, w_pad=0.04, h_pad=0.04)
        save_png(fig, os.path.join(explain_dir, "gnnexplainer_review_sheet_combined.png"), dpi=600, transparent=False, pad=0.02)

def train_epoch(model, loader, optimizer, device, lambda_denom=0.0, eps=1e-8, max_grad_norm=1.0):
    model.train()
    total_loss, n_graphs = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred_all, pred_denom_log = model(batch)
        y_all = batch.y.view(batch.num_graphs, -1)
        denom_true = batch.denoms.view(batch.num_graphs, -1)
        denom_log_true = torch.log(torch.clamp(denom_true, min=eps))
        loss = F.mse_loss(pred_all, y_all) + lambda_denom * F.mse_loss(pred_denom_log, denom_log_true)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs
    return math.sqrt(total_loss / n_graphs) if n_graphs else float('nan')

def evaluate_with_total_loss(model, loader, device, C_curve, C_scalar, L, lambda_denom=0.0, eps=1e-8):
    model.eval()
    total_curve_loss, total_denom_loss, n_graphs = 0.0, 0.0, 0
    all_y_curves, all_pred_curves, all_y_scalar, all_pred_scalar = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_norm, pred_denom_log = model(batch)
            B = batch.num_graphs
            y_norm = batch.y.view(B, -1)
            total_curve_loss += F.mse_loss(pred_norm, y_norm, reduction='sum').item()
            denom_true = batch.denoms.view(B, -1)
            denom_log_true = torch.log(torch.clamp(denom_true, min=eps))
            total_denom_loss += F.mse_loss(pred_denom_log, denom_log_true, reduction='sum').item()
            n_graphs += B
            y_norm_np, denoms_true_np = y_norm.cpu().numpy(), denom_true.cpu().numpy()
            pred_norm_np, pred_denoms_np = pred_norm.cpu().numpy(), np.exp(pred_denom_log.cpu().numpy())
            curve_len = C_curve * L
            y_curves_norm = y_norm_np[:, :curve_len].reshape(B, C_curve, L)
            pred_curves_norm = pred_norm_np[:, :curve_len].reshape(B, C_curve, L)
            y_curves = y_curves_norm * denoms_true_np[:, :C_curve, None]
            pred_curves = pred_curves_norm * pred_denoms_np[:, :C_curve, None]
            y_scalar = y_norm_np[:, curve_len:] * denoms_true_np[:, C_curve:]
            pred_scalar = pred_norm_np[:, curve_len:] * pred_denoms_np[:, C_curve:]
            all_y_curves.append(y_curves)
            all_pred_curves.append(pred_curves)
            all_y_scalar.append(y_scalar)
            all_pred_scalar.append(pred_scalar)
    if n_graphs == 0:
        return None, float('inf')


    curve_output_count = C_curve * L + C_scalar
    denom_output_count = C_curve + C_scalar
    avg_curve_loss = total_curve_loss / (n_graphs * curve_output_count)
    avg_denom_loss = total_denom_loss / (n_graphs * denom_output_count)
    avg_total_loss = avg_curve_loss + lambda_denom * avg_denom_loss
    results = (np.vstack(all_y_curves), np.vstack(all_pred_curves), np.vstack(all_y_scalar), np.vstack(all_pred_scalar))
    return results, math.sqrt(avg_total_loss)


def get_seven_hex_features_from_json(json_path):

    from seven_descriptor_shap import extract_seven_descriptors
    return extract_seven_descriptors(json_path, is_inside_hexagon)

def plot_shap_analysis(X_test, Y_pred, X_feat_names, outdir,
                       title="Seven-descriptor linear-proxy SHAP", surrogate="linear", *,
                       sample_ids=None, fidelity_fraction=0.20, seed=42, predictor_path=None):

    from seven_descriptor_shap import FEATURE_NAMES, analyze_seven_descriptors
    if list(X_feat_names) != FEATURE_NAMES:
        raise ValueError("SHAP requires the original five descriptors plus two orientation fractions")
    if surrogate != "linear":
        raise ValueError("The revised manuscript uses a seven-descriptor linear proxy")
    return analyze_seven_descriptors(
        X_test, Y_pred, outdir, sample_ids=sample_ids, seed=seed,
        fidelity_fraction=fidelity_fraction, predictor_path=predictor_path)


def plot_training_history(history_df, outdir):

    epochs = history_df['epoch'].to_numpy(dtype=float)
    train_loss = history_df['train_total_rel_pct'].to_numpy(dtype=float)
    val_df = history_df.dropna(subset=['val_total_rel_pct'])
    val_epochs = val_df['epoch'].to_numpy(dtype=float)
    val_loss = val_df['val_total_rel_pct'].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(3.10, 2.15), dpi=600)

    ax.plot(epochs, train_loss, color=PAPER_BLUE, alpha=0.20, linewidth=0.65, zorder=1)
    ax.plot(epochs, smooth_array(train_loss, 0.74), color=PAPER_BLUE,
            linewidth=1.65, label="Training", zorder=3)

    if val_loss.size:
        ax.plot(val_epochs, val_loss, color=PAPER_ORANGE_LIGHT, alpha=0.20, linewidth=0.65, zorder=1)
        ax.plot(val_epochs, smooth_array(val_loss, 0.74), color=PAPER_ORANGE_LIGHT,
                linewidth=1.65, linestyle="--", label="Validation", zorder=3)

    finite_vals = np.concatenate([
        train_loss[np.isfinite(train_loss)],
        val_loss[np.isfinite(val_loss)] if val_loss.size else np.array([])
    ])
    if finite_vals.size:
        ymax = np.nanpercentile(finite_vals, 97) * 1.08
        ymin = max(0.0, np.nanpercentile(finite_vals, 2) * 0.75)
        if np.isfinite(ymax) and ymax > ymin:
            ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Epoch", fontsize=8.8)
    ax.set_ylabel("RMSE loss (%)", fontsize=8.8)
    ax.tick_params(axis="both", labelsize=7.6)
    style_axes(ax, full_box=True)
    ax.legend(loc="upper right", fontsize=7.4, handlelength=1.6, borderpad=0.2)
    fig.tight_layout(pad=0.22)

    save_png(fig, os.path.join(outdir, "training_loss_curve.png"), dpi=600, transparent=False)

def curve_value_at_strain(curves, strain_grid, target_strain=0.5):
    values = np.asarray(curves, dtype=float)
    grid = np.asarray(strain_grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or values.shape[-1] != len(grid):
        raise ValueError("Curve and strain-grid dimensions do not match")
    if not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
        raise ValueError("Strain grid must be finite and strictly increasing")
    if target_strain < grid[0] - 1e-8 or target_strain > grid[-1] + 1e-8:
        raise ValueError("Strain grid does not cover the requested endpoint")
    flat = values.reshape(-1, len(grid))
    return np.asarray([np.interp(target_strain, grid, row) for row in flat]).reshape(values.shape[:-1])


def plot_accuracy_scatter(pred_curve_dir, outdir):

    print("\n--- 生成预测精度散点图 / publication parity plot ---")
    pred_files = glob.glob(os.path.join(pred_curve_dir, '*.csv'))
    if not pred_files:
        return

    pairs = defaultdict(list)
    for file_path in pred_files:
        df = pd.read_csv(file_path)
        try:
            true_col = next(c for c in df.columns if c.lower().endswith('_true'))
            pred_col = next(c for c in df.columns if c.lower().endswith('_pred'))
            base_name = true_col[:-5] if true_col.lower().endswith('_true') else 'SEA'
            endpoint = curve_value_at_strain(
                df[[true_col, pred_col]].to_numpy(dtype=float).T,
                df['strain'].to_numpy(dtype=float), 0.5)
            pairs[base_name].append((float(endpoint[0]), float(endpoint[1])))
        except (StopIteration, IndexError, ValueError):
            continue

    for name, data_pairs in pairs.items():
        data = np.asarray(data_pairs, dtype=float)
        if data.shape[0] < 2:
            continue

        actual, pred = data[:, 0], data[:, 1]
        r2 = r2_score(actual, pred)
        rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))

        lim_min = float(min(np.nanmin(actual), np.nanmin(pred)))
        lim_max = float(max(np.nanmax(actual), np.nanmax(pred)))
        pad = (lim_max - lim_min) * 0.075 if lim_max > lim_min else max(abs(lim_max), 1.0) * 0.075
        lims = [lim_min - pad, lim_max + pad]

        fig, ax = plt.subplots(figsize=(2.55, 2.55), dpi=600)
        ax.scatter(
            actual, pred,
            s=9.0,
            color=PAPER_BLUE,
            alpha=0.72,
            edgecolors=WHITE,
            linewidths=0.20,
            zorder=3,
        )
        ax.plot(lims, lims, linestyle="--", color=GRAY_LINE, linewidth=0.95, zorder=2)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")

        display_name = "SEA (J/kg)" if "sea" in str(name).lower() else str(name).replace("_", " ")
        ax.set_xlabel(f"FEM {display_name} at 0.5 strain", fontsize=8.2)
        ax.set_ylabel(f"GNN {display_name} at 0.5 strain", fontsize=8.2)
        ax.tick_params(axis="both", labelsize=7.2)
        style_axes(ax, full_box=True)
        ax.text(
            0.055, 0.95,
            f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.1f}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=7.4,
            bbox=dict(boxstyle="round,pad=0.18", fc=WHITE, ec=LIGHT_GRAY, lw=0.65),
        )

        fig.tight_layout(pad=0.20)
        safe_name = sanitize_name(name)
        save_png(fig, os.path.join(outdir, f'accuracy_scatter_{safe_name}.png'), dpi=600, transparent=False)
        print(f"{display_name} parity plot (R²={r2:.3f}) saved.")

def plot_selected_test_samples(true_curves, pred_curves, strain_grid, graph_ids, outdir, num_samples=3):

    print("\n--- 生成代表性曲线拟合图 / publication representative curves ---")
    img_dir = os.path.join(outdir, 'sample_comparisons')
    os.makedirs(img_dir, exist_ok=True)

    n_total = len(graph_ids)
    if n_total == 0:
        return

    scores = np.asarray([np.nanmax(true_curves[i, 0, :]) for i in range(n_total)], dtype=float)
    order = np.argsort(scores)

    if n_total >= 3:
        selected_indices = [int(order[0]), int(order[n_total // 2]), int(order[-1])]
        labels = ["Low SEA", "Medium SEA", "High SEA"]
    elif n_total == 2:
        selected_indices = [int(order[0]), int(order[-1])]
        labels = ["Low SEA", "High SEA"]
    else:
        selected_indices = [int(order[0])]
        labels = ["Representative"]

    n_select = len(selected_indices)

    fig, axes = plt.subplots(1, n_select, figsize=(2.25 * n_select, 1.95), squeeze=False, dpi=600)
    axes = axes.ravel()
    selected_rows = []

    for k, idx in enumerate(selected_indices):
        gid = graph_ids[idx]
        y_true = true_curves[idx, 0, :]
        y_pred = pred_curves[idx, 0, :]

        ax = axes[k]
        ax.plot(strain_grid, y_true, color=DARK_TEXT, linewidth=1.50, label="FEM")
        ax.plot(strain_grid, y_pred, color=PAPER_BLUE, linewidth=1.50, linestyle="--", label="GNN")
        ax.set_xlabel("Strain", fontsize=8.2)
        if k == 0:
            ax.set_ylabel("SEA (J/kg)", fontsize=8.2)
        ax.tick_params(axis="both", labelsize=7.1)
        style_axes(ax, full_box=True)
        ax.text(0.045, 0.95, labels[k], transform=ax.transAxes,
                ha="left", va="top", fontsize=7.4, color=DARK_TEXT)
        if k == 0:
            ax.legend(loc="lower right", fontsize=6.8, handlelength=1.45)


        fig_i, ax_i = plt.subplots(figsize=(2.25, 1.95), dpi=600)
        ax_i.plot(strain_grid, y_true, color=DARK_TEXT, linewidth=1.55, label="FEM")
        ax_i.plot(strain_grid, y_pred, color=PAPER_BLUE, linewidth=1.55, linestyle="--", label="GNN")
        ax_i.set_xlabel("Strain", fontsize=8.5)
        ax_i.set_ylabel("SEA (J/kg)", fontsize=8.5)
        ax_i.tick_params(axis="both", labelsize=7.4)
        style_axes(ax_i, full_box=True)
        ax_i.legend(loc="lower right", fontsize=7.0, handlelength=1.5)
        fig_i.tight_layout(pad=0.18)
        save_png(
            fig_i,
            os.path.join(img_dir, f'{k+1}_{labels[k].lower().replace(" ", "_")}_{sanitize_name(gid)}_comparison.png'),
            dpi=600,
            transparent=False,
        )

        selected_rows.append({
            "graph_id": gid,
            "category": labels[k],
            "fem_max_sea": float(np.nanmax(y_true)),
        })

    fig.tight_layout(pad=0.12, w_pad=0.45)
    save_png(fig, os.path.join(outdir, "representative_curve_fits.png"), dpi=600, transparent=False)

    pd.DataFrame(selected_rows).to_csv(
        os.path.join(outdir, "selected_low_mid_high_samples.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"代表性曲线拟合图已保存至: {outdir}")

def run_experiment(args, experiment_name="single_run"):
    set_seed(args.seed)
    print(f"\n=======================================================")
    print(f"STARTING EXPERIMENT: {experiment_name}")
    print(f"=======================================================")
    os.makedirs(args.outdir, exist_ok=True)
    csv_files = glob.glob(os.path.join(args.data_dir, '*.csv'))
    max_strains = [np.max(s) for s, _, _, _, _, _, _, _ in (read_csv_new(f) for f in csv_files) if s is not None and len(s) > 0]

    max_s = max(max_strains) if max_strains else 1.0
    strain_grid = np.linspace(0, max_s, args.L)
    ds = GraphCurveDataset(args.graphs_dir, args.data_dir, strain_grid)
    C_curve, C_scalar, L = 1, 0, args.L
    Y_data = [ds.get(i).y_raw.numpy() for i in range(len(ds))]
    Y_curves = np.array([y[:C_curve * L].reshape(C_curve, L) for y in Y_data])
    denom = np.zeros((len(ds), C_curve + C_scalar))
    denom[:, :C_curve] = np.nanmean(np.abs(Y_curves), axis=2)
    denom[~np.isfinite(denom)] = 1.0; denom[denom <= 0] = 1.0
    if not bool(args.use_denom_head): denom[:] = 1.0

    from torch.utils.data import Dataset as TorchDataset
    class WrappedDataset(TorchDataset):
        def __init__(self, base_ds, denom_array):
            super().__init__()
            self.base_ds = base_ds
            self.denom_array = denom_array
        def __len__(self):
            return len(self.base_ds)
        def __getitem__(self, idx):
            data = self.base_ds[idx]
            d = self.denom_array[idx]
            y_raw = data.y_raw.numpy()
            curves = y_raw[:C_curve * L].reshape(C_curve, L)
            scalar = y_raw[C_curve * L:]
            norm_curves = (curves / d[:C_curve, None]).ravel()
            norm_scalar = scalar / d[C_curve:] if C_scalar > 0 else np.array([], dtype=np.float32)
            data.y = torch.tensor(np.concatenate([norm_curves, norm_scalar]), dtype=torch.float)
            data.denoms = torch.tensor(d, dtype=torch.float)
            return data

    wrapped_ds = WrappedDataset(ds, denom)
    train_idx, test_idx = train_test_split(list(range(len(ds))), test_size=args.test_size, random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=args.val_frac / (1 - args.test_size), random_state=args.seed)
    train_set, val_set, test_set = [torch.utils.data.Subset(wrapped_ds, i) for i in [train_idx, val_idx, test_idx]]
    g = torch.Generator()
    g.manual_seed(args.seed)
    loaders = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=lambda x: np.random.seed(args.seed + x), generator=g),
        'val': DataLoader(val_set, batch_size=args.batch_size, worker_init_fn=lambda x: np.random.seed(args.seed + x), generator=g),
        'test': DataLoader(test_set, batch_size=args.batch_size, worker_init_fn=lambda x: np.random.seed(args.seed + x), generator=g)
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_data = wrapped_ds[0]
    model = GINCurveRegressorWithDenom(
        in_channels=sample_data.x.shape[1], out_dim=C_curve * L + C_scalar, denom_channels=C_curve + C_scalar,
        hidden=args.hidden, num_layers=args.num_layers, dropout=args.dropout,
        num_global_features=int(sample_data.global_feat.shape[0]) if hasattr(sample_data, "global_feat") else 0,
        use_global_feat=bool(args.use_global_feat), pooling_type=args.pooling_type,
        use_denom_head=bool(args.use_denom_head), use_edge_feat=bool(args.use_edge_feat),
        decoder_type=args.decoder_type, seq_len=L, out_channels_curve=C_curve
    ).to(device)

    model.num_global_features = int(sample_data.global_feat.shape[0]) if hasattr(sample_data, "global_feat") else 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, best_val, best_state_path = [], float('inf'), os.path.join(args.outdir, 'best_model.pth')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, loaders['train'], optimizer, device, args.lambda_denom, max_grad_norm=1.0)
        _, val_total_loss_rmse = evaluate_with_total_loss(model, loaders['val'], device, C_curve, C_scalar, L, args.lambda_denom)
        if val_total_loss_rmse < best_val:
            best_val = val_total_loss_rmse
            torch.save(model.state_dict(), best_state_path)
        print(f"Epoch {epoch:03d} | Train RMSE: {train_loss*100:.2f}% | Val RMSE: {val_total_loss_rmse*100:.2f}% | Time: {time.time()-t0:.1f}s")
        history.append({'epoch': epoch, 'train_total_rel_pct': train_loss*100, 'val_total_rel_pct': val_total_loss_rmse*100})

    if os.path.exists(best_state_path): model.load_state_dict(torch.load(best_state_path, map_location=device))
    test_results, test_total_loss = evaluate_with_total_loss(model, loaders['test'], device, C_curve, C_scalar, L, args.lambda_denom)

    pred_curve_dir = os.path.join(args.outdir, 'pred_curves')
    os.makedirs(pred_curve_dir, exist_ok=True)
    test_r2, test_mae_abs, test_rmse_abs, test_end_rel_err_pct = 0.0, float('nan'), float('nan'), float('nan')

    if test_results:
        test_curves_true, test_curves_pred, _, _ = test_results
        test_ids = [ds.get(i).graph_id for i in test_idx]
        true_endpoints = curve_value_at_strain(test_curves_true[:, 0, :], strain_grid, 0.5)
        pred_endpoints = curve_value_at_strain(test_curves_pred[:, 0, :], strain_grid, 0.5)
        if len(true_endpoints) > 1: test_r2 = r2_score(true_endpoints, pred_endpoints)

        for i, gid in enumerate(test_ids):
            pd.DataFrame({'strain': strain_grid, "SEA(J/kg)_true": test_curves_true[i, 0, :], "SEA(J/kg)_pred": test_curves_pred[i, 0, :]}).to_csv(os.path.join(pred_curve_dir, f'{gid}.csv'), index=False)

        if test_ids:
            print("\n--- 正在为 SHAP 分析提取单晶格特征 ---")
            X_test_single_hex = []
            Y_test_single_hex = []
            shap_sample_ids = []
            valid_feat_names = ['N_nodes', 'E_edges', 'Mean_Edge_Len', 'Std_Edge_Len', 'Sparsity', 'Orientation_fraction_0_60deg', 'Orientation_fraction_60_120deg']

            for i_enum, idx_ds in enumerate(test_idx):
                gid = test_ids[i_enum]
                json_path = os.path.join(args.graphs_dir, f"{gid}.json")
                if os.path.exists(json_path):
                    feats = get_seven_hex_features_from_json(json_path)
                    if feats is not None:
                        X_test_single_hex.append(feats)
                        Y_test_single_hex.append(test_curves_pred[i_enum, 0, L - 1])
                        shap_sample_ids.append(gid)
                    else:
                        print(f"跳过 {gid} 的 SHAP 分析，因为未成功提取出完整单晶格。")

            if len(X_test_single_hex) > 0:
                X_test_single_hex = np.array(X_test_single_hex)
                Y_test_single_hex = np.array(Y_test_single_hex)
                plot_shap_analysis(X_test_single_hex, Y_test_single_hex, valid_feat_names,
                                   args.outdir, title="Seven-descriptor linear-proxy SHAP",
                                   surrogate=getattr(args,"shap_surrogate","linear"),
                                   sample_ids=shap_sample_ids,
                                   fidelity_fraction=getattr(args,"shap_fidelity_fraction",0.20),
                                   seed=args.seed,predictor_path=best_state_path)
            else:
                print("未能提取出任何单晶格特征，SHAP 分析跳过。")

        plot_selected_test_samples(test_curves_true, test_curves_pred, strain_grid, test_ids, args.outdir, num_samples=3)
        plot_accuracy_scatter(pred_curve_dir, args.outdir)
        abs_metrics = compute_absolute_curve_metrics(test_curves_true, test_curves_pred)
        test_mae_abs, test_rmse_abs, test_end_rel_err_pct = abs_metrics['mae_abs'], abs_metrics['rmse_abs'], abs_metrics['end_rel_err_pct']

        if int(args.run_explainer) == 1:
            run_gnnexplainer_on_test(model, wrapped_ds, ds, test_idx, strain_grid, args, device)

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(args.outdir, 'metrics.csv'), index=False, encoding='utf-8-sig')
    plot_training_history(history_df, args.outdir)
    print(f"Finished: {experiment_name}. Best Val: {best_val:.3f}, Test R2: {test_r2:.2f}")
    return {
        'name': experiment_name, 'params': {k: getattr(args, k) for k in TUNING_GRID.keys()},
        'best_val_rmse': best_val, 'test_rmse_norm': test_total_loss, 'test_r2': test_r2,
    }

def run_tuning_mode(base_args):
    keys, values = zip(*TUNING_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    tuning_results = []
    base_outdir = base_args.outdir
    for i, params in enumerate(combinations):
        current_args = copy.deepcopy(base_args)
        exp_name = "_".join([f"Edge_{'True' if v else 'False'}" if k == 'use_edge_feat' else f"Dec_{v.upper()}" if k == 'decoder_type' else f"{k}_{v}" for k, v in params.items()])
        for k, v in params.items(): setattr(current_args, k, v)
        current_args.outdir = os.path.join(base_outdir, exp_name)
        try: tuning_results.append(run_experiment(current_args, experiment_name=exp_name))
        except Exception as e: print(f"!!! Experiment failed: {e}")
    if tuning_results:
        summary_df = pd.DataFrame([ {**r['params'], 'best_val_rmse': r['best_val_rmse'], 'test_r2': r['test_r2']} for r in tuning_results ]).sort_values(by='test_r2', ascending=False)
        summary_df.to_csv(os.path.join(base_outdir, 'tuning_summary.csv'), index=False)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, default='single', choices=['single', 'tuning'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--graphs_dir', default='graph')
    p.add_argument('--data_dir', default='data')
    p.add_argument('--outdir', default='results')
    p.add_argument('--L', type=int, default=100)
    p.add_argument('--epochs', type=int, default=180)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--num_layers', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--test_size', type=float, default=0.15)
    p.add_argument('--val_frac', type=float, default=0.15)
    p.add_argument('--eval_freq', type=int, default=1)
    p.add_argument('--lambda_denom', type=float, default=1.0)
    p.add_argument('--pooling_type', type=str, default='dual', choices=['mean', 'dual'])
    p.add_argument('--use_global_feat', type=int, default=1, choices=[0, 1])
    p.add_argument('--use_denom_head', type=int, default=1, choices=[0, 1])
    p.add_argument('--use_edge_feat', type=int, default=1, choices=[0, 1])
    p.add_argument('--decoder_type', type=str, default='cnn', choices=['mlp', 'cnn'])
    p.add_argument('--run_explainer', type=int, default=1, choices=[0, 1])
    p.add_argument('--explain_num_graphs', type=int, default=5)
    p.add_argument('--explain_target_strain', type=float, default=0.5)
    p.add_argument('--explainer_epochs', type=int, default=200)
    p.add_argument('--explainer_topk_edges', type=int, default=30)
    p.add_argument('--shap_surrogate',default='linear',choices=['linear'])
    p.add_argument('--shap_fidelity_fraction',type=float,default=0.20,
                   help='Independent descriptor-surrogate holdout fraction; no GNN retraining needed for post-hoc use.')
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == 'tuning': run_tuning_mode(args)
    else: run_experiment(args, experiment_name="single_run")

if __name__ == "__main__": main()
