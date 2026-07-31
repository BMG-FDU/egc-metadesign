# EGC-MetaDesign

> Eulerian Graph Constraints Enable Manufacturing-Aware Design of Mechanical Metamaterials

---

A computational pipeline for **Eulerian-graph-constrained inverse design of continuously printable mechanical metamaterials**. Graph-theoretic printability constraints are embedded directly into topology generation, ensuring every candidate is printable in a single continuous toolpath before any simulation or optimization. A GNN surrogate is then trained on finite-element data and coupled with curriculum reinforcement learning to discover high-SEA lattice architectures.

基于欧拉图约束的连续可打印力学超材料逆向设计计算流程。将图论可打印性约束嵌入拓扑生成环节，保证所有候选构型在仿真和优化前即满足单次连续打印条件；在此基础上训练 GNN 代理模型并结合课程强化学习搜索高吸能 lattice 构型。

---

## Pipeline

```
Eulerian Topology Generation
        │
        ▼
Array Assembly & Eulerianization
        │
        ▼
FEA Simulation (Abaqus, B31 beams, Yeoh TPU)
        │
        ▼
GNN Surrogate (GINEConv + CNN decoder)
        │
        ▼
Curriculum RL (Maskable PPO: LOOP → LINK → NODE)
        │
        ▼
FEA Reanalysis & Experiment
```

## Files

| File | Description |
|---|---|
| `SIMULATION.py` | Abaqus FEA wrapper: job generation, submission, SEA–strain extraction |
| `GNN.py` | GNN surrogate: GINEConv encoder + CNN decoder, SEA–strain curve prediction |
| `lattice_env.py` | RL environment (Gymnasium): topology editing actions + GNN reward |
| `lattice_state.py` | State encoding: graph → RL policy features |
| `train_ppo.py` | PPO training with 3-stage curriculum and action masking |
| `best_structure_tracker.py` | Best-structure logging during RL training |
| `compare_all_methods.py` | Method benchmarking: PPO vs. random/hill-climbing/greedy |

## Quick Start

```bash
# 1. Generate FEA training data
python SIMULATION.py

# 2. Train GNN surrogate
python GNN.py

# 3. Run curriculum RL inverse design
python train_ppo.py

# 4. Compare optimization methods
python compare_all_methods.py
```

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0, PyTorch Geometric ≥ 2.5
- NetworkX, NumPy
- Abaqus 2024 (FEA only)
- Stable-Baselines3, Gymnasium

## Dataset & Models

**Training data**: 6,000 Double-loops Eulerian arrays with FEA-computed SEA–strain curves (0–50% tensile strain). Split 70/15/15 (seed 42).

| | Specification |
|---|---|
| Material | TPU-95A, 3rd-order Yeoh hyperelastic |
| Element | B31 beam, 1×1 mm² uniform section |
| GNN architecture | 8× GINEConv (dim=256) → dual pooling → 1D CNN decoder + scale head |
| GNN test R² | 0.98 |
| RL algorithm | Maskable PPO, 152 actions, 3-stage curriculum |
| RL reward | GNN-predicted SEA @ 50% strain + potential-based shaping |

## Citation

```bibtex
@article{ren2025eulerian,
  title   = {Eulerian graph constraints enable manufacturing-aware design of mechanical metamaterials},
  author  = {Ren, Li and Yang, Yunhao and Ren, Jing and Chen, Xin and Shao, Zhengzhong and Yao, Jingrong and Ling, Shengjie},
  year    = {2025}
}
```

## License

MIT
