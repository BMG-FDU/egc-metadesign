# Structural checks (Fig. S8; R1-8 and R2-3)

The saved primary table contains 885 test structures and their nearest neighbors among 4,129 training structures. It retains the original saved endpoint FEA and GNN predictions. No checkpoint is loaded and no model is trained by these scripts.

- `nearest_neighbor_885.csv`: primary 2% spatial-scale results.
- `three_equal_count_groups.csv`: three groups of 295 samples and their MAEs.
- `tolerance_sensitivity_summary.csv`: 1%, 2% and 4% spatial-scale statistics.
- `panel_a_histogram_data.csv`, `panel_b_plot_data.csv`: Fig. S8 plotting data.
- `duplicate_check_result.json`: saved cross-split exact and centered geometry/connectivity checks.
- `source_manifest.json`: historical file provenance.

From the package root:

```bash
python results/structural_similarity/check_physical_duplicates.py
python results/structural_similarity/run_analysis.py --sensitivity
```

The nearest-neighbor command recomputes the geometry comparison from the packaged JSON files and fixed split manifest; this is the longer computation. Outputs go to `results/structural_similarity/outputs/`. The common honeycomb frame is excluded; loading-relative orientation is retained. The duplicate check ignores node IDs, node order, edge order and translation, with coordinates rounded to 1e-6 and 1e-5.

本目录仅收录已报告的去重结果、Fig. S8作图数据及尺度敏感性统计。命令从代码包根目录运行，使用包内结构与固定划分，无需历史工作目录；预测值沿用已保存的原始结果，不重新训练模型。
