# SHAP inputs and reported results / SHAP输入及已报告结果

Used for Fig. 5(f) and Fig. S10. / 对应Fig. 5(f)和Fig. S10。

| File / 文件 | Content / 内容 |
| --- | --- |
| descriptor_inputs_and_splits.csv | 885 sample IDs, seven descriptors, saved GNN endpoint predictions and fixed 566/142/177 split / 885个样本、七描述符、保存的GNN端点预测及固定划分 |
| proxy_parameters.csv | Five- and seven-descriptor linear-proxy parameters, training means/scales and intercepts / 五与七描述符线性代理参数、拟合均值与尺度、截距 |
| fidelity_metrics.csv | Fitting, validation and independent holdout metrics for both proxies / 两种代理的拟合、验证及独立留出指标 |
| proxy_holdout_predictions.csv | Same 177 holdout samples for Fig. S10(a,b), both proxies / 两种代理在同一177个留出样本上的预测 |
| seven_descriptor_SHAP_values.csv | Exact seven-descriptor linear SHAP values for 177 holdout samples / 177个留出样本的七描述符精确线性SHAP值 |
| Fig5f_beeswarm_plot_data.csv | 1,239 plotted points with descriptor, SHAP, position and colour / 1,239个绘图点的描述符、SHAP、位置及颜色 |
| seven_descriptor_importance.csv | Mean absolute SHAP contributions / SHAP平均绝对贡献 |
| FigS10c_fit_correlation.csv | Seven-descriptor Pearson matrix from the 566 fitting samples / 566个拟合样本的七描述符Pearson相关矩阵 |
| source_manifest.json | Original sources, hashes and saved-prediction model hash / 原始来源、哈希及保存预测所对应的模型哈希 |

From the package root / 在代码包根目录运行：

```powershell
python run_descriptor_shap.py --input-table results/shap/descriptor_inputs_and_splits.csv --outdir outputs/shap --no-plots
```

This uses the saved GNN predictions and fixed `proxy_split` assignments; it does not retrain or load a GNN. Omit `--no-plots` to additionally generate plots. The five-descriptor baseline is retained because it is plotted in Fig. S10(a). `Sparsity` denotes graph sparsity, 1 - 2E/[N(N-1)].

命令直接使用保存的GNN预测与`proxy_split`固定划分，不重新训练或加载GNN。去掉`--no-plots`可同时出图。五描述符基线保留用于Fig. S10(a)。`Sparsity`表示图稀疏度1 - 2E/[N(N-1)]。

No additional GNN checkpoint is included here. The packaged historical predictor weights remain at `results/best_model.pth`. The original checkpoint hash associated with the cached predictions is retained in `source_manifest.json`.

本目录不增加GNN权重；代码包提供的历史预测器权重仍位于`results/best_model.pth`。缓存预测的原始模型哈希保留在`source_manifest.json`中。
