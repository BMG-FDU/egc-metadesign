# Additional validation data

`cross_family/` contains the three reported external families (100 samples each), their structure JSON files, original FEA response CSVs, saved GNN curve/endpoint predictions and metrics. `sample_manifest.csv` maps each sample to its packaged files and hashes. Corresponds to Fig. S16 and Table S12.

`expanded_numerical/` contains 37 low-performance and 36 intermediate-performance valid samples: structure JSON files, original FEA SEA curves, matched FEA/GNN curves and endpoint results. `curve_plot_data.csv`, `scatter_plot_data.csv` and `plot_samples.csv` are the ten displayed examples in Fig. S9. Overall accuracy is calculated from all 73 valid samples, not just the displayed examples.

From the repository root:

```powershell
python results/validation/summarize.py
python results/validation/cross_family/plot.py
python results/validation/expanded_numerical/plot.py
```

These commands summarize and plot archived predictions; they do not retrain the GNN or run Abaqus. Dependencies: numpy, pandas, scikit-learn, matplotlib. Existing repository model weights are not duplicated here. Historical source locations are recorded in `source_manifest.json`.

## 中文说明

`cross_family/`：三个外部结构族，各100个；含结构JSON、原FEA响应、已保存的GNN曲线与端点预测，以及样本对应清单。对应Fig. S16和Table S12。

`expanded_numerical/`：37个低性能、36个中等性能有效样本；含结构JSON、原FEA的SEA曲线、配对预测曲线和端点数据。作图文件对应Fig. S9展示的十个示例。全组精度使用全部73个有效样本计算。

上述命令仅汇总、绘制已有结果，不重新训练或仿真；不重复放置权重、未使用候选结构或大型求解文件。
