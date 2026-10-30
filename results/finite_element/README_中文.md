# Figs. S3–S7 有限元补充材料

保留实际图表数据、必要提取/计算脚本及代表性输入；未包含 ODB、失败任务或弃用的0.49敏感性系列。输入保持历史求解设置，不统一改写为新设置。S4/S5采用材料泊松比0.475，基准梁截面泊松比0、敏感性截面泊松比0.475。S7历史计算设置见 source_settings.json。

|目录|内容|
|---|---|
|Fig_S3|三种结构×四档网格，共12个B31输入、响应曲线和网格统计|
|Fig_S4|五种结构的名义梁、无接触/有接触实体输入，以及体积匹配截面0.475梁输入；响应、接触和能量比数据|
|Fig_S5|P1–P3基准/边界条件/截面0.475输入；局部应变及敏感性数据|
|Fig_S6|80个结构的载荷与应变能统计；每族一个代表结构及其载荷、能量输出输入；原提取脚本|
|Fig_S7|实际用于分解图的152条响应、加载功/名义质量/SEA数据，五个占比的代表性输入|

## 使用已有数据重新计算/绘图

在本文件夹执行（普通Python，依赖numpy、pandas、matplotlib）：

```powershell
python reproduce_summary.py S3
python Fig_S4/plot_0475.py
python Fig_S5/plot_0475.py
python Fig_S6/plot_fig_s6.py
python reproduce_summary.py S7
```

S3/S7输出位于各自reproduced目录，S4/S5位于output目录，S6沿用脚本中的输出目录。附带图用于核对数据；论文最终排图以SI为准。

## 按需重算一个输入（不会批量启动）

```powershell
python run_input.py Fig_S4/jobs/P1_no_contact.inp --abaqus abaqus --cpus 4
```

Abaqus可执行程序不在PATH时，用`--abaqus`指定本机abaqus.bat。每条命令等待当前任务结束。输入保持原分析步骤；本次整理未重新求解。

S4/S5基准和边界输入的ODB留在各自jobs目录后，执行：

```powershell
abaqus python Fig_S4/extract_results.py
abaqus python Fig_S5/extract_results.py
```

截面0.475输入的ODB留在各自section0475/jobs目录，然后在相应section0475目录执行`abaqus python extract.py .`。曲线由RF2–U2积分，不使用响应校正系数。

任意代表输入可按extraction_metadata.csv的高度和名义质量提取：

```powershell
abaqus python extract_curve.py path/to/model.odb --height 100 --mass-kg <该模型质量> --output extracted.csv
```

S6重新生成一个代表算例并提取场量：先切换到一个独立输出目录，再以绝对路径运行 `abaqus cae noGUI=<本目录>/Fig_S6/shared/abaqus_worker.py -- <本目录>/Fig_S6/02_Load_path/config.json DL_01`；能量任务将配置路径替换为03_Energy_distribution/config.json。另三族ID见representative_samples.csv。该入口沿用原建模与SF/ELSE提取流程。

脚本运行及数值汇总已核对，未重新运行Abaqus。
