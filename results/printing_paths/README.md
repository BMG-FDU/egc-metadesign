# Route comparison and repeated printing-time measurements / 路径比较与重复打印时间

## Route comparison / 寻路比较

`route_comparison/route_algorithms.py` retains the original Hierholzer (Basic), Momentum and Look-ahead functions from the historical Fig. 2(d) script. Look-ahead weight is 0.5. `graph/pattern8_single_5x3_0.json` is its historical example input.

从历史 Fig. 2(d) 脚本提取原有 Hierholzer（Basic）、Momentum、Look-ahead 函数，算法未改。Look-ahead 权重为0.5。附当时的示例结构。基础算法按邻接表顺序选择边，未新增随机抽样步骤，保留其实际名称。

From the repository root / 在代码包根目录运行：

```powershell
python results/printing_paths/route_comparison/run_comparison.py
```

Outputs in `route_comparison/output/`: `paths.json` (node sequences), `turning_angles.csv` (individual turns), and `route_comparison.csv` (summary). Each route is checked against every input edge instance, including parallel edges. Angles include every internal transition in the returned sequence, including auxiliary-edge transitions; the closing end-to-start seam is not added separately. Coordinates are normalized for angle calculations; no physical path length or deposition commands are inferred. These outputs reproduce traversal of the supplied historical example, rather than substitute for the manuscript's aggregate plotting data.

输出含三种方法的节点路径、逐转角数据及汇总，并核查每条输入边（含平行边）恰好覆盖一次。转角包括返回路径中各相邻边之间的转向（含辅助边），不另加闭合首尾转角。坐标归一化仅用于角度计算；不推算实际打印长度或沉积命令。输出是所附历史示例的运行结果，不替换论文已有统计图的数据。

## Repeated printing time / 三次重复打印时间

`printing_time/data/turning_time_replicates_raw.csv` preserves the original measurements: 18 angles (0–170°, 10° increments), three trials per angle, 54 measurements. Columns: angle, trial 1, trial 2, trial 3, original mean. The script checks that the recomputed means agree with the recorded means and uses sample SD (`ddof=1`).

原始CSV保留18个转角、每个转角3次、共54次记录；列为转角、三次时间、原记录均值。脚本核对均值，并计算样本标准差，不剔除数据。

```powershell
python results/printing_paths/printing_time/process_turning_time.py
```

`printing_time/output/` contains `turning_time_statistics.csv` and the reproduced PNG/PDF/SVG figure. The historical plotting style is retained.

输出为逐次时间、均值和标准差统计表，以及原绘图样式的PNG/PDF/SVG图件。

Dependencies / 依赖：`numpy`, `matplotlib` (see `requirements.txt`). Original paths, SHA-256 hashes and packaging changes are recorded in `sources.json`.

## Deposition commands / 沉积命令

Saved representative G-code, excerpts and export entry: [instructions](gcode_commands/README.md).
代表性G-code、命令节选及导出入口见上述说明。
