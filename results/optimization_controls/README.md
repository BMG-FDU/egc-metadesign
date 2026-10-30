# Optimization-control figure data / 优化对照绘图数据

The files correspond to Figs. S17–S18 and Tables S13–S14 in the revised Supplementary Information. SEA values are in J/kg; SD denotes sample standard deviation (ddof=1). Source paths and file hashes are recorded in `sources.json`.

以下文件对应修订版SI中的Figs. S17–S18及Tables S13–S14。SEA单位为J/kg，SD为样本标准差（ddof=1）；源路径和文件哈希见`sources.json`。

| Folder / 文件夹 | File / 文件 | Contents / 内容 |
| --- | --- | --- |
| Fig_S17 | training_curves.csv | Panel (a), plotted means and SD / (a)训练曲线均值与SD |
| Fig_S17 | best_so_far_curves.csv | Panel (b), plotted means and SD / (b)历史最优曲线均值与SD |
| Fig_S17 | per_seed_results.csv | Values from three runs used in the table / 表格采用的三次运行数值 |
| Fig_S17 | table_summary.csv | Table S13: four-condition statistics / Table S13四组统计 |
| Fig_S18 | individual_curves.csv | Three runs per condition at plotted call counts / 各组三次运行在绘图调用数处的数值 |
| Fig_S18 | mean_SD_curves.csv | Four plotted mean and SD curves / 四组绘图均值与SD |
| Fig_S18 | table_summary.csv | Table S14: final best SEA statistics / Table S14最终最优SEA统计 |

Fig. S17 uses four conditions: 850/1000 reference values with clipping on/off. Panel (a) uses the preceding 5,000-call averaging window sampled every 500 calls; panel (b) is unsmoothed. The late-window table statistic is the mean episode-best SEA for episode-end call counts in (90,000, 100,000].

Fig. S17采用850/1000参考值×截顶开关四种设置。(a)按此前5,000次调用窗口平均、每500次调用采样；(b)不平滑。表格末窗统计为回合结束调用数位于(90,000, 100,000]内的回合最优SEA均值。

Fig. S18 uses three runs per condition. Curves show best-so-far SEA from 1,000 to 100,000 evaluations; the figure uses a logarithmic vertical axis.

Fig. S18各组采用三次运行，曲线为1,000至100,000次评价区间的历史最优SEA，图中纵轴为对数轴。
