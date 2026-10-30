# IJMS代码与数据（README中文对照）

本文件与英文README的结构、文件路径及运行命令对应。

## 文件说明

- `hexagonal_generator.py`及其支持模块：晶格生成与欧拉路线构造。
- `GNN.py`：GNN训练、预测与可解释性分析。
- `train_ppo.py`、`compare_all_methods.py`及其支持模块：逆向设计与算法比较。
- `SIMULATION.py`：Abaqus/Explicit动态显式仿真，分析步时长为0.05 s，采用平滑阶跃加载幅值。
- `seven_descriptor_shap.py`及`run_descriptor_shap.py`：七描述符线性代理模型的SHAP分析。
- `graph/`和`data/`：5,899组配对的结构JSON文件与响应CSV文件。
- `splits/data_split_manifest.csv`：训练集（4,129）、验证集（885）与测试集（885）的样本划分，以及相对文件路径和文件哈希。
- `results/best_model.pth`：恢复的原始历史GNN预测模型；其SHA-256与已保存的优化日志一致。
- `results/retrained_full/best_model.pth`：重训Full模型（端点R² = 0.9774），对应Fig. 5b；训练和测试记录位于`results/retraining_records/`。
- `configs/model_checkpoints.json`：两套模型的名称、权重路径及SHA-256哈希。
- `abaqus_inputs/`：用于Fig. S5基准曲线的三个历史B31 Abaqus/Explicit输入文件（`P1_nominal.inp`至`P3_nominal.inp`），分析步时长为0.05 s，目标工程应变为0.50；`input_manifest.json`记录其来源。
- `configs/gnn_training_record.json`：历史GNN重新训练的参数记录；本次重新训练的配置与其训练记录一同保存。
- `requirements.txt`和`configs/current_environment.json`：本次报告的依赖版本；`configs/recorded_environment.json`保留此前的环境记录。
- `configs/ppo_current_defaults.json`和`environment_current_defaults.json`：当前代码中的PPO及环境参数。

## 安装环境

所报告的Python环境为Python 3.11.0、PyTorch 2.8.0+cu129。完整版本记录见`configs/current_environment.json`；`configs/recorded_environment.json`保留此前的环境记录。从代码包根目录安装：

```powershell
conda create -n ijms python=3.11.0 -y
conda activate ijms
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements.txt
```

CUDA 12.9安装命令参照[PyTorch 2.8官方说明](https://pytorch.org/get-started/previous-versions/#v280)。有限元任务另需安装Abaqus 2024；Abaqus结果提取脚本使用其自带Python运行。依赖文件按本次提供的实际版本填写，本次整理未重新安装环境测试。

## 运行代码

在安装所需依赖的Python环境中，从本目录运行以下命令。Abaqus仿真需要Abaqus运行环境及有效许可证。

生成每条边仅表示一次的物理结构：

```powershell
python hexagonal_generator.py --purpose mechanics --points-per-edge 10 --seed 42 --convert-to-inp --output-dir generated_mechanics
```

生成欧拉路线图：

```powershell
python hexagonal_generator.py --purpose path --points-per-edge 10 --seed 42 --output-dir generated_path
```

训练GNN，同时保留已提供的模型权重：

```powershell
python GNN.py --graphs_dir graph --data_dir data --outdir outputs/gnn_retraining --seed 42 --epochs 180 --batch_size 8 --use_global_feat 1 --run_explainer 0
```

训练使用代码中实现的随机划分流程，设置`seed=42`；提供划分清单以供核对。其余参数见配置文件及命令行帮助。

使用所提供的模型运行四算法比较：

```powershell
python compare_all_methods.py --gnn_model results/best_model.pth --gnn_budget 100000 --seed 42 --ppo_variants loop_link_node --save_dir outputs/comparison
```

该命令使用`results/best_model.pth`中的历史GNN预测模型，输出保存到`outputs/comparison`。

末段窗口均值的定义为：选取回合结束时累计GNN调用数落在评价预算最后10%区间内的回合，计算这些回合最优SEA值的算术平均。

准备独立的仿真工作目录，将`SIMULATION.py`复制至该目录，并将待分析的INP模型放入其`graph/`子文件夹。在该工作目录中运行：

```powershell
abaqus cae noGUI=SIMULATION.py
```

七描述符分析的独立入口：

```powershell
python run_descriptor_shap.py --input-table results/shap/descriptor_inputs_and_splits.csv --outdir outputs/shap
```

加载保存的GNN预测模型，在固定测试集上进行评估，无需重新训练：

```powershell
python evaluate_gnn.py
```

输出保存到`outputs/original_full_evaluation`，包括全曲线与0.5应变端点指标、曲线性质检查及逐样本结果。默认加载包内所提供的历史GNN模型权重。

单独评估重训模型：

```powershell
python evaluate_gnn.py --checkpoint results/retrained_full/best_model.pth --outdir outputs/retrained_full_evaluation
```

## 已保存的五种子优化记录

`results/optimization_logs/`收录PPO、Greedy、Random和Hill Climbing在种子19、37、42、48、57下的运行记录，每次运行包含100,000次GNN评价。每组包含`evaluations.csv`、`episode_ends.csv`和原始运行元数据；清单保留源文件哈希及原运行记录的模型哈希。

```powershell
python summarize_optimization_logs.py
```

脚本校验已有日志，在`results/optimization_logs/summary/`输出逐次运行及五种子汇总：首次严格超过750 J/kg的评价次数、最终最优SEA，以及回合结束调用数满足`90000 < count <= 100000`的回合最优SEA均值。跨种子标准差采用`ddof=1`。汇总对象为这里保存的五种子运行。

SI中的Figs. S17–S18及Tables S13–S14采用的优化对照数据位于`results/optimization_controls/`，文件说明见该目录的中英对照README。

SHAP输入、固定代理划分、拟合参数、保真度结果及Fig. 5(f)/Fig. S10绘图数据位于`results/shap/`，见其中的中英对照README。

## 结构相似度与物理图去重

脚本、Fig. S8数据及运行命令见`results/structural_similarity/README.md`。

新增验证的结构、FEA响应、已保存的GNN预测及脚本见 `results/validation/README.md`。

Figs. S3–S7有限元补充材料与运行方式：[说明](results/finite_element/README_中文.md)。

三种寻路策略比较脚本及三次重复打印时间数据：[运行说明](results/printing_paths/README.md)。

实验数据另见上传附件 `Data_S1.xlsx`：`FigS2_TPU_Raw`、`FigS2_FullMaterialCurves` 为材料试验及拟合数据；`FigS13_TOP1_experiment` 至 `FigS13_TOP5_experiment` 为晶格实验均值和标准差曲线；`Fig6K_ExperimentData`、`TableS9_CandidateMass` 为质量和性能汇总。

## 复现索引

除有限元说明中要求切换到相应目录的命令外，均从代码包根目录运行。详细参数保留在前述对应说明中。汇总已有结果不需要重新训练或提交Abaqus任务。

| 任务及审稿意见 | 输入 | 命令或说明 | 输出／对应图表 |
| --- | --- | --- | --- |
| 拓扑及路线生成；R1-2、R1-12、R2-1 | 生成参数、`boundary_template_5x3.json` | 上方mechanics/path生成命令 | `generated_mechanics/`、`generated_path/`；Fig. 1 |
| GNN训练；R1-8 | `graph/`、`data/`及`configs/`参数 | 上方GNN训练命令 | `outputs/gnn_retraining/`；Fig. 5b已有记录：`results/retraining_records/` |
| 固定测试集评估；R1-7、R1-8 | `results/best_model.pth`、`splits/data_split_manifest.csv`、`graph/`、`data/` | `python evaluate_gnn.py` | `outputs/original_full_evaluation/`；全曲线与端点指标 |
| 优化运行；R1-9、R2-4 | 历史预测器及拓扑编辑器 | 上方四算法运行命令 | `outputs/comparison/`；新运行与已有记录分开保存 |
| 已有五种子统计；R1-9、R2-4 | `results/optimization_logs/` | `python summarize_optimization_logs.py` | `results/optimization_logs/summary/`；Fig. 6i,j及Fig. S12 |
| 优化对照；R1-9 | `results/optimization_controls/` | [数据说明](results/optimization_controls/README.md) | Figs. S17–S18、Tables S13–S14；已保存绘图曲线与汇总 |
| 七描述符SHAP；R1-10、R2次要3 | `results/shap/descriptor_inputs_and_splits.csv` | `python run_descriptor_shap.py --input-table results/shap/descriptor_inputs_and_splits.csv --outdir outputs/shap` | `outputs/shap/`；Fig. 5f及Fig. S10 |
| 去重与最近邻分析；R1-8、R2-3 | 包内结构、固定划分及已保存预测 | [运行命令](results/structural_similarity/README.md) | `results/structural_similarity/outputs/`；Fig. S8 |
| 跨家族与扩展数值验证；R2-3、R2-5 | `results/validation/cross_family/`、`expanded_numerical/` | [汇总及绘图命令](results/validation/README.md) | 各目录中的汇总与图件；Figs. S9、S16及Table S12 |
| 有限元补充分析；R1-4、R1-6、R2-2 | `results/finite_element/` | [仿真、提取及绘图命令](results/finite_element/README_中文.md) | Figs. S3–S7；输出位置见对应说明 |
| 路径比较、打印时间及G-code；R1-2、R1-11、R2-1、R2-6 | 历史示例结构、时间记录及路径步骤表 | [路径与时间命令](results/printing_paths/README.md)；[G-code导出](results/printing_paths/gcode_commands/README.md) | 路径和转角汇总、重复打印时间图、代表性G-code |
| 实验数据；R1-5、R2次要2 | `Data_S1.xlsx` | 实验数据另见上传附件`Data_S1.xlsx`，工作表位置见前文 | 材料拟合、晶格响应、质量和性能汇总 |

## 代码包版本

仓库地址：https://github.com/BMG-FDU/egc-metadesign

本地修订包版本见`RELEASE.json`；`FILE_MANIFEST_SHA256.csv`记录发布文件的相对路径、大小及SHA-256。发布清单不包含Python缓存及运行生成的`outputs/`目录。将本次修订包上传至该仓库后，正文和回稿中引用的地址才能取得这些文件。
