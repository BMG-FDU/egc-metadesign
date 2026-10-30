# IJMS Code and Data

## Files

- `hexagonal_generator.py` and supporting modules: lattice generation and Eulerian route construction.
- `GNN.py`: GNN training, prediction, and interpretability analysis.
- `train_ppo.py`, `compare_all_methods.py`, and supporting modules: inverse design and algorithm comparison.
- `SIMULATION.py`: Abaqus/Explicit dynamic simulation with a 0.05 s step and a smooth-step loading amplitude.
- `seven_descriptor_shap.py` and `run_descriptor_shap.py`: SHAP analysis of a seven-descriptor linear proxy.
- `graph/` and `data/`: 5,899 paired structural JSON files and response CSV files.
- `splits/data_split_manifest.csv`: sample assignments for the training (4,129), validation (885), and test (885) sets, together with relative file paths and file hashes.
- `results/best_model.pth`: Historical GNN predictor restored from the original optimization checkpoint. Its SHA-256 matches the saved optimization logs.
- `results/retrained_full/best_model.pth`: Retrained Full model (endpoint R² = 0.9774), corresponding to Fig. 5b; training and test records are in `results/retraining_records/`.
- `configs/model_checkpoints.json`: checkpoint paths, model names and SHA-256 hashes.
- `abaqus_inputs/`: three historical B31 Abaqus/Explicit input files used for the Fig. S5 baseline curves (`P1_nominal.inp`–`P3_nominal.inp`), with a 0.05 s step and target engineering strain of 0.50; `input_manifest.json` records their sources.
- `configs/gnn_training_record.json`: parameter records for the historical GNN retraining run; the current retraining configuration is saved with its training records.
- `requirements.txt` and `configs/current_environment.json`: reported package versions and installation dependencies; `configs/recorded_environment.json` retains the earlier environment record.
- `configs/ppo_current_defaults.json` and `environment_current_defaults.json`: PPO and environment parameters in the current code.

## Installation

The reported Python environment is Python 3.11.0 with PyTorch 2.8.0+cu129. Exact reported package versions are in `configs/current_environment.json`; `configs/recorded_environment.json` retains the earlier environment record. Install from the package root:

```powershell
conda create -n ijms python=3.11.0 -y
conda activate ijms
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements.txt
```

The CUDA 12.9 installation command follows the [official PyTorch 2.8 instructions](https://pytorch.org/get-started/previous-versions/#v280). Abaqus 2024 must be installed separately for finite-element jobs; run Abaqus extraction scripts with its own Python interpreter. The dependency file records the reported package versions; a fresh installation has not been tested during packaging.

## Running the Code

Run the commands below from this directory in a Python environment with the required dependencies installed. Abaqus simulations require an Abaqus environment and a valid license.

Generate physical structures with each edge represented once:

```powershell
python hexagonal_generator.py --purpose mechanics --points-per-edge 10 --seed 42 --convert-to-inp --output-dir generated_mechanics
```

Generate Eulerian route graphs:

```powershell
python hexagonal_generator.py --purpose path --points-per-edge 10 --seed 42 --output-dir generated_path
```

Train the GNN while preserving the supplied model weights:

```powershell
python GNN.py --graphs_dir graph --data_dir data --outdir outputs/gnn_retraining --seed 42 --epochs 180 --batch_size 8 --use_global_feat 1 --run_explainer 0
```

Training uses the random-split procedure implemented in the code with `seed=42`; the split manifest is provided for verification. See the configuration files and command-line help for the remaining parameters.

Run a new four-algorithm comparison using the supplied historical GNN predictor:

```powershell
python compare_all_methods.py --gnn_model results/best_model.pth --gnn_budget 100000 --seed 42 --ppo_variants loop_link_node --save_dir outputs/comparison
```

This command uses `results/best_model.pth` (historical GNN predictor). Outputs are saved to `outputs/comparison`.

The final-window mean is the arithmetic mean of episode-best SEA values whose episode-end GNN-call counts fall in the last 10% of the evaluation budget.

Prepare a separate simulation working directory, copy `SIMULATION.py` into it, and place the INP models to be analyzed in its `graph/` subfolder. From that working directory, run:

```powershell
abaqus cae noGUI=SIMULATION.py
```

Standalone entry point for the seven-descriptor analysis:

```powershell
python run_descriptor_shap.py --input-table results/shap/descriptor_inputs_and_splits.csv --outdir outputs/shap
```

Evaluate the supplied historical GNN predictor on the fixed test split without retraining:

```bash
python evaluate_gnn.py
```

Outputs in `outputs/original_full_evaluation`: whole-curve and 0.5-strain endpoint metrics, curve checks, and per-sample results.

Evaluate the retrained model separately:

```powershell
python evaluate_gnn.py --checkpoint results/retrained_full/best_model.pth --outdir outputs/retrained_full_evaluation
```

## Saved five-seed optimization runs

`results/optimization_logs/` contains the saved PPO, Greedy, Random and Hill Climbing runs for seeds 19, 37, 42, 48 and 57, with 100,000 GNN evaluations per run. Each run includes `evaluations.csv`, `episode_ends.csv` and source metadata. The manifest retains source-file and model hashes recorded for these runs.

```powershell
python summarize_optimization_logs.py
```

The command verifies the saved logs and writes per-run and five-seed summaries to `results/optimization_logs/summary/`: first evaluation strictly exceeding 750 J/kg, final best SEA, and the mean of episode-best SEA values whose episode-end call counts satisfy `90000 < count <= 100000`. Aggregate standard deviations use `ddof=1`. These are summaries of the saved five-seed runs.

Optimization-control data used in Figs. S17–S18 and Tables S13–S14 are in `results/optimization_controls/`; see its bilingual README for file descriptions.

SHAP inputs, fixed proxy splits, fitted parameters, fidelity results and Fig. 5(f)/Fig. S10 plotting data are in `results/shap/`; see its bilingual README.

## Structural similarity and duplicate checks

Scripts, saved Fig. S8 data and run commands: `results/structural_similarity/README.md`.

Validation structures, FEA responses, saved GNN predictions and scripts: `results/validation/README.md`.

Finite-element supplementary materials for Figs. S3–S7: [instructions](results/finite_element/README.md).

Route-comparison scripts and three-repeat printing-time measurements: [instructions](results/printing_paths/README.md).

Experimental data are supplied separately in `Data_S1.xlsx`: `FigS2_TPU_Raw` and `FigS2_FullMaterialCurves` (material tests/fitting), `FigS13_TOP1_experiment`–`FigS13_TOP5_experiment` (lattice mean/SD curves), `Fig6K_ExperimentData` and `TableS9_CandidateMass` (mass/performance summaries).

## Reproduction index

All commands below run from the package root, except the finite-element commands explicitly run from their own directory. Detailed settings remain in the linked instructions above. Reproducing saved summaries does not require new training or Abaqus jobs.

| Task and review points | Input | Command or instructions | Output / associated figures and tables |
| --- | --- | --- | --- |
| Topology and route generation; R1-2, R1-12, R2-1 | Generator parameters and `boundary_template_5x3.json` | Mechanics/path commands under Running the Code | `generated_mechanics/`, `generated_path/`; Fig. 1 |
| GNN training; R1-8 | `graph/`, `data/`; parameters in `configs/` | GNN training command above | `outputs/gnn_retraining/`; archived Fig. 5b records: `results/retraining_records/` |
| Fixed-test GNN evaluation; R1-7, R1-8 | `results/best_model.pth`, `splits/data_split_manifest.csv`, `graph/`, `data/` | `python evaluate_gnn.py` | `outputs/original_full_evaluation/`; whole-curve and endpoint metrics |
| Optimization; R1-9, R2-4 | Historical checkpoint and topology editor | Four-algorithm command above | `outputs/comparison/`; new runs are separate from the archived runs |
| Archived five-seed statistics; R1-9, R2-4 | `results/optimization_logs/` | `python summarize_optimization_logs.py` | `results/optimization_logs/summary/`; Fig. 6i,j and Fig. S12 |
| Optimization controls; R1-9 | `results/optimization_controls/` | [Data descriptions](results/optimization_controls/README.md) | Figs. S17–S18, Tables S13–S14; saved plotting curves and summaries |
| Seven-descriptor SHAP; R1-10, R2 minor 3 | `results/shap/descriptor_inputs_and_splits.csv` | `python run_descriptor_shap.py --input-table results/shap/descriptor_inputs_and_splits.csv --outdir outputs/shap` | `outputs/shap/`; Fig. 5f and Fig. S10 |
| Duplicate/nearest-neighbor checks; R1-8, R2-3 | Packaged graphs, fixed splits and archived predictions | [Commands](results/structural_similarity/README.md) | `results/structural_similarity/outputs/`; Fig. S8 |
| Cross-family and extended numerical validation; R2-3, R2-5 | `results/validation/cross_family/`, `expanded_numerical/` | [Summary and plotting commands](results/validation/README.md) | Summaries and plots in those two folders; Figs. S9, S16 and Table S12 |
| Finite-element supplementary analyses; R1-4, R1-6, R2-2 | `results/finite_element/` | [Abaqus, extraction and plotting commands](results/finite_element/README.md) | Figs. S3–S7; outputs as specified in the linked instructions |
| Route comparison, printing time and G-code; R1-2, R1-11, R2-1, R2-6 | Historical graph, time measurements and route-step tables | [Route/time commands](results/printing_paths/README.md); [G-code export](results/printing_paths/gcode_commands/README.md) | Route/angle summaries, repeated-printing-time plot and representative G-code |
| Experimental data; R1-5, R2 minor 2 | `Data_S1.xlsx` | Experimental data are provided in the uploaded attachment `Data_S1.xlsx`; worksheet locations are listed above | Material fitting, lattice response curves, mass and performance summaries |

## Package version

Repository: https://github.com/BMG-FDU/egc-metadesign

This local revision package is identified in `RELEASE.json`. `FILE_MANIFEST_SHA256.csv` records the relative path, size and SHA-256 of each release file. Python caches and generated `outputs/` directories are excluded from the release inventory. Upload this revision package to the repository to make the files available at the address cited in the manuscript and response.
