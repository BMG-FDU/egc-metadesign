# Finite-element supplementary materials: Figs. S3–S7

This folder contains the plotted/reported data, extraction scripts and representative Abaqus inputs. ODBs, failed jobs and abandoned section-Poisson-0.49 sensitivity runs are excluded. Historical input settings are preserved. For S4/S5, material Poisson ratio is 0.475; baseline and sensitivity beam-section ratios are 0 and 0.475. S7 historical settings are recorded in Fig_S7/source_settings.json.

- S3: 12 B31 inputs (three structures, four mesh sizes), curves and mesh statistics.
- S4: five nominal beams, contacting/non-contacting solids and volume-matched section-0.475 beams, with response/contact/energy data.
- S5: baseline, boundary and section-0.475 inputs and local-strain/sensitivity data.
- S6: load-sharing and elastic-energy data for 80 structures, plus one representative per family and the original SF/ELSE extraction workflow.
- S7: 152 plotted response records and work/mass/SEA decomposition, with one representative input per topology fraction.

From this directory, with numpy, pandas and matplotlib installed:

```text
python reproduce_summary.py S3
python Fig_S4/plot_0475.py
python Fig_S5/plot_0475.py
python Fig_S6/plot_fig_s6.py
python reproduce_summary.py S7
```

Outputs are written under reproduced (S3/S7), output (S4/S5), or the S6 plotting script output directory. These plots support numerical checks; the SI contains the final publication layout.

Run one existing input, synchronously:

```text
python run_input.py Fig_S4/jobs/P1_no_contact.inp --abaqus abaqus --cpus 4
```

Use --abaqus to specify the local Abaqus launcher if needed. Inputs retain their original analysis steps. After solving S4/S5 baseline/boundary inputs in their jobs directories, run `abaqus python Fig_S4/extract_results.py` or its S5 counterpart. For section0475, keep ODBs in section0475/jobs, change to that section0475 directory, then run `abaqus python extract.py .`.

For individual ODB curves, use the height and mass in extraction_metadata.csv:

```text
abaqus python extract_curve.py path/to/model.odb --height 100 --mass-kg <model_mass> --output extracted.csv
```

For an S6 representative, change to a separate working directory and run `abaqus cae noGUI=<absolute_folder>/Fig_S6/shared/abaqus_worker.py -- <absolute_folder>/Fig_S6/02_Load_path/config.json DL_01`. Use 03_Energy_distribution/config.json for elastic-energy extraction. Other family IDs are listed in representative_samples.csv.

The data-summary/plotting entries were executed successfully during packaging; Abaqus analyses were not re-run.
