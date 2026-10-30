These records correspond to the retrained Full model currently saved as results/retrained_full/best_model.pth (SHA-256: 63b0c20b734ea9ba2581b89ca231e539a1ff3767dab3d2a881e817174227e5fb).
training.log and metrics_from_training_log_2dp.csv contain 180 epochs. Per-epoch losses were logged to two decimal places; full-precision history was not saved.
Fig5b_training_loss_data_5cols.csv contains epoch, raw training/validation losses and EMA-smoothed trends (weight 0.74). Loss values are percentages.
test_metrics.json contains the saved independent test evaluation for the same 885 test samples. No evaluation or training was rerun during this collection.
