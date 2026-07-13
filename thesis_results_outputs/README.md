# Thesis results outputs

This directory was generated from the `smart_cam/results` tree.

- `figures/` contains PDF and PNG figures. Use the PDF files in LaTeX.
- `tables/` contains aggregated CSV tables and `chapter5_key_metrics.json`.
- `reports/figure_manifest.csv` maps each figure to its experiment and source data.
- `reports/data_quality_report.txt` records missing raw files, skipped figures, and known limitations.

The script calculates aggregate privacy leakage from target-frame counts:

`100 * sum(leaked_target_frames) / sum(visible_target_frames)`

It uses frame-level and output-event files when available and falls back to run summaries when necessary.
