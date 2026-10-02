# S2H mAP Improvement and Figure 4 Run Plan

## First make the comparison valid

The current paper reports five completed target domains, but the support
protocol and checkpoint selection need to be made consistent before interpreting
the mAP gaps. The paper specifies K-instance support for non-DIOR domains and
K-image support for DIOR. `prepare_s2h_data.py --protocol paper` now implements
that rule for the datasets handled by the script. The previous `image` default
is retained for compatibility, so the protocol must be selected explicitly.

On the training server, prepare fresh support JSONs into a separate output
directory first; do not overwrite the data used by the existing reported runs:

```bash
python tools/s2h/prepare_s2h_data.py \
  --source-root /path/to/source_datasets \
  --output-root /path/to/s2h_protocol_paper \
  --protocol paper --seed 20250919
```

Inspect each generated JSON's `info.s2h_split_protocol` and effective support
counts. For instance splits, the script enforces exactly K selected annotations
per class. It also records annotations removed from selected images; manually
inspect that count and representative images because strict instance sampling
can leave other visible objects unlabeled. For DIOR, all annotations on selected
images are retained. Run the existing split-leakage checker against the new
train/test JSONs before training.

The server handover confirms there is no independent validation set:
`val_dataloader` and `test_dataloader` both use the query `test.json`. Selecting
an epoch by this val mAP therefore selects on the test set. The recommended
paper policy is to report the fixed final epoch for every method, matching the
`latest.pth` policy used by the reproduced Domain-RAG baseline. For the current
configs this means epoch 30 for ArTaxOr, DIOR, NEU-DET, and UODD, and epoch 5
for Clipart1k and FISH. The training hook should not use `save_best`, and the
ablation and sweep scripts should test that fixed checkpoint. This changes
reported values, so all old table entries must be refreshed or marked
provisional. `CHECKPOINT_POLICY=best` remains available only for diagnostics;
it is not a paper-results setting.

Re-run the Grounding DINO fine-tuning baseline and S2H with identical supports,
seeds, query set, schedule, and evaluator. Treat the old table values as
provisional until the corrected protocol reproduces the baseline within a
reasonable range. Use at least the five support seeds claimed in the paper for
the main table; keep the same seeds paired across methods and report mean and
standard deviation.

## Figure 4 sensitivity experiment

The existing sweep scripts define a one-factor-at-a-time Clipart1k 5-shot
experiment with seeds 3407, 3408, and 3409:

| Quantity | Values | Default |
| --- | --- | --- |
| FFCP context rank | 1, 2, 4, 8, 16 | 4 |
| Hard update radius | 0.25, 0.5, 1, 2, 4 | 1 |
| Authority ceiling `g_max` | 0.1, 0.25, 0.5, 0.75, 1.0 | 0.5 |
| Uncertainty threshold `theta` | 0.3, 0.4, 0.5, 0.6, 0.7 | 0.5 |
| Consistency weight | 0, 0.05, 0.1, 0.2 | 0.1 |

This is 72 runs. Pin the support split, initialization checkpoint, and training
budget. Verify the sweep script's resolved checkpoint and knowledge paths
before launching; retain per-seed logs and scalar files. Generate the plot and
CSVs only after all expected runs have completed:

```bash
DATASET=clipart1k SHOT=5 SEEDS="3407 3408 3409" \
  PARAMS="g_max theta con_weight ffcp_rank hard_radius" \
  bash tools/s2h/analysis/run_sweep.sh

python tools/s2h/analysis/summarize_sweep.py \
  --root cat_work_dir/sweep --dataset clipart1k --shot 5 \
  --params g_max theta con_weight ffcp_rank hard_radius \
  --seeds 3407 3408 3409 \
  --out work_dirs/s2h_analysis/fig4_hyperparam_clipart1k_5shot
```

Before replacing the Fig. 4 placeholder, check that each parameter/value has
three successful runs and that the summary's default markers match the
configuration used in the main results. Describe sensitivity from the measured
curves; do not claim a broad stable region unless multiple neighboring values
actually fall within the predeclared tolerance.

## mAP improvement experiments after protocol repair

Do not change several parts of the method at once. First save a clean corrected
baseline, then compare each candidate against it with paired support seeds and
the same training/evaluation budget. Run a small pilot on Clipart1k, DeepFish,
and NEU-DET, including at least one domain where current gains are modest; only
expand a change that improves mean mAP without a large regression on another
domain.

1. **ATAR correction strength.** The current `raw_alpha` starts near zero
   (`init_alpha_bias=-8`), which may make the residual too weak early in
   training. Compare the current initialization with a short warm-up or a
   better-scaled initialization. Log alpha, gate strength, and per-class AP to
   catch saturation or a correction that only raises scores.
2. **Reliability calibration.** Hard attribute-coverage cutoffs can suppress
   useful prototypes. Compare the current threshold with a continuous
   reliability weight and a conservative floor; retain a no-correction path
   for unsupported classes.
3. **Soft prototype aggregation.** Compare the coordinate-wise median against
   normalized mean and a robust/trimmed mean. Keep normalization and all other
   knowledge-building settings fixed.
4. **Localization residual.** Only after the classification-side changes are
   stable, test a bounded box residual. Track AP50/AP75 and localization errors
   as well as mAP so a classification gain cannot hide worse boxes.

For each candidate, record config, source commit or file hashes, support
manifest, seed, selected checkpoint policy, mAP/AP50/AP75, and per-class AP.
Use the five paper seeds for claims; use additional runs to confirm finalists.
Choose one shared cross-domain configuration rather than tuning each target
domain independently. Update the paper tables, Fig. 4, and method claims only
from completed, protocol-matched runs.

## Server handoff status

`tools/s2h/HANDOVER.md` supplies the training workspace, conda environment,
seven-GPU host, Grounding DINO checkpoint, BERT directory, and framework
versions. The remaining operational check is the exact raw source root used by
`prepare_s2h_data.py`; run `bash tools/s2h/collect_handover_info.sh` on the
server and use its reported source path. Keep the generated protocol-fix data
in a staging output first, inspect its support statistics and dropped
annotations, then activate it for runs without overwriting the old-result data.

The checkpoint-selection decision is settled: fixed final-epoch evaluation is
the paper policy. Because query/test is also the only val set and the
reproduced baseline uses `latest.pth`, all formal runs use
`CHECKPOINT_POLICY=final`; `best` is diagnostic only. No credentials or large
file uploads are needed in chat; paths and command output are sufficient.
