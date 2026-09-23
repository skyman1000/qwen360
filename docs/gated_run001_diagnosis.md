# Gated run001 diagnosis and run002

Job 61346 completed all 200 steps and generation. Job 61345 was cancelled during
preparation; its log does not identify who or what cancelled it. No exception in
the completed run explains the ineffective conditioning.

Run001 fit evaluation: 0.07554136356338859 at steps 0, 100 and 200, for every
conditioning mode. Dev: 0.0735615799203515, likewise unchanged. These are means
over three scheduler fractions. First-step gate gradients were 2.7e-11 to 9.0e-11;
second-step encoder gradient sums were around 1e-17 to 1e-15. Final gates were
around 3e-6 to 8e-6 in magnitude. Zero-gate identity passed exactly. The original
nonzero-gradient checks established graph connectivity, not effective learning.

Both fit and dev images preserve the ERP appearance, but correct and position-
scrambled world images look nearly identical. They do differ from the no-world
images. Thus sampling is not literally proven invariant, and visual difference
from no-world is not evidence that object positions were learned.

The old branch normalized its attention query but added an uncalibrated residual
directly to the native hidden stream. Actual hidden/update RMS was not logged,
so a scale mismatch is a working diagnosis, not a measured activation fact.
Adam epsilon 1e-6 also heavily attenuates updates when gradients are around 1e-10.
Training ran branch attention under BF16 autocast; sampling did not explicitly
use the same autocast for that external float32 branch. This is an additional
numerical discrepancy, not proof of the sole cause of image differences.

Run002 uses `hidden + tanh(gate) * stopgrad(RMS(hidden)) * CrossAttn(...)`, with
per-token RMS over channels. Gate initialization remains exactly zero. It keeps
the original encoders, three block locations, losses, frozen epoch005, captions,
resolution and sampling settings. Adam epsilon is 1e-8. The revised branch uses
BF16 autocast on CUDA in both training and sampling; accumulation is float32.
Checkpoint configs record residual_scale, and the sampler uses `none` for old
checkpoints so existing weights are not silently reinterpreted.

The first two gradient reports now record hidden RMS, raw/scaled update RMS,
residual RMS, effective delta after dtype conversion, and changed-element
fraction. Evaluation also records the maximum absolute world/no-world prediction
difference and stops if it is zero across all evaluated samples after training.
This measures whether the branch affects predictions; it does not prove spatial
control. No further geometry build or dataset download is justified by this run.

Submit from the project root:

```bash
sbatch qwen_pano/caupano/tools/submit_gated_fit.sh
```

This reuses gated_epoch005 manifests/RGB/text cache, automatically runs CPU
regressions, trains from scratch to fit_run002 and generates
images_run002_step200. Existing run001 outputs remain intact. The trainer does
not resume an existing nonempty run002. Inspect the new diagnostics and images
before increasing steps or claiming learned world control.

The assistant has only inspected source and existing results. New regression
tests, training and inference are left for the user to execute.
