# Revised Stage 3 order: separate training budget, data and architecture

## Multi-building expansion (current recommended next run)

`python -m qwen_pano.caupano.tools.multiscan plan` freezes a source-train-only
selection: existing sT4fr6TAbpF plus four additional fit buildings and one
heldout-world building. New buildings have 60..180 source-train panoramas,
no source-test membership, and local stitched captions. SHA256 ranking with
seed 42 selects buildings/views before model results are available. Build at
most 60 panoramas per fit building and 12 in the heldout building. This size
restriction is a resource choice, not representative dataset sampling.

Run the generated `download_missing.sh` using the existing authorized downloader;
it requests only missing archives and preserves its interactive terms prompt.
Archives still cover entire scans, even though expensive ERP/mesh projections
are limited to selected views. `multiscan build --threads 4` reuses existing
per-panorama outputs but writes separate batch manifests under multiscan_v1/built.
Unchanged inputs are required for reuse. Failed/review-required views are reported
and excluded, with no replacement selected based on downstream model performance.

`multiscan export` reuses the source-split exporter and native RGB preparation.
It reserves two assembled views per fit building and four views from the heldout
building as dev (14 total). Fit must retain at least ten views per building;
actual counts and source exclusions are visible in build_summary.json and protocol.json.
The rest of the assembled heldout views are unused. Cross-building holdout is for
the world module, not an independent Qwen-Pano benchmark: base exposure is recorded.

`sbatch qwen_pano/caupano/tools/submit_gated_fit.sh multiscan` caches text then
trains the unchanged gated baseline from scratch for 4000 steps, saving and
evaluating every 1000. This is an initial fixed budget, not a claim of convergence
or equal sample exposure with run003. Evaluate two fixed fit views per building
plus all 14 dev views; eval.jsonl separates same_building and heldout_building.
Final previews include both groups. No type balancing or spatial bias is enabled.
Preparation, tests, downloads, building and training are left for the user to run.

## Next experiment: existing source-train expansion

Use `prepare_gated_fit --expand-source-train` with parent `panfusion_epoch005`
and a new output `gated_epoch005_train32`, then submit
`submit_gated_fit.sh expand32`. This uses 32 fit samples and the unchanged four
dev samples, all from source train in sT4fr6TAbpF. It does not use source-test
buildings. Preparation checks source membership, pairing and stored coordinate
consistency; it does not establish physical correctness of every ERP pose.

Keep run003's gated architecture, hidden-RMS scaling and frozen epoch005;
type balancing and spatial bias remain disabled. Evaluate the original eight
fit and four dev samples, preserving their latent sampling order and isolating
additional latent RNG draws. The run saves/evaluates every 1000 steps through
4000: compare step 1000 at equal optimizer-step budget and step 4000 at equal
125 passes per training image. The latter uses four times the optimizer steps,
so it is not a data-only comparison. Runtime parameter ownership, zero-gate
identity and initial gradient routing are checked within the training job.
These checks and this revised preparation have not been run by the assistant.

Success requires correct-world advantages over shuffled/constant controls,
not merely improved loss relative to no-world. This remains a single-building
engineering experiment with known base-training overlap, not independent
generalization evaluation. Older balancing proposals below are historical
ablations, not the recommended next run.

## Parameter and optimizer audit; current submission defaults

The actual run003 safetensors header contains 5,772,291 world parameters:
80,640 object-related; 201,472 relation-related; 292,096 layout-related;
66,304 final projection; and three 1,710,593-parameter cross-attention branches.
These are new parameters, not unfrozen Qwen/epoch005 weights. The training code
freezes the entire Transformer after loading LoRA and optimizes only world
parameters. Backpropagation through frozen Transformer operations is necessary
for the new branches; it does not imply updates to frozen parameters.

Future training writes parameter_report.json after checking requires_grad,
disjoint parameter identities and exact optimizer ownership. Training config
freeze/count fields now come from that check rather than a literal flag. The
CPU reader below checks saved tensor counts without loading Qwen or training:

```bash
python -m qwen_pano.caupano.tools.parameter_report \
  --run qwen_pano/outputs/caupano/experiments/gated_epoch005/fit_run003_budget \
  --weights qwen_pano/outputs/caupano/experiments/gated_epoch005/fit_run003_budget/world-step01000.safetensors
```

Submission now accepts `baseline` (default, no type balancing), `balanced`
(explicit ablation), and `audit` (read-only run003 inference). Baseline and
balanced use separate output directories. This supersedes the earlier default
balanced submission described below. No need to rerun the same eight-image
training just to verify parameter counts. New tests were written, not executed
by the assistant. This count establishes neither a sample-complexity threshold
nor the cause of weak world control.

## Attention audit completed: run004 tests type-count balancing only

The two same-caption fit samples allocate about 78–87% attention to relations
and 0.1–0.6% to layout at the sampled first denoising step. Relations comprise
about 89% of tokens; hence this is not evidence of a learned per-token preference
for relations. Layout mass is lower even than its small count-based prior.
Across the samples, cosine similarity of pooled normalized world features is
about 0.984; pooled attended features in block 30 reach 0.999923. These are means,
not aligned token/query comparisons or proof of complete representation collapse.
Combined with constant-world evaluation, they justify a count-prior ablation,
not a claim that the sole root cause has been established.

Default submission now trains fit_run004_balanced from scratch for the same
1000 steps as run003, using `--balance-world-types`. Attention logits subtract
log of each token's type count (objects, four layout tokens, relations). This
gives equal prior mass to nonempty types when their logits are equal; learned
attention is not constrained to equal mass. Duplicating every token in a type
does not alter the ideal real-valued attention result. No new parameters, data
schema, loss, spatial prior or base-model changes are introduced. Feature
rounding stays as in run003; count metadata stays FP32 and travels explicitly
through checkpoint inputs. Old checkpoint configs default to unbalanced mode.

`sbatch qwen_pano/caupano/tools/submit_gated_fit.sh` runs CPU regressions, training,
and matched-seed sampling with automatic attention reports. Outputs are
gated_epoch005/fit_run004_balanced and images_run004_balanced_step1000. The `audit`
argument still inspects the existing run003 checkpoint. No tests or new training
have been run by the assistant. Evaluate correct-vs-shuffled/constant world and
GT placement, not layout attention mass alone. If this controlled modification
does not improve conditional usefulness, do not stack spatial changes blindly;
prioritize the planned scene-diversity and representation comparisons.

This is an explicit adapter refinement, not a clause required by the detailed
plan. It preserves Stage 3's GT-world input, text path, frozen base and original
generation objective; intervention losses and a Reasoner remain out of scope.

## Run003 completed: next action is an audit, not another training run

At step 1000, fit correct/no-world/shuffled/constant flow MSE is respectively
0.07381507 / 0.07554136 / 0.07381757 / 0.07381436. Dev is
0.07226494 / 0.07356158 / 0.07226386 / 0.07226567. This indicates predominantly
shared adaptation, not demonstrated scene-specific control. At sigma=0.5 seven
of eight fit samples narrowly favour correct over shuffled worlds, so complete
conditional collapse has not been established either. One fixed noise seed and
three sigmas are diagnostic measurements, not statistical significance tests.

Run `sbatch qwen_pano/caupano/tools/submit_gated_fit.sh audit` from the project
root. It reuses step1000 weights and generates only the first two fit samples,
whose captions are identical, at the same seed. The optional report samples 64
actual image queries per branch on the first positive-text transformer call,
recording attention mass by token type, entropy, world/value/attended means and
within-sequence variation. Report: gated_epoch005/attention_run003/attention_report.json.
This is a limited high-noise inference probe, not all training timesteps.
The reporting path has a regression test for unchanged predictions; the
assistant has not run it or inference. The `audit` argument exits before training.

Use this evidence to select a single next experiment: investigate feature/V
collapse if scene differences disappear there; test type balancing alone only
if real attention supports that hypothesis; otherwise prioritize a manageable
source-train diversity expansion with fixed evaluation identities. Do not enable
both spatial bias and balancing or add intervention objectives on this evidence.

The sections below record the preceding experiment design.

This supersedes the immediate spatial-prior run suggested in
gated_run002_diagnosis.md. The spatial code remains optional and is not enabled
by submit_gated_fit.sh. Existing submitted jobs are not modified or stopped.

Detailed plan sections 34–40 specify GT world tokens, preserved text, frozen
Qwen-Pano, gated middle-block injection as an adapter option, and the original
generation objective. Sections 35–38 represent geometry in token features;
section 39.2 does not require explicit spatial attention biases or token-count
balancing. Those are experimental additions, compatible with the broad objective
but not a proven necessary correction or an exact implementation of the document.

Run001 had a measured numerical scale problem, fixed by run002. Run002 is learning
but has not demonstrated world-specific control. Its 200 batch-one steps expose
each of eight fit images about 25 times, with stochastic noise and timesteps.
The learning curve still improves between steps 100 and 200. All images come
from one building; three share the exact bathroom caption, and two are nearby
viewpoints. Eight panoramas are not eight independent buildings or necessarily
eight independent rooms. This severely limits diversity and generalization.

Small data can also be easier to memorize, so it does not by itself explain
failure to fit the training examples. Increasing training steps and increasing
scene diversity answer different questions and should not be conflated.

No attention statistics establish relation-token domination. Token multiplicity
is a potential softmax prior, not proof of attention mass. Explicit angular bias
may help but makes assumptions about localization by OBB centres, particularly
for walls, floors and extended objects. Introducing it together with balancing
would confound two architecture changes before budget/data effects are known.

## Immediate implemented experiment

Reuse run002 architecture, RMS scaling, learning rate, epsilon, frozen checkpoint,
eight fit/four dev images, text, geometry and stochastic original panorama loss.
Train fresh for 1000 steps; evaluate every 250. This is a bounded budget test,
not a claim that 1000 steps guarantee convergence. No optimizer state was saved
in run002, so this is explicitly not an exact resume. The extra constant-world
evaluation and stratified summaries affect reporting, not the training objective.

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
sbatch qwen_pano/caupano/tools/submit_gated_fit.sh
```

Outputs: gated_epoch005/fit_run003_budget and images_run003_budget_step1000.
Automatic sampling uses two fit/two dev samples and correct, absent, position-
scrambled and whole-world-swapped conditions at the same seed. Tests and training
are not executed by the assistant.

## Subsequent steps, not yet implemented

1. Inspect fit learning and correct-vs-shuffled/constant-world margins by noise
   level, plus generated placement. Do not interpret denoising MSE or visible
   changes alone as successful control. Same-caption bathroom samples are a
   useful stronger comparison because their text is identical.
2. If fit control emerges but dev fails, prioritize diversity. Keep the baseline
   architecture and fixed dev manifest; use remaining valid source-train samples
   as an inexpensive intermediate experiment, then add a manageable selection of
   new buildings. Freeze the evaluation identities to avoid train/dev overlap.
   Do not use the 71 source-test samples as extra training examples.
3. If sufficient fit training still shows generic adaptation, inspect world
   feature variance and actual attention mass/output changes under matched-caption
   swaps. Compare a learned constant condition baseline if the existing constant-
   world inference ablation is inconclusive: they are not the same experiment.
4. Then test angular bias and type balancing separately, at matched data and
   budgets. Complete section 41 controls for wrong category/relation and no
   geometry; do not add intervention losses or a Reasoner at this stage.

Existing run002 evidence neither establishes that all geometry is correct nor
indicates a reason to rebuild it wholesale. If model diagnostics implicate a
particular RGB/world pairing, inspect that sample's overlay and convention
instead of restarting the complete data pipeline.
