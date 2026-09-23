# Run002: learning restored, scene-specific control unproven

Run002 completed. Measured hidden RMS is 4.36e6–6.89e6 whereas raw branch output
RMS is about 0.16. This confirms the run001 scale mismatch. First-step gate
gradients now range from 1.26e-4 to 4.65e-4. The encoder gradients on step two
are nonzero at useful numerical scales; the zero-initialized gates intentionally
block encoder gradients on step one. Neither base nor epoch005 is trained.

At step 200, fit flow MSE changes from 0.07554136 (no world) to 0.07503304
(correct world), about 0.67%; dev changes from 0.07356158 to 0.07315529,
about 0.55%. Shuffled-world MSE is 0.07503301 on fit and 0.07315378 on dev.
Correct world therefore has no demonstrated advantage over wrong world.
Position-scrambled images also look nearly identical to correct-world images.
This supports generic adaptation as a hypothesis, not learned spatial control.
The reports do not establish that all preceding geometry is correct, but give
no new reason to discard the dataset, rebuild geometry or abandon GT conditioning.

The former cross-attention receives image features and unordered world tokens.
Object angles are MLP inputs, but the new attention has no explicit ERP
query-position to object-position prior. It can learn this implicitly, so this
is not a mathematical impossibility. It is a difficult learning problem for
eight samples and 200 steps. Each fit sample has 26–56 object tokens versus
220–506 relation tokens and four layout tokens. A single softmax also makes
token multiplicity an unwanted prior. Neither diagnosis proves the only cause.

## Run003 changes

Keep the run002 scale correction, frozen epoch005, resolution, captions,
optimizer settings, losses and 200-step budget. Train fresh for a comparable
small-fit experiment, with optional `--spatial-world`:

- Append five nonlearned metadata values to each world token internally:
  three direction components, inverse angular width squared and a log count
  prior. These are stripped before learned K/V projections and are explicit
  checkpoint inputs. They do not modify the dataset schema or saved parameters.
- Add `(cos(angle)-1)/sigma^2` to object attention logits. Query directions use
  the existing canonical ERP pixel-centre convention and repeated circular
  boundary columns. Width uses object size/distance, limited to 0.35–1.2 radians.
  This is a soft pilot prior, not an exact visibility mask or OBB projection.
- Layout tokens are global. Relations receive a broad direction prior at their
  subject. Subtract log of the token count within each type, removing the prior
  advantage of a type having more tokens; actual attention weights remain learned.
- Add constant-world evaluation and prediction differences against incorrect
  world conditions; sample whole-world swaps as well as position scrambling.

These are local adapter changes within Stage 3. The spatial prior and type
balancing are introduced together for a feasibility run; separate ablations
would be necessary to attribute improvements to either individually. No auxiliary
loss forces invalid worlds to look worse, and image differences alone still do
not prove correct control. Position scrambling remains an inconsistent diagnostic.

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
sbatch qwen_pano/caupano/tools/submit_gated_fit.sh
```

Submission reuses all existing assets, runs CPU regression checks and trains
`gated_epoch005/fit_run003`, then samples `images_run003_step200`. Existing runs
retain their original behaviour via checkpoint config defaults. New code/tests
have not been executed by the assistant; the user runs them. Assess fit against
shuffled/constant worlds and inspect correct placement, then assess dev. Do not
claim independent generalization from these overlap-exposed samples.
