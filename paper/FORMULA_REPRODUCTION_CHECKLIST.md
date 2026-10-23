# CT-WM action-conditioned reward formula checklist (v12)

Revision: `ctm-action-conditioned-reward-v12`, 2026-10-22. There are 75 equation labels, no duplicate
or dangling labels, and 69 protected per-label expressions exactly equal
to v11. Native task/reward specification and every CTM recurrence formula remain
unchanged. Only reward conditioning signatures and two first-layer definitions
change; the exact previous NLL algebra remains after stripping the extra action
argument. Historical compiler/test/diagnostic results are not current evidence.

| Formula group | Current contract | Verification boundary |
|---|---|---|
| All CTM memory and readout formulas | Exact v11 gated recurrence, histories, time scales, activity direct path, spread synchrony | Source equality; normal integration regressions |
| Reward first layer | U[p,z]+sum omega_k W_k nu, original norm→act | Independent NumPy reference, exact parameter increase, no extra hidden layer |
| Discrete/continuous action weights | Flattened onehot/soft action or affine [1,a] | No argmax; factored categories, zero-action baseline and gradients |
| Real arrival timing | Shift once, valid carry; zero is_first, same clip as transition | Physical/window boundary test, incoming rather than outgoing action |
| Imagination timing | reward_action=[zero;action[:-1]], temporary head metadata | Successor-state/action pairing, initial reward excluded, same complete CTM state |
| Predictions and all losses | Same255binNLL, native targets, masks/averaging/scales/optimizers | Protected equation/algebra checks; joint finite model+AC update |
| Frozen v11 diagnostic | No backbone updates; grouped train/validation/test split | Prior-only paired correct/wrong gap/ranking, group-bootstrap uncertainty and equally retrained old-head control; report diagnostic sampling |
| Runtime/checkpoints | v12 first-layer/action contract | Schema rejection, save/restore/resume; old training is not modified |

Fitting a head on frozen features tests whether that readout can learn the task
under its stated supervision. It is not a full-agent convergence or policy-success
result. Current receipts record 31 passed CPU tests, native train/resume smoke, successful
source compilation, and the frozen-head experiment; see code docs/VALIDATION.md.
The frozen comparison improves ranking but not NLL consistently.
