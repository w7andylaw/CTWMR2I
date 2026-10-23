# CT-WM v12 formula and implementation alignment

Schema: `ctm-action-conditioned-reward-v12`, 2026-10-22. The inventory contains 75 equation labels;
69 protected per-label expressions and the native problem/reward section
match v11 exactly. Four signatures add aligned action conditioning; the reward
NLL algebra, prediction distributions, model/tick/actor/critic losses and all
coefficients remain unchanged. Two equations define only the replacement first
reward layer. Function references avoid old source line numbers.

| Formula group | Implementation responsibility | Current contract |
|---|---|---|
| world_model_state, CTM transition and reset | NRSM.initial/_transition/_obs_step | Exact v11 gated multi-tick dual-history mechanism; no additional SSM/state |
| private_nlm/write_gate, time constants/write calibration | PrivateNLM, NRSM | Exact v11 candidate/gate, .1 budget, bounded T and expm1 calibration |
| sync selection/state/context | NRSM._sync_indices/window_synchronization/sync_projection | Exact v11 fast/slow spread136pairs and raw64+projected64 deter |
| latent/control_feature | WorldModel, ImagActorCritic, VFunction | Decoder/cont/policy/online+slow value inputs [deter,z]192 unchanged |
| reward_feature/action_weights/first_layer | reward_nets.ActionConditionedMLP, WorldModel.reward | U[p,z]+sum omega_k W_k nu, same norm/act/bias/later layers/255binoutput; no new hidden layer |
| prediction_factorization/reward_loss | WorldModel.loss/reward | Extra explicit incoming-action condition; identical native targets/two-hotNLL/averaging/scales |
| transition_action_clip and real reward alignment | NRSM._action, WorldModel.loss | Same shifted incoming action, physical/window is_first mask, clipped convention; no outgoing-action error |
| imagined_reward and action alignment | WorldModel.imagine/reward | Ephemeral [zero;action[:-1]] paired with states; only successors affect returns; no recurrent metadata |
| continuation/return/model/tick/actor/critic objectives | Inherited WorldModel/behavior code | Exact v11 expressions and coefficients, no extra loss or optimizer |
| runtime metadata | jaxagent.py | v12 first-layer/action-alignment contract; reject production v11 checkpoints |

Inventories establish source coverage, not learned reward or policy success.
Frozen-v11 cue/action-group test splits, held-out discrimination, current runtime
tests and source compilation require their separate receipts. No pass count or
diagnostic score is inherited from v11. Multiplicative action-conditioned reading
is prior art (Oh et al.,2015); no independent CMM/SSM memory is imported.
