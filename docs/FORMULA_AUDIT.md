# CT-WM v12 source formula audit

Revision: `ctm-action-conditioned-reward-v12`, 2026-10-22.

The inventory contains **75 equation labels**, no duplicates or dangling
references, and **69 protected per-label statements** byte-identical to
v11. The entire native problem/task/reward specification is byte-identical.
Only reward_feature, prediction_factorization, nrsm_reward_loss and
imagined_reward add explicit action conditioning. Two new equations specify
action-bank weights and the replacement first layer. Removing the extra action
argument recovers the exact previous reward NLL algebra. Two-hot targets,
255-bin CE/readout, every model/tick/actor/critic objective, return/continuation,
weights and optimizer semantics are unchanged.

The old imagined reward and continuation share an align group. The inventory
retains that full group; individual continuation statement equality is checked
independently. Source SHA-256: `6694ed11434196b00b2088e1daba825424ab4811d7780183818a9125d03ae6fb`.

This maps source formulas to responsibilities; it is not a runtime pass count,
frozen-feature result or learned control improvement. Current compile/tests and
frozen-v11 diagnostic results are separate receipts. Compiled numbers stay null.

| Label | TeX line | Code responsibility | Scope |
|---|---:|---|---|
| `eq:benchmark_pomdp` | 68 | `recall2imagine/embodied/envs/*; embodied/core/driver.py; native task specification` | Native task definitions preserved; wrappers require their own runtime evidence. |
| `eq:benchmark_transition` | 73 | `recall2imagine/embodied/envs/*; embodied/core/driver.py; native task specification` | Native task definitions preserved; wrappers require their own runtime evidence. |
| `eq:benchmark_observation` | 79 | `recall2imagine/embodied/envs/*; embodied/core/driver.py; native task specification` | Native task definitions preserved; wrappers require their own runtime evidence. |
| `eq:benchmark_history` | 84 | `recall2imagine/embodied/envs/*; embodied/core/driver.py; native task specification` | Native task definitions preserved; wrappers require their own runtime evidence. |
| `eq:benchmark_return` | 89 | `recall2imagine/embodied/envs/*; embodied/core/driver.py; native task specification` | Native task definitions preserved; wrappers require their own runtime evidence. |
| `eq:benchmark_discount` | 97 | `recall2imagine/embodied/envs/*; embodied/core/driver.py; native task specification` | Native task definitions preserved; wrappers require their own runtime evidence. |
| `eq:world_model_state` | 135 | `recall2imagine/nrsm_nets.py: NRSM.initial, NRSM._transition, NRSM._obs_step` | Current CTM-only structured carry/reset specification; dynamic verification recorded separately. |
| `eq:rssm_deterministic` | 146 | `recall2imagine/nets.py: RSSM.img_step (baseline comparison)` | Baseline context, not a claim that NRSM contains a GRU. |
| `eq:ct_prior_context` | 151 | `recall2imagine/nrsm_nets.py: NRSM.initial, NRSM._transition, NRSM._obs_step` | Current CTM-only structured carry/reset specification; dynamic verification recorded separately. |
| `eq:latent_feature` | 159 | `recall2imagine/agent.py: latent_input_config, WorldModel; behaviors.py` | Unchanged decoder/continuation/policy/value [deter,stoch] state vector, default192; no action condition added to these heads. |
| `eq:control_feature` | 164 | `recall2imagine/agent.py: latent_input_config, WorldModel; behaviors.py` | Unchanged decoder/continuation/policy/value [deter,stoch] state vector, default192; no action condition added to these heads. |
| `eq:reward_feature` | 169 | `recall2imagine/reward_nets.py: ActionConditionedMLP/ActionConditionedLinear; agent.py: WorldModel.reward` | Only reward first linear layer changes: shared U[p,z] plus sum_k action_weight_k W_k raw_activity; discrete soft/onehot flatten, continuous affine [1,a]; original norm/act/bias/later layers/output retained, no argmax/detach/new state. |
| `eq:reward_action_weights` | 179 | `recall2imagine/reward_nets.py: ActionConditionedMLP/ActionConditionedLinear; agent.py: WorldModel.reward` | Only reward first linear layer changes: shared U[p,z] plus sum_k action_weight_k W_k raw_activity; discrete soft/onehot flatten, continuous affine [1,a]; original norm/act/bias/later layers/output retained, no argmax/detach/new state. |
| `eq:reward_first_layer` | 188 | `recall2imagine/reward_nets.py: ActionConditionedMLP/ActionConditionedLinear; agent.py: WorldModel.reward` | Only reward first linear layer changes: shared U[p,z] plus sum_k action_weight_k W_k raw_activity; discrete soft/onehot flatten, continuous affine [1,a]; original norm/act/bias/later layers/output retained, no argmax/detach/new state. |
| `eq:transition_action_clip` | 204 | `recall2imagine/nrsm_nets.py: NRSM._action; agent.py: WorldModel reward-action alignment` | Inherited flattened action clipping and scaled gradient unchanged; reward uses the identical reset-masked incoming action, with no argmax. |
| `eq:ct_tick_input` | 211 | `recall2imagine/nrsm_nets.py: NRSM.initial, NRSM._transition, NRSM._obs_step` | Current CTM-only structured carry/reset specification; dynamic verification recorded separately. |
| `eq:ct_tick_initialization` | 211 | `recall2imagine/nrsm_nets.py: NRSM.initial, NRSM._transition, NRSM._obs_step` | Current CTM-only structured carry/reset specification; dynamic verification recorded separately. |
| `eq:ct_synapse_glu` | 222 | `recall2imagine/nrsm_nets.py: _NavigationSynapseBlock, NRSM._transition` | Unchanged two-block navigation GLU/LN synapse and first input width258 at default two-action task. |
| `eq:ct_synapse_layernorm` | 227 | `recall2imagine/nrsm_nets.py: _NavigationSynapseBlock, NRSM._transition` | Unchanged two-block navigation GLU/LN synapse and first input width258 at default two-action task. |
| `eq:ct_synapse_tick` | 238 | `recall2imagine/nrsm_nets.py: _NavigationSynapseBlock, NRSM._transition` | Unchanged two-block navigation GLU/LN synapse and first input width258 at default two-action task. |
| `eq:ct_pre_history` | 238 | `recall2imagine/nrsm_nets.py: PrivateNLM.__call__, NRSM._transition` | Per-neuron shared ELU hidden; tanh candidate and independent sigmoid gate; write_w/write_b zero-init gate=.5; gate is temporary. |
| `eq:ct_neuron_model` | 247 | `recall2imagine/nrsm_nets.py: PrivateNLM.__call__, NRSM._transition` | Per-neuron shared ELU hidden; tanh candidate and independent sigmoid gate; write_w/write_b zero-init gate=.5; gate is temporary. |
| `eq:private_nlm` | 253 | `recall2imagine/nrsm_nets.py: PrivateNLM.__call__, NRSM._transition` | Per-neuron shared ELU hidden; tanh candidate and independent sigmoid gate; write_w/write_b zero-init gate=.5; gate is temporary. |
| `eq:ct_write_gate` | 258 | `recall2imagine/nrsm_nets.py: PrivateNLM.__call__, NRSM._transition` | Per-neuron shared ELU hidden; tanh candidate and independent sigmoid gate; write_w/write_b zero-init gate=.5; gate is temporary. |
| `eq:ct_time_constants` | 266 | `recall2imagine/nrsm_nets.py: NRSM time-scale/write calibration and _transition` | Bounded .1..100 environment-step T; rho=exp(-1/(K*T)); expm1 kappa; additive beta_w*kappa*gate*candidate; beta_w=.1; append updated activity, no activity LN/detach. |
| `eq:ct_write_calibration` | 275 | `recall2imagine/nrsm_nets.py: NRSM time-scale/write calibration and _transition` | Bounded .1..100 environment-step T; rho=exp(-1/(K*T)); expm1 kappa; additive beta_w*kappa*gate*candidate; beta_w=.1; append updated activity, no activity LN/detach. |
| `eq:ct_activity_retention` | 282 | `recall2imagine/nrsm_nets.py: NRSM time-scale/write calibration and _transition` | Bounded .1..100 environment-step T; rho=exp(-1/(K*T)); expm1 kappa; additive beta_w*kappa*gate*candidate; beta_w=.1; append updated activity, no activity LN/detach. |
| `eq:ct_post_history` | 288 | `recall2imagine/nrsm_nets.py: NRSM time-scale/write calibration and _transition` | Bounded .1..100 environment-step T; rho=exp(-1/(K*T)); expm1 kappa; additive beta_w*kappa*gate*candidate; beta_w=.1; append updated activity, no activity LN/detach. |
| `eq:ct_environment_decay` | 296 | `recall2imagine/nrsm_nets.py: NRSM._transition; independent analytic/numerical references` | Direct rho^K identity; constant-gate/candidate write calibration only; conservative value bound about10.05 at defaults, not a full Jacobian or learned-memory guarantee. |
| `eq:ct_sync_selection` | 305 | `recall2imagine/nrsm_nets.py: NRSM._sync_indices, window_synchronization` | Fixed rint(linspace(0,D-1,D_s)) indices; lexicographic upper triangle including self-pairs; initially fast/slow spread, no dynamic sorting; existing finite-window weighted formula. |
| `eq:ct_sync_weights` | 313 | `recall2imagine/nrsm_nets.py: NRSM._sync_indices, window_synchronization` | Fixed rint(linspace(0,D-1,D_s)) indices; lexicographic upper triangle including self-pairs; initially fast/slow spread, no dynamic sorting; existing finite-window weighted formula. |
| `eq:ct_sync_state` | 321 | `recall2imagine/nrsm_nets.py: NRSM._sync_indices, window_synchronization` | Fixed rint(linspace(0,D-1,D_s)) indices; lexicographic upper triangle including self-pairs; initially fast/slow spread, no dynamic sorting; existing finite-window weighted formula. |
| `eq:ct_sync_context` | 331 | `recall2imagine/nrsm_nets.py: NRSM._transition, sync_projection` | deter=[unprojected activity; tanh(sync_projection([activity,sync]))]; require deter>D; default64+(200→64)=128; no extra head/state. |
| `eq:ct_persistent_carry` | 331 | `recall2imagine/nrsm_nets.py: NRSM.initial, NRSM._transition, NRSM._obs_step` | Current CTM-only structured carry/reset specification; dynamic verification recorded separately. |
| `eq:ct_dynamics_predictor` | 345 | `recall2imagine/nrsm_nets.py: prior/posterior statistics; nets.py: distributions; jaxutils.py: OneHotDist` | Existing context-conditioned q and prior sampling, unimix and straight-through semantics. |
| `eq:ct_encoder` | 345 | `recall2imagine/nrsm_nets.py: prior/posterior statistics; nets.py: distributions; jaxutils.py: OneHotDist` | Existing context-conditioned q and prior sampling, unimix and straight-through semantics. |
| `eq:categorical_unimix` | 352 | `recall2imagine/nrsm_nets.py: prior/posterior statistics; nets.py: distributions; jaxutils.py: OneHotDist` | Existing context-conditioned q and prior sampling, unimix and straight-through semantics. |
| `eq:ct_latent_distributions` | 357 | `recall2imagine/nrsm_nets.py: prior/posterior statistics; nets.py: distributions; jaxutils.py: OneHotDist` | Existing context-conditioned q and prior sampling, unimix and straight-through semantics. |
| `eq:categorical_st` | 363 | `recall2imagine/nrsm_nets.py: prior/posterior statistics; nets.py: distributions; jaxutils.py: OneHotDist` | Existing context-conditioned q and prior sampling, unimix and straight-through semantics. |
| `eq:zero_state_reset` | 371 | `recall2imagine/nrsm_nets.py: NRSM.initial, NRSM._transition, NRSM._obs_step` | Current CTM-only structured carry/reset specification; dynamic verification recorded separately. |
| `eq:tick_latent_cost` | 383 | `recall2imagine/nrsm_nets.py: tick_statistics; agent.py: WorldModel.loss` | Tick objective and derivatives unchanged; scale default zero. |
| `eq:tick_certainty` | 383 | `recall2imagine/nrsm_nets.py: tick_statistics; agent.py: WorldModel.loss` | Tick objective and derivatives unchanged; scale default zero. |
| `eq:error_tick_weight` | 391 | `recall2imagine/nrsm_nets.py: tick_statistics; agent.py: WorldModel.loss` | Tick objective and derivatives unchanged; scale default zero. |
| `eq:certainty_tick_weight` | 391 | `recall2imagine/nrsm_nets.py: tick_statistics; agent.py: WorldModel.loss` | Tick objective and derivatives unchanged; scale default zero. |
| `eq:nrsm_tick_loss` | 391 | `recall2imagine/nrsm_nets.py: tick_statistics; agent.py: WorldModel.loss` | Tick objective and derivatives unchanged; scale default zero. |
| `eq:credit_path` | 401 | `recall2imagine/nrsm_nets.py: tick_statistics; agent.py: WorldModel.loss` | Tick objective and derivatives unchanged; scale default zero. |
| `eq:prediction_factorization` | 413 | `recall2imagine/agent.py: WorldModel.loss, WorldModel.reward; reward_nets.ActionConditionedMLP` | Action-conditioning signature only: aligned clipped incoming action, native arrival reward, same255bin two-hot NLL/target/averaging/weights/optimizer; reset/windowfirst mask both transition and head. |
| `eq:symlog` | 423 | `recall2imagine/jaxutils.py: symlog, symexp, DiscDist; nets.py: Dist` | Inherited prediction transforms/targets/readout, unchanged. |
| `eq:nrsm_observation_loss` | 431 | `recall2imagine/agent.py: WorldModel.loss; nets.py: dyn_loss, rep_loss; jaxutils.py: distributions` | Existing objective and coefficients; reward changes only conditioning features. |
| `eq:nrsm_image_loss` | 431 | `recall2imagine/agent.py: WorldModel.loss; nets.py: dyn_loss, rep_loss; jaxutils.py: distributions` | Existing objective and coefficients; reward changes only conditioning features. |
| `eq:twohot_target` | 441 | `recall2imagine/jaxutils.py: symlog, symexp, DiscDist; nets.py: Dist` | Inherited prediction transforms/targets/readout, unchanged. |
| `eq:twohot_loss` | 447 | `recall2imagine/jaxutils.py: symlog, symexp, DiscDist; nets.py: Dist` | Inherited prediction transforms/targets/readout, unchanged. |
| `eq:twohot_readout` | 447 | `recall2imagine/jaxutils.py: symlog, symexp, DiscDist; nets.py: Dist` | Inherited prediction transforms/targets/readout, unchanged. |
| `eq:nrsm_reward_loss` | 454 | `recall2imagine/agent.py: WorldModel.loss, WorldModel.reward; reward_nets.ActionConditionedMLP` | Action-conditioning signature only: aligned clipped incoming action, native arrival reward, same255bin two-hot NLL/target/averaging/weights/optimizer; reset/windowfirst mask both transition and head. |
| `eq:nrsm_done_loss` | 462 | `recall2imagine/agent.py: WorldModel.loss; nets.py: dyn_loss, rep_loss; jaxutils.py: distributions` | Existing objective and coefficients; reward changes only conditioning features. |
| `eq:dynamics_kl_loss` | 469 | `recall2imagine/agent.py: WorldModel.loss; nets.py: dyn_loss, rep_loss; jaxutils.py: distributions` | Existing objective and coefficients; reward changes only conditioning features. |
| `eq:representation_kl_loss` | 469 | `recall2imagine/agent.py: WorldModel.loss; nets.py: dyn_loss, rep_loss; jaxutils.py: distributions` | Existing objective and coefficients; reward changes only conditioning features. |
| `eq:model_loss` | 476 | `recall2imagine/agent.py: WorldModel.loss; nets.py: dyn_loss, rep_loss; jaxutils.py: distributions` | Existing objective and coefficients; reward changes only conditioning features. |
| `eq:imagined_actor` | 490 | `recall2imagine/agent.py: WorldModel.imagine; nrsm_nets.py: NRSM.img_step; behaviors.py` | Full CTM histories carried; action causes successor reward through transition; initial reward excluded. |
| `eq:imagined_context` | 490 | `recall2imagine/agent.py: WorldModel.imagine; nrsm_nets.py: NRSM.img_step; behaviors.py` | Full CTM histories carried; action causes successor reward through transition; initial reward excluded. |
| `eq:imagined_latent` | 490 | `recall2imagine/agent.py: WorldModel.imagine; nrsm_nets.py: NRSM.img_step; behaviors.py` | Full CTM histories carried; action causes successor reward through transition; initial reward excluded. |
| `eq:discrete_action_sample` | 502 | `recall2imagine/agent.py: ImagActorCritic; nets.py: Dist; jaxutils.py: distributions` | Inherited policy distribution, unchanged. |
| `eq:continuous_action_sample` | 508 | `recall2imagine/agent.py: ImagActorCritic; nets.py: Dist; jaxutils.py: distributions` | Inherited policy distribution, unchanged. |
| `eq:continuous_logprob` | 516 | `recall2imagine/agent.py: ImagActorCritic; nets.py: Dist; jaxutils.py: distributions` | Inherited policy distribution, unchanged. |
| `eq:imagined_reward` | 526 | `recall2imagine/agent.py: WorldModel.imagine/reward; behaviors.py: unchanged extrinsic reward slicing` | Successor state and its generating clipped action jointly condition reward; ephemeral reward_action=[zero;action[:-1]], initial reward excluded, no new recurrent field. |
| `eq:imagined_continuation` | 526 | `recall2imagine/agent.py: WorldModel.imagine; behaviors.py: inherited continuation weighting` | Individual continuation expression byte-identical to v11; shares an align group with changed reward-conditioning signature. Native target/mode/gamma handling unchanged. |
| `eq:lambda_return` | 536 | `recall2imagine/agent.py: ImagActorCritic, VFunction; jaxutils.py: lambda_return, Moments, SlowUpdater` | Inherited actor/critic objective and gradient routing, unchanged. |
| `eq:return_normalization` | 542 | `recall2imagine/agent.py: ImagActorCritic, VFunction; jaxutils.py: lambda_return, Moments, SlowUpdater` | Inherited actor/critic objective and gradient routing, unchanged. |
| `eq:actor_loss` | 550 | `recall2imagine/agent.py: ImagActorCritic, VFunction; jaxutils.py: lambda_return, Moments, SlowUpdater` | Inherited actor/critic objective and gradient routing, unchanged. |
| `eq:value_loss` | 558 | `recall2imagine/agent.py: ImagActorCritic, VFunction; jaxutils.py: lambda_return, Moments, SlowUpdater` | Inherited actor/critic objective and gradient routing, unchanged. |
| `eq:slow_value_update` | 567 | `recall2imagine/agent.py: ImagActorCritic, VFunction; jaxutils.py: lambda_return, Moments, SlowUpdater` | Inherited actor/critic objective and gradient routing, unchanged. |
| `eq:ct_time_initialization` | 722 | `recall2imagine/nrsm_nets.py: NRSM time-scale/write calibration and _transition` | Bounded .1..100 environment-step T; rho=exp(-1/(K*T)); expm1 kappa; additive beta_w*kappa*gate*candidate; beta_w=.1; append updated activity, no activity LN/detach. |
| `eq:ct_unrolled_activity` | 732 | `recall2imagine/nrsm_nets.py: NRSM._transition; independent analytic/numerical references` | Direct rho^K identity; constant-gate/candidate write calibration only; conservative value bound about10.05 at defaults, not a full Jacobian or learned-memory guarantee. |
| `eq:ct_constant_write` | 739 | `recall2imagine/nrsm_nets.py: NRSM._transition; independent analytic/numerical references` | Direct rho^K identity; constant-gate/candidate write calibration only; conservative value bound about10.05 at defaults, not a full Jacobian or learned-memory guarantee. |
| `eq:ct_activity_bound` | 747 | `recall2imagine/nrsm_nets.py: NRSM._transition; independent analytic/numerical references` | Direct rho^K identity; constant-gate/candidate write calibration only; conservative value bound about10.05 at defaults, not a full Jacobian or learned-memory guarantee. |
