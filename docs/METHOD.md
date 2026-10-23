# CT-WM: action-conditioned first reward layer (v12)

Schema: `ctm-action-conditioned-reward-v12`, 2026-10-22.
This is the current method; earlier numerical evidence remains specific to its
named revision. The new variant retains CTM multi-tick computation and carries
pre/post histories across environment steps. It adds no independent SSM.

## State and unchanged interfaces

The predictive state is b=(c,z), c=(deter,Hpre,Hpost). Both histories are [B,D,M],
oldest to newest. Current activity is Hpost[..., -1]; it is not duplicated in an
extra recurrent tensor. `logit` caches the categorical distribution. Gate and
candidate values are temporary computations, not states.

At defaults D64/M8/K4, z is 8×8 categorical and deter remains128. Decoder,
continuation, actor, online critic and slow critic still read [deter,stoch],
width192. Reward keeps that base vector but changes only its first linear map
to use the aligned generating action. No hidden layer, memory module or extra
head is added. Every gated CTM equation/parameter contract below matches v11.

## Tick computation and independent write gate

Carry both histories; nu0=Hpost[..., -1]. The fixed environment-transition input
u=[previous_deter,flatten(previous_z),incoming_action_clipped] is shared across
all K ticks. Current observation is absent. The existing action encoding and
scaled stop-gradient clipping convention remain unchanged.

For each tick, using the same parameters:

    alpha = NavigationBlock2(NavigationBlock1(concat(nu_previous,u)))
    Hpre = shift_append(Hpre,alpha)
    hidden_d = ELU(W1_d Hpre[d,:]+b1_d)
    candidate_d = tanh(w2_d^T hidden_d+b2_d)
    gate_d = sigmoid(write_w_d^T hidden_d+write_b_d)
    nu = rho*nu_previous + write_scale*kappa*gate*candidate
    Hpost = shift_append(Hpost,nu)
    sync = finite_window_synchronization(Hpost,spread_indices)
    deter_tau = concat(nu,tanh(sync_projection(concat(nu,sync))))

Candidate and gate share the existing private ELU hidden layer but have distinct
output parameters per neuron. `write_w` and `write_b` initialize to zero, so every
gate is initially .5. Hidden/candidate retain per-neuron Glorot uniform weights
and zero biases. No hidden layer or recurrent memory module is added.

NavigationBlock remains Linear(2D)→GLU(D)→affine LayerNorm(D), population variance,
epsilon1e-5 and no dropout. Its kernels/biases keep uniform ±1/sqrt(fan_in), LN1/0.
The two-action default input remains64+128+64+2=258. No LayerNorm, tanh, clipping
or detach is applied to the retained activity as a whole. Only updated activity
is written into post-history; candidate/gate are diagnostics when exported.

## Retention and calibrated writing

Existing positive bounded time constants retain environment-step units:

    f_d = sigmoid(time_constant_raw[d])
    T_d = exp(log(T_min)+f_d*log(T_max/T_min))
    rho_d = exp(-1/(K*T_d))
    kappa_d = expm1(-1/(K*T_d)) / expm1(-1/T_d)

Defaults are T_min=.1, T_max=100. Bounds must be finite and 0<T_min<T_max.
Initialization f=linspace(.01,.99,D), D1=.5, gives log-spaced T strictly inside
the bounds. T is a global learned parameter, distinct from the input-dependent
gate. `write_scale=.1` is a fixed configured budget, not a learned parameter or
loss scale. The expm1 ratio avoids cancellation in 1-exp(-x).

    rho^K = exp(-1/T)
    nu_K = rho^K*nu0 + write_scale*kappa*sum_j rho^(K-j)*gate_j*candidate_j

When BOTH gate and candidate are constant within the K ticks, exactly:

    nu_K = exp(-1/T)*nu0 + write_scale*gate*candidate

Thus the calibrated one-step write is not forced small by a large T. These are
conditional/direct-path identities. In the actual nonlinear system, gate and
candidate change with evolving activity/history; changing K changes computation.
The full Jacobian is not rho^K. Independent parameters permit selective admission
but do not establish learned selectivity, useful retention or gradient stability.
The .1 budget is conservative, not proven optimal; gate=.5 initially gives a
constant-signal one-step contribution .05*candidate and changes fast-unit scale.

This is additive rather than convex interpolation. Activity can exceed [-1,1].
With zero reset, bounded candidate/gate and configured T bounds, exact arithmetic
gives abs(nu)<=write_scale/(1-exp(-1/T_max)), about10.05 at defaults. This follows
from abs(nu')<=rho*abs(nu)+write_scale*kappa and remains a value bound even if
parameters change while bounds hold. It is not a stability/gradient/control
guarantee. Synchronization products and tanh projection saturation require
monitoring; no global activity normalization hides those effects.

## Spread-pair synchronization and raw activity readout

Use fixed zero-based indices np.rint(np.linspace(0,D-1,D_s)).astype(int), rounding
ties to even. D_s must be within1..D; D_s1 selects index0. Default D64/D_s16 gives
0,4,8,13,17,21,25,29,34,38,42,46,50,55,59,63. All upper-triangular pairs, including
self-pairs, are in lexicographic order, giving136 features. Since T is initially
ordered, these cover initially fast/slow neurons and cross-products. The mapping
is not dynamically sorted, and learned T may reorder later.

For history columns l=0,...,M-1, the original statistic remains:

    decay_ij = 4*sigmoid(sync_decay_raw_ij)
    weight_l,ij = exp(-decay_ij*(M-1-l))
    sync_ij = sum_l weight_l,ij*Hpost_i,l*Hpost_j,l / sqrt(sum_l weight_l,ij)

Effective decay initializes .01; initial zero slots remain in the denominator.
This finite-window statistic is recomputed, not accumulated over all past time.
It is distinct from activity rho and write_scale.

`deter = concat(raw_activity,tanh(sync_projection([raw_activity,sync])))`.
Require deter_width>D. The existing projection output shrinks to deter_width-D;
default200→64 plus raw64 still gives128. The first64 coordinates are an identity
path, with no projection/normalization/saturation. No extra prediction head or
state is added. This makes stored activity available to existing heads without
proving that their learned mappings use a task cue or action correctly.

## Filtering, reset and imagination

Filtering still draws q(z_t|deter_t,e(x_t)); imagination draws p(z_t|deter_t).
Current observations update z only, influencing deterministic memory at the next
transition. No f_post or auxiliary decoder is added. This posterior differs from
R2I's default non-recurrent q(z_t|x_t) and must be disclosed in comparisons.

Physical/window resets zero both histories, deter, z/logits and incoming action.
Global time constants/gate parameters are not reset. Sampled-window first-record
resets retain the reference finite-context approximation; unrelated batch carry
must not leak. Online state persists throughout a real episode and resets at its actual boundary.

Real and imagined paths share every gated tick and the complete state. Detached
B*T starts, horizon15, terminal weighting and action alignment are unchanged.
Action h produces successor h+1; reward reads that successor and the same
generating clipped action. Temporary reward_action=[zero;action[:-1]] is aligned
with the imagined state array, and the initial dummy reward is excluded. It is
head metadata, not a recurrent field or policy input. Checkpoints save parameters/optimizers/RNG/counters,
not live environment histories; a resume begins new episodes.

## Action-conditioned first reward layer

Split the existing deter=[nu,p] into raw CTM activity nu64 and projected features
p64; chi=[p,flatten(z)] remains128. Replace only the existing first linear map:

    q = U @ chi + sum_k omega(action)_k * W_k @ nu + bias
    h1 = configured_activation(configured_norm(q))

Discrete omega is flattened soft/onehot action, including factored categories;
there is no argmax/detach. Continuous omega=[1,flatten(action)] supplies an affine
bank with an activity-dependent zero-action baseline. Norm/activation and their
ordering follow the inherited MLP. As before, layer norm omits dense bias and
retains its own affine parameters; without norm, the linear bias convention is
unchanged. Later hidden layers, the255bin output, initialization family, NLL,
weights and optimizers are unchanged. In the two-action small preset, old64→512
activity weights become two64→512 maps, adding32768weights; no extra hidden layer.

The reward argument is the incoming action that produced the arrival state and
native reward, not that record's outgoing action. Training right-shifts replay
actions using valid carry, masks every physical/window is_first to zero, and
uses the same NRSM flattened clipping convention as the transition. An artificial
window-first arrival reward remains in the original loss despite omitted history;
no new mask/reweighting is added. Imagination pairs state[h+1] with action[h].
Initial and final-array conventions retain the inherited return construction.

Action-feature multiplicative prediction is prior work (Oh et al.,2015,§3.2,
https://arxiv.org/abs/1507.08750). This application is not a claim that the bank is
novel or that CMM proved sparse-reward success. No CMM long-term store is added.
Frozen v11 features can test readout learning independently: split whole paired
cue/action groups before fitting, freeze the backbone, retain the reward NLL,
and report held-out prior/posterior correct-minus-wrong rewards and ranking.
Any balanced-terminal diagnostic is labelled separately from production replay
sampling and policy evaluation. Completed results and limitations are recorded in ACTION_REWARD_VALIDATION.md
and its experiment receipt; no end-to-end policy improvement is inferred.

## Unchanged rewards, objectives and attribution

    L_model = mean(L_observation+L_reward+L_cont+.5*L_dyn+.1*L_rep)
              + nrsm_tick_scale*L_tick
    L_dyn = max(1,KL(stop_gradient(q_final)||p_final))
    L_rep = max(1,KL(q_final||stop_gradient(p_final)))

Native task rewards/signs, vector/image decoders, 255-bin symlog two-hot reward
and value distributions, continuation1-is_terminal, all loss coefficients,
tick weighting/derivatives, actor/critic objectives, optimizers, replay, reset
loss masks, exploration and training budgets are unchanged. Tick scale remains0;
multi-tick computation remains active. No retention loss, auxiliary loss or
reward reweighting is introduced. Only first-layer reward parameterization changes.

Private temporal networks, navigation synapse and synchronization are adapted
from CTM. Controlled retention/multiple time scales are inspired by SSMs; separate
input admission and retention are a general gated-recurrence principle. This
particular gated CTM update is not original CTM, S3M, a diagonal SSM or imported
HiPPO; it has no associative parallel-scan or inherited S3M stability guarantee.
Existing nonlinear feedback still requires environment-time sequential scans.

## Compatibility and verification

Schema v12 rejects previous schemas for production training; the reward-bank
weights and aligned-action contract differ. A frozen-v11 diagnostic may read its
own archived schema explicitly without production checkpoint migration. Use a new output directory; editing source does not migrate
an existing training process or checkpoint. Old experiment records are preserved.

FORMULA_ALIGNMENT/AUDIT and the paper inventory map current expressions. Current
runtime/compiler/test results belong in the versioned v12 validation receipt;
v11 pass counts and diagnostics do not transfer. No learning gain, GPU throughput
or successful cue retention is claimed from source changes alone.
