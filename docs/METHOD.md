# CT-WM migrated variant

This folder implements the migration decision accepted in the conversation:
use the R2I training framework, replace S3M with NRSM, retain the baseline
training objectives, and test internal-tick supervision separately.
It is **not** an implementation of every formula in the earlier 64-equation
manuscript, which still contains posterior correction, an auxiliary branch,
Gaussian reward regression, and scalar value regression. That manuscript
requires a separate revision before it describes this implementation.

## Recurrent state and internal computation

The only persistent world-model state is b_t=(c_t,z_t). R2I-compatible names
are `deter` and `stoch`; `logit` is a distribution cache, not extra memory.
At every transition concatenate the previous c, flattened z, and incoming
action. Learned tanh projections initialize neuronal activity and its finite
history. Synchronization accumulators start at zero at every environment step.
Each of K thought ticks executes the synapse network, history shift, private
per-neuron temporal MLP, exponential synchronization, and context projection.
The final context is c_t=cbar_t. There is no f_post gate and no auxiliary
reconstruction branch.

For selected pairs i<=j, rho=softplus(beta) and

    N_tau = exp(-rho) N_(tau-1) + nu_i,tau nu_j,tau
    Z_tau = exp(-rho) Z_(tau-1) + 1
    S_tau = N_tau / sqrt(Z_tau)

The shared prior p(z|c_tau) and posterior q(z|c_tau,encoder(x_t)) are evaluated
at each internal tick during training. Only the final posterior is sampled
for a real step. Imagination samples the final prior without observations.
This posterior is conditioned on the context; it deliberately differs from
R2I's default non-recurrent q(z|x). That difference must also be disclosed in
comparisons, rather than attributing every effect to synchronization alone.

Internal neuronal buffers are never carried across environment steps. Actual
episode resets and replay-window initial resets clear c, z, and incoming
actions. A single compiled sequential scan processes environment time; this
nonlinear NRSM does not inherit S3M's associative parallel scan.

## Loss and gradient routing

The world-model objective is

    L_model = L_R2I_base + nrsm_tick_scale * L_tick
    L_R2I_base = L_predictions + 0.5 L_dyn + 0.1 L_rep
    L_dyn = max(1, KL(stop_gradient(q_final) || p_final))
    L_rep = max(1, KL(q_final || stop_gradient(p_final)))

The displayed coefficients are repository defaults; task presets can override
them. Prediction heads, image/vector reconstruction likelihoods, symlog reward
and value distributions, return normalization, slow critic regularization and
EMA, actor losses, optimizers, and default 0.01 categorical unimix are inherited
from R2I. Their origin must be credited to the relevant Dreamer/R2I sources.
There is no additional standalone raw final KL term and no auxiliary loss.
Reset records are treated by the upstream objective, not the old manuscript's
incoming-transition mask.

Additional tick regularization uses the same mixture-adjusted distributions:

    ell_tau = KL(q_tau || p_tau)
    C_tau = 1 - H(p_tau) / (J log C)
    w_tau = [softmax(-alpha_error * ell)_tau
             + softmax(alpha_cert * certainty)_tau] / 2
    L_tick = mean_records sum_tau w_tau ell_tau

All weight derivatives remain enabled. The final
tick is included in this additional regularizer, so it does receive additional
regularization beyond the baseline balanced KL. Low KL or high certainty alone
does not establish accurate predictions or good control.
The added raw tick KL is not clipped at the free-nat threshold: below that
threshold it can still produce a gradient even when the two baseline KL terms
are flat. Thus the base implementation is retained, but the complete objective
does not retain its free-information region. This is an experimental design
choice to evaluate with the off/uniform/adaptive ablations, not a stability
guarantee.

Modes are `adaptive`, `uniform`, and `off`; scale zero also removes the term.
Off mode still computes diagnostics, so it is a loss ablation, not a claim of
removing all tick-head computation. Use identical budgets and seeds for a
comparison. Also compare K=1 against K>1 and report parameters and compute.

The model, actor, and critic have separate optimizers. Actor computations may
differentiate through fixed world-model computations for continuous actions;
the actor optimizer does not update the world-model parameters. R2I flattens
all B*T states for imagination; terminal starts have zero trajectory weight.
The continuation head predicts 1-is_terminal, and gamma is applied once during
imagination/return calculation.

## Attribution and claims

- DreamerV3: world-model training and imagination-based actor-critic machinery.
  https://arxiv.org/abs/2301.04104
- R2I/S3M: supplied engineering framework, replay and domain configurations.
  https://arxiv.org/abs/2403.04253
  https://github.com/chandar-lab/Recall2Imagine
- CTM: neuron-level temporal models, synchronization representations, and the
  inspiration for error/certainty-based internal-tick supervision.
  https://arxiv.org/abs/2505.05522
- NRSM is the proposed adaptation into a recurrent generative world model.
  Novelty and performance remain research questions, not outcomes of a smoke
  test. Neither replay length nor extra internal ticks guarantee long memory.

The original projects and their output directories are not modified.
