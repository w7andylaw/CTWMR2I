# CT-WM migrated variant

This folder implements the migration decision accepted in the conversation:
use the R2I training framework, replace S3M with NRSM, retain the baseline
training objectives, and test internal-tick supervision separately.
The manuscript has been revised together with the persistent-memory model.
The earlier posterior correction, auxiliary reconstruction, Gaussian reward
regression, and scalar value regression are not part of this implementation.

## Recurrent state and internal computation

The persistent world-model state is b_t=(c_t,z_t), with structured deterministic
memory c_t=(m_t,Hpre_t,Hpost_t). The two histories have shape [D,M]. The compact
vector m_t is a synchronization readout cache, not an independently updated
memory. R2I-compatible state fields are `deter`, `pre_trace`, `post_trace`, and
`stoch`; `logit` only caches categorical distribution parameters. Each history
is carried across environment steps. Zero sentinels initialize a new episode
or sampled replay window; transitions do not regenerate histories.

At the start of a transition:

    Hpre_(t,0) = Hpre_(t-1), Hpost_(t,0) = Hpost_(t-1)
    nu_(t,0) = Hpost_(t-1)[..., -1]
    u_t = concat(m_(t-1), vec(z_(t-1)), a_(t-1))

Each of K thought ticks uses the same synapse and private neuronal networks:

    alpha = synapse(concat(nu, u_t))
    Hpre = shift_append(Hpre, alpha)
    nu_i = private_NLM_i(Hpre_i)
    Hpost = shift_append(Hpost, nu)

For selected pairs i<=j, the finite-window synchronization is

    rho_ij = clip(beta_ij, 0, 4)
    w_l,ij = exp(-rho_ij * (M - 1 - l)), l = 0,...,M-1
    S_ij = sum_l(w_l,ij * Hpost_i,l * Hpost_j,l) / sqrt(sum_l(w_l,ij))
    m = tanh(linear(S))

This follows the finite pre/post-activation histories of the official CTM RL
implementation. It does not combine that window with an unbounded recurrent
synchronization accumulator. beta starts at zero (no decay). The synapse,
conditioning on the previous latent/action, and generative p/q heads are our
world-model adaptation, not an exact copy of CTM's full architecture.

The shared prior p(z|m_tau) and posterior q(z|m_tau,encoder(x_t)) are evaluated
at each internal tick during training. Only the final posterior is sampled
for a real step. Imagination samples the final prior without observations.
This posterior is conditioned on the context; it deliberately differs from
R2I's default non-recurrent q(z|x). That difference must also be disclosed in
comparisons, rather than attributing every effect to synchronization alone.

Actual episode resets and replay-window initial resets clear the complete c,
z, distribution cache, and incoming action. The observation x_t only enters
q; it can influence neuronal history at the next transition through z_t.
Real filtering and imagination use exactly the same deterministic update;
imagination must carry both histories from its starting posterior. Heads and
actor/critic read only concat(m,vec(z)), never flattened history buffers.

M counts internal ticks, not environment steps and not a guaranteed maximum
memory span. Beyond the explicit windows, information must be learned through
recurrent activity. Independent replay windows cannot share the preceding
batch's histories; the sampler's forced first-record reset clears them. This
does not recover cues outside the sampled window. A single compiled sequential
scan processes environment time; NRSM does not inherit S3M's parallel scan.

## Loss and gradient routing

The world-model objective is

    L_model = L_R2I_base + nrsm_tick_scale * L_tick
    L_R2I_base = L_predictions + 0.5 L_dyn + 0.1 L_rep
    L_dyn = max(1, KL(stop_gradient(q_final) || p_final))
    L_rep = max(1, KL(q_final || stop_gradient(p_final)))

`nrsm_tick_scale` defaults to zero: tick statistics remain available, but only
the R2I base objective contributes to training. Multi-tick transitions and
persistent neuronal histories remain active and receive gradients through
the final prediction and base KL losses. Adaptive weights do not fuse or
select the recurrent state. The displayed base coefficients are repository
defaults; task presets can override them. Prediction heads, image/vector
reconstruction likelihoods, symlog reward
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

When a positive scale explicitly enables this optional term, all weight
derivatives remain enabled. The final tick is included, so it receives additional
regularization beyond the baseline balanced KL. Low KL or high certainty alone
does not establish accurate predictions or good control.
The added raw tick KL is not clipped at the free-nat threshold: below that
threshold it can still produce a gradient, if its scale is positive, even when
the two baseline KL terms are flat. A positive scale therefore removes the
base objective's free-information region from the complete objective. The
default zero scale preserves that region. Nonzero scales are experimental
choices to validate, not a stability guarantee.

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

Legacy transition-local NRSM checkpoints are incompatible. The wrapper records
the `persistent-traces-v2` architecture schema and rejects older metadata;
start this architecture in a new log directory. Existing output directories
are preserved.
