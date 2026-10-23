# Historical only: persistent-traces-v2 formula checklist

The text below is the 2026-10-20 trace-only checklist. Its 64 numbered equations, 57 labels, compiled numbers, and test claims do not apply to slow-ssm-thought-read-v4. It is preserved for provenance; use the current checklist and current label inventory.

# CT-WM 持久记忆版本公式与实现对应

当前编译稿有64个公式编号、57个公式标签；部分align中间行没有独立标签。
以下逐组列出所有编号并指向实现。接口/目标定义、上游继承模块和新增NRSM机制分别注明；
对应与短程数值检查不等于已复现任务成绩。当前项目是Desktop/CTWMR2I，已不使用旧UAV/HER代码。

| 编号 | 标签与代码路径 | 验证范围 |
|---|---|---|
| 1 | `eq:benchmark_pomdp`：`recall2imagine/embodied/envs/bsuite.py; recall2imagine/embodied/core/driver.py` | 原生环境/接口定义；本轮真实CLI验证BSuite，其他域见TASK_SUPPORT |
| 2 | `eq:benchmark_transition`：`recall2imagine/embodied/envs/bsuite.py; recall2imagine/embodied/core/driver.py` | 原生环境/接口定义；本轮真实CLI验证BSuite，其他域见TASK_SUPPORT |
| 3 | `eq:benchmark_observation`：`recall2imagine/embodied/envs/bsuite.py; recall2imagine/embodied/core/driver.py` | 原生环境/接口定义；本轮真实CLI验证BSuite，其他域见TASK_SUPPORT |
| 4 | `eq:benchmark_history`：`recall2imagine/embodied/envs/bsuite.py; recall2imagine/embodied/core/driver.py` | 原生环境/接口定义；本轮真实CLI验证BSuite，其他域见TASK_SUPPORT |
| 5 | `eq:benchmark_return`：`recall2imagine/embodied/envs/bsuite.py; recall2imagine/embodied/core/driver.py` | 原生环境/接口定义；本轮真实CLI验证BSuite，其他域见TASK_SUPPORT |
| 6 | `eq:benchmark_discount`：`recall2imagine/embodied/envs/bsuite.py; recall2imagine/embodied/core/driver.py` | 原生环境/接口定义；本轮真实CLI验证BSuite，其他域见TASK_SUPPORT |
| 7 | `eq:world_model_state`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization` | NumPy逐tick/跨步carry/reset/因果/JIT/梯度测试通过 |
| 8 | `eq:rssm_deterministic`：`recall2imagine/nets.py:RSSM` | 对照算子定义，非NRSM新增机制 |
| 9 | `eq:ct_prior_context`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization` | NumPy逐tick/跨步carry/reset/因果/JIT/梯度测试通过 |
| 10 | `eq:latent_feature`：`recall2imagine/agent.py:WorldModel.imagine; recall2imagine/nrsm_nets.py:NRSM.img_step` | 完整起点trace、同转移、gamma/terminal和起点梯度测试通过 |
| 11 | `eq:transition_action_clip`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization` | NumPy逐tick/跨步carry/reset/因果/JIT/梯度测试通过 |
| 12, 13, 14 | `eq:ct_tick_input`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization`<br>`eq:ct_tick_initialization`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization` | NumPy逐tick/跨步carry/reset/因果/JIT/梯度测试通过 |
| 15, 16, 17, 18 | `eq:ct_synapse_tick`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization`<br>`eq:ct_pre_history`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization`<br>`eq:ct_neuron_model`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization`<br>`eq:ct_post_history`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization` | NumPy逐tick/跨步carry/reset/因果/JIT/梯度测试通过 |
| 19 | `eq:private_nlm`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization` | NumPy逐tick/跨步carry/reset/因果/JIT/梯度测试通过 |
| 20 | `eq:ct_sync_weights`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization` | NumPy逐tick/跨步carry/reset/因果/JIT/梯度测试通过 |
| 21 | `eq:ct_sync_state`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization` | NumPy逐tick/跨步carry/reset/因果/JIT/梯度测试通过 |
| 22, 23 | `eq:ct_sync_context`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization`<br>`eq:ct_persistent_carry`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization` | NumPy逐tick/跨步carry/reset/因果/JIT/梯度测试通过 |
| 24, 25 | `eq:ct_dynamics_predictor`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization`<br>`eq:ct_encoder`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization` | NumPy逐tick/跨步carry/reset/因果/JIT/梯度测试通过 |
| 26 | `eq:categorical_unimix`：`recall2imagine/nets.py:RSSM._stats; recall2imagine/jaxutils.py:OneHotDist` | 继承R2I；categorical及联合更新测试通过 |
| 27, 28 | `eq:ct_latent_distributions`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization` | NumPy逐tick/跨步carry/reset/因果/JIT/梯度测试通过 |
| 29 | `eq:categorical_st`：`recall2imagine/nets.py:RSSM._stats; recall2imagine/jaxutils.py:OneHotDist` | 继承R2I；categorical及联合更新测试通过 |
| 30 | `eq:zero_state_reset`：`recall2imagine/nrsm_nets.py:NRSM, PrivateNLM, window_synchronization` | NumPy逐tick/跨步carry/reset/因果/JIT/梯度测试通过 |
| 31, 32 | `eq:tick_latent_cost`：`recall2imagine/nrsm_nets.py:tick_statistics, categorical_kl`<br>`eq:tick_certainty`：`recall2imagine/nrsm_nets.py:tick_statistics, categorical_kl` | KL数值、完整权重梯度和有限差分测试通过 |
| 33, 34, 35, 36 | `eq:error_tick_weight`：`recall2imagine/nrsm_nets.py:tick_statistics, categorical_kl`<br>`eq:certainty_tick_weight`：`recall2imagine/nrsm_nets.py:tick_statistics, categorical_kl`<br>`eq:nrsm_tick_loss`：`recall2imagine/nrsm_nets.py:tick_statistics, categorical_kl` | KL数值、完整权重梯度和有限差分测试通过 |
| 37 | `eq:credit_path`：`recall2imagine/nrsm_nets.py:tick_statistics, categorical_kl` | KL数值、完整权重梯度和有限差分测试通过 |
| 38 | `eq:symlog`：`recall2imagine/nets.py; recall2imagine/jaxutils.py; recall2imagine/agent.py:WorldModel.loss` | 继承R2I分布；向量/图像/联合训练数值检查通过 |
| 39, 40 | `eq:nrsm_observation_loss`：`recall2imagine/nets.py; recall2imagine/jaxutils.py; recall2imagine/agent.py:WorldModel.loss`<br>`eq:nrsm_image_loss`：`recall2imagine/nets.py; recall2imagine/jaxutils.py; recall2imagine/agent.py:WorldModel.loss` | 继承R2I分布；向量/图像/联合训练数值检查通过 |
| 41 | `eq:twohot_target`：`recall2imagine/nets.py; recall2imagine/jaxutils.py; recall2imagine/agent.py:WorldModel.loss` | 继承R2I分布；向量/图像/联合训练数值检查通过 |
| 42, 43 | `eq:twohot_loss`：`recall2imagine/nets.py; recall2imagine/jaxutils.py; recall2imagine/agent.py:WorldModel.loss`<br>`eq:twohot_readout`：`recall2imagine/nets.py; recall2imagine/jaxutils.py; recall2imagine/agent.py:WorldModel.loss` | 继承R2I分布；向量/图像/联合训练数值检查通过 |
| 44 | `eq:nrsm_done_loss`：`recall2imagine/nets.py; recall2imagine/jaxutils.py; recall2imagine/agent.py:WorldModel.loss` | 继承R2I分布；向量/图像/联合训练数值检查通过 |
| 45, 46 | `eq:dynamics_kl_loss`：`recall2imagine/nets.py:RSSM.dyn_loss, rep_loss; recall2imagine/agent.py:WorldModel.loss`<br>`eq:representation_kl_loss`：`recall2imagine/nets.py:RSSM.dyn_loss, rep_loss; recall2imagine/agent.py:WorldModel.loss` | 继承balanced/free-nats基础目标并增加tick；核心与集成测试通过 |
| 47 | `eq:model_loss`：`recall2imagine/nets.py:RSSM.dyn_loss, rep_loss; recall2imagine/agent.py:WorldModel.loss` | 继承balanced/free-nats基础目标并增加tick；核心与集成测试通过 |
| 48, 49, 50 | `eq:imagined_actor`：`recall2imagine/agent.py:WorldModel.imagine; recall2imagine/nrsm_nets.py:NRSM.img_step`<br>`eq:imagined_context`：`recall2imagine/agent.py:WorldModel.imagine; recall2imagine/nrsm_nets.py:NRSM.img_step`<br>`eq:imagined_latent`：`recall2imagine/agent.py:WorldModel.imagine; recall2imagine/nrsm_nets.py:NRSM.img_step` | 完整起点trace、同转移、gamma/terminal和起点梯度测试通过 |
| 51 | `eq:discrete_action_sample`：`recall2imagine/nets.py:Dist; recall2imagine/agent.py:Agent.policy` | 离散/连续联合更新及policy测试通过；eval保留采样 |
| 52, 53 | `eq:continuous_action_sample`：`recall2imagine/nets.py:Dist; recall2imagine/agent.py:Agent.policy` | 离散/连续联合更新及policy测试通过；eval保留采样 |
| 54 | `eq:continuous_logprob`：`recall2imagine/nets.py:Dist; recall2imagine/agent.py:Agent.policy` | 离散/连续联合更新及policy测试通过；eval保留采样 |
| 55, 56 | `eq:imagined_reward`：`recall2imagine/agent.py:WorldModel.imagine; recall2imagine/nrsm_nets.py:NRSM.img_step`<br>`eq:imagined_continuation`：`recall2imagine/agent.py:WorldModel.imagine; recall2imagine/nrsm_nets.py:NRSM.img_step` | 完整起点trace、同转移、gamma/terminal和起点梯度测试通过 |
| 57 | `eq:lambda_return`：`recall2imagine/agent.py:ImagActorCritic, VFunction; recall2imagine/jaxutils.py:Moments, SlowUpdater` | 继承R2I Actor-Critic；联合更新、WM参数隔离及恢复测试通过 |
| 58, 59, 60, 61 | `eq:return_normalization`：`recall2imagine/agent.py:ImagActorCritic, VFunction; recall2imagine/jaxutils.py:Moments, SlowUpdater` | 继承R2I Actor-Critic；联合更新、WM参数隔离及恢复测试通过 |
| 62 | `eq:actor_loss`：`recall2imagine/agent.py:ImagActorCritic, VFunction; recall2imagine/jaxutils.py:Moments, SlowUpdater` | 继承R2I Actor-Critic；联合更新、WM参数隔离及恢复测试通过 |
| 63 | `eq:value_loss`：`recall2imagine/agent.py:ImagActorCritic, VFunction; recall2imagine/jaxutils.py:Moments, SlowUpdater` | 继承R2I Actor-Critic；联合更新、WM参数隔离及恢复测试通过 |
| 64 | `eq:slow_value_update`：`recall2imagine/agent.py:ImagActorCritic, VFunction; recall2imagine/jaxutils.py:Moments, SlowUpdater` | 继承R2I Actor-Critic；联合更新、WM参数隔离及恢复测试通过 |

新版结构需从新的logdir重新训练，旧局部历史checkpoint不兼容。
CPU单元/集成/包装器/回放测试：48通过、1跳过（POPGym可选依赖缺失）。
真实BSuite短训练与恢复：160条记录、30次模型及AC更新；不代表记忆学习成绩。
完整GPU批次、长期收敛及其他未安装模拟器的验证仍需独立执行。
