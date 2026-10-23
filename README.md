> 独立 v12 备份。备份内容和已有训练恢复说明见 [BACKUP.md](BACKUP.md)。

# CT-WM on the Recall2Imagine framework

当前版本为 `ctm-action-conditioned-reward-v12`。CTM 自己保存跨环境步神经历史，
并在每次状态转移内进行多 tick 计算；已移除独立 SSM、`ltm`、memory backbone
及读取 SSM 的预测和策略旁路。训练框架、原生任务、回放、奖励定义、基础预测损失、
balanced/free-nats KL、symlog 奖励/价值分布和 Actor-Critic 沿用当前 R2I 接入。

## 当前机制

每个 tick 用现有 GLU/LayerNorm synapse 和私有 NLM 的共享隐藏层，分别计算候选活动和独立写门：

```text
candidate, write_gate = PrivateNLM(pre_trace)
rho_i = exp(-1 / (K * T_i))
kappa_i = expm1(-1 / (K * T_i)) / expm1(-1 / T_i)
activity = rho * activity + write_scale * kappa * write_gate * candidate
post_trace = append(post_trace, activity)
sync = synchronization(post_trace, fixed_spread_neuron_pairs)
deter = concat(activity, tanh_projection(concat(activity, sync)))
```

`T_i` 是每个神经元可学习的时间常数，单位为环境步。正时间常数采用有界 log-space
参数化，默认范围 0.1–100，初始化在 log-space 中从范围的 1% 到 99% 等距分布。
当神经元只有一个时使用中点。K 次更新的直接旧活动通路系数为 `exp(-1/T_i)`；
候选活动仍依赖非线性反馈，因此这不是线性 SSM 的整体稳定性或长期记忆保证。
独立写门是输入相关的 sigmoid 输出，门输出权重和偏置初始化为零，初始值 0.5。
`write_scale=.1` 是固定写入预算，作为配置保存。若一个环境步内候选和门不变，
K 次更新的总写入为 `write_scale * write_gate * candidate`，不再被 `1-exp(-1/T_i)`
额外缩小；一般情况下各 tick 的候选和门会变化。选择性写入仍需学习。

跨环境步传递 `pre_trace`、`post_trace` 和派生读出 `deter`，当前活动直接取
`post_trace[..., -1]`，没有新增递归记忆张量。递归活动本身不做 LayerNorm；
LayerNorm 仍位于现有 synapse 内。每个新回合恢复零神经历史。
新更新不是凸插值，活动不再限制于 [-1,1]；默认参数的保守前向幅度界约 10.05。
`deter` 的前 64 维原样复制当前活动，后 64 维为活动与同步的 tanh 投影。
同步从全部 64 个神经元中固定等距选择 16 个（含两端），形成 136 个上三角配对，
覆盖初始快慢尺度；不随训练中的时间常数动态排序。保持 `deter > neurons`。
观测仍只进入 `q(z|deter,encoder(observation))`，下一转移通过 z 影响神经动态。
真实过滤和 imagination 使用同一 `_transition`。

decoder、continuation、actor 和 critic 继续读取 `[deter,stoch]`。
reward 保留相同状态特征，并用产生当前奖励的 incoming action 调节第一层对原始活动的映射：

```text
deter = concat(activity, projected)
reward_h0 = existing_activation_and_norm(
    shared_kernel @ concat(projected, flatten(stoch))
    + sum_k incoming_action[k] * action_kernel[k] @ activity)
```

离散 one-hot 动作使用加权求和，不使用 argmax；后续奖励隐藏层、255-bin 输出与 NLL 不变。
连续动作使用 `[1, flatten(action)]` 的仿射条件权重，零动作仍保留基础活动映射。
没有额外隐藏层、记忆或监督标签。当前两动作、64活动、512隐藏宽度只多32768个权重。
训练仍把 outgoing action 右移为 incoming action，reset 行清零；想象状态
`state[t+1]` 对应 `action[t]` 产生的奖励。想象轨迹用临时 `reward_action` 保存这个对齐，
它不加入递归状态；首个 dummy 奖励仍排除。255-bin symlog 分布及其 NLL 不变。
没有新增辅助 loss、奖励重加权或额外记忆网络；tick scale 默认 0。

定义见 [docs/METHOD.md](docs/METHOD.md)，逐式对应见
[docs/FORMULA_ALIGNMENT.md](docs/FORMULA_ALIGNMENT.md)，验证见
[docs/VALIDATION.md](docs/VALIDATION.md)。这是借鉴 SSM 时间尺度思想的 CTM 变体，
不声称原版 CTM/S3M 的原样复现或已有收敛保证。

## 代码结构

```text
CTWM.py                       统一训练入口
recall2imagine/nrsm_nets.py    CTM 神经递推、NLM、同步读出和 tick 诊断
recall2imagine/reward_nets.py  奖励头第一层的动作条件读出
recall2imagine/agent.py        世界模型和 Actor-Critic
recall2imagine/configs.yaml    上游预设、ctwm、ctwm_smoke
recall2imagine/embodied/       环境、回放、日志和训练
scripts/                      GPU 和任务接口检查
tests/                        数值、集成和保存恢复测试
docs/                         方法、公式、验证和任务说明
```

上游 GRU/S3M 的文件保留以支持框架中的其他模型，CTWM 的 `nrsm` 路径不调用它们。

## RTX 5090 / WSL 运行

当前已运行的旧进程继续使用旧结构。v12 的奖励头第一层参数已改变，必须使用新的输出
目录；旧输出还包含 replay 内嵌模型状态，不能只删除顶层 checkpoint 后复用。
v12 checkpoint 只支持同架构、同配置恢复，拒载 v11 及更早版本。安装说明见 [docs/GPU_RUNTIME.md](docs/GPU_RUNTIME.md)。

```bash
cd /mnt/c/Users/Administrator/Desktop/v12
source /home/administrator/.venvs/ctwm-r2i-gpu/bin/activate
unset LD_LIBRARY_PATH CUDA_ROOT CUDA_PATH JAX_PLATFORMS JAX_PLATFORM_NAME
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$PWD/outputs/.jax_cache"

python scripts/gpu_check.py --require-gpu && \
python -u CTWM.py --configs small ctwm \
  --jax.platform gpu --jax.precision float32 --jax.prealloc False \
  --task bsuite_memory_len/30 --envs.amount 1 --envs.parallel none \
  --batch_size 32 --batch_length 1024 --replay_size 10000000 \
  --imag_horizon 15 --run.train_ratio 1024 --nrsm_tick_scale 0 \
  --seed 0 --run.steps 1000000 \
  --logdir outputs/memory_len30_v12_fresh001
```

B32/T1024、1M 总预算、seed0 和原生稀疏奖励沿用当前实验。`small` 调整共享网络容量，
CTM 大小由 `ctwm` 配置控制。`memory_len/30` 的 30 是原生记忆长度，计数包含 reset
记录。首次执行需要图编译，CPU 短测不能代替正式 GPU 吞吐或收敛测量。

## 默认 CTM 配置

| 项目 | ctwm 默认值 |
|---|---|
| 数值精度 | float32，无混合精度 |
| 神经动态 | 64 neurons，pre/post 各 8 ticks 历史，每环境步 4 ticks |
| 神经元时间常数 | `time_constant_min=.1`、`time_constant_max=100`，单位环境步 |
| Synapse | 两个 Linear → GLU → LayerNorm 块，输出 64 维 |
| 私有 NLM | 每神经元独立，共享隐藏层输出 tanh 候选和 sigmoid 写门；门初值 0.5 |
| 写入预算 | `write_scale=.1`；使用稳定 expm1 比值校准 K 次写入 |
| 同步 | 全部神经元间等距选 16 个、136 对；现有 learnable decay 及其公式保留 |
| 确定性读出 | 原始活动 64 维直通＋活动/同步投影 64 维，合计 deter128 |
| 随机 latent | 8×8 categoricals |
| 预测/策略/价值状态输入 | `[deter,stoch]`，默认 192 维；奖励首层额外由 aligned incoming action 调节 |
| 回放 | FIFO/LFS，10,000,000 条记录 |
| 序列与想象 | 默认 batch4/length1024；正式命令 B32；H15 |
| 基础 KL | dyn0.5、rep0.1，free threshold1；任务预设可覆盖 |
| Tick KL | 默认 scale0，不加入训练目标 |

`T_i` 属于模型参数，随 checkpoint 保存；它不是随回合变化的状态。
在线神经历史和环境不保存到 checkpoint，因此恢复不能精确接续回合中的记忆。
配置优先级为 defaults → presets → CLI，建议 `small`/任务预设在前、`ctwm` 在后。

终端使用 tqdm，指标写入 metrics.jsonl，每回合原生回报写入 scores.jsonl。
`run.log_every`、`run.save_every` 单位为秒。默认不上传 WandB。
旧版验证只代表对应旧结构，当前通过项目以 docs/VALIDATION.md 为准。

```bash
python -m unittest discover -s tests -v
python scripts/smoke_env.py --task bsuite_memory_len/4 bsuite_discounting_chain/10 \
  --steps 256 --seed 0
```

模型测试应显式使用 CPU；GPU 环境检查用 `scripts/gpu_check.py --require-gpu`。
TensorFlow 不是模型执行引擎。其他任务依赖与支持边界见
[docs/TASK_SUPPORT.md](docs/TASK_SUPPORT.md)。保留上游 [LICENSE](LICENSE) 与
[UPSTREAM_README.md](UPSTREAM_README.md)。
