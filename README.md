# CT-WM on the Recall2Imagine framework

独立迁移工程：NRSM 使用 JAX/Ninjax 实现；基础预测损失、balanced/free-nats
KL、symlog 奖励与价值头、Actor-Critic 和回放基础设施沿用 R2I。
确定性记忆持续保存神经元输入/输出历史；保留可消融的 tick consistency loss，
**没有后验记忆校正和辅助重构分支**。
原来的 CTWM 和 Recall2Imagine 文件夹不受此工程修改。

这是一份可运行的结构迁移。短程验证不代表复现了论文成绩，也不保证优于 R2I。
论文已同步为结构化记忆版本，方法与继承损失的定义同步更新。
数学定义、继承关系和差异见 [docs/METHOD.md](docs/METHOD.md)。

## 结构

```text
CTWM.py                       统一训练入口
recall2imagine/nrsm_nets.py    NRSM、private NLM、同步计算和 tick loss
recall2imagine/agent.py        R2I 世界模型与 Actor-Critic 接入
recall2imagine/configs.yaml    上游任务预设、ctwm、ctwm_smoke
recall2imagine/embodied/       环境、回放、日志、训练和 checkpoint
scripts/                      GPU 检查、任务接口检查
tests/                        公式、集成、环境、保存恢复测试
docs/                         方法说明、验证记录和支持范围
```

## RTX 5090 / WSL 运行

使用 Python 3.11 的独立环境。不要把 GPU 依赖装进原来的 `dreamer`，也不要在同一
环境叠加安装 CPU 和 GPU 两份 requirements。已准备环境的具体说明见
[docs/GPU_RUNTIME.md](docs/GPU_RUNTIME.md)。

```bash
cd /mnt/c/Users/Administrator/Desktop/CTWMR2I
source /home/administrator/.venvs/ctwm-r2i-gpu/bin/activate
unset LD_LIBRARY_PATH CUDA_ROOT CUDA_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python scripts/gpu_check.py --require-gpu
XLA_FLAGS=--xla_gpu_autotune_level=0 python -u CTWM.py --configs ctwm ctwm_smoke \
  --jax.platform gpu \
  --task bsuite_memory_len/4 \
  --logdir outputs/gpu_smoke_user \
  --run.steps 200
```

`ctwm_smoke` 使用小模型、2×8 序列、H=3、Uniform replay，仅供启动验证。
它是 CPU 预设，所以 GPU 运行时必须加 `--jax.platform gpu`。
上面的短测试关闭 XLA 自动调优，便于先确认训练能运行；正式命令保留默认调优，
两者都通过了小模型验证，但尚未比较稳定训练吞吐。

正式尺寸例子（去掉 `ctwm_smoke`；首次执行需要编译）：

```bash
python -u CTWM.py --configs small ctwm \
  --jax.platform gpu --jax.prealloc False \
  --task bsuite_memory_len/30 \
  --envs.amount 1 --envs.parallel none \
  --batch_size 4 --batch_length 1024 \
  --replay_size 10000000 --imag_horizon 15 --run.train_ratio 1024 \
  --logdir outputs/memory_len30_persistent_seed0 \
  --seed 0 --run.steps 1000000
```

`bsuite_memory_len/30` 的 30 是需要记忆的原生长度，**不是 sweep 索引**。
另一个任务可以用 `--task bsuite_discounting_chain/100`，将100步延迟链设为
原生1.1奖励链。`small` 调整共享编码器/预测头/策略的容量，NRSM 仍使用 ctwm 的
默认尺寸；做基线比较时要报告实际参数量和计算成本。

POPGym 任务另装可选依赖，使用原生任务名：

```bash
python -m pip install -r requirements-tasks.txt
python -u CTWM.py --configs popgym ctwm \
  --jax.platform gpu --jax.prealloc False \
  --task gym_popgym:popgym-RepeatPreviousEasy-v0 \
  --logdir outputs/repeat_previous_easy_seed0 \
  --seed 0 --run.steps 50000000
```

这里显式采用 v1 论文50M预算，覆盖上游 POPGym preset 的200M默认值。

恢复训练使用相同的任务、配置和 `--logdir`，并把 `--run.steps` 设置成更大的
**总目标**。加载 checkpoint 后会继续优化器与随机数状态。环境进度和数据预取
队列会重新建立，因此整个交互过程不是逐位确定的恢复。
旧 TensorFlow checkpoint 和“每步重建神经历史”的旧 NRSM checkpoint 均不能加载到
当前持久历史版本。请使用新的 logdir；不会自动转换旧权重或覆盖旧训练结果。
正在运行的旧进程不会因磁盘代码更新而切换结构，需重新启动新实验。

## 默认训练配置

| 项目 | ctwm 默认值 |
|---|---|
| dtype | float32；没有 float16/bfloat16 混合精度 |
| NRSM | 64 neurons，pre/post 两个历史窗口各8 ticks，4 ticks/环境步，读出128，z=8×8 |
| 回放 | FIFO/LFS，10,000,000 条记录 |
| 训练序列 | batch 4，length 1024 |
| 想象 | 所有 batch 记录作为起点，终止起点权重为0；H=15 |
| KL | dyn 0.5 / rep 0.1，free threshold 1（任务预设可覆盖） |
| tick | adaptive 诊断，scale 0；默认不加入训练损失 |
| continuation | 预测 1-terminal，回报计算中乘一次 gamma |
| gamma / lambda | 1-1/333 ≈0.997 / 0.95 |

完整确定性状态为 `c=(deter,pre_trace,post_trace)`，其中 `deter` 是同步读出缓存。
真实交互、回放过滤和想象 rollout 都持续传递两个历史窗口；episode/reset 和
独立训练窗口边界重置全部状态。窗口8表示内部ticks，并非保证只能或必能记住
8个环境步。长期保留仍需要学习。有限窗口同步初始不衰减，学习的衰减率限制在[0,4]。

不再沿用旧 UAV 项目的 HER、500 想象起点、50×10 序列或单一500万步预算。
任务原生奖励保持不变。配置优先级为 `defaults → --configs 中的先后顺序 → CLI参数`；
建议任务/容量预设在前，`ctwm` 在后，`ctwm_smoke` 最后。

正式 LFS 回放会预分配数据文件，上游默认另保留2GiB checkpoint前缀。向量任务
也会有这部分启动IO；图像10M回放可能需要上百GB。`--replay_dir` 可指向 Linux
文件系统中的存储目录以减少 `/mnt/c` 的IO开销，实际磁盘占用须按任务评估。

## Tick 消融

```bash
# 仅 NRSM + 基础目标
python CTWM.py --configs small ctwm --task bsuite_memory_len/30 \
  --nrsm_tick_scale 0 \
  --logdir outputs/nrsm_no_tick --run.steps 1000000

# 所有 tick 均匀监督
python CTWM.py --configs small ctwm --task bsuite_memory_len/30 \
  --nrsm.tick_mode uniform --nrsm_tick_scale 1 \
  --logdir outputs/nrsm_uniform_tick --run.steps 1000000
```

默认 `nrsm_tick_scale=0`：保留 adaptive 权重和 KL 诊断，仅优化 R2I 基础目标。
如需验证额外 adaptive 正则，显式设置 `--nrsm_tick_scale 1`；uniform 正则
也需显式启用非零 scale。保持任务、种子、容量、训练预算一致。正则包含最终
tick；关闭 loss 保留多 tick 转移、持久神经历史和诊断计算，不代表移除这些
计算成本。权重为0时，adaptive 权重不参与训练，也不融合或选择最终状态。

## 日志和测试

终端有 tqdm 预采集/训练进度、累计更新数、回放大小；每回合打印长度和原生回报。
训练指标写入 `metrics.jsonl`，回报写入 `scores.jsonl`；模型、优化器、RNG及计数
保存在 `checkpoint.ckpt`，LFS另保存回放状态。正常完成时会保存最终训练权重。
`--run.log_every` 和 `--run.save_every` 的单位是**秒**，不是交互步。
首次编译可能暂时没有进度增长；日志中的 `Tracing ...` 表示正在编译。
入口默认使用 `outputs/.jax_cache` 保存编译缓存；可通过
`JAX_COMPILATION_CACHE_DIR` 指定其他路径。首次完整GPU图可能需要数分钟。
上游交互计数包括 reset 记录，记录数不应直接冒充纯环境动作次数。

不再默认启用 WandB 上传。某些没有正/负奖励样本的诊断分组会输出NaN，表示
该分组没有样本；损失和梯度是否有限需独立判断。

```bash
python -m unittest discover -s tests -v
python scripts/smoke_env.py --task bsuite_memory_len/4 \
  bsuite_discounting_chain/10 --steps 256 --seed 0
```

单元测试强制CPU；GPU运行检查用 `scripts/gpu_check.py --require-gpu`。
CPU参考环境（Python3.11）使用 `requirements.txt`；GPU独立环境使用
`requirements-gpu.txt`。TensorFlow不是新工程的执行引擎，也不是必需依赖。
已验证范围和未验证视觉任务见 [docs/TASK_SUPPORT.md](docs/TASK_SUPPORT.md)。
本次验证汇总见 [docs/VALIDATION.md](docs/VALIDATION.md)。

保留上游 [LICENSE](LICENSE) 和 [UPSTREAM_README.md](UPSTREAM_README.md)。
R2I、DreamerV3、CTM 的方法来源在 [docs/METHOD.md](docs/METHOD.md) 中逐项说明。
