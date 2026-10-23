# v12 独立备份

本目录恢复的是 `ctm-action-conditioned-reward-v12`，不是将 v13 重命名。
代码、配置、测试、公式文档、Git 历史和历史输出均为独立文件副本。
v13 修改过的源文件、测试和论文已从修改前的 SHA256 校验备份恢复。

- `outputs/memory_len30_ctm_gated_seed0_fresh001/`：备份时稳定的 checkpoint、完整 replay 和日志。
- `backup_reference/validated_20261023_070802/`：此前已独立评估的成功 v12 权重与评估结果，作为固定参考。
- `paper/`：对应 v12 的 LaTeX 源码和依赖。`reference_build_artifacts/` 是原目录已有编译产物，未验证与恢复后的 v12 源码一致，不作为 v12 编译结果。
- `BACKUP_MANIFEST.json`：文件 SHA256、恢复来源和备份时间。
- `BACKUP_VALIDATION.json`：checkpoint、回放和独立加载检查结果。

仅省略可再生成的 Python/XLA 缓存、瞬时锁文件，以及 v13 新增的专属文档。
没有改动、停止或重启原目录的训练。该副本是备份时的快照，不会自动同步后续训练。
Python/GPU 依赖仍使用机器上现有的 WSL 环境；源码、权重、回放和新缓存不共享原项目。

## 从本副本已有 checkpoint 继续训练

以下命令尚未运行。请在需要时自行执行，避免与当前 GPU 训练争用资源。
参数保持该实验的 small/ctwm、B32、T1024、seed0 和 1M 总步数。

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
  --logdir outputs/memory_len30_ctm_gated_seed0_fresh001
```

从零重跑使用 README 中的新输出目录；不要删除此备份里的 checkpoint 或 replay。
