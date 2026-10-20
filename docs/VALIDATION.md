# 迁移验证记录

验证日期：2026-10-19。机器：WSL2 / RTX 5090 D v2。
以下是结构、数值和运行验证，不是任务学习成绩。

| 检查 | 实际结果 | 证据 |
|---|---|---|
| CPU公式、损失、环境、采样种子、集成和checkpoint | 38项全部通过，97.626秒 | `all_tests.log` |
| 新增Uniform持久化/重复恢复/时序检查 | 3项全部通过，0.116秒 | `replay_tests.log` |
| GPU float32运算、梯度、TFP和Agent导入 | GPU_CHECK_OK | `gpu_probe.json` |
| GPU离散/连续联合更新与图像损失 | 2项、2个子测试通过，391.21秒 | `gpu_integration.log` |
| 真实CPU BSuite训练与恢复 | 最终172条记录，19次更新，已保存 | `train_smoke.log`、`train_resume_verified.log` |
| 真实GPU BSuite训练与恢复 | 132条/8更新恢复到164条/16更新，已保存 | `gpu_train_smoke.log`、`gpu_train_resume.log` |

默认GPU自动调优用于两项集成测试；真实GPU CLI短测试使用了
`XLA_FLAGS=--xla_gpu_autotune_level=0`，单独验证关闭调优时的运行路径。
首次GPU图编译开销较大。上述耗时混合了编译、测试和执行，不能作为稳定训练吞吐。

所有记录到的模型/actor/critic优化器损失和梯度范数均为有限值。
无正/负奖励样本时，上游按该奖励分组的诊断可能为NaN；这与总损失不同。
本轮短训练为小模型、2×8序列、H=3，未执行正式4×1024配置。
记录计数包含reset，不能当作纯动作次数。

恢复检查覆盖了模型、优化器、RNG、累计更新数以及精确checkpoint步数。
环境和预取队列会重新建立，LFS尾部不满chunk的交互不等于完整可恢复轨迹。
因此这里只保证已测的训练状态恢复，不宣称整条在线轨迹逐位复现。

迁移过程中修复并回归验证的工程问题包括：

- 静态神经元配对索引不再被GPU编译隐式读回。
- 非法checkpoint在验证失败时保留当前参数、RNG和计数。
- LFS先恢复回放，再由完整checkpoint恢复最新模型和精确步数。
- Uniform重复恢复不再扩张数据；同UUID快照去重；chunk按真实successor链恢复。
- 预采集离散和连续动作受配置种子控制。
- 正常结束会保存最终权重，日志保存失败不再静默吞掉。

BSuite两个任务族和POPGym九个任务通过真实环境接口检查。
DMC、Atari和MemoryMaze的实际环境运行尚未验证；长期学习效果、完整批次显存和
吞吐仍需独立实验。旧论文需同步到本版无f_post/aux的损失与训练定义。

结构化结果在 `validation_report.json`；源代码复制/修改信息和哈希在
`PROVENANCE.json`。原始两个项目未被本次迁移修改。
