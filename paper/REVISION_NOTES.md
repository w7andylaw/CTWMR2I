# CT-WM：奖励头首层动作条件读出（v12）

日期：2026-10-22。Schema：`ctm-action-conditioned-reward-v12`。
本记录仅描述相对 v11 的改动；不把旧测试或训练结果当作新版证据。

- CTM 跨步双历史、独立写门、时间常数、多 tick、活动直通 deter、快慢同步配对逐式不变；不新增 SSM、CMM、记忆状态或隐藏层。
- 仅把奖励 MLP 第一层替换为 `U[p,vec(z)] + sum_k omega(a)_k W_k nu`；原 Norm→SiLU 顺序、bias 规则、后续隐藏层和 255-bin 输出不变。二动作 small512 默认增加32768个权重。
- 离散动作保留 onehot/straight-through 值和梯度，不用 argmax；连续动作使用 `[1,a_flat]` 的仿射 bank，零动作仍保留活动映射。
- 奖励头读取产生当前到达状态/奖励的 incoming action。训练将 replay outgoing actions 右移一次，physical/window is_first 都归零，并用与 NRSM 相同的裁剪。想象 `reward_action=[zero;action[:-1]]` 与状态逐项配对，初始奖励仍排除。
- 动作仅是临时奖励头输入，不加入 recurrent carry、policy 或 value 输入。decoder/continuation/actor/online+slow critic 仍读 `[deter,z]`。
- native reward、255-bin two-hot NLL、预测目标、loss mask/平均方式、所有 KL/tick/model/actor/critic loss及系数、优化器全部不变。论文仅显式补充 reward conditioning 参数。
- 引用 Oh 等2015的动作条件乘性交互；不声称该交互本身是创新，不声称 CMM 已验证稀疏奖励。没有导入 CMM 长期存储。
- 冻结 v11 特征验证需按 cue/action 配对组划分训练/验证/测试，冻结 backbone，仅拟合读出并报告 prior-only 正确/错误动作奖励与排序。不将该诊断等同策略成功率；结果另附收据，未完成前不填写分数。
- 新架构使用新目录/新 schema；本次不停止已有训练、不清缓存、不修改 v11 checkpoint。编译、数值/集成测试和诊断结果由独立新版记录保存。

验证补记：新版 TeX 编译成功。冻结 v11 特征、三个读出训练种子、相同训练预算的 prior 留出动作排序率从50.24%升至60.19%，但平均 NLL 从1.36680变为1.36967；只能说明部分读出能力改善，不能声称稀疏训练收敛。完整协议与全部种子见代码目录 docs/ACTION_REWARD_VALIDATION.md。
