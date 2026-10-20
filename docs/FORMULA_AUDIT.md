# CT-WM 逐公式审查

**结论：NRSM 的主要递推是闭合的，未发现明显的基本代数、奖励时间索引或 lambda-return 递推错误；但当前论文不能作为最新迁移模型的准确说明。**
当前最需要处理的是：旧方法与新方法版本分叉、raw tick KL 的信息压缩风险、同步与加权梯度的解释边界。数学上能定义的 loss，不等于已证明有利于长期记忆。

本次只读审查，没有修改论文或运行源码，没有启动训练。代码采用用户移动后的路径 `C:/Users/Administrator/Desktop/CTWMR2I`。审查对象仍是原论文目录的主 `ijcai26.tex`；新代码文件夹中没有另一份主 TeX。源码 SHA256 与现有公式清单一致，64条编号公式全部核对，53个label与 `.aux` 编号一致。PDF共9页；实际含公式的第1–5页已渲染检查，未发现明显的公式编号错位或裁切。以下源码行号与 PDF 行号不同。

## 优先处理的发现

### 1. 论文仍然描述已取消的后验校正与aux分支（高优先级，版本一致性）

原稿(28)–(31)定义 learned f_post，(46)–(48)定义其对应的辅助解码。新代码直接令 c_t=cbar_t，这两组已经不属于当前方法。摘要、引言、架构表、训练说明和结论也仍保留旧机制。

若在新定义下保留同一posterior z、同一组heads的aux，两个分支输入完全相同。原配置会把重构权重从 `10L_x+L_r+L_delta` 改成 `11L_x+2L_r+2L_delta`，没有独立对齐约束。按已确认的迁移版，删除旧分支、将确定性记忆更新写成 c_t=cbar_t。

f_post本身并不是数学错误。它还会给当前观测一个不经过z的KL约束的确定性解码捷径，所以旧目标不应额外被宣称为标准生成模型ELBO；原稿目前没有直接声称ELBO，这一点不作为虚构错误。

### 2. 新增raw tick KL会绕开R2I的free-bits区间（高优先级，设计风险）

当前基础目标为 `0.5 max(1,KL(sg(q)||p)) + 0.1 max(1,KL(q||sg(p)))`。低于1 nat时，这两项对该KL的梯度为0。额外tick项使用未截断的raw KL，并且包含最终tick；它仍会对q、p和权重产生梯度。

一个可复算的数学例子：令4个tick相同，J=1、C=2、q=(0.6,0.4)、p=(0.5,0.5)。KL约0.0201355<1，两基础项为常数，但L_tick仍等于0.0201355并推动q/p靠近。这个例子说明阈值被额外目标绕开，不证明真实训练一定坍缩：prior改善也能降低KL。

因此可以写“基础KL实现沿用R2I”，不能写“完整目标保持了低KL区间不压缩信息的保护”。建议先比较lambda_tick=0、uniform和adaptive，联合检查任务回报、cue probe、posterior信息利用和prior预测。对q stop-gradient或给tick阈值都是新的研究设计，不能当作当前已有方法或保证有效的修复。

### 3. KL一致性、自信和长期credit必须分开（解释与证据）

(32)是prior/posterior的一致性；低KL也可能由q忽略观测产生。(33)是低熵自信；它不等于正确率。(38)是乘积求导恒等式，只说明权重导数被保留，不证明长期信用分配改善。

对error分支，设 L_e=sum_i w_i ell_i，有

    dL_e/dell_i = w_i [1-alpha_error(ell_i-L_e)].

取ell=(0,3)、alpha_error=1，第二个分量导数约-0.088104。这说明动态加权并不等价于固定正权重逐tick监督。**反例只针对error分支，不能据此断言完整默认混合目标必然产生负梯度。**

另外，KL按J个变量求和，而certainty已归一化到[0,1]。两种alpha都取1不表示尺度平衡；改变J/C时error权重的锐度也会变。建议用“consistency-based tick supervision/weighting”，把性能结论留给消融。

### 4. 同步公式正确，但sqrt(Z)不是相关系数归一化（解释与消融）

(17)与(18)–(20)严格等价，tau>=1时分母为正，无零除。因为nu在[-1,1]，有

    |S_tau| <= sqrt(Z_tau),
    Z_tau = (1-exp(-rho*tau))/(1-exp(-rho)).

所有活动为1、rho趋近0时，S_tau趋近sqrt(tau)。因此改变K或rho会同时改变活动聚合范围和幅度，可能影响后续tanh饱和；不能把K消融全部解释为推理深度变化。保留当前公式可以，但应称为指数加权共活动/同步特征，而非Pearson相关度。

默认beta=0使decay=1/2，K4时Z=1.875，幅度上界约1.3693。16个同步神经元的136配对中，16个是自配对能量通道，120个是交叉通道。这都不是错误，但属于解释与可复现细节。

### 5. 内部tick历史与环境长期记忆是两件事（能力边界）

M8/K4时，最后历史矩阵仍有4列learned seed history和4列本步生成的输入。内部history、N/Z/S每环境步重建；跨环境记忆仅通过c/z递归。K个thought ticks不是向未来推进K个环境步，也不是额外获得K份观测。

固定参数下cbar=T(u)，数据处理关系意味着内部计算不能恢复已从u=(c_prev,z_prev,a_prev)中不可逆丢失的历史信息。它可以重新组织已有信息、增加转移表达能力；长期保持能力仍需cue/sequence任务、不同环境延迟与上下文长度来检验。论文已写“representation不假定为精确充分统计量”和“长期收益待实验”，应保留这些界限。

### 6. 概率、预测头和控制部分需要同步到真实R2I训练定义（版本一致性）

- (24)(26)需要加epsilon=0.01的uniform mixture：p=(1-epsilon)softmax(l_p)+epsilon/C，q同理。tick KL/entropy也基于这个混合分布。
- (40)–(45)的categorical/Gaussian定义不是当前MultiDecoder与symlog reward heads。输入one-hot化不会自动使解码损失成为categorical cross-entropy。单位Gaussian NLL为half-SSE；若直接用SSE则对应不同方差/权重，1/2不能作为加法常数省略。
- continuation head当前预测kappa=1-is_terminal，想象折扣是gamma*hat_kappa。(6)(58)原旧版“直接预测gamma*kappa”也能自洽，本次没有发现旧式重复乘gamma；修订时只保留一套语义。
- (56)(62)是整个样本经过tanh的Gaussian；当前代码为Normal(tanh(mean),std)，然后由转移/环境clip。不是同一种分布，不能保留tanh Jacobian作为当前策略的logprob。若研究tanh-Gaussian本身，原Jacobian公式正确；native affine动作缩放还需说明常数项。
- (59)(60)(63)应写明实际online critic bootstrap、percentile-normalized advantage、distributional CE+slowreg及每步EMA；旧稿target bootstrap、scalar MSE、100次hard-copy属于另一版设计。
- 代码名为mean()的symlog readout实际为symexp(sum_b p_b bin_b)。严格说它一般不等于sum_b p_b symexp(bin_b)，应明确点预测读出，不能未经定义当作原尺度分布的精确数学期望。
- 原稿按sum(w)归一化，代码是先乘stopped weight再对固定记录数mean；终止起点过滤与零权重展开也有分母/计算量差别。若采用R2I框架，应写真实surrogate和平均方式。
- 表格和正文的encoder/heads为2×128 ELU，但迁移版共享网络取R2I配置及各domain/capacity preset（默认可为5层1024、SiLU/LayerNorm）；actor/value学习率默认也为3e-5而非旧稿统一1e-4。NRSM内部128维不代表整个Agent都是128宽，必须报告实际解析配置与总参数量。

### 7. reset、padding与评估不是只有一句话就已实现（协议一致性）

原稿区分O/T并排除窗口首incoming reward/cont，且声称padding冻结状态。当前R2I式目标对全长采样记录直接mean，没有这套mask；目前采样器提供全长窗口，不需要padding。窗口首reward仍是真实已发生的值，学习有限上下文下的条件分布并非捏造奖励，但可产生aliasing/噪声；不能说已经实现旧稿承诺的mask。

原稿写latent和action采用mode评估，当前过滤与eval action仍sample。两者都可作为协议，但成绩复现依赖这一选择。训练窗口重置与真实终止应继续区分，不能把截断当失败或将时间限制自动设为terminal。

### 8. CTM来源必须在论文中补充（归属）

当前TeX/Bib没有CTM/Darlow引用。private NLM、内部时间轴、神经同步及error/certainty监督的基础思路均应明确引用CTM；R2I论文也明确基础训练沿用DreamerV3。论文的候选贡献应集中在NRSM如何构造生成转移、递归记忆和对应监督上。此次审查不是完整的新颖性检索，不能据此证明任何组合“首次提出”。

## 64条编号公式检查表

“成立”表示在给定假设下数学可定义，不代表已证明性能改善。需要同步的行不应误记为“代数错误”；应删除的行是依据已经确认的新版本选择。

| 编号 | 内容 | 数学判断 | 对照当前版本的处理与细节 | TeX行 |
|---:|---|---|---|---:|
| 1 | 基准任务POMDP | 成立 | 可保留：标准POMDP定义；终止和截断由后续wrapper约定补充。 | [L65](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:65) |
| 2 | 原生转移和奖励 | 成立 | 补清定义：联合转移奖励核可用；建议定义P_R与分别列出的P、R的关系。r_(t+1)索引正确。 | [L70](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:70) |
| 3 | 部分可观测接口 | 成立 | 可保留：观测由当前环境状态产生，符合部分可观测接口。 | [L76](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:76) |
| 4 | 可用因果历史 | 成立 | 可保留：因果历史没有未来量。模型只使用其压缩表示，不应称为精确充分统计量。 | [L81](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:81) |
| 5 | 通用折扣回报目标 | 成立 | 补清期望约定：折扣回报与r_(t+1)一致；观测核的随机性应说明隐含在期望内。gamma选择与原生未折扣评价可能偏好不同策略。 | [L86](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:86) |
| 6 | 原生续接与折扣 | 成立 | 统一折扣语义：delta=gamma*kappa作为有效折扣定义正确。当前head学kappa，想象时再乘gamma；不要继续把head目标写成delta。 | [L94](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:94) |
| 7 | 唯一持久潜状态 | 成立 | 可保留：c,z是唯一持久潜状态。logit是分布缓存；内部活动/历史/同步量不跨环境步持久化。 | [L131](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:131) |
| 8 | RSSM参考算子 | 成立 | 可保留并引用：标准RSSM对照；NRSM替换transition operator，不是消除deterministic memory。 | [L138](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:138) |
| 9 | SyncCTM转移 | 成立 | 可保留：prior transition仅依赖上一c/z/action，不读取当前观测。K个thought ticks仍只推进一个环境步。 | [L143](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:143) |
| 10 | 转移输入拼接 | 成立 | 补维度：d_u=d_c+JC+d_a；onehot或连续动作进入同一拼接。c、z和a的职责不重复。 | [L155](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:155) |
| 11 | 初始神经活动 | 成立 | 可保留：W_init为D×d_u，nu_0初始化已闭合。它可以与历史初始化独立，不要求nu_0=g(A_0)。 | [L156](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:156) |
| 12 | 历史窗口初始化 | 成立 | 说明种子历史：W_A为DM×d_u；生成的是learned seed history。默认M8/K4时最后仍有4列种子，不是8个真实过去环境步。 | [L164](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:164) |
| 13 | synapse更新 | 成立 | 可保留：synapse输入D+d_u，输出D；各tick共享参数，ELU隐藏和线性输出已说明。 | [L170](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:170) |
| 14 | 历史shift | 成立 | 可保留：移除最老一列并追加当前alpha，历史维度始终D×M。 | [L173](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:173) |
| 15 | 逐神经元时间模型 | 成立 | 可保留并引用CTM：每神经元只处理自己的M长历史；没有额外跨神经元权重共享。 | [L175](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:175) |
| 16 | private NLM公式 | 成立 | 可保留并引用CTM：私有M→H→1网络维度正确；每神经元独立参数，跨tick和环境步共享。 | [L180](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:180) |
| 17 | 同步求和定义 | 成立但需收紧解释 | 解释尺度：指数加权共活动特征合法，但不是Pearson/cosine相关系数；绝对值上界sqrt(Z)，不一定小于1。K与rho同时影响尺度。 | [L188](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:188) |
| 18 | 同步分子递推 | 成立 | 可保留：N的零初始化和指数递推与显式加权和严格一致。 | [L195](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:195) |
| 19 | 同步权重递推 | 成立 | 可保留：Z从0开始，每tick加1，tau>=1时Z>=1，没有零除。 | [L196](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:196) |
| 20 | 同步归一化 | 成立但需收紧解释 | 解释尺度：N/sqrt(Z)与(17)等价；不是除以Z的加权均值，也不是L2归一化相关系数。 | [L197](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:197) |
| 21 | 同步投影 | 成立 | 可保留：136个选定配对映射到128维context，tanh读出共享。自配对属于能量特征。 | [L202](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:202) |
| 22 | 最终prior context | 成立 | 可保留：仅最后tick的context进入环境时间递归；中间监督不增加持久state。 | [L204](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:204) |
| 23 | 先验分类分布 | 成立 | 可保留：J个categorical条件独立是模型近似，不表示真实环境因子独立。 | [L213](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:213) |
| 24 | 先验softmax | 成立但定义不完整 | 补unimix：纯softmax没有写出当前实现0.01 uniform mixture；tick KL和certainty也用混合后的概率。 | [L215](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:215) |
| 25 | 后验分类分布 | 成立 | 可保留：q(z\|cbar,x)的因子化合法；它与R2I默认q(z\|x)不同，应作为结构选择披露。 | [L217](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:217) |
| 26 | 后验概率 | 成立但定义不完整 | 补unimix：posterior同样需要(1-epsilon)*softmax+epsilon/C，才能逐式对应当前实现。 | [L219](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:219) |
| 27 | 直通分类样本 | 成立 | 说明有偏估计与评估：ST前向hard、反向soft是合法代理梯度，不是无偏重参数估计。正文mode评估与当前sample评估不一致。 | [L225](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:225) |
| 28 | 学习式后验校正 | 成立但属于旧设计 | 删除并改为c=cbar：可学习后验记忆校正已经取消。继续保留会描述另一个算法，并引入不受z的KL限制的观测捷径。 | [L233](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:233) |
| 29 | 后验门控 | 成立但属于旧设计 | 随(28)删除：sigmoid门控本身没有数学问题；当前模型没有该参数或运算。 | [L238](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:238) |
| 30 | 后验候选 | 成立但属于旧设计 | 随(28)删除：tanh候选记忆本身合法；当前模型没有该运算。 | [L239](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:239) |
| 31 | 后验融合 | 成立但属于旧设计 | 随(28)删除：门控凸融合合法，但与当前c=cbar的模型不同。 | [L240](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:240) |
| 32 | tick KL | 成立但需收紧解释 | 保留为consistency：是KL(q\|\|p)，不是对真实环境的预测误差；低KL既可能来自更好prior，也可能来自q丢弃信息。 | [L252](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:252) |
| 33 | prior certainty | 成立但需收紧解释 | 保留为confidence：归一化负熵范围[0,1]正确；低熵表示自信，不代表预测正确或校准良好。 | [L259](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:259) |
| 34 | 误差权重 | 成立但需实验验证 | 保留开关并消融：偏好低KL的ticks；完整权重梯度不是固定正权重监督。KL按J求和，温度尺度与[0,1]的certainty不同。 | [L265](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:265) |
| 35 | 确定性权重 | 成立但需实验验证 | 保留开关并消融：偏好低熵prior；默认alpha=1时最大权重比受e限制，与error分支的尺度明显不同。 | [L269](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:269) |
| 36 | 两权重融合 | 成立 | 可保留：两个归一化权重的凸组合仍非负且总和1；这不等于对两种指标完成了数值校准。 | [L273](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:273) |
| 37 | tick辅助损失 | 成立但有信息压缩风险 | 优先验证：额外raw tick KL绕过基础free-bits阈值，且包括末tick；不能说总目标保留了KL<1时的信息自由区间。 | [L277](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:277) |
| 38 | 完整tick梯度 | 成立但解释过强 | 收紧credit表述：正确的乘积求导恒等式；它说明梯度存在，不证明长时信用分配或每个tick误差单调改善。 | [L283](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:283) |
| 39 | 统一latent feature | 成立 | 可保留：对应deter和展平stoch特征；与新actor/critic输入一致。 | [L298](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:298) |
| 40 | 按观测模态定义的共享解码器 | 成立但属于旧likelihood | 按R2I heads重写：categorical CE与单位Gaussian是合法模型；当前onehot观测仍走symlog MSE，image走SSE，并非自动Cat decoder。 | [L303](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:303) |
| 41 | 奖励预测分布 | 成立但属于旧likelihood | 改为symlog/two-hot定义：单位Gaussian reward本身合法；新reward head使用symlog bins，不能继续用该NLL。 | [L311](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:311) |
| 42 | 有效续接折扣 | 成立但约定须统一 | 定义native continuation：sigmoid有效；当前预测kappa而非gamma*kappa。旧直接sigmoid也未强制hat_delta<=gamma。 | [L314](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:314) |
| 43 | 原生观测负对数似然 | 成立 | 同步真实观测损失：NLL定义正确。单位方差Gaussian对应half-SSE；正文仅说SSE需明确1/2，乘数不能作为加法常数忽略。 | [L319](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:319) |
| 44 | 奖励损失 | 成立但属于旧likelihood | 随reward head重写：half-MSE确实对应(41)省略常数的NLL；当前symlog two-hot CE不是这个损失。 | [L321](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:321) |
| 45 | 续接损失 | 成立但约定须统一 | 目标改为native kappa：soft-target BCE可用，但不是原生terminal标签的普通Bernoulli观测NLL；当前head目标为1-is_terminal。 | [L323](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:323) |
| 46 | 校正前上下文辅助解码特征 | 成立但在新模型冗余 | 删除：c=cbar后等于主feature，使用同一个posterior z时没有独立约束。 | [L330](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:330) |
| 47 | 共享模态头辅助观测/奖励/续接损失 | 成立但在新模型冗余 | 删除：同一输入、同一z、同一heads重复预测，只是在重调各损失权重。 | [L336](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:336) |
| 48 | 辅助条件似然目标 | 成立但属于旧目标 | 删除：旧O/T分开平均作为设计可用；新版本没有aux分支。 | [L345](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:345) |
| 49 | world model损失 | 成立但属于旧目标 | 重写基础目标：新基础KL是0.5 dyn + 0.1 rep的stop-gradient/free-nats形式；新头、mask约定也不同。 | [L355](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:355) |
| 50 | 含tick与辅助预测的模型损失 | 成立但属于旧目标 | 改为L_base+lambda_tick*L_tick：去aux，明确额外tick包含final tick，保留off/uniform/adaptive作为待验证扩展。 | [L359](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:359) |
| 51 | 想象actor输入 | 成立 | 可保留：当前状态选择当前动作，因果顺序正确。actor输入stop-gradient最好在公式或梯度路由中写清。 | [L370](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:370) |
| 52 | 想象context转移 | 成立 | 可保留：只用当前c/z/a推进下一context，不读取未来观测。 | [L372](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:372) |
| 53 | 想象先验采样 | 成立 | 可保留并补label：从下一步prior采样z，与前后状态递推一致。 | [L375](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:375) |
| 54 | 想象状态推进 | 成立 | 更新邻近正文：想象c=cbar正确；新真实路径也c=cbar，不再存在观测后的确定性记忆校正规则。 | [L376](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:376) |
| 55 | 原生离散动作采样 | 成立 | 同步动作编码约定：onehot categorical正确；若描述MultiDiscrete，当前适配器可展平为联合类别，与独立因子heads的假设有别。 | [L384](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:384) |
| 56 | 原生连续动作采样 | 成立但属于不同策略 | 按实际Normal+clip重写：论文是tanh变换整个Gaussian样本；代码是tanh均值的Normal，再经动作clip，二者不是同一种分布。 | [L387](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:387) |
| 57 | 想象奖励 | 成立 | 定义symlog读出：用后继状态预测incoming reward，时间索引正确；新头须明确symexp(E[bin])读出，别未经定义称原尺度分布期望。 | [L394](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:394) |
| 58 | 想象折扣 | 成立但约定须统一 | 改为gamma*hat_kappa：旧head直接预测有效折扣时无需再乘gamma；当前head预测native continuation时必须乘一次gamma。 | [L396](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:396) |
| 59 | lambda return | 成立 | 同步bootstrap来源：lambda递推与terminal截断正确；旧稿用target critic，新代码用online critic readout。 | [L401](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:401) |
| 60 | 按动作空间选择的actor梯度 | 成立但属于不同surrogate | 同步R2I actor目标：代码使用percentile-normalized advantage；连续分支也减baseline；权重乘后普通mean，不是按sum(w)归一化。 | [L410](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:410) |
| 61 | 离散/连续策略熵 | 成立但分布须同步 | 同步实际entropy：categorical/MC entropy定义合理；新Normal entropy不是tanh-squashed entropy。因子动作可补entropy求和。 | [L420](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:420) |
| 62 | 多维tanh Gaussian对数概率 | 成立但属于不同策略 | 若沿用R2I则删除此变换式：tanh-Gaussian Jacobian符号和稳定式正确；native affine scaling密度另有常数。当前未对整个样本做tanh。 | [L428](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:428) |
| 63 | value损失 | 成立但属于旧critic | 改为distributional CE+slowreg：half-MSE和stopped target合理，但当前critic是symlog distributional损失及EMA，每100次hard-copy描述过期。 | [L437](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:437) |
| 64 | 完整目标与梯度路由 | 成立 | 保留分块梯度说明：三项加总可作记账；模型/actor/critic只更新各自参数，连续动作仍保留经过固定模型的导数。 | [L444](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex:444) |

## 依据与建议修订顺序

1. 以已确认的迁移版为目标：c=cbar，删除f_post和aux，模型目标写成L_base+lambda_tick L_tick。
2. 定义基础heads、unimix、balanced/free-nats KL、native continuation、真实actor/critic与平均方式；更新表格、摘要和结论。
3. 保留NRSM核心初始化与递推，补张量维度和CTM引用，收紧同步和credit解释。
4. 先验证tick正则是否保留了任务所需信息，再决定是否修改其梯度或权重方案。没有实验根据时不增加“保证长期记忆”表述。

本次没有发现lambda-return的reward到达索引、terminal截断、旧版gamma使用次数或三组参数梯度归属的明显错误。BSuite gamma=.997与100步bonus1.1的目标冲突已经在原稿披露：相对即时1，晚奖励的折扣值约0.817；不能把这种失败都归因于记忆或credit。

关键本地依据：

- [主论文源码](C:/Users/Administrator/Desktop/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation/ijcai26.tex)，以及同目录PDF、aux和formula_inventory.json。
- [NRSM与tick计算](C:/Users/Administrator/Desktop/CTWMR2I/recall2imagine/nrsm_nets.py:23)。
- [实际世界模型loss](C:/Users/Administrator/Desktop/CTWMR2I/recall2imagine/agent.py:193)。
- [实际连续策略分布](C:/Users/Administrator/Desktop/CTWMR2I/recall2imagine/nets.py:600)。
- [实际actor与critic](C:/Users/Administrator/Desktop/CTWMR2I/recall2imagine/agent.py:326)。
- [配置](C:/Users/Administrator/Desktop/CTWMR2I/recall2imagine/configs.yaml:80)。

原始文献（仅依赖一手资料）：

- [R2I：世界模型目标与Actor-Critic](https://arxiv.org/html/2403.04253v1)。
- [DreamerV3：free bits、unimix与distributional value](https://arxiv.org/html/2301.04104v2)。
- [CTM：神经元时序与同步表征](https://arxiv.org/html/2505.05522v1)。

机器可读的逐式表达式、行号和判定见FORMULA_AUDIT.json；CSV可用于逐项修订跟踪。
审查固定的TeX SHA256：`18bceb851cf3e2ec6cb93f634f008b8e05eb6cc00a3670eacaa03df945cf9383`。
