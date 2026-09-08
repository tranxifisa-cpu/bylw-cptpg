# 非平稳 CPT-PG 跟踪验证环境说明

本文档说明 `mvp_cpt_pg/synthetic_env.py` 和 `scripts/run_synthetic_tracking.py` 中的半合成非平稳强化学习环境。当前主实验已改为“沪深300真实因子 + 半合成收益”：用真实 A 股沪深300成分股行情和估值/交易因子构造状态特征，再按可控市场阶段生成收益，用于快速验证动态参考点 CPT-PG 在非平稳市场中能否跟踪时变目标函数的一阶驻点邻域。

## 目标

该环境只验证算法机制，不涉及 LLM、多 Agent、用户偏好和偏好约束。主实验涉及真实 A 股数据，但不直接使用真实未来收益作为学习目标，而是使用真实因子和真实残差噪声构造半合成收益，因此既保留真实市场横截面结构，又保留可控的非平稳阶段和可解释的“真机制”。

核心问题是：在外生市场分布缓慢漂移，同时参考点内生更新的情况下，CPT-PG 的平方梯度范数是否能保持可控，平均平方梯度范数是否随时间下降，累计平方梯度范数是否呈次线性增长。主实验使用归一化财富终值作为目标随机变量。

## 环境结构

环境包含现金和沪深300风险资产：

- 现金资产位于动作向量第 0 维，收益率恒为 0。
- 半合成主实验默认从沪深300最新成分中筛选数据完整的股票，可通过 `--risky-assets` 控制最多使用多少只。
- 投资组合权重是 `N+1` 维 simplex 向量，包含现金权重和股票权重，权重非负且总和为 1。
- 初始组合为全现金。
- 每期先用滑动窗口估计 CPT 目标和 CPT 策略梯度，再更新策略参数，然后采样当期组合权重。
- 组合价值用 price-relative 推进，即 `y_t = 1 + r_t`，现金的 price-relative 恒为 1。
- 环境保存上一期组合权重，并把上一期权重作为策略特征之一。

现金维度和交易成本处理参考 FinRL portfolio environment 的设计。FinRL 中动作空间大小为 `1 + stock_dim`，现金位于第 0 维；TRF 交易成本通过 transaction remainder factor 扣减组合价值。

PGPortfolio 论文 `1706.10059` 的核心环境结构也是组合权重、未来 price-relative、上一期权重和交易成本共同决定下一期组合价值。当前合成环境迁移的是这些环境结构，不迁移其神经网络训练目标。

代码来源：

- `reference_paper/finrl_portfolio_env.py`
- `mvp_cpt_pg/synthetic_env.py`
- `scripts/run_synthetic_tracking.py`

## 半合成市场生成

主实验固定使用半合成沪深300环境。构造流程如下：

- 通过 Tushare `index_weight` 读取沪深300（默认 `000300.SH`）最新成分股。
- 读取这些成分股的真实 `daily` 和 `daily_basic` 缓存数据。
- 剔除在所需窗口内缺失日线、估值、换手率、量比或流通市值的股票。
- 用真实历史数据计算严格滞后的因子特征，避免使用当日收盘后的未来信息。
- 不直接把真实未来收益作为目标收益，而是用真实因子暴露、真实市场噪声和真实横截面残差生成半合成收益。

半合成收益形式为：

```text
R_{i,t+1}^{semi}
= market_mu(regime_t)
+ factor_strength * beta(regime_t)^T factor_{i,t}
+ noise_scale(regime_t) * real_residual_{i,t+1}
+ small_noise_{i,t+1}
```

其中 `factor_{i,t}` 来自真实沪深300数据，`real_residual` 来自真实 A 股横截面收益残差，`market_mu`、`beta` 和 `noise_scale` 按市场阶段人为设定。`factor_strength` 控制半合成因子信号强度，默认值为 `2.5`。单期收益经过 `tanh` 压缩到 `[-return_bound, return_bound]`，默认 `return_bound=0.08`。

这种设计的意义是：状态特征和噪声来自真实市场，阶段机制是可控的，因此可以明确观察算法是否在已知市场突变后出现“冲击-响应-重跟踪”。

### 市场阶段

半合成市场每 500 个正式评估 step 循环一次。每个 500 step 周期划分为四个阶段：

- 阶段一：`uptrend_momentum`。上涨市场，动量因子有效，过去涨得好的股票继续更容易上涨。
- 阶段二：`oscillation_reversal`。震荡市场，反转因子有效，过去短期跌得多的股票更容易反弹。
- 阶段三：`negative_shock`。市场突变，整体均值转负，真实残差噪声放大。
- 阶段四：`style_switch_lowvol_quality`。风格切换，动量弱化，低波动、质量和价值因子变得更有效。

每期 trace 会输出 `market_regime` 字段，用于分段分析梯度范数、平均平方梯度范数、财富和参考点适应。若运行超过 500 个正式评估 step，第 501 个 step 会重新进入 `uptrend_momentum`。

从 MDP 角度看，该环境是非平稳 MDP。原因是收益分布参数随时间变化，状态转移核和奖励分布会随 `t` 改变。即使策略参数不变，同一个组合在不同时间面对的收益分布也不同。若只用静态历史收益或固定转移核，这个条件就无法成立。

## 状态特征

每个资产每期有以下特征：

- `bias`：常数项。
- `cash_indicator`：现金指示变量，现金为 1，风险资产为 0。
- `momentum_signal`：真实历史 20 日动量，严格滞后一期。
- `reversal_signal`：真实历史 5 日反转信号，严格滞后一期。
- `low_vol_signal`：真实历史 20 日低波动信号，严格滞后一期。
- `value_signal`：真实 PB 估值因子，低 PB 更偏价值，严格滞后一期。
- `turnover_signal`：真实换手率因子，严格滞后一期。
- `quality_signal`：真实流通市值和量比构造的稳定性/质量代理因子，严格滞后一期。
- `cycle_signal`：真实市场滚动收益状态，严格滞后一期。
- `prev_weight_signal`：上一期组合权重，即 portfolio vector memory。

除现金和常数项外，横截面因子先做逐日 z-score，再限制在 `[-1, 1]` 内，用于满足特征有界假设。

## 策略函数

当前脚本支持三种策略参数化：

- `softmax`：线性得分加高斯噪声，然后 softmax 得到权重。
- `sparsemax`：线性得分加高斯噪声，然后 sparsemax 得到稀疏权重。
- `dirichlet`：线性得分映射到 Dirichlet 参数 `alpha`。梯度和目标函数估计阶段从 Dirichlet 分布随机采样；真实执行阶段由 `dirichlet_execution_mode` 控制，可用 `sample` 或均值 `mean`。

Dirichlet 参数映射为：

```text
alpha_i = exp(theta^T x_i)
```

该映射连续可微，且 `alpha_i > 0`，不再需要人为设置 Dirichlet 参数上下界。

## CPT 估计和策略梯度

每期使用当前参考点 `r_t` 和当前策略参数 `theta_t`，在最近 `h` 期滑动窗口上采样收益轨迹。

采样分为两批：

- `n_t` 条轨迹用于估计当前 CPT 目标值和 CPT 权重分位数。
- `m_t` 条轨迹用于估计 score function 策略梯度。

目标函数估计使用概率权重函数 `w` 本身。

策略梯度中的 CPT 权重项使用概率权重函数导数 `w'`。

当前代码复用主算法中的：

- `compute_cpt_objective`
- `compute_cpt_gradient`
- `quantile_tail_sum`
- `cpt_probability_weight`
- `cpt_probability_weight_derivative`

每条轨迹的目标随机变量是 `h` 步后的归一化财富终值：

```text
Z_{t,h}(theta) = normalized_wealth_t * product_{k=t-h}^{t-1}(mu_k * w_k(theta)^T y_k)
```

其中 `normalized_wealth_t = wealth_t / initial_wealth`，`mu_k` 是 TRF 交易成本后的剩余因子，`w_k` 是策略采样出的组合权重，`y_k` 是 price-relative 向量。

四个方法的目标为：

```text
Dynamic CPT-PG:    J_t(theta) = CPT_{r_t}(Z_{t,h}(theta)), eta_gain > eta_loss
Symmetric CPT-PG:  J_t(theta) = CPT_{r_t}(Z_{t,h}(theta)), eta_gain = eta_loss
Static CPT-PG:     J_t(theta) = CPT_1(Z_{t,h}(theta))
Expected PG:       J_t(theta) = E[Z_{t,h}(theta)]
Exponential PG:    J_t(theta) = E[u_exp(Z_{t,h}(theta))]
```

## 参考点更新

动态参考点使用非对称适应规则，且参考点和财富都使用归一化财富口径：

```text
r_{t+1} = r_t + eta_gain * max(W_{t+1} - r_t, 0) - eta_loss * max(r_t - W_{t+1}, 0)
```

其中 `W_{t+1}` 是当期收盘后的归一化财富。初始值为 `r_0 = 1.0`。

静态参考点实验可通过 `--static-reference` 启用，此时参考点保持为 0。

## 交易成本

合成环境支持 TRF 交易成本。默认 `--transaction-cost-bps 10.0`，用于让主实验包含交易摩擦。

若要加入交易成本，可设置：

```powershell
--transaction-cost-bps 10
```

交易成本会通过 transaction remainder factor 扣减当期组合可投资价值。

## 输出文件

每次运行会生成一个独立目录：

```text
artifacts/synthetic/run_YYYYMMDD_HHMMSS_{policy}_{reference}_seed{seed}
```

目录包含：

- `synthetic_trace.csv`：逐期结果。
- `synthetic_summary.csv`：单次运行汇总。
- `config_snapshot.json`：配置快照。
- `tracking_diagnostics.png`：梯度追踪诊断图。
- `objective_and_wealth.png`：目标函数、财富和参考点图。

## 关键结果字段

- `market_regime`：当前半合成市场阶段。
- `portfolio_return`：当期组合净收益率。
- `normalized_wealth`：归一化财富，等于当前财富除以初始财富。
- `relative_wealth`：归一化财富减去更新前参考点。
- `cash_weight`：当期策略输出的现金权重。
- `turnover`：当期单边换手率，计算为 `0.5 * sum(abs(w_t - w_{t-1}))`。
- `pgr`：盈利实现比例，采用 Odean 次数口径，等于 `realized_gain_count / (realized_gain_count + paper_gain_count)`。
- `plr`：亏损实现比例，采用 Odean 次数口径，等于 `realized_loss_count / (realized_loss_count + paper_loss_count)`。
- `disposition_spread`：处置效应代理指标，等于 `pgr - plr`。
- `realized_gain_count`：开盘价高于买入成本且本期被卖出的风险资产数量。
- `realized_loss_count`：开盘价低于买入成本且本期被卖出的风险资产数量。
- `paper_gain_count`：开盘价高于买入成本且本期继续持有的风险资产数量。
- `paper_loss_count`：开盘价低于买入成本且本期继续持有的风险资产数量。
- `aggregate_pgr`、`aggregate_plr`：summary 文件中的全运行期次数汇总口径 PGR/PLR。
- `gradient_norm`：当前 CPT 策略梯度估计的 L2 范数。
- `dynamic_local_regret_term`：Dynamic Local Regret 的单期项，按最近 `evaluation_horizon` 期梯度向量做指数时间平滑后取平方范数，平滑系数为 0.9。
- `dynamic_local_regret`：Dynamic Local Regret，等于 `dynamic_local_regret_term` 的逐步累计和。
- `average_dynamic_local_regret`：Dynamic Local Regret 的逐步平均值。
- `squared_gradient_norm`：当前梯度范数平方。
- `cumulative_squared_gradient_norm`：累计平方梯度范数。
- `average_squared_gradient_norm`：平均平方梯度范数。
- `objective_estimate`：滑动窗口下的 CPT 目标函数估计值。
- `reference_point`：当前参考点。
- `reference_drift`：相邻两期参考点变化幅度。
- `environment_drift`：外生均值漂移变化幅度。
- `window_approximation_proxy`：当前漂移和滑动窗口均值漂移之间的距离，用于近似观察窗口误差。
- `theta_norm`：策略参数 L2 范数。
- `theta_boundary_share`：保留为兼容字段；当前不做参数裁剪，取值固定为 0。

## 处置效应代理指标

处置效应用来观察策略是否更倾向于卖出盈利资产、继续持有亏损资产。

当前合成环境按上一期持仓、当期开盘价、平均买入成本和当期目标权重计算。现金资产不计入 PGR/PLR。

```text
pgr = realized_gain_count / (realized_gain_count + paper_gain_count)
plr = realized_loss_count / (realized_loss_count + paper_loss_count)
disposition_spread = pgr - plr
```

风险资产的盈亏状态用当期开盘价相对平均买入成本判断。新买入资产的买入成本取当期开盘价；加仓时用开盘价更新平均买入成本；减仓时保留剩余仓位的平均买入成本；清仓后清除该资产成本基准。

若 `disposition_spread > 0`，说明盈利资产实现比例高于亏损资产实现比例，存在处置效应迹象。若 `disposition_spread < 0`，说明策略更倾向于实现亏损资产。

该指标是交易行为代理指标，不参与 CPT-PG 参数更新。

## 假设逐条审核

### A1 动态参考点漂移有界

满足条件：收益率被 `tanh` 压缩到有界区间，归一化财富由有界 price-relative 连乘得到，参考点更新使用 `eta_gain` 和 `eta_loss` 的凸组合式更新。

当前状态：基本满足。

需要注意：若把 `return_bound` 设得过大或运行期数过长，归一化财富和参考点漂移仍会明显增大。建议报告 `reference_drift` 的最大值、均值和 95% 分位数。

### A2 滑动窗口近似误差有界

满足条件：半合成市场的阶段参数、真实残差噪声缩放和收益压缩边界均显式有界。

当前状态：基本满足。

需要注意：阶段切换附近的窗口误差会短期升高，这正是 tracking 实验要观察的区域。当前用 `window_approximation_proxy` 作为误差代理，还没有计算理论误差上界。

### MDP 非平稳性

满足条件：环境生成时使用显式时间变化的阶段参数、真实残差噪声缩放和真实滞后因子。由于收益分布参数随时间变化，给定同样状态和动作，下一期收益分布仍会随时间改变。

当前状态：满足。

需要注意：当前非平稳性是可控的合成非平稳性。它适合验证 tracking 机制，不能直接替代真实市场实证。

### A3 CPT 价值函数和概率权重函数条件

满足条件：实验中的归一化财富终值由有界收益生成，在有限窗口内有界；CPT 目标值使用概率权重函数 `w`，策略梯度权重使用 `w'`。

当前状态：部分满足。

需要注意：当前幂型价值函数在全实数域上无全局上界，依赖有限窗口内的归一化财富有界来满足实验域有界。当前 TK 概率权重函数在端点附近存在导数放大，代码通过概率下界处理数值稳定，但这属于实验实现约束。若论文假设要求严格的全局 Lipschitz 可微和导数 Lipschitz，需要在理论部分使用平滑概率权重函数，或明确分析域为 `[epsilon, 1-epsilon]`。

### A4 策略关于参数可微且得分函数有界

满足条件：特征被裁剪到 `[-1, 1]`，Dirichlet 的 `alpha=exp(theta^T x)` 映射连续可微且严格为正，参数更新使用最大绝对分量归一化梯度方向。

当前状态：部分满足。

需要注意：指数参数化下 `theta` 不再投影，因此严格有界 score 仍依赖额外理论假设或实验域控制。Dirichlet score 含有 `log(w_i)`，当采样权重接近 0 时理论上无界。代码用 `1e-300` 做数值下界，只能保证程序不会报错。若需要严格满足有界 score 假设，建议增加 interior simplex 混合：

```text
w_exec = (1 - K * delta) * w_sample + delta
```

其中 `K` 是资产总数，`delta` 是极小正数。

### A5 一阶驻点集合满足误差界

满足条件：当前输出了 `gradient_norm` 和 `squared_gradient_norm`，可以观察梯度是否下降。

当前状态：尚未直接验证。

需要注意：当前没有直接计算 `dist(theta_t, S_t)`。合成环境资产和参数维度较小，可以后续加入网格搜索或多起点数值优化，近似每期的一阶驻点集合 `S_t`，再验证平方梯度范数和距离之间的关系。

### A6 两个采样数固定且可调大

满足条件：脚本显式提供 `--cpt-samples` 和 `--gradient-samples`，默认分别为 256 和 32。

当前状态：满足。

需要注意：固定采样数下只能验证有限样本 tracking 表现。若要贴近渐近结论，需要做采样数敏感性分析，例如 `n_t=256,512,1024` 和 `m_t=32,64,128`。

### A7 更新步长固定

满足条件：脚本内部设置 `gamma_exponent=0.0`，实际更新步长恒为 `gamma0`。

当前状态：满足。

需要注意：固定步长通常对应跟踪一阶驻点邻域。若希望梯度精确趋近 0，需要递减步长和更强平稳条件。当前论文目标是非平稳 tracking，固定步长更容易解释。

### A8 用户偏好完全不涉及

满足条件：脚本不读取偏好路径，不调用 agent，不使用用户反馈。

当前状态：满足。

## prewarm 的含义

`prewarm_steps` 的作用是给第一期 CPT-PG 更新准备历史窗口。

例如 `prewarm_steps=30`，`evaluation_horizon=15` 时，正式评估第 1 期开始前，环境已经生成了至少 15 期历史收益和状态。算法用这些历史样本估计第一期的 CPT 目标和策略梯度。

prewarm 不更新 `theta`，不记录为评估期收益，也不等同于单独训练阶段。它只是历史缓冲区，确保第一个评估日就有完整滑动窗口。

## 偏差来源

- 滑动窗口偏差：当前目标函数用最近 `h` 期归一化财富终值近似，市场漂移越快，偏差越大。
- 阶段切换偏差：切换附近历史窗口同时包含旧分布和新分布。
- 有界收益偏差：`tanh` 压缩收益会降低极端尾部事件对 CPT 的影响。
- 概率下界偏差：CPT 权重导数在端点附近被数值下界截断。
- 梯度归一化偏差：参数更新只保留最大绝对分量归一化后的梯度方向，弱化不同目标函数之间的梯度尺度差异。
- 策略参数化偏差：softmax、sparsemax、Dirichlet 的探索分布不同，导致梯度估计的方差和方向可能不同。

## 方差来源

- 动作采样方差：Dirichlet 采样和高斯噪声会带来不同方差。
- CPT 分位数方差：`n_t` 越小，排序分位数越不稳定。
- score function 方差：`m_t` 越小，梯度估计波动越大。
- 市场冲击方差：共同冲击和个体冲击会造成收益轨迹分布变化。
- 阶段切换方差：阶段切换附近梯度范数可能出现尖峰。
- 参考点反馈方差：动态参考点会改变相对收益分布，进而改变 CPT 权重项。

## 建议验证顺序

先运行动态参考点和静态参考点的 Dirichlet 策略，确认 tracking 指标是否合理。再运行 softmax 和 sparsemax，检查结论是否依赖策略参数化。

推荐先做半合成 120 期小规模检查：

```powershell
python scripts\run_synthetic_tracking.py --policy-normalizer dirichlet --methods dynamic_cpt_pg symmetric_cpt_pg static_cpt_pg expected_return_pg exponential_utility_pg --seeds 3407 --steps 120 --prewarm-steps 30 --risky-assets 300 --evaluation-horizon 15 --cpt-samples 256 --gradient-samples 32 --gamma0 0.05 --dirichlet-execution-mode mean --factor-strength 2.5
```

主实验看 500 期、10 个 seeds：

```powershell
python scripts\run_synthetic_tracking.py --policy-normalizer dirichlet --methods dynamic_cpt_pg symmetric_cpt_pg static_cpt_pg expected_return_pg exponential_utility_pg --seeds 29 147 592 1201 2027 3141 3407 4513 4703 8089 --steps 500 --prewarm-steps 30 --risky-assets 300 --evaluation-horizon 15 --cpt-samples 256 --gradient-samples 32 --gamma0 0.05 --dirichlet-execution-mode mean --factor-strength 2.5
```

如果平均平方梯度范数下降、累计平方梯度范数的 log-log 斜率小于 1，并且阶段切换后梯度能保持可控，说明该半合成环境下的 tracking 现象成立。若动态非对称参考点在正常非平稳阶段切换后比对称/静态参考点更快恢复，说明动态参考点机制对渐进式市场变化有额外解释力。

## 路径依赖诊断实验

`scripts/run_path_dependence.py` 用于验证同一最终财富下，不同历史路径是否导致不同参考点和后续行为。

脚本构造两条历史：

- `peak_then_crash`：先涨到高位，再下跌到共同终点。
- `crash_then_recovery`：先跌到低位，再反弹到共同终点。

两条路径进入正式环境前具有相同的归一化财富，但参考点不同。之后二者进入同一个非平稳市场，并同时运行动态参考点 CPT-PG 和静态参考点 CPT-PG。静态方法的参考点固定为 1.0，因此两条历史路径在静态方法下会给出相同轨迹，这是用来证明路径记忆来自动态参考点的对照组。若动态参考点确实产生路径依赖，图中应看到：

- 初始 `reference_point` 明显不同。
- 初始 `relative_wealth` 明显不同。
- 早期 `cash_weight` 或风险暴露不同。
- 随时间推进，参考点和风险暴露差异逐步收敛。
- `dynamic_minus_static.png` 中同一路径下动态方法减静态方法的差值不恒为 0。

运行示例：

```powershell
python scripts\run_path_dependence.py --policy-normalizer dirichlet --seed 3407 --steps 120 --prewarm-steps 30 --risky-assets 30 --evaluation-horizon 15 --cpt-samples 256 --gradient-samples 32 --gamma0 1.0 --transaction-cost-bps 10
```
