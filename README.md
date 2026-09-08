# Dynamic Reference CPT-PG Portfolio Experiment

本项目用于验证动态参考点 CPT-PG 在非平稳投资组合环境中的在线跟踪能力。当前代码只保留算法、数据构建、实验运行、结果汇总和绘图，不包含用户偏好模块，也不会默认调用 LLM。

## Methods

`scripts/run_mvp.py` 支持五类方法：

- `dynamic_cpt_pg`：非对称动态参考点 CPT-PG。
- `symmetric_cpt_pg`：对称参考点适应 CPT-PG，`eta_gain` 和 `eta_loss` 取均值。
- `static_cpt_pg`：静态参考点 CPT-PG。
- `expected_return_pg`：以收益随机变量期望为目标的策略梯度。
- `exponential_utility_pg`：以指数效用期望为目标的风险敏感策略梯度。

已移除旧逻辑：用户偏好路径、多 Agent、默认 LLM 调用、偏好硬约束、偏好特征、外部文本资讯模块、候选池 Top-K、`mean_variance`、`equal_weight`、Dirichlet tempered 模式。

## Core Flow

每个评估交易日的主流程：

1. 读取当前可观测市场状态和上一期投资组合权重。
2. 在 `rolling_window` 模式下，用当前日期前最近 `h=evaluation_horizon` 个交易日估计目标函数和策略梯度。
3. 在 `on_policy_episode` 模式下，策略先执行一个 `h` 日有限时域 episode，再用 episode 轨迹估计目标函数和策略梯度。
4. 用 `n_t` 条轨迹估计目标函数值。
5. 用独立的 `m_t` 条轨迹估计 score-function 策略梯度；开启 `--shared-cpt-gradient-samples` 时，目标函数和梯度共用同一批轨迹。
6. 用最近 `h` 期梯度做指数时间平滑，形成实际更新方向。
7. 默认用最大绝对分量归一化后的平滑梯度更新 `theta`；使用 `--disable-gradient-normalization` 可改为直接用平滑梯度。
8. 用更新后的 `theta` 生成连续组合权重。
9. 收盘后按 open-to-close 收益和交易成本结算账户财富。
10. 动态参考点方法按归一化财富口径更新参考点。

## CPT Estimation

CPT-PG 的目标函数是有限时域收益轨迹形成的随机变量的 CPT 效用。当前主口径为归一化财富：

```text
objective_estimate = CPT({Z_i - r_t}_{i=1}^{n_t})
```

其中 `Z_i` 是第 `i` 条轨迹结束时的归一化财富，`r_t` 是当期参考点。

目标函数值使用 CPT 概率权重函数 `w`。代码位置：

- `compute_cpt_objective()`
- `cpt_probability_weight()`
- `gain_value()`
- `loss_value()`

CPT 策略梯度估计使用：

```text
g_t = 1 / m_t * sum_j psi(Z_j - r_t) * score_j
```

其中 `psi` 是 CPT 策略梯度中的权重项。该权重项通过 gain/loss utility 分位数差分构造，并使用概率权重函数导数 `w'`。代码位置：

- `compute_cpt_gradient()`
- `cpt_policy_gradient_weight()`
- `quantile_tail_sum()`
- `cpt_probability_weight_derivative()`

因此当前口径是：目标函数值用 `w`，CPT 策略梯度权重项用 `w'`。

## Policy Function

默认策略函数是 Dirichlet policy：

1. 每只股票有一行市场特征 `x_i(s_t)`。
2. 策略参数 `theta` 表示各特征的重要性。
3. 股票打分为 `z_i = x_i(s_t)^T theta`。
4. 打分映射到 Dirichlet 参数：

```text
alpha_i = exp(z_i)
```

5. 梯度和目标函数估计阶段从 `Dirichlet(alpha)` 随机抽样。
6. 真实执行阶段由 `--dirichlet-execution-mode` 控制：

- `sample`：从 `Dirichlet(alpha)` 随机抽样。
- `mean`：使用 `alpha / sum(alpha)`。

保留 `softmax` 和 `sparsemax` 作为策略函数对照。二者使用 `z_i + Gaussian noise` 得到 latent score，再归一化为非负且和为 1 的权重。

当前策略特征只包含市场因子，不包含用户偏好字段：

- `momentum_score`
- `value_score`
- `quality_score`
- `low_vol_score`
- `mean_reversion_score`
- `qbot_boll_reversion_score`
- `qbot_rsi_reversal_score`
- `qbot_macd_trend_score`
- `qbot_rsrs_timing_score`
- `balanced_score`

## Data

需要设置 `TUSHARE_TOKEN` 或 `Tushare_Token`。当前数据模块使用：

- `stock_basic`
- `trade_cal`
- `daily`
- `daily_basic`
- `index_weight`

外部文本资讯模块已移除。

### Survivorship-Free Universe

项目支持每日可交易股票池缓存，用于降低 survivorship bias：

```powershell
python scripts\prefetch_survivorship_free_universe.py --sleep-seconds 0.35
```

运行实验时通过 `--universe-by-date-path` 启用。

每日股票池过滤条件：

- `list_date <= trade_date`
- `delist_date` 为空或 `delist_date > trade_date`
- 非金融行业
- 当日 `daily` 有行情
- 若字段可用，则 `amount > 0` 且 `vol > 0`

### Strict Clean Mode

`--strict-drop-missing-stocks` 会剔除任一必需市场字段缺失的股票。该模式用于干净数据实验，主要读取本地 raw cache。

## Module Logic

`mvp_cpt_pg/config.py`

集中管理实验默认配置，包括日期、样本数、步长、CPT 参数、Dirichlet 参数、方法列表和输出目录。

`mvp_cpt_pg/market_data.py`

负责 Tushare 数据读取、缓存、股票池构建、行情 panel 构建、严格清洗和技术因子生成。输出 `MarketDataset`，包含 `universe`、`panel`、`source_status`、`trade_dates`。

`mvp_cpt_pg/actions.py`

负责把市场状态转换为策略特征矩阵，构造连续权重动作，统计持仓和调仓计划。

`mvp_cpt_pg/strategies.py`

实现所有优化方法。核心包括 CPT 目标估计、CPT 策略梯度、期望收益 PG、指数效用 PG、Dirichlet/softmax/sparsemax 策略、交易成本、参考点更新、组合价值结算。

`mvp_cpt_pg/runner.py`

负责完整实验流程。包括加载数据、读取初始持仓、逐日运行策略、实时写入 trace、summary、aggregate 和 plots。

`mvp_cpt_pg/metrics.py`

从 `daily_trace.csv` 汇总 run 级别和 method 级别指标，包括财富、Sharpe、最大回撤、梯度范数、平方梯度范数、动态局部遗憾、theta 规模、参考点漂移等。

`mvp_cpt_pg/plots.py`

根据 trace 实时生成财富、参考点、收益率、梯度范数、动态局部遗憾和平方梯度范数图。

`mvp_cpt_pg/synthetic_env.py`

构建半合成非平稳环境，使用真实沪深300因子和半合成收益验证算法跟踪能力。

## Scripts

`scripts/run_mvp.py`

主实验入口，支持方法选择、seed、日期窗口、样本数、步长、策略函数、严格清洗、每日股票池缓存等 CLI 参数。

`scripts/run_synthetic_tracking.py`

半合成非平稳环境入口，支持多方法、多 seed、三类策略函数、on-policy episode 和结果聚合图。

`scripts/run_path_dependence.py`

路径依赖诊断实验入口。

`scripts/run_grid_search.py`

用于分阶段并行运行参数搜索。会启动多个 `run_mvp.py` 子进程，并写入 grid log 和 manifest。

`scripts/prefetch_market_cache.py`

预取交易日、日线行情和 `daily_basic`。

`scripts/prefetch_survivorship_free_universe.py`

预处理每日可交易股票池缓存。

`scripts/deepseektest.py`、`scripts/Qwentest.py`

手动验证外部模型接口，不参与主实验流程。

## Key CLI Options

- `--methods`：要运行的方法。
- `--seeds`：随机种子列表。
- `--dry-run-days`：只运行评估期前 N 个交易日。
- `--initial-capital`：账户初始预算。
- `--initial-holdings`：初始持仓 CSV，字段至少为 `ts_code,buy_price,shares`。
- `--universe-by-date-path`：每日可交易股票池缓存。
- `--index-universe-code`：使用固定指数成分股票池，例如沪深300为 `000300.SH`。
- `--strict-drop-missing-stocks`：剔除缺失特征股票。
- `--fixed-sample-counts`：固定 `n_t` 和 `m_t`。
- `--shared-cpt-gradient-samples`：目标函数和策略梯度共用同一批轨迹。
- `--cpt-sample-base`：CPT 目标函数估计样本数初值。
- `--gradient-sample-base`：策略梯度估计样本数初值。
- `--gamma0`：策略梯度步长。
- `--gamma-exponent`：步长衰减指数，`0` 表示固定步长。
- `--disable-gradient-normalization`：关闭最大绝对分量归一化。
- `--estimation-mode {rolling_window,on_policy_episode}`：估计模式。
- `--policy-normalizer {dirichlet,softmax,sparsemax}`：策略权重生成方式。
- `--policy-temperature`：softmax/sparsemax 温度。
- `--policy-noise-scale`：softmax/sparsemax 的高斯探索噪声。
- `--dirichlet-execution-mode {sample,mean}`：Dirichlet 真实执行模式。
- `--fixed-asset-count`：每个 method/seed 固定抽取一次股票池。
- `--bootstrap-asset-count`：每次采样时 bootstrap 抽取股票位置。
- `--eta-gain`、`--eta-loss`：动态参考点上行和下行适应速度。

## Example Commands

真实 A 股主实验：

```powershell
python scripts\run_mvp.py `
  --methods dynamic_cpt_pg symmetric_cpt_pg static_cpt_pg expected_return_pg exponential_utility_pg `
  --seeds 29 147 3141 `
  --prewarm-start 20221201 `
  --prewarm-end 20221230 `
  --evaluation-start 20230103 `
  --evaluation-end 20260531 `
  --evaluation-horizon 5 `
  --estimation-mode on_policy_episode `
  --initial-capital 1000000 `
  --index-universe-code 000300.SH `
  --fixed-sample-counts `
  --shared-cpt-gradient-samples `
  --gradient-sample-base 64 `
  --gamma0 0.05 `
  --gamma-exponent 0 `
  --policy-normalizer dirichlet `
  --dirichlet-execution-mode mean `
  --strict-drop-missing-stocks
```

半合成非平稳实验：

```powershell
python scripts\run_synthetic_tracking.py `
  --policy-normalizer dirichlet `
  --estimation-mode on_policy_episode `
  --methods dynamic_cpt_pg symmetric_cpt_pg static_cpt_pg expected_return_pg exponential_utility_pg `
  --seeds 29 147 592 1201 2027 3141 3407 4513 4703 8089 `
  --steps 650 `
  --prewarm-steps 30 `
  --risky-assets 100 `
  --evaluation-horizon 5 `
  --cpt-samples 256 `
  --gradient-samples 256 `
  --shared-cpt-gradient-samples `
  --gamma0 0.05 `
  --transaction-cost-bps 10 `
  --dirichlet-execution-mode mean `
  --factor-strength 2.5
```

## Outputs

每次运行会创建独立目录：

```text
artifacts/results/run_YYYYMMDD_HHMMSS/
```

主要输出：

- `traces/daily_trace.csv`：逐日 trace。
- `tables/summary_by_run.csv`：方法与 seed 级别汇总。
- `tables/summary_by_method.csv`：方法级别聚合。
- `tables/stock_info_source_status.csv`：数据源和清洗状态。
- `plots/*.png`：自动结果图。
- `config_snapshot.json`：本次运行配置快照。

关键字段：

- `wealth`：账户财富曲线。
- `day_return_rate`：总账户日净收益率。
- `investment_return_rate`：已投资组合日净收益率。
- `reference_point`：动态参考点。
- `objective_estimate`：当期收益随机变量上的目标函数估计。
- `offline_cpt_common_ref`：统一参考点口径下的离线 CPT 重评估。
- `gradient_norm`：策略梯度估计 L2 范数。
- `dynamic_local_regret_term`：最近 `evaluation_horizon` 期梯度指数时间平滑后的平方范数，平滑系数为 0.9。
- `dynamic_local_regret`：`dynamic_local_regret_term` 的累计和。
- `average_dynamic_local_regret`：动态局部遗憾逐日平均值。
- `average_squared_gradient_norm`：平均平方梯度范数。
- `cumulative_squared_gradient_norm`：累计平方梯度范数。
- `theta_norm`、`theta_max_abs`、`theta_boundary_share`：参数规模诊断。
- `holding_count`：有效持仓股票数。
- `turnover`：股票权重变化的单边换手率。
- `policy_normalizer`：`dirichlet`、`softmax` 或 `sparsemax`。

## Git Hygiene

建议提交：

- `mvp_cpt_pg/`
- `scripts/`
- `README.md`
- `requirements.txt`
- `SYNTHETIC_ENVIRONMENT.md`

不要提交：

- `artifacts/cache/`
- `artifacts/inputs/`
- `artifacts/results/`
- `reference_paper/`
- `.agents/`
- `.aris/`
- `.claude/`
- PDF、临时结果、本地环境文件、token。

提交前检查：

```powershell
git status --short
git diff --cached --name-only
```
