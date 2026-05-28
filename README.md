# Dynamic Reference CPT-PG Portfolio Experiment

本项目用于验证面向 A 股投资组合的动态参考点 CPT-PG 在线优化框架。核心问题是：在同一市场状态、同一用户偏好路径和同一实验设置下，动态参考点 CPT-PG 是否能稳定跟踪时变目标，并在财富、风险、用户偏好适配等指标上形成可解释结果。

代码保留核心算法、数据构建、实验运行、结果汇总和绘图。缓存、实验结果、LLM 缓存、参考论文、本地 Agent 状态不应提交到 Git。

## Methods

`scripts/run_mvp.py` 支持六类方法：

- `dynamic_cpt_pg`：动态参考点 CPT-PG，偏好动态更新。
- `static_cpt_pg`：静态参考点 CPT-PG，偏好冻结。
- `dynamic_cpt_pg_frozen_pref`：动态参考点 CPT-PG，偏好冻结。
- `static_ref_dynamic_pref_cpt_pg`：静态参考点 CPT-PG，偏好动态更新。
- `expected_return_pg`：以收益随机变量的期望为目标的策略梯度。
- `exponential_utility_pg`：以指数效用期望为目标的风险敏感策略梯度。

已移除旧实验逻辑：外部文本资讯读取模块、候选池 Top-K 筛选、`mean_variance`、`equal_weight`、Dirichlet 第三类执行模式。

## Core Algorithm

每个评估交易日 `t` 的主流程：

1. 开盘前读取 `t-1` 交易日可观测市场状态。
2. 读取预生成偏好路径，或调用用户 Agent 与偏好 Agent 得到当期偏好。
3. 用 `< t` 的最近 `h=evaluation_horizon` 个交易日构造滑动窗口。
4. 用 `n_t` 条窗口收益轨迹估计目标函数值。
5. 用独立的 `m_t` 条窗口收益轨迹估计策略梯度。
6. 按 `theta_{t+1}=clip(theta_t+gamma_t g_t)` 更新策略参数。
7. 用更新后的 `theta_{t+1}` 生成当期连续组合权重。
8. 投顾 Agent 输出解释，或在偏好路径实验中写入 deterministic trace。
9. 收盘后用当日 open-to-close 收益结算净盈亏和净收益率。
10. 动态参考点方法用当日已投资组合净收益率更新参考点。

## CPT Estimation

CPT-PG 的目标函数是滑动窗口收益轨迹形成的收益随机变量的 CPT 效用。当前收益随机变量口径为每条 `h` 步轨迹的累计已投资组合净收益率。

目标函数估计：

```text
objective_estimate = CPT({R_i - r_t}_{i=1}^{n_t})
```

该目标函数值使用 CPT 概率权重函数 `w` 本身。代码位置：

- `compute_cpt_objective()`
- `cpt_probability_weight()`
- `gain_value()`
- `loss_value()`

CPT 策略梯度估计：

```text
g_t = 1 / m_t * sum_j psi(R_j - r_t) * score_j
```

其中 `psi` 是 CPT 梯度中的权重项。该权重项通过 `n_t` 条 CPT 样本构造 gain/loss utility 分位数差分，并使用概率权重函数导数 `w'`。代码位置：

- `compute_cpt_gradient()`
- `cpt_policy_gradient_weight()`
- `quantile_tail_sum()`
- `cpt_probability_weight_derivative()`

因此当前口径是：目标函数值用 `w`，CPT 策略梯度权重项用 `w'`。

## Policy Function

默认策略函数是 Dirichlet policy。流程如下：

1. 每只股票有一行特征 `x_i(s_t)`。
2. 策略参数 `theta` 表示各特征的重要性。
3. 股票打分为 `z_i = x_i(s_t)^T theta`。
4. 打分映射到 Dirichlet 参数：

```text
alpha_i = alpha_min + (alpha_max - alpha_min) * sigmoid(z_i)
```

5. 根据 `dirichlet_execution_mode` 输出权重：

- `sample`：从 `Dirichlet(alpha)` 随机抽样。
- `mean`：使用 `alpha / sum(alpha)`。

保留 `softmax` 和 `sparsemax` 作为对照策略。它们使用 `z_i + Gaussian noise` 得到 latent score，再归一化为非负且和为 1 的权重。

## Preference Modes

偏好字段包括：

- `risk_budget`
- `max_single_weight`
- `turnover_cap`
- `diversification_target`
- `style_tilt`

当前有三类偏好模式：

- 默认模式：偏好作为硬约束进入权重投影。`preference_features_enabled=False`，所以 style 默认不进入特征。
- `--preference-features-only`：偏好只进入策略特征，跳过硬约束投影。
- `--disable-preference-constraints`：禁用偏好硬约束，也禁用偏好特征。

若希望 style 和换手率只影响策略打分，可以使用 `--preference-features-only`。

## Data

需要设置 `TUSHARE_TOKEN` 或 `Tushare_Token`。当前数据模块只使用 Tushare：

- `stock_basic`
- `trade_cal`
- `daily`
- `daily_basic`

外部文本资讯读取模块已经移除。

### Survivorship-Free Universe

项目支持每日可交易股票池缓存，用于降低 survivorship bias：

```powershell
python scripts\prefetch_survivorship_free_universe.py --sleep-seconds 0.35
```

运行实验时通过 `--universe-by-date-path` 启用：

```text
artifacts/cache/universe_by_date/survivorship_free_universe_20241215_20260512.csv
```

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

集中管理实验默认配置，包括日期、样本数、步长、CPT 参数、Dirichlet 参数、LLM 配置、方法列表和输出目录。

`mvp_cpt_pg/market_data.py`

负责 Tushare 数据读取、缓存、股票池构建、行情 panel 构建、严格清洗、技术因子生成。输出 `MarketDataset`，包含 `universe`、`panel`、`source_status`、`trade_dates`。

`mvp_cpt_pg/actions.py`

负责把市场状态转换为策略特征矩阵，构造连续权重动作，执行偏好硬约束投影，统计持仓、调仓计划和约束违反原因。

`mvp_cpt_pg/strategies.py`

实现所有优化方法。核心包括 CPT 目标估计、CPT 策略梯度、期望收益 PG、指数效用 PG、Dirichlet/softmax/sparsemax 策略、交易成本、参考点更新、组合价值结算。

`mvp_cpt_pg/runner.py`

负责完整实验流程。包括加载数据、读取初始持仓、读取偏好路径或调用 Agent、逐日运行策略、实时写入 trace、summary、aggregate 和 plots。

`mvp_cpt_pg/agents.py`

封装用户 Agent、偏好 Agent、投顾 Agent。使用预生成偏好路径时，逐轮 LLM 交互会跳过。

`mvp_cpt_pg/llm.py`

封装 OpenAI-compatible LLM 调用、缓存、JSON 解析和错误处理。

`mvp_cpt_pg/schemas.py`

定义偏好、硬约束、用户反馈、投顾输出等结构化数据格式。

`mvp_cpt_pg/metrics.py`

从 `daily_trace.csv` 汇总 run 级别和 method 级别指标，包括财富、Sharpe、最大回撤、梯度范数、平方梯度范数、theta 规模、参考点漂移等。

`mvp_cpt_pg/plots.py`

根据 trace 实时生成财富、参考点、收益率、梯度范数、平方梯度范数、偏好和换手率图。

`mvp_cpt_pg/raw_cache.py`

提供 DataFrame 原始数据缓存，避免重复请求 Tushare。

`mvp_cpt_pg/utils.py`

提供目录创建、稳定哈希、z-score、权重归一化、JSON 提取、环境变量读取等通用函数。

`mvp_cpt_pg/progress.py`

封装 tqdm 进度条。

## Scripts

`scripts/run_mvp.py`

主实验入口，支持方法选择、seed、日期窗口、样本数、步长、策略函数、Dirichlet 参数、偏好模式、严格清洗、每日股票池缓存等 CLI 参数。

`scripts/run_grid_search.py`

用于分阶段并行运行参数搜索。会启动多个 `run_mvp.py` 子进程，并写入 grid log 和 manifest。

`scripts/prefetch_market_cache.py`

预取交易日、日线行情和 `daily_basic`，并生成市场上下文 CSV，供合成偏好路径使用。

`scripts/prefetch_survivorship_free_universe.py`

预处理每日可交易股票池缓存。

`scripts/generate_preference_paths.py`

根据市场上下文生成保守、均衡、激进三类动态偏好路径。

`scripts/deepseektest.py`、`scripts/test.py`

手动验证 LLM 接入，不参与主实验流程。

## Key CLI Options

- `--methods`：要运行的方法。
- `--seeds`：随机种子列表。
- `--dry-run-days`：只运行评估期前 N 个交易日。
- `--initial-capital`：用户预算上限。
- `--initial-holdings`：初始持仓 CSV，字段至少为 `ts_code,buy_price,shares`。
- `--preference-path`：预生成动态偏好路径。
- `--universe-by-date-path`：每日可交易股票池缓存。
- `--strict-drop-missing-stocks`：剔除缺失特征股票。
- `--fixed-sample-counts`：固定 `n_t` 和 `m_t`。
- `--cpt-sample-base`：CPT 目标函数估计样本数初值，默认 `256`。
- `--gradient-sample-base`：策略梯度估计样本数初值，默认 `32`。
- `--gamma0`：策略梯度初始步长。
- `--gamma-exponent`：步长衰减指数，`0` 表示固定步长。
- `--policy-normalizer {dirichlet,softmax,sparsemax}`：策略权重生成方式。
- `--policy-temperature`：softmax/sparsemax 温度。
- `--policy-noise-scale`：softmax/sparsemax 的高斯探索噪声。
- `--dirichlet-alpha-min`：Dirichlet 参数下界。
- `--dirichlet-alpha-max`：Dirichlet 参数上界。
- `--dirichlet-execution-mode {sample,mean}`：Dirichlet 执行模式。
- `--fixed-asset-count`：每个 method/seed 固定抽取一次股票池。
- `--bootstrap-asset-count`：每次采样时 bootstrap 抽取股票位置。
- `--preference-features-only`：偏好只进特征，跳过硬约束投影。
- `--disable-preference-constraints`：禁用偏好硬约束和偏好特征。
- `--eta-gain`、`--eta-loss`：动态参考点上行和下行适应速度。

## Example Command

```powershell
python scripts\run_mvp.py `
  --methods dynamic_cpt_pg static_cpt_pg expected_return_pg exponential_utility_pg `
  --seeds 4703 `
  --dry-run-days 60 `
  --initial-capital 1000000 `
  --preference-path artifacts\inputs\preference_path_balanced_240d.csv `
  --fixed-sample-counts `
  --strict-drop-missing-stocks `
  --policy-normalizer dirichlet `
  --dirichlet-execution-mode sample
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
- `reference_point`：动态参考点，按已投资组合净收益率更新。
- `objective_estimate`：当期收益随机变量上的目标函数估计。
- `offline_cpt_common_ref`：统一参考点口径下的离线 CPT 重评估。
- `gradient_norm`：策略梯度估计 L2 范数。
- `average_squared_gradient_norm`：平均平方梯度范数。
- `cumulative_squared_gradient_norm`：累计平方梯度范数。
- `projected_gradient_mapping_norm`：参数裁剪后的梯度映射范数。
- `theta_norm`、`theta_max_abs`、`theta_boundary_share`：参数规模和边界诊断。
- `holding_count`：有效持仓股票数。
- `turnover`：按股票权重变化计算的单边换手率。
- `constraint_violation_reason`：约束违反原因。
- `policy_normalizer`：`dirichlet`、`softmax` 或 `sparsemax`。
- `preference_features_only`：偏好是否只作为特征。
- `preference_constraints_disabled`：是否禁用偏好硬约束和偏好特征。

## Git Hygiene

建议提交：

- `mvp_cpt_pg/`
- `scripts/`
- `README.md`
- `requirements.txt`

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
