# Dynamic Reference CPT-PG Portfolio Experiment

本项目用于验证一个面向 A 股投资组合的多 Agent 动态参考点 CPT-PG 框架。核心实验问题是：在同一市场状态和同一用户偏好路径下，动态参考点 CPT-PG 是否比静态参考点或普通 PG 基线表现出更好的训练稳定性、用户适配性和投资结果。

代码保留核心算法、数据构建、实验运行和绘图汇总。缓存、结果文件、LLM 缓存、参考 PDF、本地 Agent 状态不应提交到 Git。

## Methods

`scripts/run_mvp.py` 支持：

- `dynamic_cpt_pg`：动态参考点 CPT-PG，偏好动态更新。
- `static_cpt_pg`：静态参考点 CPT-PG，偏好冻结。
- `dynamic_cpt_pg_frozen_pref`：动态参考点 CPT-PG，偏好冻结。
- `static_ref_dynamic_pref_cpt_pg`：静态参考点 CPT-PG，偏好动态更新。
- `expected_return_pg`：期望收益策略梯度。
- `exponential_utility_pg`：指数效用风险敏感策略梯度。

已移除的旧实验逻辑包括候选池 Top-K 筛选、`mean_variance`、`equal_weight`、临时聊天/烟测/eta 分析脚本。

## Algorithm

每个评估交易日 `t` 的流程：

1. 开盘前读取 `t-1` 交易日可观测市场状态。
2. 用户偏好路径或偏好 Agent 给出 `style_tilt` 和硬约束。
3. CPT-PG 用 `< t` 的滑动历史窗口估计当期目标函数和策略梯度。
4. 参数按 `theta_{t+1} = clip(theta_t + gamma_t * g_t)` 更新。
5. 策略从更新后的参数抽样连续投资组合权重。
6. 投顾 Agent 或 deterministic trace 输出推荐解释。
7. 收盘后用当日 open-to-close 收益结算净收益率，并更新动态参考点。

策略是连续权重策略。默认 latent policy 为：

```text
z = Phi(s) theta + epsilon,  epsilon ~ N(0, sigma^2 I)
w_raw = normalizer(z)
w = project_to_constraints(w_raw)
```

`normalizer` 默认是 `softmax`，也可用 `sparsemax` 做稀疏化对照实验。高斯噪声负责探索，归一化负责把 latent 分数映射到非负且和为 1 的组合权重。

## Data

### Tushare

需要设置 `TUSHARE_TOKEN` 或 `Tushare_Token`。使用的数据包括：

- `stock_basic`
- `trade_cal`
- `daily`
- `daily_basic`
- 可选 `news`

### Akshare

用于市场新闻和股票信息源。新闻聚合为市场级情绪特征，并通过 `style_tilt` 和新闻压力调整策略特征矩阵。

### Survivorship-Free Universe

为避免只用当前上市股票倒推历史导致 survivorship bias，项目支持预处理每日可交易股票池：

```powershell
$env:HTTP_PROXY='http://127.0.0.1:7897'
$env:HTTPS_PROXY='http://127.0.0.1:7897'

python scripts\prefetch_survivorship_free_universe.py --sleep-seconds 0.35
```

输出示例：

```text
artifacts/cache/universe_by_date/survivorship_free_universe_20241215_20260512.csv
```

每日股票池过滤条件：

- `list_date <= trade_date`
- `delist_date` 为空或 `delist_date > trade_date`
- 非金融行业
- 当日 `daily` 有行情
- 若字段可用，则 `amount > 0` 且 `vol > 0`

运行实验时通过 `--universe-by-date-path` 启用该缓存。缓存文件不提交 Git。

### Strict Clean Mode

`--strict-drop-missing-stocks` 会在构建 panel 后剔除任一必需非新闻字段缺失的股票。该模式用于干净数据实验，不会实时请求接口，主要使用本地 raw cache。

## LLM Agents

系统包含三个 Agent：

- 用户 Agent：生成用户反馈。
- 偏好 Agent：把语义反馈转为下一期偏好和约束。
- 投顾 Agent：解释优化器输出的组合调仓动作。

默认 LLM 配置在 `mvp_cpt_pg/config.py`。如果使用预生成偏好路径，例如：

```text
artifacts/inputs/preference_path_balanced_240d.csv
```

则运行时跳过逐轮 LLM 交互，更适合可复现实验。

## Key CLI Options

- `--methods`：要运行的方法。
- `--seeds`：随机种子列表。
- `--dry-run-days`：只运行评估期前 N 个交易日。
- `--initial-capital`：用户预算上限。
- `--initial-holdings`：初始持仓 CSV，字段至少为 `ts_code,buy_price,shares`。
- `--preference-path`：预生成动态偏好路径。
- `--universe-by-date-path`：每日无幸存者偏差股票池缓存。
- `--strict-drop-missing-stocks`：剔除缺失特征股票。
- `--fixed-sample-counts`：固定 CPT 目标函数样本数 `n_t` 和梯度样本数 `m_t`。
- `--cpt-sample-base`：CPT 目标函数估计样本数初值，默认 `256`。
- `--gradient-sample-base`：策略梯度估计样本数初值，默认 `32`。
- `--gamma0`：策略梯度初始步长。
- `--gamma-exponent`：步长衰减指数，`0` 表示固定步长。
- `--policy-noise-scale`：latent 高斯探索噪声。
- `--policy-normalizer {softmax,sparsemax}`：latent 到原始权重的归一化方式。
- `--disable-preference-constraints`：禁用用户硬约束投影，但保留 `style_tilt` 对策略特征打分的影响。
- `--eta-gain`、`--eta-loss`：动态参考点上行和下行适应速度。
- `--enable-tushare-news`：启用 Tushare 新闻补充。

## Recommended Commands

### Sparsemax 对照实验

```powershell
$env:HTTP_PROXY='http://127.0.0.1:7897'
$env:HTTPS_PROXY='http://127.0.0.1:7897'

python scripts\run_mvp.py `
  --methods dynamic_cpt_pg static_cpt_pg dynamic_cpt_pg_frozen_pref static_ref_dynamic_pref_cpt_pg `
  --seeds 29 147 3141 `
  --initial-capital 1000000 `
  --preference-path artifacts\inputs\preference_path_balanced_240d.csv `
  --fixed-sample-counts `
  --gamma0 2 `
  --gamma-exponent 0 `
  --strict-drop-missing-stocks `
  --universe-by-date-path artifacts\cache\universe_by_date\survivorship_free_universe_20241215_20260512.csv `
  --policy-normalizer sparsemax
```

### 禁用偏好硬约束但保留 style

```powershell
python scripts\run_mvp.py `
  --methods dynamic_cpt_pg static_cpt_pg `
  --seeds 29 147 3141 `
  --initial-capital 1000000 `
  --preference-path artifacts\inputs\preference_path_balanced_240d.csv `
  --fixed-sample-counts `
  --gamma0 2 `
  --gamma-exponent 0 `
  --strict-drop-missing-stocks `
  --disable-preference-constraints `
  --universe-by-date-path artifacts\cache\universe_by_date\survivorship_free_universe_20241215_20260512.csv
```

### 60 天 PG 基线对比

```powershell
python scripts\run_mvp.py `
  --methods dynamic_cpt_pg expected_return_pg exponential_utility_pg `
  --seeds 4703 `
  --dry-run-days 60 `
  --initial-capital 1000000 `
  --preference-path artifacts\inputs\preference_path_balanced_240d.csv `
  --fixed-sample-counts `
  --strict-drop-missing-stocks
```

## Outputs

每次运行会创建独立目录：

```text
artifacts/results/run_YYYYMMDD_HHMMSS/
```

主要输出：

- `traces/daily_trace.csv`：逐日 trace。
- `tables/summary_by_run.csv`：方法 × seed 汇总。
- `tables/summary_by_method.csv`：方法层面聚合。
- `tables/stock_info_source_status.csv`：数据源和清洗状态。
- `plots/*.png`：自动结果图。
- `config_snapshot.json`：本次运行配置快照。

重要字段：

- `wealth`：账户财富曲线。
- `investment_return_rate`：当日已投资组合净收益率。
- `reference_point`：动态参考点，按净收益率口径更新。
- `objective_estimate`：当期采样随机变量上的目标函数估计。
- `offline_cpt_common_ref`：统一参考点口径下的离线 CPT 重评估。
- `gradient_norm`：策略梯度估计 L2 范数。
- `average_squared_gradient_norm`：平均平方梯度范数。
- `cumulative_squared_gradient_norm`：累计平方梯度范数。
- `theta_norm`、`theta_max_abs`、`theta_boundary_share`：参数规模和边界诊断。
- `holding_count`：有效持仓股票数。
- `turnover`：股票调仓权重之和。
- `constraint_violation_reason`：约束违反原因。
- `policy_normalizer`：`softmax` 或 `sparsemax`。
- `preference_constraints_disabled`：是否禁用偏好硬约束。

## Project Structure

```text
mvp_cpt_pg/
  actions.py       连续权重动作、特征矩阵、约束投影、交易摘要
  agents.py        用户、偏好、投顾 Agent
  config.py        默认实验配置
  llm.py           OpenAI-compatible LLM client
  market_data.py   Tushare/Akshare 数据读取、缓存、特征构造
  metrics.py       汇总指标
  plots.py         结果图生成
  raw_cache.py     DataFrame 原始缓存
  runner.py        实验主流程
  schemas.py       Agent 和偏好结构
  strategies.py    CPT-PG、期望 PG、指数效用 PG
  utils.py         通用工具

scripts/
  run_mvp.py                            主实验入口
  run_grid_search.py                    网格搜索入口
  prefetch_market_cache.py              预取市场行情和新闻缓存
  prefetch_survivorship_free_universe.py 每日可交易股票池缓存
  generate_preference_paths.py          生成合成用户偏好路径
```

## Git Hygiene

建议提交：

- `mvp_cpt_pg/`
- `scripts/`
- `README.md`
- `requirements.txt`
- 论文框架 tex 文件

不要提交：

- `artifacts/cache/`
- `artifacts/inputs/`
- `artifacts/results/`
- `reference_paper/`
- `.agents/`
- `.aris/`
- `.claude/`
- PDF、临时结果和本地环境文件

提交前检查：

```powershell
git status --short
git diff --cached --name-only
```

确认 staged 文件中没有缓存、结果、PDF、token 或本地私有状态。
