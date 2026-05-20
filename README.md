# Dynamic Reference CPT-PG Portfolio Experiment

本项目用于验证一个面向 A 股投资组合推荐的多 Agent 动态参考点 CPT-PG 框架。核心目标是比较动态参考点 CPT-PG、静态参考点 CPT-PG、期望收益 PG、指数效用 PG 在同一市场数据和同一用户偏好路径下的训练稳定性与投资表现。

当前代码保留的是核心算法与实验流程。结果文件、缓存数据、LLM 缓存和论文参考 PDF 不应提交到 Git。

## 当前保留的方法

`scripts/run_mvp.py` 支持以下方法：

- `dynamic_cpt_pg`：动态参考点 CPT-PG，偏好动态更新。
- `static_cpt_pg`：静态参考点 CPT-PG，偏好冻结。
- `dynamic_cpt_pg_frozen_pref`：动态参考点 CPT-PG，偏好冻结。
- `static_ref_dynamic_pref_cpt_pg`：静态参考点 CPT-PG，偏好动态更新。
- `expected_return_pg`：期望收益 PG。
- `exponential_utility_pg`：指数效用 PG。

已删除的内容：

- 候选池筛选逻辑。
- `mean_variance` 和 `equal_weight` 基线。
- PGM 相关指标。
- 临时脚本 `agent_chat.py`、`smoke_test.py`、`analyze_eta.py`。

## 默认实验区间

默认日期在 `mvp_cpt_pg/config.py` 中：

- `prewarm`: `20250414` 到 `20250514`
- `evaluation`: `20250515` 到 `20260512`
- `evaluation_horizon`: `15`
- `initial_capital_amount`: `1000000`
- `cpt_sample_base`: `256`
- `gradient_sample_base`: `32`
- `gamma0`: `3.0`
- `gamma_exponent`: `0.51`
- `policy_noise_scale`: `0.1`
- `exponential_risk_aversion`: `0.5`

`prewarm` 用于准备首个评估日之前的历史窗口和初始市场上下文。`evaluation` 是真实输出每日推荐、观测收益、更新参考点和记录 trace 的区间。

## 数据来源

### Tushare

用于读取：

- A 股股票列表。
- 交易日历。
- 日线行情 `daily`。
- 每日基础指标 `daily_basic`。
- 可选新闻快讯 `news`。

环境变量需要设置其中一个：

```powershell
$env:TUSHARE_TOKEN="你的token"
```

或使用系统环境变量 `Tushare_Token`。

### Akshare

用于读取市场新闻和股票信息源。当前 Akshare 新闻进入市场情绪特征，并影响策略特征矩阵中的风格倾向。

### 严格清洗模式

默认运行保留原始数据填补逻辑。若希望剔除含缺失字段的股票，使用：

```powershell
--strict-drop-missing-stocks
```

该模式会：

- 跳过 `daily_basic` 整体为空的交易日。
- 在 `prewarm.start` 到 `evaluation.end` 内检查每只股票的必需非新闻字段。
- 任一必需字段缺失就剔除该股票。
- 将清洗结果写入 `tables/stock_info_source_status.csv`。

### Tushare 新闻补充

默认不开启。若要用 Tushare 新闻快讯补充 Akshare 新闻，使用：

```powershell
--enable-tushare-news
```

该选项会额外读取 `sina`、`10jqka`、`eastmoney`、`cls` 四类 Tushare 快讯源。由于 Tushare 新闻接口可能有权限和频率限制，建议先确认本地 token 权限。

## LLM Agent

项目包含三个 Agent：

- 用户 Agent：模拟用户输入和语义反馈。
- 偏好 Agent：从用户反馈中生成下一期偏好和约束。
- 投顾 Agent：解释优化器输出的投资组合权重调整动作。

默认 LLM 配置在 `mvp_cpt_pg/config.py`：

- `model`: `deepseek-v4-flash`
- `base_url`: `https://api.deepseek.com`
- `enable_thinking`: `True`
- `reasoning_effort`: `max`

可通过命令行覆盖：

```powershell
python scripts\run_mvp.py --llm-model 模型名 --llm-base-url 兼容OpenAI的base_url
```

如果使用预生成偏好路径，例如 `artifacts\inputs\preference_path_balanced_240d.csv`，运行时不需要每轮调用偏好 Agent。

## 初始资金和持仓

无初始持仓时：

- 不传 `--initial-holdings`。
- 初始参考点为 `0`。
- `--initial-capital` 表示用户预算上限。
- 初始组合为现金，优化器根据偏好和市场状态逐步建仓。

有初始持仓时：

```powershell
--initial-holdings artifacts\inputs\initial_holdings.csv
```

CSV 至少包含：

```csv
ts_code,buy_price,shares
000001.SZ,10.5,1000
```

当前逻辑会用首个评估日前最近可用收盘价计算初始组合权重和已有市值。

## 常用运行命令

### 60 天固定采样数，均衡用户，seed 4703

```powershell
python scripts\run_mvp.py --methods dynamic_cpt_pg expected_return_pg exponential_utility_pg --seeds 4703 --dry-run-days 60 --initial-capital 1000000 --preference-path artifacts\inputs\preference_path_balanced_240d.csv --fixed-sample-counts
```

### CPT 相关四个方法，240 天，均衡用户

```powershell
python scripts\run_mvp.py --methods dynamic_cpt_pg static_cpt_pg dynamic_cpt_pg_frozen_pref static_ref_dynamic_pref_cpt_pg --seeds 4703 --initial-capital 1000000 --preference-path artifacts\inputs\preference_path_balanced_240d.csv --fixed-sample-counts
```

### 使用严格干净股票池

```powershell
python scripts\run_mvp.py --methods dynamic_cpt_pg expected_return_pg exponential_utility_pg --seeds 4703 --dry-run-days 60 --initial-capital 1000000 --preference-path artifacts\inputs\preference_path_balanced_240d.csv --fixed-sample-counts --strict-drop-missing-stocks
```

### 启用 Tushare 新闻补充

```powershell
python scripts\run_mvp.py --methods dynamic_cpt_pg expected_return_pg exponential_utility_pg --seeds 4703 --dry-run-days 60 --initial-capital 1000000 --preference-path artifacts\inputs\preference_path_balanced_240d.csv --fixed-sample-counts --enable-tushare-news
```

## 关键命令行参数

- `--methods`：指定运行方法。
- `--seeds`：指定随机种子。
- `--dry-run-days`：只运行评估期前 N 个交易日。
- `--initial-capital`：用户预算上限。
- `--initial-holdings`：初始持仓 CSV。
- `--preference-path`：预生成用户偏好路径。
- `--evaluation-horizon`：滑动历史窗口长度。
- `--cpt-sample-base`：CPT 目标函数估计样本数初值。
- `--gradient-sample-base`：策略梯度估计样本数初值。
- `--fixed-sample-counts`：固定 `n_t` 和 `m_t`，用于对照实验。
- `--gradient-diagnostic-repeats`：同一期重复估计梯度的次数，用于估计梯度噪声。
- `--gradient-diagnostic-use-mean-update`：用重复估计的均值更新参数。
- `--eta-gain`：参考点上行适应速度。
- `--eta-loss`：参考点下行适应速度。
- `--gamma0`：初始策略梯度步长。
- `--gamma-exponent`：步长衰减指数。
- `--policy-noise-scale`：策略抽样噪声。
- `--exponential-risk-aversion`：指数效用 PG 风险厌恶参数。
- `--strict-drop-missing-stocks`：启用严格缺失剔除。
- `--enable-tushare-news`：启用 Tushare 新闻补充。

## 输出文件

每次运行会创建独立结果目录：

```text
artifacts/results/run_YYYYMMDD_HHMMSS/
```

主要输出：

- `traces/daily_trace.csv`：逐日 trace，包含收益、参考点、目标函数估计值、梯度范数、参数向量、权重和交易计划。
- `tables/summary_by_run.csv`：按方法和 seed 汇总。
- `tables/summary_by_method.csv`：按方法聚合多 seed 结果。
- `tables/stock_info_source_status.csv`：Tushare、Akshare、清洗模式的数据源状态。
- `plots/*.png`：自动生成的结果图。
- `config_snapshot.json`：本次运行的完整配置快照。

## 主要评价指标

- `wealth`：账户财富曲线。
- `investment_return_rate`：当日已投资组合净收益率。
- `reference_point`：动态参考点，当前按净收益率口径更新。
- `objective_estimate`：当前方法在当期采样随机变量上的目标函数估计值。
- `offline_cpt_common_ref`：统一参考点口径下的离线 CPT 重评估。
- `gradient_norm`：当前策略梯度估计的 L2 范数。
- `average_squared_gradient_norm`：平方梯度范数均值。
- `cumulative_squared_gradient_norm`：平方梯度范数累积和。
- `theta_norm`：策略参数向量范数。
- `theta_max_abs`：策略参数最大绝对值。
- `theta_boundary_share`：参数触及裁剪边界的比例。
- `holding_count`：当前有效持仓股票数。
- `turnover`：当期调仓换手。
- `rating_mean`、`adoption_rate`：用户反馈表现。

## 项目结构

```text
mvp_cpt_pg/
  actions.py       连续权重动作、约束投影、交易摘要
  agents.py        用户、偏好、投顾 Agent 调用
  config.py        默认实验配置
  llm.py           OpenAI 兼容 LLM 客户端
  market_data.py   Tushare 和 Akshare 数据读取、缓存、特征构造
  metrics.py       汇总指标
  plots.py         结果图生成
  raw_cache.py     DataFrame 缓存
  runner.py        实验主流程
  schemas.py       Agent 输入输出结构
  strategies.py    CPT-PG、期望 PG、指数效用 PG
  utils.py         通用工具

scripts/
  run_mvp.py                 主运行入口
  run_grid_search.py         网格搜索入口
  prefetch_market_cache.py   预取市场缓存
  generate_preference_paths.py 生成合成用户偏好路径
```

## Git 提交建议

建议提交：

- `mvp_cpt_pg/`
- `scripts/`
- `README.md`
- `requirements.txt`
- `paper_framework.tex`
- `framework-update.tex`

建议忽略：

- `artifacts/cache/`
- `artifacts/results/`
- `artifacts/plots/`
- `artifacts/tables/`
- `artifacts/traces/`
- `reference_paper/`
- `.agents/`
- `.aris/`
- `.claude/`
- `__pycache__/`

提交前检查：

```powershell
git diff --cached --name-only
```

确认输出中没有缓存、结果文件、PDF 和本地 Agent 状态文件。
