# 实验 5--7：外部有效性与行为拓展

这三个实验与 E1--E4 分开运行。E1--E4 回答理论跟踪、估计器和消融问题；E5--E7 不把真实市场绩效或模拟满意度当作动态局部遗憾定理的证明。

## E5：沪深300真实市场回测

输入是 `artifacts/cache/raw/tushare` 中点时可用的日行情、日基本面、指数权重和公告后财务指标。资产池在测试开始日前固定，因子只使用执行日前可获得的信息：20日动量、5日反转、20日波动率、估值、流动性、规模、已公告质量因子和上一期权重。训练轨迹使用决策日前的历史块，真实执行路径与学习轨迹使用同一个随机 Dirichlet actor，不切换为均值动作。交易成本按 `c * ||w_t-w_{t-1}||_1` 从财富中扣除。

比较对象是 DRCPT-PG、对称参考点 CPT-PG、静态参考点 CPT-PG、期望财富 PG、指数效用 PG 和等权基线；HS300 指数权重与逆波动率只用于数据处理或资产池构造时的参考，不作为回测方法。四联图依次报告财富轨迹、回撤与恢复路径、终值财富--最大回撤--换手气泡图，以及平均总换手与累计交易成本。表格另外报告年化收益、波动、Sharpe、最大回撤、收益/换手比和终值现金比例。该实验回答“方法能否迁移到真实行情并保留经济决策差异”，不用于证明 DLR 次线性。

```powershell
python scripts/run_external_validity.py --experiment 5 `
  --cache artifacts/cache/raw/tushare --panel artifacts/inputs/paper_hs300_full.csv `
  --output results/e5_hs300_e3sync_expcompatible_beta6_c10 `
  --start-date 20230103 --test-start 20250102 `
  --end-date 20260529 --assets 300 --pool-seed 2022 `
  --seeds 29 147 3141 42 3407 592 7 101 2024 2025 `
  --horizon 5 --gamma 0.08 --trajectory-budget 512 `
  --evaluation-n 512 --evaluation-m 256 --eta-gain 0.2 --eta-loss 0.05 `
  --cost 0.001 --trade-fraction 0.2 `
  --policy-sharpness 6 --dirichlet-scale 10
```

上述命令对应当前正式 E5 manifest。`beta=6` 只放大资产间得分差异，`c0=10` 只调节共同浓度尺度；二者都保留指数 Dirichlet 参数化。当前缓存若无法覆盖完整的成分股行情，脚本会在构造固定池时失败，不会静默换股。允许停牌日沿用上次报价时，`--allow-suspension-carry` 已由脚本默认开启，具体冻结记录写入面板元数据。行情收益是 `close/pre_close-1` 的价格相对收益，不冒充含分红总收益。

## E6：动态参考点与处置效应对齐

优先读取 `behavior/processing_process.zip` 中已完成除权除息处理的 `6purchase/xrxddisposition_*.pkl`。这些文件仍保留单位 `(投资者, 股票, 调仓事件)`、调仓前后权重和 `stock_pool_beforexrxdpurchase`，因此不需要再次读取约780万行原始Excel。只有处理包不存在时才回退到原始调仓记录。每个样本都是调仓前已经持有的股票，实际结果统一定义为下一次调仓时目标权重是否下降。静态买入成本直接来自处理文件，动态参考点在事件序列上递推，且只使用当前决策前的信息。

四个模型使用完全相同的样本、实际减持结果、控制变量、时间切分和 Logistic 模型，只替换参考点表示：无参考点、静态买入成本、对称适应和非对称适应。公共控制包括持仓年龄、事件频数、前期权重、价格、主动性、投资者历史减持率、历史机会数，以及严格滞后一期的市场、持仓和社交汇总；当前决策不能生成自己的预测变量。样本按日期划分训练、验证和测试，置信区间使用同一组投资者重抽样的配对 cluster bootstrap。

E6 不再把原论文按静态买入成本定义的 PGR、PLR 当作不同参考点模型的共同因变量。对测试集中同一批实际减持决策，分别按照静态、对称动态和非对称动态参考点划分盈利与亏损状态，再计算盈利状态减持率、亏损状态减持率及二者之差。Figure E6 的 A 面板比较三种参考点相对无参考点模型的样本外 Log loss 和 Brier 改进；B 面板比较各参考点划分下实际盈利/亏损减持率；C 面板报告参考点条件下的处置差异；D 面板按训练期历史减持率和交易活跃度分组报告增量预测价值。只有当非对称参考点同时提供稳定的样本外预测增益和正的处置差异时，才能说它与实际决策更一致；该实验不作因果识别。

```powershell
python scripts/run_external_validity.py --experiment 6 `
  --behavior-root behavior --output results/e6_reference_behavior_eta041 `
  --max-opportunities 0 --bootstrap 200 --eta-gain 0.4 --eta-loss 0.1 --seed 2026
```

首次运行直接读取处理压缩包并在 E6 输出根目录生成 `opportunities.parquet`，不再重复解析原始 Excel。若已有同一 `eta_gain/eta_loss` 口径的完整机会表，可显式传入 `--e6-opportunities-input` 复用；不得跨参考点参数复用。正式结果必须使用 `--max-opportunities 0`，并检查 `opportunities.json` 和 `model/manifest.json`。

## E7：异质性投资者模拟交互

用行为汇总表、投资者画像、持仓股票数和关注网络的静态统计聚类出若干投资者类型。E6 训练期的盈利减持率、亏损减持率和处置差异按投资者机会数做收缩，小样本投资者不会因少数交易被判为极端类型；E6 的全局回归系数不作为聚类特征。输出标签使用可解释的行为描述，不使用 `type_x` 之类的无语义编号。先在历史留出段检查类型画像能否复现持股数、交易频率和 PGR/PLR，再把抽样投资者放入逐期交互。

E7 从 E5 保存的逐期参数和确定性随机流精确重放现金加 300 只股票的完整推荐权重。每个五日 episode 开始时，先根据当时可见的首日推荐、历史行为和滞后满意度决定是否采纳，随后才观察本期收益；采纳后连续五日执行该方法的推荐，拒绝则连续五日保持期初组合。两条路径分别逐日扣除适用成本并更新动态参考点，第五日比较终端前景价值。因此本期任何收益和未来推荐都不会进入期初采纳概率。顾问策略采用 E5 中冻结的参考点速度，投资者参考点采用 E6 行为校准速度，二者在 manifest 中分别记录，不宣称为同一组参数。

E7 图的前三个面板分别展示投资者类型画像、按类型分解的 episode 采纳率，以及采纳条件下的五日期末满意度；第四个面板将每种方法在所有投资者类型上的平均采纳率与条件满意度画成气泡图，气泡大小表示平均兼容度。表格另外报告把拒绝记为零的总体交互价值，避免把“是否采纳”和“采纳后是否满意”混成同一个主指标。

```powershell
python scripts/run_external_validity.py --experiment 7 `
  --behavior-root behavior `
  --e5-run results/e5_hs300_e3sync_expcompatible_beta6_c10 `
  --e6-model results/e6_reference_behavior_eta041/model `
  --output results/e7_heterogeneous_agents_reproduction `
  --clusters 4 --agents-per-type 16 --e6-shrinkage 20 --seed 2026
```

E7 保留 E5 的 seed 级市场路径，在同一投资者、seed 和 episode 上对所有方法使用共同随机采纳抽样；总体结果按真实校准样本中的类型占比加权。`post_sent.pkl` 没有可靠的逐帖时间戳，因此不会被解释为严格的滞后社交因果变量。`agents-per-type` 控制每类抽取的真实校准投资者数量；正式结果应固定参数后运行，不根据输出反向选择代理数量。E7 的满意度和采纳率是参数化反事实模拟，不是观察到的真实用户满意度或真实采纳行为。

若 E6 结果目录不包含 `opportunities.parquet`，E7 入口会按 E6 模型的参考点参数从行为数据重建机会表，并核对样本行数。复现时使用新的输出目录，保留论文所用的正式结果。

E5 和 E7 的 manifest 位于结果根目录，E6 的模型 manifest 位于 `model/manifest.json`。审计时先检查 manifest，再检查源数据哈希和时间切分，最后解释经济指标；不要把 E5 的 Sharpe、E6 的行为预测准确率或 E7 的模拟满意度与 E1--E4 的理论 DLR 数值混排。
