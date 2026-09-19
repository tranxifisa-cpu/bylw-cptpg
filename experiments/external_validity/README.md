# 实验 5--7：外部有效性与行为拓展

这三个实验与 E1--E4 分开运行。E1--E4 回答理论跟踪、估计器和消融问题；E5--E7 不把真实市场绩效或模拟满意度当作动态局部遗憾定理的证明。

## E5：沪深300真实市场回测

输入是 `artifacts/cache/raw/tushare` 中点时可用的日行情、日基本面、指数权重和公告后财务指标。资产池在测试开始日前固定抽取，因子只使用执行日前可获得的信息：20日动量、5日反转、20日波动率、估值、流动性、规模、已公告质量因子和上一期权重。训练轨迹使用历史块采样，真实执行按当前日期的策略均值动作，交易成本按 `c * ||w_t-w_{t-1}||_1` 从财富中扣除。

比较对象是 DRCPT-PG、对称参考点 CPT-PG、静态参考点 CPT-PG、期望财富 PG、指数效用 PG，以及等权、沪深300指数权重和逆波动率基线。主要输出为财富、回撤、收益风险散点、换手和累计成本，以及年化收益、波动、Sharpe、最大回撤、收益/换手比和终值现金比例。该实验回答“方法能否迁移到真实行情并保留经济决策差异”，不用于证明 DLR 次线性。

```powershell
python scripts/run_external_validity.py --experiment 5 `
  --cache artifacts/cache/raw/tushare --panel artifacts/inputs/e5_hs300_full.csv `
  --output results/e5_hs300_real --start-date 20230103 --test-start 20250102 `
  --end-date 20260529 --assets 300 --pool-seed 2022 `
  --seeds 29 147 3141 42 3407 592 7 101 2024 2025 `
  --horizon 5 --gamma 0.05 --trajectory-budget 256 `
  --evaluation-n 512 --evaluation-m 256 --cost 0.001 --trade-fraction 0.2
```

当前缓存若无法覆盖完整的成分股行情，脚本会在构造固定池时失败，不会静默换股。允许停牌日沿用上次报价时，运行命令中的 `--allow-suspension-carry` 已由脚本默认开启，具体冻结记录写入面板元数据。行情收益是 `close/pre_close-1` 的价格相对收益，不冒充含分红总收益；若要使用总收益，应先提供经过审计的总收益面板并另行接入。

## E6：动态参考点与处置效应对齐

优先读取 `behavior/processing_process.zip` 中已完成除权除息处理的 `6purchase/xrxddisposition_*.pkl`。这些文件仍保留单位 `(投资者, 股票, 调仓事件)`、调仓前后权重和 `stock_pool_beforexrxdpurchase`，因此不需要再次读取约780万行原始Excel。只有处理包不存在时才回退到原始调仓记录。只有股票在调仓前确实持有，且目标权重下降时才构造减持标签；标签不使用原始汇总表中的 PGR/PLR。静态买入成本直接来自处理文件，动态参考点在事件序列上递推，且只使用当前事件前的信息。四个预测模型共享控制变量，只替换参考点表示：无参考点、静态买入成本、对称适应、非对称适应。按时间划分训练、验证、测试，按投资者聚类自助法给出置信区间。

主要指标是 log loss、Brier、AUROC/AUPRC、校准斜率，以及统一标签下的 PGR、PLR 和 PGR-PLR。只有当非对称动态模型在测试集校准与 PGR/PLR 误差上同时优于静态、对称和无参考点模型时，才能说它与已有行为数据更一致；这不是因果识别，也不是对真实用户满意度的测量。

```powershell
python scripts/run_external_validity.py --experiment 6 `
  --behavior-root behavior --output results/e6_behavior_alignment `
  --max-opportunities 0 --bootstrap 200 --seed 2026
```

首次运行直接读取处理压缩包并生成 `opportunities.parquet`，不再重复解析原始Excel。若处理包缺失，脚本才会回退到原始工作簿。可先用 `--max-opportunities 10000` 做入口冒烟测试；正式结果必须用 `0`，并检查 `opportunities.json` 和 `model/manifest.json`。

## E7：异质性投资者模拟交互

用行为汇总表、投资者画像、持仓股票数和关注网络的静态统计聚类出若干投资者类型。先在历史留出段检查聚类能否复现持仓数、交易频率和 PGR/PLR 的分布，再把每个类型放入“市场观测--策略推荐--投资者效用--是否采纳--下一期市场”的模拟循环。满意度是明确给定的个体 CPT 型效用减去换手和推荐风险不匹配惩罚；采纳是该效用规则生成的模拟概率。因此输出用于比较不同方法对不同类型的适配性，不得写成真实用户满意度或真实采纳率。

```powershell
python scripts/run_external_validity.py --experiment 7 `
  --behavior-root behavior --e5-daily results/e5_hs300_real/daily.csv `
  --output results/e7_heterogeneous_agents --clusters 4 --seed 2026
```

每个目录都包含 `manifest.json`、原始数据范围、模型输出和图。审计时先检查 manifest，再检查源数据哈希和时间切分，最后解释经济指标；不要把 E5 的 Sharpe、E6 的行为预测准确率或 E7 的模拟满意度与 E1--E4 的理论 DLR 数值混排。
