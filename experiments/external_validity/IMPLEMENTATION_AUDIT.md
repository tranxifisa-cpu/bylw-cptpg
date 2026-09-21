# E5--E7 实现核对

| 实验 | 输入 | 中间过程 | 输出 | 可支持的结论 | 不支持的结论 |
|---|---|---|---|---|---|
| E5 真实市场 | Tushare 行情、日基本面、公告后 ROE/ROA/ROIC、HS300固定资产池、等权基线 | 测试日前固定资产池；因子严格滞后；学习方法使用 centered-LOO + independent randomized correction 的 hybrid 估计器；统一交易成本 | `daily.csv`、`summary.csv`、`quality_coverage.csv`、财富/回撤/气泡/换手成本四联图 | 真实行情下的财富、风险、回撤、换手和成本差异 | 不能用真实市场绩效证明 DLR 定理或因果优越性 |
| E6 行为对齐 | `processing_process.zip` 中的 `6purchase` 事件级持仓与调仓结果、除权后买入成本、滞后控制变量 | 对同一批实际减持决策使用相同控制、模型和时间切分，只替换无/静态/对称/非对称参考点表示；按各参考点重新划分盈利亏损状态；按投资者配对 bootstrap；按训练期减持率和活跃度分组 | `metrics.csv`、`reference_behavior_summary.csv`、两类 bootstrap 输出、`subgroup_metrics.csv`、A--D行为对齐图 | 哪种参考点对实际减持决策具有更高样本外解释力，以及其划分下是否呈现盈利减持率高于亏损减持率 | 不能宣称因果识别、真实满意度或把按静态成本定义的传统 PGR/PLR 当作机制胜负标准 |
| E7 异质性投资者 | 行为汇总、画像、持仓数、关注网络、E6训练期个体收缩参数、E5完整组合推荐 | 精确重放E5的301维权重；只用E6训练期逐投资者PGR/PLR/处置差异聚类；每个五日期初在观察收益前决定是否采纳；两条路径逐日更新参考点，期末计算采纳条件前景价值增量；总体按真实类型占比加权 | `e5_recommendations.parquet`、`agent_assignments.csv`、`agent_traces.parquet`、类型与总体汇总、E7四联图 | 不同数据校准模拟投资者类型对各方法的相对兼容性、模拟采纳与五日模拟满意度 | 不能写成真实用户采纳/满意，不能把顾问与投资者参考点速度称为同一参数 |

## 时间与信息约束

- E5 的财务指标使用 `ann_date < trade_date`，行情因子使用执行日前的历史窗口。
- E6 的当前调仓不能参与当前买入成本和参考点，除权除息只在除权日之前的持仓成本上递推。
- E7 的社交网络是静态画像；当前数据没有可靠的逐帖时间戳，因此不把它伪装成动态社交因果变量。一个 episode 内的全部五日收益只在期初采纳决策之后用于满意度与状态更新。
- 三项实验都写入 `manifest.json`，记录数据范围、时间切分、方法和解释边界。

## 正式结果状态

- E5 正式结果位于 `results/e5_hs300_e3sync_expcompatible_beta6_c10`：固定 300 只沪深 300 成分股、10 个 policy seeds、hybrid 估计器和随机 Dirichlet 执行，使用 `gamma=0.08`、`eta_gain=0.20`、`eta_loss=0.05`、`beta=6`、`c0=10`。
- E6 正式模型位于 `results/e6_reference_behavior_eta041/model`：使用完整机会样本、`eta_gain=0.40`、`eta_loss=0.10` 和 200 次投资者聚类 bootstrap。
- E7 正式结果位于 `results/e7_heterogeneous_agents`：使用正式 E5 推荐和 E6 行为校准，包含 4 类、每类 16 名模拟投资者，并按 E5 的五日 episode 评价采纳与条件满意度。
- `python -m py_compile mvp_cpt_pg/external_validity.py scripts/run_external_validity.py` 与正式入口测试均已通过。

冒烟输出不是论文结果，不应保留在 `results/` 或用于结果汇总。正式结果解释仍须服从上表中的“可支持的结论”和“不支持的结论”边界。
