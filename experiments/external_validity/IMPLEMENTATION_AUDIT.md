# E5--E7 实现核对

| 实验 | 输入 | 中间过程 | 输出 | 可支持的结论 | 不支持的结论 |
|---|---|---|---|---|---|
| E5 真实市场 | Tushare 行情、日基本面、公告后 ROE/ROA/ROIC、HS300 权重 | 测试日前固定资产池；因子严格滞后；五个学习方法逐 episode 更新；三类经济基线独立执行；统一交易成本 | `daily.csv`、`summary.csv`、`quality_coverage.csv`、E5 四联图 | 真实行情下的财富、风险、回撤、换手和成本差异 | 不能用真实市场绩效证明 DLR 定理或因果优越性 |
| E6 行为对齐 | `processing_process.zip` 中的 `6purchase` 事件级处理结果、画像控制变量 | 直接使用除权后的调前成本；重建持有中的减持机会；标签是目标权重下降；四种参考点模型按时间切分；按投资者bootstrap | `opportunities.parquet`、`metrics.csv`、`pgr_plr_summary.csv`、校准图 | 非对称动态参考点是否更能预测既有行为数据中的减持/PGR/PLR | 不能宣称识别了因果处置效应或测量了真实满意度 |
| E7 异质性投资者 | 行为汇总、画像、持仓数、关注网络、E5 方法推荐 | 训练期聚类；留出期校验类型分布；方法推荐进入显式 CPT 型效用和采纳概率 | `agent_type_profiles.csv`、`agent_method_summary.csv`、E7 四联图 | 不同模拟投资者类型对方法的相对适配性和稳健性 | 不能把模拟采纳率写成真实用户采纳率 |

## 时间与信息约束

- E5 的财务指标使用 `ann_date < trade_date`，行情因子使用执行日前的历史窗口。
- E6 的当前调仓不能参与当前买入成本和参考点，除权除息只在除权日之前的持仓成本上递推。
- E7 的社交网络是静态画像；当前数据没有可靠的逐帖时间戳，因此不把它伪装成动态社交因果变量。
- 三项实验都写入 `manifest.json`，记录数据范围、时间切分、方法和解释边界。

## 已完成的验证

- `python -m py_compile mvp_cpt_pg/external_validity.py scripts/run_external_validity.py` 通过。
- E5 使用现有30资产面板和单seed小样本运行通过，学习方法、三个基线、交易成本、summary和图均生成。
- E6 优先使用处理压缩包中的100条机会和10次cluster bootstrap运行通过；正式结果必须使用 `--max-opportunities 0`，不会重新读取原始Excel。
- E7 使用E5冒烟输出和3类模拟投资者运行通过。
- `git diff --check` 通过。

冒烟结果不用于论文结论，只用于验证入口、字段和时间切分；正式结论必须基于正式参数重新运行并由结果审计。
