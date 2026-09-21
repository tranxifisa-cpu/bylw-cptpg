# DRCPT-PG 论文实验仓库

本仓库当前只以 E1--E7 证据链为论文主线。历史 MVP、旧 multilevel 实验和旧中英文论文仍可能保留用于追溯，但不属于当前复现入口。

## 当前方法口径

- 策略：指数参数化 Dirichlet factor policy，`alpha_i = c0 * exp(beta * q_i^T theta)`。
- 动作：学习轨迹和真实执行路径使用同一个随机 Dirichlet actor；不存在“训练 sample、执行 mean”的主线口径。
- CPT 梯度估计器：centered LOO base + independent raw randomized correction，代码中统一称为 `hybrid`。
- 参考点：DRCPT-PG 使用非对称动态参考点；对照包括对称动态参考点和静态参考点。
- 组合：现金加风险资产的权重和为 1，交易通过统一的惯性动作映射和交易成本模型执行。

Hybrid 估计器直接用于 E2--E5。E1 是机制实验，不估计策略梯度；E6 是真实投资者行为预测；E7 重放 E5 推荐并模拟异质投资者交互，不重新训练策略。

## E1--E7 证据链

| 实验 | 问题 | 当前入口 |
| --- | --- | --- |
| E1 | 路径依赖参考点是否改变组合评价与选择 | `experiments/estimator_final/e1_e4_final/build_e1_final.py` |
| E2 | Hybrid CPT 梯度估计器的偏差、方差、正态性和预算标度 | `experiments/estimator_final/e1_e4_final/run_e2_final.py` |
| E3 | 连续慢变环境中的一阶驻点跟踪和 DLR | `experiments/estimator_final/e1_e4_final/run_e3_slow_continuous.py` |
| E4 | 参考点速度、时域和优化参数的稳健性 | `experiments/estimator_final/e1_e4_final/run_e4_dense_incremental.py` |
| E5 | 真实沪深 300 市场中的经济外部有效性 | `scripts/run_external_validity.py --experiment 5` |
| E6 | 不同参考点对真实处置行为的样本外解释力 | `scripts/run_external_validity.py --experiment 6` |
| E7 | 数据校准异质投资者的模拟采纳与满意度 | `scripts/run_external_validity.py --experiment 7` |

E1--E4 的冻结实现、证明说明和复现资产位于 `experiments/estimator_final/`。E5--E7 的运行合同位于 `experiments/external_validity/README.md`。

## 正式结果

- E5：`results/e5_hs300_e3sync_expcompatible_beta6_c10`
- E6：`results/e6_reference_behavior_eta041`
- E7：`results/e7_heterogeneous_agents`

正式 E5 使用 300 只固定沪深 300 成分股、10 个 policy seeds、`h=5`、`gamma=0.08`、`eta_gain=0.20`、`eta_loss=0.05`、`beta=6`、`c0=10` 和 hybrid 估计器。E6 使用 `eta_gain=0.40`、`eta_loss=0.10`；E7 将 E5 的顾问参数与 E6 的投资者行为参数分开记录。

## 当前论文

当前英文源文件位于 `output_paper/DRCPT_PG_Full_Paper_E1_E4/`。其理论、算法与 E1--E7 已和 hybrid 主线对齐；E5--E7 正文使用上述正式结果，并保持市场、行为与交互三类外部有效性结论和 E1--E4 理论证据分层解释。

以下目录不是当前论文入口：

- `output_paper/DRCPT_PG_v2_project/`：导师早期 multilevel 草稿。
- `output_paper/应统论文/`、`output_paper/应统作文模板/`：保留的旧中文稿和模板。
- `experiments/estimator_final/source/vendor/`：冻结依赖快照，只用于 E1--E4 复现与溯源。

不要使用旧目录中的默认参数或命令生成论文结果。

## 数据与环境

E5 使用 `artifacts/inputs/paper_hs300_full.csv` 和 `artifacts/cache/raw/tushare`。Tushare 拉取需要环境变量 `TUSHARE_TOKEN` 或 `Tushare_Token`。E6--E7 使用 `behavior/` 中的处理后投资者行为数据。

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

正式运行前应确认目标输出目录不存在或为空，并在完成后检查每个结果目录中的 `manifest.json`。
