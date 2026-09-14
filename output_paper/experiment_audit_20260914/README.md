# 论文实验统一审核包

整理日期：2026-09-14。先读本页，再查reports中的详细报告，最后回到results中的原始CSV与manifest。所有链接相对于本目录，可整体移动给审核者。原结果没有改写，图没有重新训练或重绘。

## 一、当前能得出的结论

| 证据 | 观测结论 | 不能推出 |
|---|---|---|
| 路径递推 | 指定两条同终值财富路径产生参考点1.104与1.032，差0.072 | 非对称机制必要；真实人类处置效应得到验证 |
| 首轮估计器 | 固定cap平均的方差随重复数约按1/M下降；当前配置plug-in MSE更低 | 无限层无偏已由有限实验认证；多层有限预算胜出 |
| 正式估计器确认 | base16、多外轨迹8在预算2048时，MSE约为plug-in的6.6至7.7倍；较同base单外轨迹下降83%至85% | 超过plug-in；理论速率意味着更小有限样本常数 |
| 最新同机制在线比较 | plugin与多外轨迹相对frozen总收益平均增加0.226和0.214个百分点 | 多层优于plugin；非对称优于对称/静态 |
| 跟踪诊断 | frozen的平均DLR也约3.5e-5，与学习组接近 | 已证明有效驻点跟踪或渐近次线性 |
| 首轮真实回放 | 有完整运行图表，但成本/持仓漂移与停牌口径待核 | 已获得真实可执行投资优势 |

最新在线比较的平均终值财富：frozen 0.978956、plugin 0.981220、单外轨迹多层0.981299、多外轨迹多层0.981092。均低于1。三种学习方法之间没有明确的配对优劣证据。共同CPT略有改善，但多重比较后不足以支持强显著性结论，详见下述报告。

## 二、阅读顺序与对应图

| 顺序/新计划归属 | 报告或原始结果 | 首选图 |
|---|---|---|
| 1. 当前五组计划 | [审核版实验计划](reports/EXPERIMENT_PLAN.md) | 计划，不是结果 |
| 2. E1及首轮总体结果 | [首轮分析](reports/paper_results_analysis_20260913.md) | [路径依赖](results/paper_e1_path/experiment1_path_dependence.png) |
| 3. E2正式估计器确认 | [估计器协议及正式结果表](reports/estimator_optimization_protocol.md) | [上涨](results/paper_e2_base16_confirm/experiment2_diagnostics_day0.png)、[震荡](results/paper_e2_base16_confirm/experiment2_diagnostics_day150.png)、[冲击](results/paper_e2_base16_confirm/experiment2_diagnostics_day290.png)、[风格切换](results/paper_e2_base16_confirm/experiment2_diagnostics_day350.png) |
| 4. E2在线补充，旧称实验三 | [最新在线分析](reports/paper_e3_confirmation_analysis_20260914.md) | [共同状态比较](results/paper_e3_estimator_compare_confirm/experiment3_common_context.png)、[自有路径跟踪](results/paper_e3_estimator_compare_confirm/experiment3_tracking_merged.png) |
| 5. 历史多参考点比较 | 首轮分析第1、2、4节 | [旧半合成跟踪](results/paper_e3_tracking/experiment3_tracking_merged.png) |
| 6. 参数验证 | [验证排序原表](results/paper_e4_validation/validation_ranking.csv) | [消融与敏感性](results/paper_e4_validation/experiment4_ablation_sensitivity.png) |
| 7. 真实回放，存在限制 | 首轮分析第4节 | 到[真实回放目录](results/paper_e5_real_proxy/)查看图与summary |

每张PNG所在目录同时保留原PDF/SVG（原运行实际提供的格式），原始CSV及运行manifest，不仅打包截图。图中的聚合常用中位数/IQR；报告表使用均值/样本标准差时已有注明，不能直接对照成同一种统计量。

## 三、数据追溯与版本

- `results/`完整保留11个paper开头运行目录，包括正式、旧版、pilot和smoke；没有把它们合并为新增独立重复。
- `paper_e2_diagnostic*_pilot*`属于探索性调参；`paper_e3_estimator_smoke*`仅验证流程，不纳入性能结论。
- `paper_e2_base2_confirm`与`paper_e2_base16_confirm`是不同多层配置，同一批plug-in和参考数据有复用，不合并成1000个独立重复。
- `paper_e3_estimator_compare_confirm`实际是同一非对称动态参考点规则下的估计器比较，frozen冻结theta而非参考点。按照新计划归入E2在线补充，不是新E3已经完成。
- `paper_e3_tracking`是旧多目标比较；自有目标CPT/DLR不能跨方法直接排名。原始图保留是为了审计，不代表认可该排名解释。
- `reports/`为整理时文档快照；历史报告的“当前源码一致”和“下一步尚未运行”均有时间背景，以新报告和新计划为准。
- `source_snapshot/`为整理时当前七个相关源文件，不是所有旧运行的历史源码。各结果manifest保留当时哈希；发现不一致只表示版本不同，不能用当前源码冒充旧版精确复现。
- `inputs/`保存本次复制的paper_hs300输入及同名前缀元数据，不包含整个Tushare缓存和behavior隐私数据。核验原始数据抓取与清洗还需要仓库数据管线及源缓存。

## 四、建议审核顺序

1. 根据`manifest.json`确认目标、参考点、策略、预算、seed和执行口径；不要仅看文件夹编号。
2. 从summary/原始向量核对报告数字；最新在线实验按seed配对，不把每期当独立样本。
3. E2核对参考精度、有限cap偏差、成本计算以及两次配置共享数据；保留plug-in更好的负结果。
4. E3核对Q残差定义、独立评价噪声与frozen；不能用训练梯度自身或平滑抵消证明真实驻点距离。
5. 检查真实回放限制、参数选择和留出是否合规，再决定哪些图能进主文。

`file_manifest.csv`列出每个复制文件的原始绝对路径、包内路径、字节数、SHA256及复制一致性。它认证复制未变化，不认证所有数学结论、程序实现或历史数据正确。README与清单为本次新增索引，不属于原运行文件。

## 五、接下来尚未完成

共同状态决策扩展、真正可辨识的跟踪正/负对照、固定完整目标与递增精度诊断、真实执行口径修复、行为字段与预测任务审核，均见新计划。没有真实投资者心理对齐、满意或采纳的已完成结果，包内不会预设这些结论。
