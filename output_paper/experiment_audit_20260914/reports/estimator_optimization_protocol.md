# 估计器优化与实验二、三更新

## 无限层与无限样本

无限条轨迹的样本均值无法在有限时间计算。随机化无限层不同：每次只抽一个有限整数L，再生成base*2**L条内轨迹；L的分布支持所有非负整数，但每次实际成本几乎必然有限。若p_L按2**(-a*L)衰减且1<a<2，在相应层差二阶矩和可积条件下，可兼顾有限期望成本、有限方差与对模拟目标的无偏性。期望无偏不代表每次输出等于真梯度。

当前实现使用有限cap，控制单次最大样本量；它仍有截断偏差。将M增加到很大能降低平均的方差，却不会消除固定cap的偏差。不能用“代码能抽到很高的层”代替对尾部成本和方差的控制。本轮不新增未经验证的无限层执行入口。

## 验证顺序

先完成固定上下文的估计器诊断，再运行正式实验二，最后运行同一参考点机制下的实验三。参考点消融继续作为另一组实验，不和估计器贡献混合。

诊断上下文：四个起始日0、150、290、350（零基索引），分别对应上涨、震荡、冲击、风格切换；各上下文财富和参考点为1、初始全现金、theta各分量0.1/sqrt(d)。这是隔离市场阶段的固定状态检验，不声称该状态是学习算法实际经历的状态。使用h=5，所有方法在每条轨迹内部按同一规则更新参考点。

## 四种有限cap估计器

| 名称 | 构造 | 主要待检验因素 |
|---|---|---|
| plugin | 一个内样本批次，共享给预算内的所有独立外轨迹 | 强基线 |
| multilevel | 原来的随机单层，外轨迹数1 | 原始实现 |
| multilevel_outer | 随机单层，同一内批次评价q条独立外轨迹 | 外轨迹平均降低score噪声 |
| split_base | 每次单独计算基础项，再在正层级中随机抽取一项修正 | 减少基础项被随机选择的波动 |

每组全样本与两半样本使用相同外轨迹，内外独立。split_base的基础项和修正项独立生成，并计入二者成本；正层级概率重新归一化。有限cap下四者的期望都等于最高层T_n的期望，因此这一轮比较的重点是方差、MSE和成本，不是声称有限cap严格无偏。

诊断要求n=base*2**cap。预算B以完整h步轨迹计：plugin用n条内轨迹、B-n条外轨迹；多层使用floor(B/单次期望成本)次独立重复。实际成本具有随机性，CSV和图用实测平均成本，不能把预算上限理解成每次绝不超出。小预算下整数舍入可能明显少用预算，特别是split_base；正式比较使用512以上预算，并同时报告实际成本和耗时。

## 已完成的小规模验证

两个pilot目录分别为artifacts/results/paper_e2_diagnostic_pilot_20260913和paper_e2_diagnostic_base16_pilot_20260913。每个上下文60次独立重复，最高层n=64；参考值使用4批、两档精度（内4096/8192、外2048/4096）。不同基础样本配置使用相同参考值和plug-in随机流，不能将它们视作两份独立plug-in重复。

预算512时的向量MSE：

| 阶段 | plugin | 原始多层 base2 | 多外轨迹 base2,q8 | 多外轨迹 base16,q8 |
|---|---:|---:|---:|---:|
| 上涨 | 1.449e-6 | 5.138e-4 | 1.720e-4 | 1.232e-5 |
| 震荡 | 1.298e-5 | 2.033e-3 | 6.720e-4 | 9.052e-5 |
| 冲击 | 4.126e-4 | 3.617e-2 | 1.068e-2 | 2.950e-3 |
| 风格切换 | 7.701e-6 | 1.932e-3 | 5.199e-4 | 6.354e-5 |

提高基础内样本并平均外轨迹明显改善了原始构造，但尚未超过plugin。split_base在base16、q8、预算512的四个上下文均弱于multilevel_outer，不宜因为结构复杂就优先采用。上述仅为探索性结果，不自动选择正式主算法；还需更多重复和更高精度参考值确认。

每个pilot已输出diagnostic_raw.csv、diagnostic_summary.csv、diagnostic_references.csv、level_raw.csv、level_summary.csv，以及四个阶段各一张四面板PDF/SVG/PNG。面板是MSE-成本、方差-成本、第一坐标带Monte Carlo区间的偏差、层修正二阶矩。所有向量坐标保留，可重新计算其他统计量，不以单一坐标代替整体MSE。

## 正式实验二

以下两条分别保持原始基础样本和提高基础样本，最高层均64。可顺序执行；自动出图。没有在本轮运行正式命令。

```powershell
python scripts/run_paper_experiments.py --experiment 2 --estimator-diagnostics --output artifacts/results/paper_e2_base2_confirm --panel artifacts/inputs/paper_hs300.csv --n 64 --base 2 --cap 5 --outer-batch 8 --replications 500 --diagnostic-days 0 150 290 350 --budget-grid 512 2048 --reference-batches 8 --reference-n 16384 --reference-m 8192 --horizon 5
```

```powershell
python scripts/run_paper_experiments.py --experiment 2 --estimator-diagnostics --output artifacts/results/paper_e2_base16_confirm --panel artifacts/inputs/paper_hs300.csv --n 64 --base 16 --cap 2 --outer-batch 8 --replications 500 --diagnostic-days 0 150 290 350 --budget-grid 512 2048 --reference-batches 8 --reference-n 16384 --reference-m 8192 --horizon 5
```

旧实验二（不加--estimator-diagnostics）仍提供维度、M与QQ检验。新诊断固定d=10，不使用旧命令的dimensions、n-grid和m-grid。偏差区间包含参考值Monte Carlo方差，但不能消除参考值自身有限n偏差；保留两档参考值是为了判断精度是否足够。若MSE接近参考不确定性，先提高参考精度。

## 改进实验三

新增--estimator-comparison：同一非对称参考点规则，比较指定估计器，并自动加入frozen。训练依然sample、执行mean，参数仍在每个时期结束后更新。frozen不计算训练梯度，训练轨迹成本为0，保留独立评价。

每个方法的自有财富/参考点路径都会分化，因此新增共同状态probe：对每个时期，令财富=参考点=1、上一期持仓全现金，市场起始日相同，以各方法更新前的theta评价同一个模拟目标。共同CPT的大小可比较；它描述在标准化共同状态下学到的策略质量，并不是各方法整条财富路径的总CPT。原来自有路径上的Q/DLR仍保存为辅助诊断，不据此做跨目标的直接排名。

共同probe保存两次独立梯度g1、g2：g1·g2在条件期望上估计有限样本梯度均值的平方范数；0.5*||g1-g2||²诊断方差迹。交叉乘积可能为负，原样保存，不裁到0；它不是精确总体残差，也没有消除有限内样本偏差。每个向量坐标和额外评价成本都记入episodes.csv。

待实验二确认后，先执行以下同构造比较。所有学习组具有相同的预计训练预算，基础样本16、cap2、n64；此处原始单外轨迹多层也使用base16，不能和旧base2直接混称相同配置。

```powershell
python scripts/run_paper_experiments.py --experiment 3 --estimator-comparison --compare-estimators plugin multilevel multilevel_outer split_base --output artifacts/results/paper_e3_estimator_compare_base16 --panel artifacts/inputs/paper_hs300.csv --seeds 29 147 3141 42 3407 592 7 101 2024 2025 --steps 500 --horizon 5 --gamma 0.05 --eta-gain 0.2 --eta-loss 0.05 --n 64 --base 16 --cap 2 --outer-batch 8 --trajectory-budget 512 --evaluation-n 8192 --evaluation-m 4096
```

新增主图experiment3_common_context包括共同CPT、独立梯度交叉乘积、噪声和实际执行财富。原来的自有路径跟踪图和逐方法图同时输出。标准化probe最初各组theta相同，评价应一致；后续差异反映参数学习差异。公共评价种子用于配对比较，不能把方法之间结果当独立样本。

实验三小规模流程测试保存在artifacts/results/paper_e3_estimator_smoke_20260913，只运行1个seed、2个时期，用于验证各构造、frozen、预算字段及出图，不用于报告性能。18项测试通过，包含离散双点分布可枚举的有限层期望检验。

本轮只改估计器和实验二/三入口，没有改真实执行环境。之前发现的实验五停牌掩码、持仓漂移和费用口径问题仍需另行修复。

## 正式实验二完成后的结论与实验三命令

2026-09-13核验paper_e2_base2_confirm和paper_e2_base16_confirm：两组均complete，各32个配置单元、每单元500次重复，共16000条原始估计记录。主键无重复、梯度坐标有限，原始向量重算MSE与summary的最大绝对差低于1e-16。七个记录的源码哈希均与核验时文件一致。两组复用相同的plugin样本及参考值，不合并成1000个独立重复。

基础16、最高内样本64、外平均8条，预算2048时：

| 阶段 | plugin MSE | 单外轨迹多层MSE | 多外轨迹多层MSE | 基础项分离MSE | 多外轨迹/plugin |
|---|---:|---:|---:|---:|---:|
| 上涨 | 4.138e-7 | 1.886e-5 | 3.187e-6 | 4.575e-6 | 7.70 |
| 震荡 | 3.172e-6 | 1.365e-4 | 2.085e-5 | 3.462e-5 | 6.57 |
| 冲击 | 1.097e-4 | 4.904e-3 | 7.968e-4 | 1.250e-3 | 7.26 |
| 风格切换 | 2.170e-6 | 9.587e-5 | 1.579e-5 | 2.375e-5 | 7.27 |

多外轨迹MSE的Monte Carlo标准误依次为1.012e-7、6.150e-7、2.273e-5、5.297e-7，plugin对应为1.849e-8、1.195e-7、3.914e-6、8.074e-8；差距远大于这些标准误的量级。在基础16的两个预算、四个阶段共8个配置中，多外轨迹优于单外轨迹及基础项分离，但全部弱于plugin。改进多层在同配置下相对单外轨迹的MSE下降约83%至85%；不能写成超过plugin。

预算由512增至2048，多外轨迹MSE下降3.77至4.06倍，与方差主导的约1/B规律一致。两个预算点只支持局部经验趋势，不能作为渐近速率的独立证明。参考均值方差迹占plugin MSE最多约1.83%，不足以解释多层与plugin的倍数差距；但冲击阶段参考梯度范数0.00234、参考均值标准误范数量级0.00142，不能用当前参考值精确确定很小的偏差或梯度方向。

下一步停止扩展估计器参数搜索，先做在线效用检验。实验三保留plugin、单外轨迹多层、多外轨迹多层及自动加入的frozen。基础项分离保留在实验二作为已测试的改进方向，暂不增加实验三计算量。下列命令取代上面四种构造的初始计划；统一预算提高至2048，减少低预算噪声。单外轨迹多层同样使用base16，它代表原始结构在新基础样本下的版本，而不是旧base2数值的复现。

```powershell
python scripts/run_paper_experiments.py --experiment 3 --estimator-comparison --compare-estimators plugin multilevel multilevel_outer --output artifacts/results/paper_e3_estimator_compare_confirm --panel artifacts/inputs/paper_hs300.csv --seeds 29 147 3141 42 3407 592 7 101 2024 2025 --steps 500 --horizon 5 --gamma 0.05 --eta-gain 0.2 --eta-loss 0.05 --n 64 --base 16 --cap 2 --outer-batch 8 --trajectory-budget 2048 --evaluation-n 8192 --evaluation-m 4096
```

本命令已运行完成。2026-09-14核验记录见`docs/paper_e3_confirmation_analysis_20260914.md`：10个seed完整；plugin和多外轨迹多层相对frozen的终值收益平均分别增加0.2263和0.2136个百分点，但二者之间没有清晰差异。共同CPT仅有小幅改善，自有路径跟踪指标与frozen接近，尚不足以证明驻点跟踪优势。下一步优先复用保存参数做高精度共同状态配对评价，不用财富排名反向挑选参考点速度，也不将多层估计器的理论性质写成已验证的有限预算性能优势。
