下面按我在本地核对到的文件和脚本整理。依据主要来自 `raw_data`、`processing_process.zip` 样本文件、`1preprocessing.py` 到 `15afterholiday.py`、`final_data`。

**1. 原始数据结构**

核心原始表是调仓明细、股票行情、用户关系、帖子、用户画像、除权除息、沪深 300。

| 数据             | 文件                                                         | 关键字段                           | 含义                                       |
| ---------------- | ------------------------------------------------------------ | ---------------------------------- | ------------------------------------------ |
| 调仓明细         | `raw_data/trading_data/tm_trading1.xlsx` 到 `tm_trading8.xlsx`，`tm_trading_plus.xlsx` | `sp`                               | 组合 ID                                    |
|                  |                                                              | `category`                         | 记录类型，样本中为 `user_rebalancing`      |
|                  |                                                              | `status`                           | 调仓状态，样本中为 `success`               |
|                  |                                                              | `updated_at_content`               | 调仓内容更新时间                           |
|                  |                                                              | `id`                               | 调仓记录 ID                                |
|                  |                                                              | `market`                           | 市场标识                                   |
|                  |                                                              | `prev_weight_adjusted`             | 调仓前权重                                 |
|                  |                                                              | `target_weight`                    | 调仓后目标权重                             |
|                  |                                                              | `price`                            | 调仓成交或记录价格                         |
|                  |                                                              | `proactive`                        | 是否主动调仓                               |
|                  |                                                              | `stock_code`                       | 股票代码，纯数字                           |
|                  |                                                              | `stock_symbol`                     | 带交易所前缀的股票代码，如 `SH600690`      |
|                  |                                                              | `stock_name`                       | 股票名称                                   |
|                  |                                                              | `stock_label`                      | 股票标签，如新股                           |
|                  |                                                              | `updated_at_rebalancing_histories` | 调仓历史更新时间                           |
| 股票日行情       | `raw_data/stock_price.zip`                                   | `Stkcd`                            | 股票代码                                   |
|                  |                                                              | `Trddt`                            | 交易日                                     |
|                  |                                                              | `Opnprc`                           | 开盘价                                     |
|                  |                                                              | `Hiprc`                            | 最高价                                     |
|                  |                                                              | `Loprc`                            | 最低价                                     |
|                  |                                                              | `Clsprc`                           | 收盘价                                     |
|                  |                                                              | `ret`                              | 股票收益率                                 |
| 组合和主理人映射 | `tm_user_sp_all.xlsx`                                        | `sp`                               | 组合 ID                                    |
|                  |                                                              | `user`                             | 主理人用户 ID                              |
| 主理人关注关系   | `tm_zhulirenguanzhu.pkl`                                     | key                                | 主理人用户 ID                              |
|                  |                                                              | value                              | 该主理人关注的用户 ID 列表                 |
| 主理人画像       | `zlr_profile.xlsx`                                           | `user_id`                          | 主理人用户 ID                              |
|                  |                                                              | `guanzhu`                          | 关注人数                                   |
|                  |                                                              | `fans`                             | 粉丝数                                     |
|                  |                                                              | `status_count`                     | 发帖数                                     |
|                  |                                                              | `ip`                               | IP 属地                                    |
|                  |                                                              | `province`                         | 省份                                       |
|                  |                                                              | `city`                             | 城市                                       |
|                  |                                                              | `gender`                           | 性别，`m/f/n` 等                           |
| 关注人画像       | `gzr_profile.xlsx`                                           | 同上                               | 用来识别关注人中的意见领袖                 |
| 主理人股票数     | `zhulirenstocks_count.xlsx`                                  | `user_id`                          | 主理人用户 ID                              |
|                  |                                                              | `stocks_count`                     | 主理人涉及或关注的股票数量                 |
| 关注人帖子       | `processing_process.zip/posts1` 和 `posts2`                  | `p_id`                             | 帖子 ID                                    |
|                  |                                                              | `user_id`                          | 发帖用户 ID                                |
|                  |                                                              | `dtime`                            | 发帖时间                                   |
| 帖子情绪         | `post_sent.pkl`                                              | 外层 key                           | 用户 ID                                    |
|                  |                                                              | 内层 key                           | 帖子 ID                                    |
|                  |                                                              | 内层 value                         | 情绪分数，正数为正向，负数为负向，0 为中性 |
| 除权除息         | `xrxd1.xls`，`xrxd2.xls`                                     | `股票代码_StkCd`                   | 股票代码                                   |
|                  |                                                              | `除权除息日_ExDt`                  | 除权除息日                                 |
|                  |                                                              | `现金红利(元)_Dividend`            | 每股现金红利                               |
|                  |                                                              | `送股比例()_StkDrate`              | 送股比例                                   |
|                  |                                                              | `转增比例()_CapIssurate`           | 转增比例                                   |
| 沪深 300         | `CSI300.xls`                                                 | `交易日期_TrdDt`                   | 指数交易日                                 |
|                  |                                                              | `收盘价(元/点)_ClPr`               | 沪深 300 收盘价                            |
|                  |                                                              | `ret`                              | 原始收益字段，脚本中重新计算对数收益       |
| 退市表           | `stock_delisting_date.xlsx`                                  | `Stkcd`                            | 退市股票代码                               |
|                  |                                                              | `Suspdate`                         | 停牌或退市起始日期                         |
|                  |                                                              | `Reason`                           | 原因                                       |
| 清洗名单         | `sps.xlsx`，`sps_cleaning.xlsx`，`removing_delisted_wrong100.xlsx` | `sp/sps/wrong`                     | 组合全集、清洗后组合、额外排除组合         |

**2. 最终主回归数据结构**

我把“最终主回归数据”理解为 `final_data/benchmark_regression_sent_group_stata.xlsx`，它是 `benchmark_regression_sent.pkl` 进一步删去明细字段后给 Stata 或回归使用的表。

字段如下：

| 字段                            | 含义                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| `day`                           | 观测日，即组合发生卖出行为的调仓日                           |
| `zhuliren`                      | 主理人用户 ID                                                |
| `RG_xrxdpurchase`               | 实现收益股票数，卖出价高于除权除息调整后买入成本             |
| `RL_xrxdpurchase`               | 实现损失股票数，卖出价低于调整后买入成本                     |
| `PG_xrxdpurchase`               | 纸面收益股票数，未卖出持仓中成本低于当日价格                 |
| `PL_xrxdpurchase`               | 纸面损失股票数，未卖出持仓中成本高于当日价格                 |
| `PGR`                           | 实现收益比例，`RG / (RG + PG)`                               |
| `PLR`                           | 实现损失比例，`RL / (RL + PL)`                               |
| `disposition`                   | 处置效应指标，`PGR - PLR`                                    |
| `mktret`                        | 观测日前 3 天沪深 300 对数收益之和                           |
| `stknum`                        | 观测日前 1 到 3 天平均持仓股票数量                           |
| `stkprice`                      | 观测日前 1 到 3 天平均持仓股票价格                           |
| `transnum`                      | 观测日前 3 天组合调仓次数                                    |
| `pos`                           | 匹配到的正向情绪帖子数                                       |
| `neg`                           | 匹配到的负向情绪帖子数                                       |
| `neu`                           | 匹配到的中性帖子数                                           |
| `fans_dv`                       | 主理人粉丝数是否高于中位数，1 为高于，0 为低于或等于         |
| `tiezi_dv`                      | 主理人发帖数是否高于中位数                                   |
| `stocks_dv`                     | 主理人股票数量是否高于中位数                                 |
| `province_city_dv`              | 地区分组，1 为北上广深，0 为其他常规省市，2 为无法归入常规省市 |
| `gender_dv`                     | 性别分组，男为 1，女为 0，其他或缺失为空                     |
| `opinion_leaders_proportion_dv` | 主理人关注人中意见领袖占比是否高于中位数                     |
| `num_posts`                     | 匹配帖子数量                                                 |

上游完整主表 `benchmark_regression_sent.pkl` 还有这些明细字段：`sp`、`zhulirenguanzhuren`、`guanzhurenzhutie`、`stock_pool`、`stock_pool_closeprice`、`stock_pool_maxprice`、`stock_pool_minprice`、`stock_pool_beforexrxdpurchase`、`stock_pool_afterxrxdpurchase`、`sent`、用户画像原始字段等。最终回归表删掉这些字段，只保留可直接回归的变量。

**3. 从原始数据到最终主回归数据的步骤**

1. 合并调仓原始表
   脚本：`1preprocessing.py`
   合并 `tm_trading1.xlsx` 到 `tm_trading8.xlsx` 和 `tm_trading_plus.xlsx`，去重，按 `sp` 拆分。新增 `day`，由 `updated_at_content` 取日期。修正少数股票代码，如 `SH601313` 改为 `SH601360`。删除新股记录，删除每只股票进入真实持仓前的无效记录，处理部分组合的缺失权重。
2. 筛选有效组合
   脚本：`2transaction_data_cleaning.py`
   排除权重异常、缺少画像、买卖方向不完整、股票代码异常、记录数过少、前后权重衔接异常的组合。输出 `sps_cleaning.xlsx`，从 11661 个组合筛到 4394 个组合。
3. 匹配主理人、关注人和帖子
   脚本：`3matching_posts.py`
   用 `tm_user_sp_all.xlsx` 找到每个组合的主理人，用 `tm_zhulirenguanzhu.pkl` 找到主理人的关注人。对每个调仓日，提取关注人在调仓日前 3 天到调仓日之间的帖子 ID，写入 `guanzhurenzhutie`。同时保存主理人关注人列表 `zhulirenguanzhuren`。
4. 构造每日股票池
   脚本：`4stock_pool.py`
   按调仓记录顺序维护 `stock_pool`。`target_weight > 0` 时把股票加入持仓池，`target_weight == 0` 时从持仓池移除。
5. 删除退市股票
   脚本：`5remove_the_delisted_stocks_from_the_stock_pool.py`
   用 `stock_delisting_date.xlsx`，在退市或停牌日期之后，从对应组合的 `stock_pool` 中移除该股票。
6. 计算除权除息调整后买入成本和处置效应基础变量
   脚本：`6purchase.py`
   读取每只股票的日行情 `close_股票代码.pkl`，给股票池补充当日收盘价、最高价、最低价。
   用 `xrxd1.xls` 和 `xrxd2.xls` 调整历史买入成本，公式逻辑是现金红利减少成本，送股和转增摊薄成本。
   对每个有卖出行为的交易日计算：
   `RG`、`RL`、`PG`、`PL`。
7. 生成组合日层面的基础回归表
   脚本：`7zhg.py`
   只保留当天存在卖出行为的记录。删除单笔调仓字段，保留组合日层级字段。计算：
   `PGR = RG / (RG + PG)`，`PLR = RL / (RL + PL)`，`disposition = PGR - PLR`。
   合并主理人画像、股票数、沪深 300 前 3 天收益、前 3 天平均持仓数量、平均持仓价格、调仓次数。
8. 合并所有组合并限制样本日期
   脚本：`8benchmark_regression.py`
   合并所有 `zhg_*.pkl`。保留 `day <= 2023-05-31` 的样本，删除 `guanzhuren_count == 0` 的主理人，输出 `benchmark_regression.pkl`。
9. 加入帖子情绪
   脚本：`10benchmark_regression_sent.py`
   用 `post_sent.pkl` 按 `guanzhurenzhutie` 找到每条匹配帖子的情绪分数。统计正向、负向、中性帖子数，得到 `pos`、`neg`、`neu`，输出 `benchmark_regression_sent.pkl`。
10. 构造异质性和分组变量，导出最终主回归表
       脚本：`11heterogeneity.py`
       根据主理人粉丝数、发帖数、股票数的中位数生成 `fans_dv`、`tiezi_dv`、`stocks_dv`。
       根据省市生成 `province_city_dv`，根据性别生成 `gender_dv`。
       用关注人画像中粉丝数的 99 分位识别意见领袖，再生成 `opinion_leaders_proportion_dv`。
       删除明细字典字段和用户画像原始字段，输出 `benchmark_regression_sent_group_stata.xlsx`，这就是最终主回归数据。

有一个需要注意的前提：帖子原始抓取和情绪分数生成过程在当前脚本里没有完整展开，当前项目把它们作为已有输入，即 `processing_process.zip/posts1`、`posts2` 和 `post_sent.pkl`。