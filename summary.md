# FinGPT Part 2 项目概述

本文是当前仓库实现的中文开发摘要，对应英文版 [DEVELOPMENT.md](/C:/Project/FinGPT/FinGPT_Part2/DEVELOPMENT.md)。内容以当前代码与 `output/` 中已生成的结果为准，重点说明真实工作流、Dashboard，以及当前回测表现。

## 1. 项目目标

这个项目实现了一条本地运行的两阶段金融新闻信号流水线：

1. `Agent 1` 读取新闻文本，生成结构化 `NewsFingerprint`
2. `Agent 2` 基于 fingerprint 生成交易方向 `TradingSignal`
3. `backtest` 模块把同样逻辑跑在历史数据集上，并计算收益与诊断指标

核心设计不是让模型“自己报分”，而是直接读取 vLLM `prompt_logprobs` 中的真实 token 对数概率，再由 Python 做确定性后处理、校准和过滤。

## 2. 当前代码结构

- [pipeline.py](/C:/Project/FinGPT/FinGPT_Part2/pipeline.py)：实时抓取新闻并串联 Agent 1 / Agent 2
- [agent1/extractor.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/extractor.py)：抽取、情绪打分、事件类型打分
- [agent1/schema.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/schema.py)：`NewsFingerprint`
- [agent2/reasoner.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/reasoner.py)：交易信号、PMI 校正、过滤规则
- [agent2/schema.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/schema.py)：`TradingSignal`
- [backtest/backtester.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/backtester.py)：端到端回测、重定价、指标统计
- [backtest/pmi_grid_search.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/pmi_grid_search.py)：离线 `pmi_alpha` / `confidence` 网格搜索
- [backtest/dataset_parser.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/dataset_parser.py)：数据集解析与规范化
- [ingestion/news_fetcher.py](/C:/Project/FinGPT/FinGPT_Part2/ingestion/news_fetcher.py)：Alpaca / Finnhub 新闻抓取

## 3. 端到端工作流

### 3.1 实时模式

入口是 [pipeline.py](/C:/Project/FinGPT/FinGPT_Part2/pipeline.py) 的 `run_pipeline(...)`。

流程如下：

1. 从 Alpaca 或 Finnhub 抓取候选新闻
2. 在抓取层做 leakage-safe 过滤
3. 每个 ticker 只保留目标时点之前最近的一篇新闻
4. `Agent 1` 生成 `NewsFingerprint`
5. `Agent 2` 生成 `TradingSignal`
6. 输出到 `output/signals_<timestamp>.json`

当前实时模式的关键约束：

- `ticker` 与 `headline` 由上游抓取层提供
- `Agent 2` 不直接重读完整新闻正文
- 默认每个 ticker 只使用一篇最新可用新闻

### 3.2 回测模式

入口是 [backtest/run_backtest.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/run_backtest.py) 与 [backtest/backtester.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/backtester.py)。

流程如下：

1. 读取 Hugging Face 数据集，或本地 `.csv` / `.parquet`
2. 统一列名为 `input`、`output`、`answer`、`ticker`
3. 解析出：
   - `ticker`
   - `headline`
   - `article_text`
   - `start_date`
   - `end_date`
   - `fingpt_label`
4. 按 batch 运行：
   - Agent 1 批量抽取
   - Agent 2 批量打信号
   - yfinance 抓取真实区间收益
   - 结果扁平化写入 CSV

回测默认是“宽容模式”：

- `fingerprint_failed` 会跳过该行
- `event_type_logits_failed` 默认退化为 `OTHER`
- `signal_failed` 会跳过该行
- `price_fetch_failed` 会保留前面结果，但该行不计入成功样本

## 4. 两个 Agent 实际在做什么

### 4.1 Agent 1

`Agent 1` 最终输出 [agent1/schema.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/schema.py) 中定义的 `NewsFingerprint`，其中最重要的字段包括：

- 元数据：`ticker`、`headline`、`source`、`published_at`
- 情绪：`sentiment_label`、`sentiment_confidence`、`sentiment_probabilities`
- 事件：`event_type`、`event_type_confidence`、`event_type_margin`
- 透传：`article_text`

当前实现里它实际上包含 3 次独立 LLM 调用：

1. guided extraction：抽 `source` / `published_at` / `companies_named`
2. sentiment scoring：直接对 `POSITIVE` / `NEGATIVE` / `NEUTRAL` 打分
3. event-type scoring：直接对 `A-G` 七类事件 token 打分

注意：

- `ticker` 不再依赖 `companies_named[0]`
- `headline` 优先保留上游传入值
- `OTHER` 不是模型输出，而是 Python 阈值规则赋值

### 4.2 Agent 2

`Agent 2` 最终输出 [agent2/schema.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/schema.py) 中定义的 `TradingSignal`，关键字段包括：

- `direction`：`long` / `neutral` / `short`
- `confidence`
- `raw_signal_logits`
- `pmi_null_logprobs`
- `signal_logits`：PMI 调整后的 logits
- `signal_probabilities`
- `signal_filter_forced_hold`
- `signal_filter_reason`

当前逻辑：

1. 用 fingerprint 信息构造紧凑 prompt
2. 对 `A/B/C` 三个策略 token 取真实 logprob
3. 如果启用 PMI，则按 `adjusted = raw - pmi_alpha * null` 修正
4. 经 `CALIBRATION_T=1.2` 做 softmax 校准
5. 再经过 confidence / margin / buy / sell 阈值过滤

映射关系：

- `A -> BUY -> long`
- `B -> HOLD -> neutral`
- `C -> SELL -> short`

## 5. 当前配置重点

主要配置在 [config.py](/C:/Project/FinGPT/FinGPT_Part2/config.py)。

当前默认值里最值得关注的是：

- `CALIBRATION_T = 1.2`
- `FINGPT_PMI_ALPHA = 1.0`
- `FINGPT_SIGNAL_MIN_CONFIDENCE = 0.0`
- `FINGPT_SIGNAL_MIN_MARGIN = 0.0`
- `FINGPT_SIGNAL_USE_COT = False`
- `FINGPT_BACKTEST_STRICT_MODE = False`
- `FINGPT_BACKTEST_BATCH_SIZE = 10`

这意味着当前默认回测更偏向：

- 使用 PMI 修正
- 不启用 Agent 2 CoT
- 先保留信号，再通过离线 grid search 调整 `alpha` 与 `confidence`

## 6. Dashboard 说明

当前仓库已有静态 Dashboard 输出，目录在 [output/dashboard](/C:/Project/FinGPT/FinGPT_Part2/output/dashboard)。

### 6.1 `pmi_alpha_grid_search_backtest.html`

文件：[pmi_alpha_grid_search_backtest.html](/C:/Project/FinGPT/FinGPT_Part2/output/dashboard/pmi_alpha_grid_search_backtest.html)

用途：

- 展示 6 个 `pmi_alpha` 值的回测结果
- 切换查看单个 alpha 的累计收益曲线
- 查看 long / neutral / short 仓位构成
- 查看总收益、每笔均值、命中率、Sharpe、Sortino、最大回撤
- 底部柱图对比所有 alpha 的总收益与 Sharpe

当前 Dashboard 中展示的 alpha 为：

- `0.00`
- `0.25`
- `0.50`
- `0.75`
- `1.00`
- `1.25`

### 6.2 `alpha_confidence_2d_grid_search.html`

文件：[alpha_confidence_2d_grid_search.html](/C:/Project/FinGPT/FinGPT_Part2/output/dashboard/alpha_confidence_2d_grid_search.html)

用途：

- 对 `pmi_alpha` 与 `signal_min_confidence` 做二维网格搜索可视化
- Heatmap 支持切换指标：
  - `Total PnL`
  - `Sharpe`
  - `Direction accuracy`
  - `Trade count`
- 点击某个网格后，可查看：
  - 总收益
  - Sharpe
  - 最大回撤
  - 方向准确率
  - trade 数
  - forced-hold 比例
  - long / neutral / short 数量
- 下方折线图展示：同一 alpha 下，不同 confidence 阈值对应的 PnL
- 下方柱图展示：不同 confidence 阈值下平均 trade 数与 forced-hold 率

### 6.3 `preview.html`

文件：[preview.html](/C:/Project/FinGPT/FinGPT_Part2/output/dashboard/preview.html)

这个页面是当前的 long-only grid search dashboard，用的是带 margin 过滤的买入信号。

策略逻辑可以概括为：

```python
adjusted_logit = raw_logit - alpha * null_logit
prob = softmax(adjusted_logit)

buy_margin = prob_buy - max(prob_hold, prob_sell)

if prob_buy >= confidence_threshold and buy_margin >= margin_threshold:
    position = 1
else:
    position = 0
```

这里的含义是：

- 只做多，不做空
- `BUY` 满足阈值时持有多头
- `HOLD` 和 `SELL` 都视为现金 / 空仓
- 相比普通 confidence 过滤，这个版本额外要求 `BUY` 对另外两类有足够 margin

当前 `preview.html` 展示的内容包括：

- best gross 组合
- best net 10bp 组合
- Top 20 组合
- 全部 grid
- 每个组合的：
  - `alpha`
  - `confidence`
  - `longs`
  - `coverage`
  - `long precision > 10bp`
  - `gross`
  - `net10bp`
  - `net20bp`
  - `Sharpe`
  - `Sortino`
  - `MaxDD`

## 7. 关于模型当前的表现

以下结论来自当前产物：

- [output/pmi_alpha_grid_search.csv](/C:/Project/FinGPT/FinGPT_Part2/output/pmi_alpha_grid_search.csv)
- [output/alpha_confidence_grid_search.csv](/C:/Project/FinGPT/FinGPT_Part2/output/alpha_confidence_grid_search.csv)
- 两个 HTML dashboard

### 7.1 基础回测覆盖情况

- 总样本 `300`
- 成功定价并进入有效统计的样本 `290`
- `10` 条为 `price_fetch_failed`
- 当前 artifact 对应的基础 run 默认 `pmi_alpha=1.0`、`signal_min_confidence=0.0`

### 7.2 `pmi_alpha` 单维扫描结论

在当前这批结果里，`pmi_alpha` 对收益影响非常大，而且默认 `1.0` 不是最优点。

表现最好的几个点：

- `alpha=0.0`：`total_pnl=0.2814`，`annualized_sharpe=0.2870`，`direction_accuracy=44.14%`
- `alpha=0.5`：`total_pnl=0.1373`
- `alpha=0.25`：`total_pnl=0.1326`

表现较差的几个点：

- `alpha=0.75`：`total_pnl=-0.3657`
- `alpha=1.0`：`total_pnl=-0.2632`
- `alpha=1.25`：`total_pnl=-0.3162`

从当前结果看，较强 PMI 修正会明显把仓位推向 `short`：

- `alpha=1.0` 时，成功样本里 `short=249`，`long=12`
- `alpha=1.25` 时，成功样本里 `short=270`，`long=7`

这说明当前模型与当前数据上，PMI 默认值偏强，存在把信号整体拉向卖出侧的风险。

### 7.3 `alpha + confidence` 二维扫描结论

二维 grid search 的最好组合是：

- `pmi_alpha=0.0`
- `signal_min_confidence=0.35`

对应结果：

- `total_pnl=0.3028`
- `annualized_sharpe=0.3101`
- `direction_accuracy=43.79%`
- `num_trades=232`
- `forced_hold_rate≈1.33%`

这比 `alpha=0.0, confidence=0.30` 略好，也明显优于当前默认 `alpha=1.0, confidence=0.0`。

整体趋势上：

- 轻微提高 `confidence` 阈值有时能改善收益和 Sharpe
- 但阈值过高会快速提升 `forced_hold_rate`
- 当 `confidence >= 0.45` 后，多数组合的 trade 数明显下降
- 当 `confidence >= 0.50` 后，很多组合已经进入“高 abstention、低覆盖”的状态

### 7.4 当前 long-only 表现简述

long-only 结果来自 [preview.html](/C:/Project/FinGPT/FinGPT_Part2/output/dashboard/preview.html)。

当前这个 long-only dashboard 的最好 gross 组合是：

- `alpha=0.05`
- `confidence=0.30`

对应结果：

- `gross=0.4306`
- `Sharpe=0.495`
- `MaxDD=0.3798`
- `longs=165`
- `coverage=56.9%`
- `long precision > 10bp = 58.8%`

同一个组合也是当前页面中的 best net 10bp：

- `net10bp=0.2656`
- `net20bp=0.1006`

从当前 long-only grid 看，几个比较清楚的现象是：

- 小幅 `alpha`，尤其 `0.00` 到 `0.25` 左右，表现整体更稳
- `confidence=0.30` 到 `0.35` 是当前较好的区间
- 阈值继续提高到 `0.45+` 后，coverage 会快速下降
- 高阈值下虽然个别 trade 质量可能更高，但样本数会明显变少，整体收益未必更优，且泛化能力值得怀疑

## 8. 想改哪里看哪里

### 改 Agent 1

- [agent1/prompt.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/prompt.py)
- [agent1/extractor.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/extractor.py)
- [agent1/schema.py](/C:/Project/FinGPT/FinGPT_Part2/agent1/schema.py)

### 改 Agent 2

- [agent2/prompt.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/prompt.py)
- [agent2/reasoner.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/reasoner.py)
- [agent2/schema.py](/C:/Project/FinGPT/FinGPT_Part2/agent2/schema.py)

### 改回测与离线搜索

- [backtest/backtester.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/backtester.py)
- [backtest/pmi_grid_search.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/pmi_grid_search.py)
- [backtest/dataset_parser.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/dataset_parser.py)
- [backtest/price_fetcher.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/price_fetcher.py)

### 改实时抓取

- [ingestion/news_fetcher.py](/C:/Project/FinGPT/FinGPT_Part2/ingestion/news_fetcher.py)
- [pipeline.py](/C:/Project/FinGPT/FinGPT_Part2/pipeline.py)
