# 1. Alpaca Weekly Backtest 输出列说明

这份文档说明 [alpaca_weekly_backtest.ipynb](/C:/Project/FinGPT/FinGPT_Part2/notebooks/alpaca_weekly_backtest.ipynb) 当前会产出的几类文件，以及每一列的含义。

主要有三类输出：

1. `dataset CSV`
   由 [alpaca_news_pipeline.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/alpaca_news_pipeline.py) 生成，作为现有 backtester 的输入。
2. `backtest results CSV`
   由 [backtester.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/backtester.py) 生成，包含 Agent 1、Agent 2、收益与跳过原因。
3. `metrics JSON`
   由 notebook 和回测辅助函数生成，是对结果 CSV 的摘要统计。

---

## 2. Backtest Results CSV 列说明

默认文件路径来自 config 里的 `backtest_output_path`，通常是：

- [output/alpaca_news_backtest_results.csv](/C:/Project/FinGPT/FinGPT_Part2/output/alpaca_news_backtest_results.csv)

这些列主要来自 [backtester.py](/C:/Project/FinGPT/FinGPT_Part2/backtest/backtester.py:48) 的 `_make_base_result()`，再加上 Agent 1 / Agent 2 / price layer 的填充。

下面按功能分组说明。

### 2.1 基础标识列

#### `ticker`
- 这条样本对应的股票代码。

#### `headline`
- 从原始 `input` 中提取的第一条 headline。
- 对 multi-news prompt，默认把第一条 headline 当作主 headline。

#### `fingerprint_ticker`
- Agent 1 最终输出的 `NewsFingerprint.ticker`。
- 正常情况下应与 dataset 的 `ticker` 一致。
- 如果这里是空，通常说明 Agent 1 没成功产出 fingerprint。

#### `signal_ticker`
- Agent 2 最终输出的 `TradingSignal.ticker`。
- 正常情况下应与 `fingerprint_ticker` 一致。

#### `start_date`
- 这条样本的实际回测开始日。
- 当前周频逻辑下：
  - `news_window_end` 之后严格下一个美股交易日。

#### `end_date`
- 这条样本的实际回测结束日。
- 当前逻辑下：
  - `start_date + holding_period_days`
  - 如果该日不是交易日，则顺延到下一个交易日。

#### `article_text`
- 传给 Agent 1 的文本摘要。
- 只保留前 120 个字符写入结果 CSV，便于预览。

#### `fingpt_label`
- 从 dataset 的 `answer` 字段解析出的标签。
- Alpaca weekly pipeline 里通常是：
  - `no_label_provided`
- 如果你用的是原始标注数据集，则可能是：
  - `up`
  - `down`
  - `neutral`

---

### 2.2 Agent 1: Sentiment 相关列

#### `sentiment_label`
- Agent 1 对新闻的情绪判断。
- 值通常为：
  - `POSITIVE`
  - `NEGATIVE`
  - `NEUTRAL`

#### `sentiment_score`
- 情绪的数值化表示。
- 映射关系：
  - `POSITIVE -> 1.0`
  - `NEGATIVE -> -1.0`
  - `NEUTRAL -> 0.0`

#### `sentiment_confidence`
- Agent 1 情绪分类的置信度。
- 来源是情绪 logits 经 softmax 后最大类别的概率。

#### `sentiment_logprob_POSITIVE`
- 模型对 `POSITIVE` 的原始对数概率。

#### `sentiment_logprob_NEGATIVE`
- 模型对 `NEGATIVE` 的原始对数概率。

#### `sentiment_logprob_NEUTRAL`
- 模型对 `NEUTRAL` 的原始对数概率。

#### `sentiment_prob_POSITIVE`
- 经过 softmax 与 calibration 后，`POSITIVE` 的概率。

#### `sentiment_prob_NEGATIVE`
- `NEGATIVE` 的概率。

#### `sentiment_prob_NEUTRAL`
- `NEUTRAL` 的概率。

---

### 2.3 Agent 1: Event Type 相关列

#### `event_type`
- Agent 1 给出的事件类别。
- 典型值包括：
  - `EARNINGS`
  - `GUIDANCE`
  - `ANALYST_RATING`
  - `LEGAL_REGULATORY`
  - `MNA`
  - `PRODUCT_BUSINESS`
  - `MACRO`
  - `OTHER`

#### `event_type_confidence`
- 事件类别 top class 的概率。

#### `event_type_margin`
- 事件类别 top1 与 top2 概率的差值。

#### `event_type_method`
- 事件类别是如何得到的。
- 常见值：
  - `logits_accepted`
  - `abstained_low_confidence`
  - `abstained_low_margin`
  - `event_type_logits_failed`

#### `event_logprob_A` ~ `event_logprob_G`
- Agent 1 对 A-G 七个事件 token 的原始对数概率。

#### `event_prob_A` ~ `event_prob_G`
- Agent 1 对 A-G 七个事件 token 的 softmax 概率。

---

### 2.4 Agent 2: Signal / PMI 相关列

这一组列同时包含 4 层信息：

1. 原始 logits
2. PMI correction 之后的 adjusted logits
3. softmax 概率
4. filter 之后的最终方向

建议把 Agent 2 的流程理解成：

1. 先得到 `A=BUY`、`B=HOLD`、`C=SELL` 的原始 `prompt_logprobs`
2. 再按 PMI 公式做 correction
3. 对 correction 后的 logits 做 softmax
4. 再应用 confidence / margin / buy-sell threshold filter
5. 最后得到真正写入结果 CSV 的 `direction`

#### `direction`
- Agent 2 最终方向。
- 当前典型值：
  - `long`
  - `neutral`
  - `short`
  - `no_signal`

#### `confidence`
- Agent 2 最终方向的置信度。

#### `raw_signal_logprob_A`
- Agent 2 对 `A=BUY` 的原始对数概率。

#### `raw_signal_logprob_B`
- Agent 2 对 `B=HOLD` 的原始对数概率。

#### `raw_signal_logprob_C`
- Agent 2 对 `C=SELL` 的原始对数概率。

#### `pmi_null_logprob_A`
- PMI prior 下 `A=BUY` 的 null-context 对数概率。

#### `pmi_null_logprob_B`
- `B=HOLD` 的 null-context 对数概率。

#### `pmi_null_logprob_C`
- `C=SELL` 的 null-context 对数概率。

#### `pmi_adjusted_logit_A`
- 经过 PMI correction 后的 `A` logit。

#### `pmi_adjusted_logit_B`
- `B` 的 PMI-corrected logit。

#### `pmi_adjusted_logit_C`
- `C` 的 PMI-corrected logit。

#### `signal_prob_A`
- `BUY` 概率。

#### `signal_prob_B`
- `HOLD` 概率。

#### `signal_prob_C`
- `SELL` 概率。

#### `pmi_alpha_used`
- 这条信号使用的 PMI alpha 超参数。

#### `calibration_T`
- softmax 使用的 calibration temperature。

#### Agent 2 correction 逻辑

当前 correction 的核心公式是：

```text
adjusted_logit = raw_logit - pmi_alpha * null_logit
prob = softmax(adjusted_logit / calibration_T)
```

对应到结果列就是：

- `raw_signal_logprob_A/B/C`
  表示模型在当前新闻上下文下，对 `BUY / HOLD / SELL` 的原始 logprob
- `pmi_null_logprob_A/B/C`
  表示模型在 null / neutral context 下，对 `BUY / HOLD / SELL` 的先验 logprob
- `pmi_adjusted_logit_A/B/C`
  表示按 `raw - alpha * null` 修正后的 logits
- `signal_prob_A/B/C`
  表示修正后再 softmax 得到的概率

几个重要结论：

- `pmi_alpha = 0.0`
  等价于完全不做 PMI correction
- `pmi_alpha > 0`
  表示开始扣除一部分模型的先验偏置
- `calibration_T`
  不改变 logits 的相对顺序，但会改变 softmax 分布的尖锐程度

可以把 PMI correction 理解成：

- `raw_signal_logprob_*`
  看的是模型原始偏好
- `pmi_adjusted_logit_*`
  看的是去掉一部分先验偏置后的偏好

#### `signal_filter_forced_hold`
- 是否被后处理规则强制打回 `HOLD`。

#### `signal_filter_reason`
- 如果被强制打回，具体原因是什么。
- 常见值：
  - `low_confidence`
  - `low_margin`
  - `buy_threshold`
  - `sell_threshold`
  - `long_only_filter`

#### Agent 2 filter 具体逻辑

在 correction 和 softmax 之后，Agent 2 会再应用一层 filter。

当前顺序是：

```text
top_idx = argmax(prob)
top_prob = prob[top_idx]
margin = top1_prob - top2_prob

if top_prob < min_confidence:
    force HOLD
elif margin < min_margin:
    force HOLD
elif raw_top == BUY and top_prob < buy_threshold:
    force HOLD
elif raw_top == SELL and top_prob < sell_threshold:
    force HOLD
else:
    keep original top class
```

这几个 reason 的含义分别是：

- `low_confidence`
  最高类别概率太低，模型自己都不够确定
- `low_margin`
  top1 和 top2 太接近，说明分类边界不清晰
- `buy_threshold`
  虽然 `BUY` 是 top1，但 BUY 概率还没高到允许开多
- `sell_threshold`
  虽然 `SELL` 是 top1，但 SELL 概率还没高到允许开空
- `long_only_filter`
  这是后续 long-only grid search / sentiment-only notebook 里额外使用的规则，不是原始 Agent 2 默认规则

如果 filter 被触发，那么最终结果会被改写成：

- `direction = neutral`
- `signal_direction = neutral`
- `signal_filter_forced_hold = True`
- `signal_filter_reason = 对应原因`

所以在读一条回测记录时，推荐按下面顺序理解：

1. 看 `raw_signal_logprob_A/B/C`
2. 看 `pmi_null_logprob_A/B/C`
3. 看 `pmi_adjusted_logit_A/B/C`
4. 看 `signal_prob_A/B/C`
5. 看 `signal_filter_forced_hold`
6. 看 `signal_filter_reason`
7. 最后再看 `direction`

#### `signal_direction`
- `direction` 的兼容别名。
- 为了兼容旧代码和 metrics 保留。

#### `signal_confidence`
- `confidence` 的兼容别名。

#### `strategy_tag`
- 策略标签。
- 当前新闻信号典型值：
  - `event_driven`
- `no_signal` 样本可能是：
  - `none`

#### 一个完整例子

假设某条样本有：

```text
raw_signal_logprob_A = -0.4
raw_signal_logprob_B = -0.8
raw_signal_logprob_C = -1.6

pmi_null_logprob_A = -1.2
pmi_null_logprob_B = -1.0
pmi_null_logprob_C = -1.1

pmi_alpha_used = 0.5
```

那么 correction 后大致相当于：

```text
A: -0.4 - 0.5 * (-1.2)
B: -0.8 - 0.5 * (-1.0)
C: -1.6 - 0.5 * (-1.1)
```

得到 `pmi_adjusted_logit_*` 后，再做 softmax，得到：

```text
signal_prob_A
signal_prob_B
signal_prob_C
```

如果这时：

- `A` 是 top1
- 但 `signal_prob_A` 没超过 `buy_threshold`

那么最终写入结果的不会是 `long`，而会被 filter 打回：

```text
direction = neutral
signal_filter_forced_hold = True
signal_filter_reason = buy_threshold
```

---

### 2.5 收益与仓位列

#### `realized_return`
- 从 `start_date` 到 `end_date` 的实际收益率。
- 来源是 price layer 对 Yahoo / yfinance 的抓取结果。

#### `position`
- 根据信号转换出的仓位。
- 映射通常是：
  - `long -> 1`
  - `short -> -1`
  - `neutral -> 0`

#### `strategy_return`
- 策略收益。
- 当前公式：
  - `position * realized_return`

---

### 2.6 原因与错误列

#### `pass_reason`
- 来自 dataset 层的原因透传。
- 例如：
  - `no_article_provided`

#### `skipped_reason`
- 这条结果为什么没有成为完整有效样本。
- 常见值包括：
  - 空字符串：正常成功
  - `no_article_provided`
  - `fingerprint_failed`
  - `signal_failed`
  - `price_fetch_failed`

#### `price_fetch_error_reason`
- price layer 失败时的更细原因。
- 常见值包括：
  - `chart_request_failed`
  - `empty_history`
  - `missing_close`
  - `download_exception`

---

## 3. Metrics JSON 字段说明

通常是：

- [output/alpaca_news_backtest_results_metrics.json](/C:/Project/FinGPT/FinGPT_Part2/output/alpaca_news_backtest_results_metrics.json)

这些字段由 [compute_metrics](C:/Project/FinGPT/FinGPT_Part2/backtest/backtester.py:486) 和 [augment_metrics_with_demo_fields](C:/Project/FinGPT/FinGPT_Part2/backtest/backtester.py:637) 生成。

### 基础回测指标

#### `total_rows`
- 总样本数。

#### `successful_rows`
- 成功完成收益评估的样本数。

#### `skip_rate`
- 跳过比例。

#### `direction_accuracy`
- 方向正确率。
- 即信号方向和未来真实涨跌方向是否一致。

#### `long_accuracy`
- 做多信号的命中率。

#### `short_accuracy`
- 做空信号的命中率。

#### `mean_strategy_return`
- 单条样本的平均策略收益。

#### `std_strategy_return`
- 单条样本策略收益的标准差。

#### `annualized_sharpe`
- 年化 Sharpe。
- 当前按周频近似，使用 `sqrt(52)`。

#### `total_pnl`
- 累计策略收益和。

#### `gross_return`
- 当前与 `total_pnl` 相同，作为兼容字段保留。

#### `max_drawdown`
- 累计策略收益序列的最大回撤。

#### `vs_fingpt_accuracy`
- 和 `fingpt_label` 的一致率。
- 在 Alpaca weekly pipeline 下通常是：
  - `no_label_provided`

### 仓位覆盖相关指标

#### `long_trade_count`
- 做多样本数。

#### `short_trade_count`
- 做空样本数。

#### `neutral_count`
- 中性样本数。

#### `num_trades`
- 交易样本数。
- 当前等于：
  - `long_trade_count + short_trade_count`

#### `coverage`
- 有仓位信号的样本占比。

#### `abstention_rate`
- 中性 / abstain 的样本占比。

#### `long_precision`
- 与 `long_accuracy` 当前等价。

#### `short_precision`
- 与 `short_accuracy` 当前等价。

#### `signal_filter_forced_hold_rate`
- 被后处理强制打回 HOLD 的比例。

### Agent 摘要指标

#### `avg_sentiment_confidence`
- Agent 1 情绪置信度平均值。

#### `avg_signal_confidence`
- Agent 2 最终信号置信度平均值。

#### `avg_pmi_alpha_used`
- 平均使用的 PMI alpha。
- 对单一回测通常就是固定值。

#### `avg_calibration_T`
- 平均 calibration temperature。

### 可选 breakdown

#### `event_type_breakdown`
- 按事件类型分组后的平均收益和样本数。

---

## 4. 快速判断一行结果是否“正常成功”

你可以用下面几列快速判断：

- `fingerprint_ticker`
  有值表示 Agent 1 成功了。
- `signal_ticker`
  有值表示 Agent 2 成功了。
- `skipped_reason`
  为空表示这条样本最终成功进入收益计算。
- `realized_return`
  有值表示 price layer 成功了。

如果一行同时满足：

- `fingerprint_ticker` 非空
- `signal_ticker` 非空
- `skipped_reason` 为空
- `realized_return` 非空

那这条样本就是一条完整成功的回测记录。

---

## 5. 建议的查看顺序

实际分析时，建议按这个顺序看：

1. 先看 `dataset CSV`
   确认 `ticker`、`window_key`、`skip_llm` 是否合理。
2. 再看 `results CSV`
   重点看：
   - `sentiment_label`
   - `event_type`
   - `direction`
   - `signal_prob_A/B/C`
   - `skipped_reason`
3. 最后看 `metrics JSON`
   重点看：
   - `total_pnl`
   - `annualized_sharpe`
   - `direction_accuracy`
   - `coverage`
   - `signal_filter_forced_hold_rate`

如果你后面还需要，我可以继续帮你再补一版“按 notebook cell 对照”的说明，也就是：

- Cell 5 看到的是 dataset 哪些列
- Cell 9 展示的是 results 哪些列
- 哪些列最适合拿去做研究图表
