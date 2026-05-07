
# Agent1 情绪信号回测结果报告

## 1. 实验背景

本轮实验的目标是评估新闻情绪分析在交易信号生成中的实际价值。此前 pipeline 使用两层 Agent：

- **Agent1**：从新闻中提取 sentiment / event / confidence 等信息；
- **Agent2**：基于 Agent1 输出进一步生成 BUY / HOLD / SELL 信号。

但在此前实验中，Agent2 的信号层表现出明显的不稳定性，尤其是在 PMI correction 后容易产生过多 short signal，而 short 方向的实际收益表现较差。因此，本轮实验主要尝试去掉 Agent2，直接使用 Agent1 的情绪输出构造更简单、可解释的交易规则。

本轮重点比较两个 Agent1-only 策略：

1. **Agent1 sentiment-only long**
2. **Agent1 confidence-filtered long**

---

## 2. 策略定义

### 2.1 Agent1 sentiment-only long

该策略直接使用 Agent1 的情绪标签：

```text
POSITIVE -> BUY
NEGATIVE / NEUTRAL -> HOLD
````

也就是说，只要 Agent1 判断新闻情绪为正面，就做多；否则不交易。

### 2.2 Agent1 confidence-filtered long

该策略进一步加入置信度过滤：

```text
POSITIVE and confidence >= 0.6 -> BUY
otherwise -> HOLD
```

这个策略的核心思想是：不使用所有 positive sentiment，而只保留 Agent1 相对更有把握的正面新闻信号。

---

## 3. 回测数据概览

本轮数据共有：

```text
total_rows = 126
successful_rows = 124
skip_rate ≈ 1.59%
```

说明整体 pipeline 运行成功率较高，只有少量样本被跳过。由于 Alpaca dataset 中没有有效的 FinGPT label，本轮 `vs_fingpt_accuracy` 被标记为 `no_label_provided`，因此不参与评价。

---

## 4. 策略一：Agent1 sentiment-only long

### 4.1 核心结果

Agent1 sentiment-only long 的主要指标如下：

```text
long_trade_count = 37
num_trades = 37
coverage ≈ 29.84%
abstention_rate ≈ 70.16%
total_pnl ≈ 0.0558
gross_return ≈ 0.0558
annualized_sharpe ≈ 0.114
max_drawdown ≈ 0.252
long_precision ≈ 48.65%
```

该策略一共进行了 37 笔 long 交易，覆盖率约为 29.8%。最终总收益约为 5.6%，但 Sharpe ratio 只有 0.114，最大回撤达到约 25.2%。

### 4.2 结果解读

这个结果说明，Agent1 的 raw positive sentiment 并不是完全无效，但直接将所有 positive sentiment 映射成 BUY 的策略质量较弱。

从 equity curve 看，该策略早期有过上涨，但随后经历了较长时间的回撤，最终收益主要依靠后期部分修复。drawdown curve 也显示其风险较高，最大回撤较大。

因此，这个策略更适合作为一个 baseline，而不是最终主策略。

### 4.3 结论

Agent1 sentiment-only long 的结果可以总结为：

```text
Agent1 的 positive sentiment 有一定方向信息，
但 raw sentiment 信号噪声较大，
直接 positive -> BUY 的策略不够稳定。
```

---

## 5. 策略二：Agent1 confidence-filtered long

### 5.1 核心结果

Agent1 confidence-filtered long 的主要指标如下：

```text
long_trade_count = 19
num_trades = 19
coverage ≈ 15.32%
abstention_rate ≈ 84.68%
total_pnl ≈ 0.3325
gross_return ≈ 0.3325
annualized_sharpe ≈ 1.285
max_drawdown ≈ 0.0613
long_precision ≈ 68.42%
```

相比 sentiment-only long，该策略交易次数从 37 笔下降到 19 笔，覆盖率从约 29.8% 降至约 15.3%。但收益显著提高，总收益从约 5.6% 提升到约 33.3%，最大回撤从约 25.2% 降至约 6.1%，Sharpe ratio 也提升到约 1.285。

### 5.2 结果解读

confidence filter 显著改善了策略表现。这说明 Agent1 的 confidence 并不是完全无意义的附属字段，而是能够帮助过滤低质量 positive sentiment。

换句话说，Agent1 在“是否正面”之外，还能通过 confidence 反映一定的信号强弱。高置信度 positive sentiment 在样本内明显比普通 positive sentiment 更有价值。

不过，这个结果也需要谨慎解读。该策略只有 19 笔交易，样本量较少，且 equity curve 呈现明显的平台期和跳升特征。这说明策略收益很可能集中在少数几笔交易上，存在较强的样本内偶然性。

### 5.3 结论

Agent1 confidence-filtered long 的结果可以总结为：

```text
高置信度 positive sentiment 在样本内表现较好，
confidence filter 能明显降低噪声和回撤，
但交易数较少，暂时不能证明泛化能力。
```

---

## 6. Mixed Strategy 的问题

从 mixed strategy 的可视化结果看，该策略整体表现较差。equity curve 明显下行，drawdown 较深，说明 long-short 混合策略目前并不稳定。

更关键的问题是，SELL 信号后的 realized return 也呈现正收益特征。这意味着：

```text
如果 SELL 信号之后资产价格上涨，
那么做空方向实际上是错误的。
```

这与此前观察一致：Agent2 或 mixed signal 层容易产生过多 short 信号，而 short side 的方向判断并不可靠。

因此，当前阶段不建议继续把 mixed long-short strategy 作为主策略。它更适合作为一个 negative baseline 或 ablation，用来说明 Agent2 signal layer / short mapping 尚未校准好。

---

## 7. Long-only Strategy 的相对优势

相比 mixed strategy，long-only 策略明显更稳。无论是 Agent1 sentiment-only long，还是 confidence-filtered long，都避免了 short side 带来的方向性错误。

尤其是 confidence-filtered long，在样本内取得了较高收益和较低回撤。该结果说明：

```text
在当前数据和模型设置下，long-only 比 mixed long-short 更合理。
```

不过，long-only 的问题是交易过于稀疏。特别是 confidence-filtered long 只有 19 笔交易，说明它更像是一个强过滤器，而不是一个持续产生交易机会的完整策略。

因此，long-only 可以作为当前更稳妥的方向，但仍不应被过度解读为已经具备稳定 alpha。

---

## 8. Event Type 观察

从 event type breakdown 看，EARNINGS 类事件在两个 Agent1 策略中都有正的平均收益：

```text
Agent1 sentiment-only long:
EARNINGS mean_return ≈ 0.005663
MACRO mean_return ≈ -0.000323

Agent1 confidence-filtered long:
EARNINGS mean_return ≈ 0.008637
MACRO mean_return ≈ 0.001799
```

在 confidence-filtered long 中，MACRO 类事件的平均收益也转为正值。 

这说明 confidence filter 可能不仅改善了个股事件新闻，也改善了宏观类新闻的筛选效果。

不过需要注意，EARNINGS 样本只有 16 行，而 MACRO 样本有 108 行，因此 event type 的结果仍然需要更大样本验证。

---

## 9. 当前结果的局限性

本轮结果有几个明显局限：

### 9.1 样本量偏小

总样本只有 126 行，successful rows 为 124 行。对于交易策略回测来说，这个样本量较小。尤其 confidence-filtered long 只有 19 笔交易，很容易受到个别样本影响。

### 9.2 标的数量较少

当前实验只覆盖少量股票 / ETF，因此难以证明策略能跨资产泛化。某些结果可能来自特定 ticker 或特定市场阶段。

### 9.3 时间窗口较短

当前回测区间较短，无法覆盖多个市场 regime。例如牛市、熊市、震荡市、利率冲击、财报季差异等都没有充分覆盖。

### 9.4 交易信号过于稀疏

confidence-filtered long 的表现较好，但覆盖率只有约 15.3%。这说明该策略并不是持续交易系统，而更像是一个高置信度机会筛选器。

### 9.5 不应使用 vs_fingpt_accuracy

由于 Alpaca dataset 中没有有效 answer label，`vs_fingpt_accuracy` 被标记为 `no_label_provided`，因此不能用于衡量策略有效性。

---

## 10. 总体结论

本轮实验最重要的结论是：

```text
Agent2 signal layer 当前不稳定；
Agent1 sentiment 更可解释、更适合作为新闻信号源；
confidence filter 能显著改善 Agent1 positive sentiment 的样本内表现；
但由于样本量和交易数较少，目前不能证明泛化能力。
```

更具体地说：

1. **Agent1 sentiment-only long 表现较弱。**
   直接 positive -> BUY 的策略收益较低、回撤较大，说明 raw sentiment 噪声较高。

2. **Agent1 confidence-filtered long 表现明显更好。**
   加入 confidence >= 0.6 后，策略收益、Sharpe 和回撤都有明显改善。

3. **Mixed long-short strategy 当前不可靠。**
   short side 方向判断较差，SELL 信号不应直接用于做空。

4. **Long-only 是当前更合理的方向。**
   在当前模型和数据下，禁用 short 或将 short collapse 成 HOLD 更稳。

5. **Agent1 更适合作为 filter，而不是完整交易系统。**
   最合理的定位是将 Agent1 作为新闻情绪过滤器，与纯价格量化策略结合。