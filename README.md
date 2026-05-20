# MVP Multi-Agent CPT-PG Experiment

This project implements a minimal viable simulation for a multi-agent A-share portfolio system with:
- Tushare market and financial data caching
- Akshare `stock_info*` market-state enrichment with missing-source isolation
- Qwen/DashScope-backed simulated user, preference, and advisor agents
- Dynamic-reference CPT-PG, static-reference CPT-PG, frozen-preference CPT-PG, mean-variance, and equal-weight baselines
- Qbot-inspired technical strategy actions, including BOLL reversion, RSI reversal, MACD trend, and RSRS timing
- Akshare news sentiment enters the action affinity used by CPT-PG sampling and gradient estimates
- Daily traces, summary tables, and four main plots

## Quick Start

1. Ensure these environment variables are available:
   - `Tushare_Token` or `TUSHARE_TOKEN`
   - `DashScope_API_Key` or `DASHSCOPE_API_KEY`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run a 10-day smoke test:
   - `python scripts/smoke_test.py`
4. Run the full experiment:
   - `python scripts/run_mvp.py`

## LLM Agent Configuration

The three runtime agents use independent OpenAI-compatible DashScope clients:
- simulated user agent
- preference agent
- advisor agent

Default LLM settings match `scripts/LLM_test.py`:
- API key: `DASHSCOPE_API_KEY`, `DashScope_API_Key`, or `DASHSCOPE_KEY`
- Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- Model: `glm-5`
- Thinking mode: enabled

Useful run examples:
- `python scripts/run_mvp.py --methods dynamic_cpt_pg --seeds 7 --dry-run-days 80`
- `python scripts/run_mvp.py --methods dynamic_cpt_pg --seeds 7 --dry-run-days 1 --initial-holdings artifacts/inputs/initial_holdings.csv`
- `python scripts/run_mvp.py --methods dynamic_cpt_pg --seeds 7 --dry-run-days 1 --initial-capital 1000000`
- `python scripts/run_mvp.py --methods dynamic_cpt_pg --seeds 7 --dry-run-days 80 --llm-model glm-5 --llm-base-url https://dashscope.aliyuncs.com/compatible-mode/v1`
- `python scripts/run_mvp.py --methods dynamic_cpt_pg --seeds 7 --dry-run-days 80 --disable-llm-thinking`

`--initial-holdings` expects `ts_code,buy_price,shares`. The first reference point is the total buy cost amount, and the starting portfolio value is computed from the latest close before the first evaluation date.

If the user has no current holdings, omit `--initial-holdings`. In that case the first reference point is `0`, and `--initial-capital` only sets the cash budget used for first-day allocation and trade amount calculation.

`evaluation_horizon` is the CPT-PG sliding history window length. With the default value `20`, the first evaluation day uses the prior 20 trade dates when they are available, so dynamic CPT-PG does not wait for 20 live interaction days before updating.

The CPT reference point and objective signal use currency amounts. The observed daily objective signal is the portfolio value after close and after proportional transaction costs.

## Outputs

Artifacts are written under `artifacts/`:
- `artifacts/cache/raw/` raw Tushare and Akshare caches
- `artifacts/cache/llm/` agent response cache
- `artifacts/results/traces/daily_trace.csv`
- `artifacts/results/tables/summary_by_run.csv`
- `artifacts/results/tables/summary_by_method.csv`
- `artifacts/results/tables/stock_info_source_status.csv`
- `artifacts/results/plots/*.png`
