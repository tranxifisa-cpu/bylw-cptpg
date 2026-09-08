# Data notes

This package is deterministic and offline. It creates `ashare_semisynthetic_panel_2023_2026.csv`, a semi-synthetic A-share-style factor panel with five stocks plus cash, market regimes, returns, and factors.

To use real Tushare data, place a local CSV at `experiments/data/user_ashare_panel.csv` with columns `date, asset, ret, momentum, reversal, volatility, value` and adapt the loader in `generate_experiments.py`.

To use real retail behavior data, place a local CSV at `experiments/data/user_retail_behavior.csv` with columns such as `user_id, date, action, position, realized_gain, realized_loss, turnover`. The shipped `retail_multiagent_evaluation.csv` is a simulated heterogeneous-agent evaluation and is not claimed to be Xueqiu raw data.
