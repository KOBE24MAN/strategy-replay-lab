# Strategy Replay Lab

An offline comparison of a tabular Q-learning policy against buy-and-hold. The demo uses a synthetic return series and common trading-cost assumptions so both policies can be evaluated on the same holdout.

## How it works

1. Train a long-or-flat policy using observed return direction and the current position as state.
2. Replay the learned policy and the baseline on a separate 120-day holdout with identical fees and slippage.
3. Report net return, maximum drawdown and traded units.

## Run locally

Requires Python 3.10+.

```bash
python portfolio_demo/strategy_replay.py
python -m unittest discover -s portfolio_demo -v
```

## Roadmap

- [ ] Accept timestamped return data and expose commission and slippage separately as command-line options.
- [ ] Export a per-step audit log with decisions, filled positions, fee deductions and the resulting equity series.
- [ ] Add a momentum baseline and compare all policies across identical holdout windows and trading-cost scenarios.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for distribution terms.
