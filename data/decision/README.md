# Decision data package

This directory contains the data needed for a first risk-aware portfolio
decision experiment built on the volatility world model.

## Available observed data

- `../dow30/dow30_daily_returns_2021_2026.csv`: 30-asset daily returns used
  for realized portfolio P&L after a decision at date (t).
- `../dow30/merged_rv_data_filled.csv`: daily RV panel used by the world
  model.
- `../dow30/merged_iv_data_filled.csv`: daily IV panel used as a forward
  risk signal.
- `fred_3m_tbill.csv`: FRED DTB3 3-month Treasury bill rate, in percent per
  year, for a cash and excess-return benchmark.
- `fred_10y_treasury.csv`: FRED DGS10 10-year Treasury constant-maturity
  rate, in percent per year, for a macro risk-state feature.

## Decision environment

The current observations support an exogenous-market decision environment:
the action changes portfolio exposure and reward, while the investor does not
change the Dow30 market transition. A daily action is a vector of portfolio
weights. The first implementation should report gross and net portfolio
returns separately and use explicit transaction-cost sensitivity.

The source archive does not contain historical bid-ask spreads, borrow rates,
order-book depth, or a survivorship-free historical Dow membership file. Those
fields must be obtained from CRSP, Compustat, Bloomberg, Refinitiv, or a
comparable licensed source before claiming implementation-grade execution or
historical-universe results. The `decision_assumptions.yaml` file records the
transparent research assumptions for a prototype.
