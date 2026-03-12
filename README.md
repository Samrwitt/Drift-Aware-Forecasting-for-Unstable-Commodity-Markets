# Drift-Aware Forecasting for Unstable Commodity Markets

This project implements a prototype forecasting pipeline designed for volatile commodity markets, using Ethiopia-inspired market conditions as the motivating environment. Traditional forecasting approaches often assume stable statistical relationships over time, but real markets—especially in developing economies—frequently experience structural breaks, supply shocks, currency fluctuations, and seasonal disruptions.

The system models commodity prices as the result of multiple interacting drivers rather than a single time series. Inputs include historical prices along with external signals such as exchange rate proxies, seasonal indicators, and shock flags.

A key feature of the system is concept-drift awareness. The pipeline continuously monitors prediction errors to detect when the underlying data distribution changes. When significant drift is detected, the model automatically retrains on recent data to adapt to the new regime.

To improve robustness, the prototype compares multiple forecasting models—including baseline persistence models, linear regression, and tree-based methods—and evaluates their performance across stable and shock periods.

Instead of producing a single deterministic forecast, the system outputs prediction intervals and confidence indicators, allowing the model to communicate uncertainty during unstable periods.

The goal of this project is not perfect prediction, but to demonstrate how a drift-aware forecasting architecture can remain useful in environments where market dynamics change frequently.
