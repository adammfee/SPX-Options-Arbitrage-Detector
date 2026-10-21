# SPX-Options-Arbitrage-Detector
Python — Linear programming model that detects arbitrage in SPX option chains and finds the optimal portfolio to exit an existing option position

**Part 1 — Arbitrage detection.** Long and short legs are separate variables so
bid–ask spreads are priced correctly. The continuous payoff space reduces to an
exact finite constraint set (every strike, plus zero, plus a slope condition on
the tail), so no discretization error. Validated against seeded crossed-quote,
monotonicity, and convexity violations.

**Part 2 — Optimal position exit.** Given an option position, finds the portfolio
of options and a zero-coupon bond whose payoff exactly cancels it, maximizing cash
received today. Routinely beats crossing the spread on the position itself by
constructing synthetic forwards from tighter-quoted strikes.
