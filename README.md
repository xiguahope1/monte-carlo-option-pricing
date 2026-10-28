# Monte Carlo Option Pricer

I built this to learn how Monte Carlo methods handle problems that don't have neat closed-form answers. It prices European, Asian and up-and-out barrier options under two models: Geometric Brownian Motion and the Heston stochastic volatility model.

## What I did
- Simulated GBM using exact log-normal steps, and Heston using full-truncation Euler–Maruyama (so the variance can't go negative).
- Checked the European prices against the Black–Scholes formula to make sure the engine was working before trusting the exotic options.
- Compared standard random sampling with Sobol quasi-random sequences to see how much faster the price converges.
- Estimated Delta using central finite differences.
- Added a small Streamlit dashboard to play with the parameters.

## What I found
- [e.g. Standard MC error fell roughly like 1/√N, as expected. Sobol reached the same accuracy with about __x fewer paths for European options, but the gain was smaller for Asian and barrier options.]
- [e.g. Barrier prices shifted as I added more time steps, because checking the barrier only at discrete times misses some crossings.]

## What I'd improve next
- Correct for the discrete barrier monitoring bias.
- Add more Greeks (Gamma, Vega) and variance reduction like control variates.

---

## Project Architecture

```text
monte-carlo-pricer/
│
├── models.py        # Stochastic processes (GBM, Heston Model)
├── payoffs.py       # Payoff strategy classes (European, Asian, Barrier)
├── engine.py        # Monte Carlo simulation engine & Greeks calculation
├── analytical.py    # Analytical Black-Scholes-Merton closed-form benchmark
├── app.py           # Streamlit web application & interactive dashboard
├── main.py          # Command-line benchmark script
├── requirements.txt # Project dependencies
└── README.md
