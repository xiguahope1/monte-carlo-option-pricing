# Quantitative Monte Carlo Option Pricing Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)

An object-oriented Monte Carlo pricing and risk simulation engine in Python. The system values European, path-dependent (Arithmetic Asian), and exotic (Knock-Out Barrier) options under both constant volatility (Geometric Brownian Motion) and stochastic volatility (Heston Model) dynamics.

It incorporates **Quasi-Monte Carlo (Sobol low-discrepancy sequences)** for variance reduction and validates convergence bounds against analytical **Black-Scholes-Merton** solutions.

---

## Key Features

* **Decoupled Architecture:** Implements the Strategy design pattern to separate stochastic dynamics (`StochasticProcess`), payoff structures (`OptionPayoff`), and execution (`MonteCarloEngine`).
* **Multi-Model Dynamics:**
  * **Geometric Brownian Motion (GBM):** Analytical log-Euler increments with Itô drift correction.
  * **Heston Stochastic Volatility:** Full-truncation Euler-Maruyama discretization capturing mean-reverting variance and spot-volatility leverage effects ($\rho$).
* **Diverse Payoff Support:**
  * European Vanilla (Call/Put)
  * Arithmetic Asian (Path-averaged monitoring)
  * Up-and-Out Barrier (Continuous trajectory threshold monitoring)
* **Quasi-Monte Carlo Sampling:** Low-discrepancy Sobol sequences via `scipy.stats.qmc` to accelerate numerical convergence from $\mathcal{O}(1/\sqrt{N})$ toward $\mathcal{O}(1/N)$.
* **Risk Greeks:** First-order sensitivity calculation ($\Delta$) via central finite differences under simulation-noise bounds.
* **Interactive Dashboard:** Full Streamlit web interface with real-time parameter tweaking, trajectory plots, terminal distributions, and convergence benchmarks.

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
