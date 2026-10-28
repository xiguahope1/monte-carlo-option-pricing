# Monte Carlo Option Pricer

I built this to learn how Monte Carlo methods handle problems that don't have neat closed-form answers. It prices European, Asian and up-and-out barrier options under two models: Geometric Brownian Motion and the Heston stochastic volatility model.

## What I did
- Simulated GBM using exact log-normal steps, and Heston using full-truncation Euler–Maruyama (so the variance can't go negative).
- Checked the European prices against the Black–Scholes formula to make sure the engine was working before trusting the exotic options.
- Compared standard random sampling with Sobol quasi-random sequences to see how much faster the price converges.
- Estimated Delta using central finite differences.
- Added a small Streamlit dashboard to play with the parameters.

## What I found
- **The engine matches Black–Scholes.** With 131,072 paths, the European call came out at 10.466 ± 0.041 vs the exact 10.451, within one standard error.
- **Sobol helps a lot, but less as the dimension grows.** Over 20 repeated runs, Sobol cut the error by ~300x for a one-step European option, but only ~5–9x once each path had 52–252 time steps (European and Asian). More steps means more dimensions, and quasi-random points lose their edge.
- **Barrier prices depend heavily on the time step.** The up-and-out call fell from 1.86 (12 steps) to 1.35 (252) to 1.26 (1,000 steps), still above thecontinuous-monitoring formula value of 1.18. Checking the barrier only at discrete times misses crossings, so the simulation overprices it.
- **Delta needs a sensible bump size.** A 0.01% bump gave a wildly noisy Delta (standard deviation ~6 across runs). A 1% bump gave 0.646 ± 0.030, close to the Black–Scholes value of 0.637. Independent random paths for the up and down prices make tiny bumps useless.
- **Heston barely moves vanilla prices here but changes the barrier a lot.** European and Asian prices were within noise of GBM, but the barrier option rose from 1.32 to 2.17. With ρ = −0.7, volatility drops as the price rises, so fewer paths hit the barrier.

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
