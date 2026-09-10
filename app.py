import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

from models import GeometricBrownianMotion, HestonModel
from payoffs import EuropeanCallPayoff, AsianArithmeticCallPayoff, UpAndOutBarrierCallPayoff
from engine import MonteCarloEngine

# Page configuration
st.set_page_config(
    page_title="Monte Carlo Option Pricer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Analytical Black-Scholes benchmark
def black_scholes_call(s0, k, t, r, sigma):
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)

st.title("Quantitative Monte Carlo Option Pricing Engine")
st.markdown("Interactive pricing workbench comparing **Geometric Brownian Motion** and **Heston Stochastic Volatility**.")
st.sidebar.header("Market Parameters")
s0 = st.sidebar.slider("Spot Price ($S_0$)", min_value=50.0, max_value=200.0, value=100.0, step=1.0)
strike = st.sidebar.slider("Strike Price ($K$)", min_value=50.0, max_value=200.0, value=100.0, step=1.0)
t_years = st.sidebar.slider("Maturity ($T$ in years)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
r = st.sidebar.slider("Risk-Free Rate ($r$)", min_value=0.0, max_value=0.15, value=0.05, step=0.005)
sigma = st.sidebar.slider("Volatility ($\sigma$)", min_value=0.05, max_value=0.60, value=0.20, step=0.01)

st.sidebar.header("Exotics & Simulation Setup")
barrier = st.sidebar.slider("Barrier Level ($B$)", min_value=float(strike), max_value=250.0, value=120.0, step=5.0)
n_paths = st.sidebar.select_slider(
    "Simulation Paths ($N$)",
    options=[4096, 8192, 16384, 32768, 65536],
    value=16384
)
n_steps = st.sidebar.slider("Time Steps ($M$)", min_value=50, max_value=365, value=252, step=1)
use_sobol = st.sidebar.checkbox("Use Sobol Quasi-Monte Carlo", value=True)

gbm = GeometricBrownianMotion(mu=r, sigma=sigma, s0=s0)
heston = HestonModel(mu=r, kappa=2.0, theta=sigma**2, xi=0.3, rho=-0.7, s0=s0, v0=sigma**2)

euro_call = EuropeanCallPayoff(strike=strike)
asian_call = AsianArithmeticCallPayoff(strike=strike)
barrier_call = UpAndOutBarrierCallPayoff(strike=strike, barrier=barrier)

# Compute Analytical Benchmark
bs_benchmark = black_scholes_call(s0, strike, t_years, r, sigma)

with st.spinner("Simulating price paths and evaluating contracts..."):
    t0 = time.perf_counter()

    # GBM Engines
    gbm_euro = MonteCarloEngine(gbm, euro_call, r, t_years)
    gbm_asian = MonteCarloEngine(gbm, asian_call, r, t_years)
    gbm_barrier = MonteCarloEngine(gbm, barrier_call, r, t_years)

    p_euro, err_euro = gbm_euro.price(n_paths, n_steps, use_sobol=use_sobol)
    p_asian, err_asian = gbm_asian.price(n_paths, n_steps, use_sobol=use_sobol)
    p_barr, err_barr = gbm_barrier.price(n_paths, n_steps, use_sobol=use_sobol)

    # Heston Engines
    h_euro = MonteCarloEngine(heston, euro_call, r, t_years)
    h_asian = MonteCarloEngine(heston, asian_call, r, t_years)
    h_barr = MonteCarloEngine(heston, barrier_call, r, t_years)

    hp_euro, herr_euro = h_euro.price(n_paths, n_steps)
    hp_asian, herr_asian = h_asian.price(n_paths, n_steps)
    hp_barr, herr_barr = h_barr.price(n_paths, n_steps)

    delta = gbm_euro.delta(epsilon=0.01, n_paths=20_000, n_steps=100)
    elapsed = time.perf_counter() - t0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Black-Scholes Exact", f"${bs_benchmark:.3f}")
col2.metric("European Call (MC)", f"${p_euro:.3f}", f"±${err_euro:.3f}")
col3.metric("Option Delta (Δ)", f"{delta:.4f}")
col4.metric("Engine Runtime", f"{elapsed:.2f}s")

st.markdown("---")

st.subheader("Price Summary Across Models")
summary_data = {
    "Contract": ["European Call", "Asian Arithmetic Call", f"Up-and-Out Barrier (B={barrier})"],
    "GBM Price": [f"${p_euro:.3f} (±{err_euro:.3f})", f"${p_asian:.3f} (±{err_asian:.3f})", f"${p_barr:.3f} (±{err_barr:.3f})"],
    "Heston Price": [f"${hp_euro:.3f} (±{herr_euro:.3f})", f"${hp_asian:.3f} (±{herr_asian:.3f})", f"${hp_barr:.3f} (±{herr_barr:.3f})"],
    "Analytic Benchmark": [f"${bs_benchmark:.3f}", "No Closed Form", "Reflect. Principle"]
}
st.table(summary_data)

tab1, tab2, tab3 = st.tabs(["Price Paths", "Terminal Distribution", "Convergence"])

dt = t_years / n_steps
time_grid = np.linspace(0, t_years, n_steps + 1)

# Tab 1: Paths Plot
with tab1:
    n_display = 35
    paths_gbm = gbm.simulate(n_paths=n_display, n_steps=n_steps, dt=dt)
    paths_heston = heston.simulate(n_paths=n_display, n_steps=n_steps, dt=dt)

    fig1, (ax_g, ax_h) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    ax_g.plot(time_grid, paths_gbm.T, color="steelblue", alpha=0.4, linewidth=1)
    ax_g.axhline(strike, color="black", linestyle="--", label=f"Strike K={strike}")
    ax_g.axhline(barrier, color="crimson", linestyle=":", label=f"Barrier B={barrier}")
    ax_g.set_title(f"GBM ({n_display} Paths)", fontweight="bold")
    ax_g.set_xlabel("Years")
    ax_g.set_ylabel("Asset Price ($)")
    ax_g.legend(loc="upper left")
    ax_g.grid(True, alpha=0.3)

    ax_h.plot(time_grid, paths_heston.T, color="darkorange", alpha=0.4, linewidth=1)
    ax_h.axhline(strike, color="black", linestyle="--", label=f"Strike K={strike}")
    ax_h.axhline(barrier, color="crimson", linestyle=":", label=f"Barrier B={barrier}")
    ax_h.set_title(f"Heston Model ({n_display} Paths)", fontweight="bold")
    ax_h.set_xlabel("Years")
    ax_h.legend(loc="upper left")
    ax_h.grid(True, alpha=0.3)

    st.pyplot(fig1)
    plt.close(fig1)

# Tab 2: Terminal Price Distribution
with tab2:
    terminal_paths = gbm.simulate(n_paths=15_000, n_steps=n_steps, dt=dt)
    st_prices = terminal_paths[:, -1]

    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    ax2.hist(st_prices, bins=70, density=True, color="steelblue", alpha=0.6, edgecolor="white")
    ax2.axvline(strike, color="crimson", linestyle="--", label=f"Strike K={strike}")
    ax2.axvline(np.mean(st_prices), color="navy", linestyle=":", label=f"Mean S_T (${np.mean(st_prices):.2f})")
    ax2.set_title("Terminal Price Distribution $S_T$ at Expiry", fontweight="bold")
    ax2.set_xlabel("Price ($)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig2)
    plt.close(fig2)

# Tab 3: Convergence
with tab3:
    st.markdown("Comparing convergence rate toward the Black-Scholes benchmark as sample size increases:")
    test_counts = [512, 1024, 2048, 4096, 8192, 16384]
    p_ests, s_ests = [], []

    for count in test_counts:
        px_p, _ = gbm_euro.price(n_paths=count, n_steps=60, use_sobol=False)
        px_s, _ = gbm_euro.price(n_paths=count, n_steps=60, use_sobol=True)
        p_ests.append(px_p)
        s_ests.append(px_s)

    fig3, ax3 = plt.subplots(figsize=(10, 4.5))
    ax3.axhline(bs_benchmark, color="black", linestyle="--", label=f"Analytical Benchmark (${bs_benchmark:.3f})")
    ax3.plot(test_counts, p_ests, marker="o", label="Standard Pseudo-MC", color="royalblue")
    ax3.plot(test_counts, s_ests, marker="s", label="Sobol Quasi-MC", color="forestgreen")
    ax3.set_xscale("log", base=2)
    ax3.set_title("Estimator Convergence vs Exact Solution", fontweight="bold")
    ax3.set_xlabel("Number of Paths (Log2 Scale)")
    ax3.set_ylabel("Price ($)")
    ax3.legend()
    ax3.grid(True, which="both", alpha=0.3)

    st.pyplot(fig3)
    plt.close(fig3)