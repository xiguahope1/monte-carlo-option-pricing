import time
from models import GeometricBrownianMotion, HestonModel
from payoffs import EuropeanCallPayoff, AsianArithmeticCallPayoff, UpAndOutBarrierCallPayoff
from engine import MonteCarloEngine

def main():
    # Parameters for the Geometric Brownian Motion
    mu = 0.05
    sigma = 0.2
    s0 = 100

    # Create a Geometric Brownian Motion process
    gbm_process = GeometricBrownianMotion(mu, sigma, s0)

    # Parameters for the Heston Model
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    v0 = 0.04
    rho = -0.7

    # Create a Heston Model process
    heston_process = HestonModel(mu, kappa, theta, xi, rho, s0, v0)

    # Parameters for the option payoffs
    strike_price = 100
    barrier_price = 120

    # Create different option payoffs
    european_call_payoff = EuropeanCallPayoff(strike_price)
    asian_arithmetic_call_payoff = AsianArithmeticCallPayoff(strike_price)
    up_and_out_barrier_call_payoff = UpAndOutBarrierCallPayoff(strike_price, barrier_price)

    # Risk-free rate and time to maturity
    risk_free_rate = 0.05
    T = 1.0

    # Create Monte Carlo engines for each process and payoff combination
    gbm_european_engine = MonteCarloEngine(gbm_process, european_call_payoff, risk_free_rate, T)
    gbm_asian_engine = MonteCarloEngine(gbm_process, asian_arithmetic_call_payoff, risk_free_rate, T)
    gbm_barrier_engine = MonteCarloEngine(gbm_process, up_and_out_barrier_call_payoff, risk_free_rate, T)

    heston_european_engine = MonteCarloEngine(heston_process, european_call_payoff, risk_free_rate, T)
    heston_asian_engine = MonteCarloEngine(heston_process, asian_arithmetic_call_payoff, risk_free_rate, T)
    heston_barrier_engine = MonteCarloEngine(heston_process, up_and_out_barrier_call_payoff, risk_free_rate, T)

    n_paths = 100000
    n_steps = 252

    runs = [
        ("GBM European Call", gbm_european_engine, heston_european_engine),
        ("GBM Asian Arithmetic Call", gbm_asian_engine, heston_asian_engine),
        ("GBM Up-and-Out Barrier Call", gbm_barrier_engine, heston_barrier_engine),
    ]

    total_start = time.perf_counter()

    for name, gbm_engine, heston_engine in runs:
        gbm_price, gbm_std_error = gbm_engine.price(n_paths, n_steps)
        heston_price, heston_std_error = heston_engine.price(n_paths, n_steps)

    total_elapsed = time.perf_counter() - total_start

    delta = gbm_european_engine.delta(epsilon=1e-4, n_paths=n_paths, n_steps=n_steps)

if __name__ == "__main__":
    main()