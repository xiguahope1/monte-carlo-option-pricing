import numpy as np
from models import StochasticProcess
from payoffs import OptionPayoff


class MonteCarloEngine:
    def __init__(self, process: StochasticProcess, payoff: OptionPayoff, risk_free_rate: float, T: float):
        self.process = process
        self.payoff = payoff
        self.risk_free_rate = risk_free_rate
        self.T = T

    def price(self, n_paths: int = 100_000, n_steps: int = 252, use_sobol: bool = False):
        # Calculate time step dt for models.py
        dt = self.T / n_steps

        # Simulate price trajectories
        paths = self.process.simulate(n_paths, n_steps, dt, use_sobol=use_sobol)

        # Evaluate terminal undiscounted payoffs: shape (n_paths,)
        payoffs = self.payoff.evaluate(paths)

        # Discount payoffs across all individual paths
        discount_factor = np.exp(-self.risk_free_rate * self.T)
        discounted_payoffs = discount_factor * payoffs

        # Estimate expected price and standard error
        price = float(np.mean(discounted_payoffs))
        std_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs)))

        return price, std_error

    def delta(self, epsilon: float = 0.01, n_paths: int = 50_000, n_steps: int = 100):
        """Computes Delta (dV/dS) using central finite differences."""
        # Use proportional bump (e.g. 1%) rather than tiny float to avoid Monte Carlo noise
        original_s0 = self.process.s0
        bump = original_s0 * epsilon

        # Price at S0 + bump
        self.process.s0 = original_s0 + bump
        price_up, _ = self.price(n_paths=n_paths, n_steps=n_steps)

        # Price at S0 - bump
        self.process.s0 = original_s0 - bump
        price_down, _ = self.price(n_paths=n_paths, n_steps=n_steps)

        # Reset original spot price
        self.process.s0 = original_s0

        # Central difference formula
        return (price_up - price_down) / (2.0 * bump)

    # Alias so both calculate_delta and delta work
    calculate_delta = delta