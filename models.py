from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import qmc, norm


class StochasticProcess(ABC):
    @abstractmethod
    def simulate(self, n_paths: int, n_steps: int, dt: float, use_sobol: bool = False) -> np.ndarray:
        """
        Simulate asset price paths.
        Returns:
            np.ndarray of shape (n_paths, n_steps + 1)
        """
        pass


class GeometricBrownianMotion(StochasticProcess):
    def __init__(self, mu: float, sigma: float, s0: float):
        self.mu = mu
        self.sigma = sigma
        self.s0 = s0

    def simulate(self, n_paths: int, n_steps: int, dt: float, use_sobol: bool = False) -> np.ndarray:
        if use_sobol:
            # Low-discrepancy Sobol sequence (requires power of 2 paths)
            power = int(np.ceil(np.log2(n_paths)))
            sampler = qmc.Sobol(d=n_steps, scramble=True)
            u = sampler.random_base2(m=power)
            u = np.clip(u, 1e-7, 1.0 - 1e-7)
            z = norm.ppf(u)
        else:
            z = np.random.normal(size=(n_paths, n_steps))

        actual_paths = z.shape[0]

        # Calculate exact log-increments
        drift = (self.mu - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt) * z
        increments = drift + diffusion

        # Build trajectory matrix
        paths = np.zeros((actual_paths, n_steps + 1))
        paths[:, 0] = self.s0
        paths[:, 1:] = self.s0 * np.exp(np.cumsum(increments, axis=1))

        return paths


class HestonModel(StochasticProcess):
    """
    Heston Stochastic Volatility Model:
      dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_t^S
      dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_t^v
    """
    def __init__(self, mu: float, kappa: float, theta: float, xi: float, rho: float, s0: float, v0: float):
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi          # Volatility of variance (vol-of-vol)
        self.rho = rho        # Asset/variance correlation
        self.s0 = s0
        self.v0 = v0          # Initial variance

    def simulate(self, n_paths: int, n_steps: int, dt: float, use_sobol: bool = False) -> np.ndarray:
        sqrt_dt = np.sqrt(dt)

        # Generate correlated Brownian shocks
        z1 = np.random.normal(size=(n_paths, n_steps))
        z2_raw = np.random.normal(size=(n_paths, n_steps))
        z2 = self.rho * z1 + np.sqrt(1.0 - self.rho ** 2) * z2_raw

        paths = np.zeros((n_paths, n_steps + 1))
        variances = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.s0
        variances[:, 0] = self.v0

        # Full-truncation Euler-Maruyama scheme
        for t in range(1, n_steps + 1):
            v_prev = np.maximum(variances[:, t - 1], 0.0)
            sqrt_v = np.sqrt(v_prev)

            # Variance process update
            dv = (self.kappa * (self.theta - v_prev) * dt +
                  self.xi * sqrt_v * sqrt_dt * z2[:, t - 1])
            variances[:, t] = np.maximum(variances[:, t - 1] + dv, 0.0)

            # Asset price update (Log-Euler step)
            log_increment = (self.mu - 0.5 * v_prev) * dt + sqrt_v * sqrt_dt * z1[:, t - 1]
            paths[:, t] = paths[:, t - 1] * np.exp(log_increment)

        return paths