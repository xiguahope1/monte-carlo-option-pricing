import numpy as np
from scipy.stats import qmc, norm       # quasi-Monte Carlo and normal distribution
from abc import ABC, abstractmethod     # abstract base class
import time

class OptionPayoff(ABC):
    @abstractmethod
    def evaluate(self, paths: np.ndarray) -> np.ndarray:
        pass

class EuropeanCallPayoff(OptionPayoff):
    def __init__(self, strike: float):
        self.strike = strike

    def evaluate(self, paths: np.ndarray) -> np.ndarray:
        return np.maximum(paths[:, -1] - self.strike, 0.0)

class AsianArithmeticCallPayoff(OptionPayoff):
    def __init__(self, strike: float):
        self.strike = strike

    def evaluate(self, paths: np.ndarray) -> np.ndarray:
        s_avg = np.mean(paths[:, 1:], axis=1)
        return np.maximum(s_avg - self.strike, 0.0)

class UpAndOutBarrierCallPayoff(OptionPayoff):
    def __init__(self, strike: float, barrier: float):
        self.strike = strike
        self.barrier = barrier

    def evaluate(self, paths: np.ndarray) -> np.ndarray:
        terminal_payoff = np.maximum(paths[:, -1] - self.strike, 0.0)
        knocked_out = np.max(paths, axis=1) >= self.barrier
        return np.where(knocked_out, 0.0, terminal_payoff)