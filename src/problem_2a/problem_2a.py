from typing import Any
import numpy as np


def magnetization(lattice: np.ndarray[float, Any]) -> float:
    return np.sum(lattice) / np.size(lattice)


def energy(lattice: np.ndarray[float, Any]) -> float:
    offset_lattice = np.zeros_like(lattice)
    offset_lattice[0, :] = lattice[-1, :]
    offset_lattice[:, 0] = lattice[:, -1]
    offset_lattice[1:-1, 1:-1] = lattice[:-2, :-2]

    return -float(np.sum(lattice * offset_lattice))


def delta_E(lattice: np.ndarray[float, Any], indices: tuple) -> float:
    i, j = indices
    bound = lattice.shape[0]
    return float(-2*lattice[i, j] * (
        lattice[(i-1) % bound, j] 
        + lattice[(i+1) % bound, j] 
        + lattice[i, (j-1) % bound] 
        + lattice[i, (j+1) % bound]
        ))



def monte_carlo(lattice: np.ndarray[float, Any], Temperature: float, steps: int):
    for _ in range(steps):
        i, j = np.random.randint(low=0, high=lattice.shape[0], size=2)

        dE = delta_E(lattice, (i, j))
        
        if dE <= 0 or np.random.random() <= np.exp(-dE / Temperature):
            lattice[i, j] *= -1


def run():...
