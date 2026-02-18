
import time
import numpy as np
import torch
from holographic_qec.dynamics.krylov import compute_krylov_complexity, build_xxz_hamiltonian

def benchmark(n):
    print(f"Benchmarking n={n}...")
    H = build_xxz_hamiltonian(n, J_xy=1.0, J_z=1.0, h=0.5)
    psi0 = np.random.randn(2**n) + 1j * np.random.randn(2**n)
    psi0 /= np.linalg.norm(psi0)
    
    start = time.time()
    result = compute_krylov_complexity(H, psi0, t_max=10.0, n_steps=100, max_krylov_dim=100)
    end = time.time()
    print(f"n={n} took {end-start:.2f} seconds")
    return end-start

if __name__ == "__main__":
    for n in [10, 11, 12]:
        benchmark(n)
