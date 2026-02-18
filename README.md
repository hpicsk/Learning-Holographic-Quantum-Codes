# Learning Holographic Quantum Codes

**Code Geometry Constrains Krylov Complexity Dynamics**

## Overview

A unified machine learning framework that reveals how quantum error-correcting code geometry constrains Krylov complexity dynamics, providing computational evidence for holographic intuitions connecting code distance to constrained operator evolution.

The framework combines:
- **Hypergraph GNN** for stabilizer code property prediction (distance MAE 0.10, 96.4% accuracy)
- **DeepONet** conditioned on GNN embeddings for Krylov complexity trajectory prediction ($R^2 = 0.67$)
- **Partial correlation analysis** revealing geometry--complexity anti-correlations ($r_{\text{partial}} = -0.60$, $p = 5.1 \times 10^{-65}$)

Across 661 stabilizer codes ($n \leq 12$), we show that code distance constrains complexity growth, with within-size correlations strengthening to $r = -0.91$ at $n=12$.

## Project Structure

```
holographic_qec/
├── codes/
│   ├── happy_codes.py        # HaPPY holographic code generation
│   ├── stabilizer.py         # Stabilizer code utilities
│   └── dataset.py            # Dataset generation for 661 codes
├── gnn/
│   ├── hypergraph_conv.py    # Hypergraph convolution layers
│   ├── code_generator.py     # Conditional code generation
│   └── embeddings.py         # Geometric embeddings
├── dynamics/
│   ├── krylov.py             # Lanczos algorithm & C_K(t)
│   ├── deeponet.py           # DeepONet architecture
│   └── hamiltonian.py        # Hamiltonian builders (XXZ, Ising, random)
├── analysis/
│   ├── correlation.py        # Partial correlation analysis
│   └── holographic.py        # AdS/CFT dictionary tests
└── utils/
    └── visualization.py      # Plotting utilities

scripts/
├── prototype.py              # Prototype validation
├── run_experiments.py        # Full experiment pipeline
├── train_gnn.py              # GNN training
├── train_deeponet.py         # DeepONet training
├── analyze_correlation.py    # Correlation analysis
├── analyze_extended.py       # Extended analysis
├── analyze_weaknesses.py     # Robustness checks
└── generate_paper_figures.py # Paper figure generation

paper/                        # NeurIPS 2026 manuscript
configs/
└── default.yaml              # Hyperparameters
```

## Installation

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

```bash
# Run the full experiment pipeline
python scripts/run_experiments.py

# Or run individual components:
python scripts/train_gnn.py              # Train hypergraph GNN
python scripts/train_deeponet.py         # Train DeepONet
python scripts/analyze_correlation.py    # Correlation analysis
python scripts/generate_paper_figures.py # Reproduce paper figures
```

## Key Results

| Component | Metric | Value |
|-----------|--------|-------|
| GNN distance prediction | MAE | 0.10 |
| GNN distance prediction | Accuracy | 96.4% |
| DeepONet (full) | $R^2$ | 0.67 |
| DeepONet (no code info) | $R^2$ | -10.1 |
| Distance vs. max $C_K$ | $r_{\text{partial}}$ | -0.60 |
| Within-size ($n=12$) | $r(d, \alpha)$ | -0.91 |
| Between-code variance | share | 99.8% |

## Dependencies

- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- Qiskit >= 1.0.0
- NumPy, SciPy, Matplotlib

## Citation

If you use this code in your research, please cite:


## License

MIT License

## Contact

For questions or collaboration, please open an issue or contact krml919@korea.ac.kr.
