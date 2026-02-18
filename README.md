# Learning Holographic Quantum Codes

Neural Networks for Emergent Geometry and Complexity Dynamics

## Overview

This project implements a unified framework combining:
- **GNN-based holographic code design** (Proposal 1)
- **DeepONet-based Krylov complexity learning** (Proposal 3)
- **Geometry-complexity correlation analysis** for AdS/CFT testing

## Project Structure

```
holographic_qec/
├── codes/                    # Quantum code generation
│   ├── happy_codes.py        # HaPPY holographic codes
│   ├── stabilizer.py         # Stabilizer code utilities
│   └── dataset.py            # Dataset generation (coming soon)
├── gnn/                      # Graph neural networks
│   ├── hypergraph_conv.py    # Hypergraph convolution layers
│   ├── code_generator.py     # Conditional generation (coming soon)
│   └── embeddings.py         # Geometric embeddings (coming soon)
├── dynamics/                 # Complexity dynamics
│   ├── krylov.py             # Lanczos algorithm & C_K(t)
│   ├── deeponet.py           # DeepONet architecture (coming soon)
│   └── hamiltonian.py        # Hamiltonian builders (coming soon)
├── analysis/                 # Correlation analysis
│   ├── correlation.py        # Statistical analysis (coming soon)
│   └── holographic.py        # AdS/CFT dictionary (coming soon)
└── utils/                    # Utilities
    ├── metrics.py            # Evaluation metrics (coming soon)
    └── visualization.py      # Plotting (coming soon)

scripts/
├── prototype.py              # Prototype validation
├── train_gnn.py              # GNN training (coming soon)
├── train_deeponet.py         # DeepONet training (coming soon)
└── analyze_correlation.py    # Analysis script (coming soon)

configs/
└── default.yaml              # Hyperparameters (coming soon)
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the prototype validation script:

```bash
python scripts/prototype.py
```

This will:
1. Generate HaPPY holographic codes
2. Train a simple GNN to predict code distance
3. Compute Krylov complexity for test Hamiltonians
4. Analyze geometry-complexity correlations
5. Generate visualization plots in `results/`

## Core Components

### 1. HaPPY Code Generator

```python
from holographic_qec.codes import create_happy_code

# Create depth-2 HaPPY code
code = create_happy_code(depth=2)
print(f"Code: {code.stabilizer_code.code_parameters}")
print(f"Geometric features: {code.get_geometric_features()}")
```

### 2. Hypergraph GNN

```python
from holographic_qec.gnn import HypergraphNN, CodeDistancePredictor

# Create model
model = CodeDistancePredictor(
    node_features=4,
    hidden_dim=64,
    num_layers=3
)

# Convert code to graph format
node_feat, edge_idx, hyperedges = code.stabilizer_code.to_graph()

# Predict distance
pred = model(torch.tensor(node_feat), hyperedges)
```

### 3. Krylov Complexity

```python
from holographic_qec.dynamics import (
    compute_krylov_complexity,
    build_xxz_hamiltonian
)

# Build Hamiltonian
H = build_xxz_hamiltonian(n_qubits=6, J_xy=1.0, J_z=0.5)

# Compute complexity curve
result = compute_krylov_complexity(H, psi0, t_max=10.0)
print(f"Growth exponent: {result.growth_exponent}")
print(f"Saturation value: {result.saturation_value}")
```

## Research Phases

### Phase 1: Code Design via GNN (Weeks 1-6)
- [x] Basic HaPPY code generation
- [x] Hypergraph convolution layers
- [x] Code distance prediction
- [ ] Full dataset generation (5000+ codes)
- [ ] Conditional code generation
- [ ] Geometric embedding extraction

### Phase 2: Complexity Dynamics via DeepONet (Weeks 7-10)
- [x] Krylov complexity computation (Lanczos)
- [x] XXZ and Ising Hamiltonians
- [ ] DeepONet architecture
- [ ] Physics-informed constraints
- [ ] Training pipeline

### Phase 3: Geometry-Dynamics Analysis (Weeks 11-14)
- [x] Basic correlation analysis
- [ ] Comprehensive statistical tests
- [ ] Phase transition detection
- [ ] AdS/CFT dictionary validation

### Phase 4: Hardware Validation (Future)
- [ ] NISQ implementation
- [ ] Noise characterization
- [ ] Prediction validation

## Key Innovations

1. **Conditional Code Generation**: First GNN-based inverse design for holographic codes
2. **Physics-Constrained Learning**: DeepONet respects stabilizer code structure
3. **Empirical Holography**: First computational test of geometry-complexity correspondence

## Dependencies

- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- Qiskit >= 1.0.0
- TensorNetwork >= 0.4.0
- NumPy, SciPy, Matplotlib

## Citation

If you use this code in your research, please cite:

```bibtex
@software{holographic_qec,
  title={Learning Holographic Quantum Codes},
  author={PhD Research Project},
  year={2026},
  url={https://github.com/xxx/holographic-qec}
}
```

## License

MIT License

## Contact

For questions or collaboration, please open an issue.
