"""
DeepONet Architecture for Krylov Complexity Learning

Implements the DeepONet (Deep Operator Network) architecture for
learning the mapping from (code geometry, Hamiltonian parameters)
to Krylov complexity dynamics C_K(t).

Architecture:
- Branch Network: Encodes code geometry and Hamiltonian parameters
- Trunk Network: Encodes time coordinate with Fourier features
- Output: C_K(t) via bilinear combination

Physics-informed constraints:
- C_K(t) >= 0 (non-negativity)
- dC_K/dt >= 0 for early times (monotonicity)
- Bounded growth rate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np
import math


class FourierFeatures(nn.Module):
    """
    Fourier feature encoding for continuous coordinates.

    Maps scalar input t to high-dimensional sinusoidal features,
    allowing the network to learn high-frequency functions.

    Based on: Tancik et al., "Fourier Features Let Networks Learn
    High Frequency Functions in Low Dimensional Domains" (2020)
    """

    def __init__(
        self,
        input_dim: int = 1,
        embed_dim: int = 64,
        scale: float = 10.0,
        learnable: bool = False
    ):
        """
        Args:
            input_dim: Dimension of input (1 for time)
            embed_dim: Output dimension (must be even)
            scale: Frequency scale parameter
            learnable: Whether to learn the frequency matrix
        """
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.scale = scale

        # Random frequency matrix
        B = torch.randn(input_dim, embed_dim // 2) * scale
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature encoding.

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            Encoded tensor (..., embed_dim)
        """
        # Project input
        x_proj = x @ self.B  # (..., embed_dim // 2)

        # Sinusoidal encoding
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class BranchNetwork(nn.Module):
    """
    Branch network for encoding problem parameters.

    Takes code embeddings and Hamiltonian parameters as input,
    outputs a vector of basis function coefficients.
    """

    def __init__(
        self,
        code_dim: int = 128,
        hamiltonian_dim: int = 8,
        hidden_dim: int = 128,
        num_basis: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_dim = code_dim + hamiltonian_dim
        self.num_basis = num_basis

        layers = []
        in_dim = self.input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_basis))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        code_embedding: torch.Tensor,
        hamiltonian_params: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode problem to basis coefficients.

        Args:
            code_embedding: Code geometry embedding (batch, code_dim)
            hamiltonian_params: Hamiltonian parameters (batch, hamiltonian_dim)

        Returns:
            Basis coefficients (batch, num_basis)
        """
        x = torch.cat([code_embedding, hamiltonian_params], dim=-1)
        return self.network(x)


class TrunkNetwork(nn.Module):
    """
    Trunk network for encoding time coordinate.

    Uses Fourier features followed by MLP to output basis functions
    evaluated at time t.
    """

    def __init__(
        self,
        trunk_dim: int = 64,
        hidden_dim: int = 128,
        num_basis: int = 64,
        num_layers: int = 3,
        fourier_scale: float = 10.0,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_basis = num_basis

        # Fourier feature encoding for time
        self.fourier = FourierFeatures(
            input_dim=1,
            embed_dim=trunk_dim,
            scale=fourier_scale
        )

        # MLP
        layers = []
        in_dim = trunk_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_basis))

        self.network = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time to basis function values.

        Args:
            t: Time values (batch, 1) or (batch,)

        Returns:
            Basis function values (batch, num_basis)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t_encoded = self.fourier(t)
        return self.network(t_encoded)


class KrylovDeepONet(nn.Module):
    """
    DeepONet for learning Krylov complexity dynamics.

    Architecture:
    - Branch: Encodes (code_embedding, hamiltonian_params) -> coefficients
    - Trunk: Encodes time t -> basis functions
    - Output: C_K(t) = <branch, trunk> + bias

    Physics-informed constraints are applied during training via
    the loss function.
    """

    def __init__(
        self,
        code_dim: int = 128,
        hamiltonian_dim: int = 8,
        trunk_dim: int = 64,
        hidden_dim: int = 128,
        num_basis: int = 64,
        num_layers: int = 3,
        fourier_scale: float = 10.0,
        dropout: float = 0.3,
        use_softplus: bool = False
    ):
        """
        Args:
            code_dim: Dimension of code embeddings (from GNN)
            hamiltonian_dim: Dimension of Hamiltonian parameters
            trunk_dim: Dimension after Fourier encoding
            hidden_dim: Hidden layer dimension
            num_basis: Number of basis functions
            num_layers: Number of layers in branch/trunk
            fourier_scale: Scale for Fourier features
            dropout: Dropout rate (default increased to 0.3 for regularization)
            use_softplus: Whether to apply softplus (disable when using normalized targets)
        """
        super().__init__()
        self.code_dim = code_dim
        self.hamiltonian_dim = hamiltonian_dim
        self.num_basis = num_basis
        self.use_softplus = use_softplus

        # Branch network
        self.branch = BranchNetwork(
            code_dim=code_dim,
            hamiltonian_dim=hamiltonian_dim,
            hidden_dim=hidden_dim,
            num_basis=num_basis,
            num_layers=num_layers,
            dropout=dropout
        )

        # Trunk network
        self.trunk = TrunkNetwork(
            trunk_dim=trunk_dim,
            hidden_dim=hidden_dim,
            num_basis=num_basis,
            num_layers=num_layers,
            fourier_scale=fourier_scale,
            dropout=dropout
        )

        # Learnable bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        code_embedding: torch.Tensor,
        hamiltonian_params: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict Krylov complexity at time t.

        Args:
            code_embedding: Code geometry embedding (batch, code_dim)
            hamiltonian_params: Hamiltonian parameters (batch, hamiltonian_dim)
            t: Time values (batch, 1) or (batch,)

        Returns:
            Predicted C_K(t) (batch, 1)
        """
        # Branch: encode problem
        branch_out = self.branch(code_embedding, hamiltonian_params)

        # Trunk: encode time
        trunk_out = self.trunk(t)

        # Combine via dot product
        output = torch.sum(branch_out * trunk_out, dim=-1, keepdim=True)
        output = output + self.bias

        # Apply softplus only if enabled (disable when using normalized targets)
        if self.use_softplus:
            output = F.softplus(output)

        return output

    def predict_trajectory(
        self,
        code_embedding: torch.Tensor,
        hamiltonian_params: torch.Tensor,
        times: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict complexity over a trajectory of times.

        Args:
            code_embedding: Single code embedding (code_dim,)
            hamiltonian_params: Single Hamiltonian params (hamiltonian_dim,)
            times: Time array (n_times,)

        Returns:
            Predicted C_K trajectory (n_times,)
        """
        n_times = times.size(0)

        # Expand inputs
        code_expanded = code_embedding.unsqueeze(0).expand(n_times, -1)
        params_expanded = hamiltonian_params.unsqueeze(0).expand(n_times, -1)

        # Predict
        predictions = self.forward(code_expanded, params_expanded, times)

        return predictions.squeeze(-1)


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function for DeepONet training.

    Combines:
    - Data fitting loss (MSE)
    - Non-negativity constraint (already in model via softplus)
    - Monotonicity constraint (dC_K/dt >= 0 for early times)
    - Growth rate bound
    """

    def __init__(
        self,
        monotonicity_weight: float = 1.0,
        growth_rate_weight: float = 0.1,
        smoothness_weight: float = 0.1,
        dt: float = 0.01
    ):
        """
        Args:
            monotonicity_weight: Weight for monotonicity violation (increased from 0.1)
            growth_rate_weight: Weight for growth rate regularization (increased from 0.01)
            smoothness_weight: Weight for smoothness regularization (new)
            dt: Time step for numerical derivative
        """
        super().__init__()
        self.monotonicity_weight = monotonicity_weight
        self.growth_rate_weight = growth_rate_weight
        self.smoothness_weight = smoothness_weight
        self.dt = dt

    def forward(
        self,
        model: KrylovDeepONet,
        code_embedding: torch.Tensor,
        hamiltonian_params: torch.Tensor,
        t: torch.Tensor,
        C_true: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute physics-informed loss.

        Args:
            model: DeepONet model
            code_embedding: Code embeddings (batch, code_dim)
            hamiltonian_params: Hamiltonian params (batch, hamiltonian_dim)
            t: Time values (batch, 1)
            C_true: Ground truth complexity (batch, 1)

        Returns:
            (total_loss, loss_dict)
        """
        losses = {}

        # Prediction at current time
        C_pred = model(code_embedding, hamiltonian_params, t)

        # Data fitting loss
        mse_loss = F.mse_loss(C_pred, C_true)
        losses['mse'] = mse_loss

        # Monotonicity constraint: dC/dt >= 0
        # Compute C at t + dt
        t_forward = t + self.dt
        C_forward = model(code_embedding, hamiltonian_params, t_forward)

        # Penalize decreasing complexity
        monotonicity_violation = F.relu(C_pred - C_forward)  # Positive if decreasing
        mono_loss = monotonicity_violation.mean()
        losses['monotonicity'] = mono_loss

        # Growth rate regularization
        # dC/dt should not be too large (prevents explosion)
        growth_rate = (C_forward - C_pred) / self.dt
        growth_loss = F.relu(growth_rate - 10.0).mean()  # Penalize if > 10
        losses['growth_rate'] = growth_loss

        # Smoothness constraint: penalize large second derivatives (prevents oscillations)
        t_backward = torch.clamp(t - self.dt, min=0.0)
        C_backward = model(code_embedding, hamiltonian_params, t_backward)

        # Approximate second derivative: (C_forward - 2*C_pred + C_backward) / dt^2
        second_deriv = (C_forward - 2 * C_pred + C_backward) / (self.dt ** 2)
        smoothness_loss = torch.abs(second_deriv).mean()
        losses['smoothness'] = smoothness_loss

        # Total loss
        total_loss = (
            mse_loss +
            self.monotonicity_weight * mono_loss +
            self.growth_rate_weight * growth_loss +
            self.smoothness_weight * smoothness_loss
        )
        losses['total'] = total_loss

        return total_loss, losses


class ComplexityDataset(torch.utils.data.Dataset):
    """
    Dataset for training DeepONet on Krylov complexity data.
    """

    def __init__(
        self,
        code_embeddings: torch.Tensor,
        hamiltonian_params: torch.Tensor,
        times: torch.Tensor,
        complexity_values: torch.Tensor
    ):
        """
        Args:
            code_embeddings: (n_samples, code_dim)
            hamiltonian_params: (n_samples, hamiltonian_dim)
            times: (n_samples, n_times) or (n_samples,)
            complexity_values: (n_samples, n_times) or (n_samples,)
        """
        self.code_embeddings = code_embeddings
        self.hamiltonian_params = hamiltonian_params
        self.times = times
        self.complexity_values = complexity_values

        # Flatten if necessary
        if times.dim() == 2:
            n_samples, n_times = times.shape
            self.code_embeddings = code_embeddings.unsqueeze(1).expand(-1, n_times, -1).reshape(-1, code_embeddings.size(-1))
            self.hamiltonian_params = hamiltonian_params.unsqueeze(1).expand(-1, n_times, -1).reshape(-1, hamiltonian_params.size(-1))
            self.times = times.reshape(-1, 1)
            self.complexity_values = complexity_values.reshape(-1, 1)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        return {
            'code_embedding': self.code_embeddings[idx],
            'hamiltonian_params': self.hamiltonian_params[idx],
            't': self.times[idx],
            'C_true': self.complexity_values[idx]
        }


class EnhancedDeepONet(nn.Module):
    """
    Enhanced DeepONet with additional features:
    - Skip connections
    - Multi-scale Fourier features
    - Attention between branch and trunk
    """

    def __init__(
        self,
        code_dim: int = 128,
        hamiltonian_dim: int = 8,
        hidden_dim: int = 128,
        num_basis: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.code_dim = code_dim
        self.hamiltonian_dim = hamiltonian_dim
        self.hidden_dim = hidden_dim
        self.num_basis = num_basis

        # Input projections
        self.code_proj = nn.Linear(code_dim, hidden_dim)
        self.hamiltonian_proj = nn.Linear(hamiltonian_dim, hidden_dim)

        # Multi-scale Fourier features for time
        self.fourier_scales = [1.0, 5.0, 10.0, 50.0]
        self.fourier_dim = 16  # Per scale
        total_fourier_dim = len(self.fourier_scales) * self.fourier_dim * 2

        self.fourier_layers = nn.ModuleList([
            FourierFeatures(1, self.fourier_dim * 2, scale=s)
            for s in self.fourier_scales
        ])
        self.time_proj = nn.Linear(total_fourier_dim, hidden_dim)

        # Branch network with skip connections
        self.branch_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.branch_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))

        # Trunk network with skip connections
        self.trunk_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.trunk_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))

        # Cross-attention between branch and trunk
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        code_embedding: torch.Tensor,
        hamiltonian_params: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with enhanced architecture."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Project inputs
        code_h = self.code_proj(code_embedding)
        ham_h = self.hamiltonian_proj(hamiltonian_params)

        # Multi-scale Fourier features for time
        fourier_feats = [layer(t) for layer in self.fourier_layers]
        time_feat = torch.cat(fourier_feats, dim=-1)
        time_h = self.time_proj(time_feat)

        # Branch processing
        branch_h = torch.cat([code_h, ham_h], dim=-1)
        for layer in self.branch_layers:
            branch_h = layer(branch_h)
            branch_h = torch.cat([branch_h, ham_h], dim=-1)
        branch_h = branch_h[:, :self.hidden_dim]  # Take first hidden_dim

        # Trunk processing
        trunk_h = time_h
        for layer in self.trunk_layers:
            trunk_h = layer(trunk_h) + time_h  # Skip connection

        # Cross-attention
        # branch_h and trunk_h: (batch, hidden_dim)
        branch_h = branch_h.unsqueeze(1)  # (batch, 1, hidden_dim)
        trunk_h = trunk_h.unsqueeze(1)

        attn_out, _ = self.cross_attention(trunk_h, branch_h, branch_h)
        combined = (attn_out + trunk_h).squeeze(1)

        # Output
        output = self.output_layer(combined) + self.bias

        # Ensure non-negativity
        output = F.softplus(output)

        return output


def create_deeponet(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create DeepONet from config.

    Args:
        config: Configuration dict with model parameters

    Returns:
        Initialized DeepONet model
    """
    model_type = config.get('type', 'standard')

    if model_type == 'standard':
        return KrylovDeepONet(
            code_dim=config.get('code_dim', 128),
            hamiltonian_dim=config.get('hamiltonian_dim', 8),
            trunk_dim=config.get('trunk_dim', 64),
            hidden_dim=config.get('hidden_dim', 128),
            num_basis=config.get('num_basis', 64),
            num_layers=config.get('num_layers', 3),
            fourier_scale=config.get('fourier_scale', 10.0),
            dropout=config.get('dropout', 0.3),
            use_softplus=config.get('use_softplus', False)
        )
    elif model_type == 'enhanced':
        return EnhancedDeepONet(
            code_dim=config.get('code_dim', 128),
            hamiltonian_dim=config.get('hamiltonian_dim', 8),
            hidden_dim=config.get('hidden_dim', 128),
            num_basis=config.get('num_basis', 64),
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown DeepONet type: {model_type}")


if __name__ == '__main__':
    print("Testing DeepONet architectures...")

    # Test standard DeepONet
    model = KrylovDeepONet(
        code_dim=64,
        hamiltonian_dim=8,
        hidden_dim=64,
        num_basis=32
    )

    batch_size = 16
    code_emb = torch.randn(batch_size, 64)
    ham_params = torch.randn(batch_size, 8)
    t = torch.rand(batch_size, 1) * 10

    output = model(code_emb, ham_params, t)
    print(f"Standard DeepONet output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Test trajectory prediction
    times = torch.linspace(0, 10, 100)
    trajectory = model.predict_trajectory(code_emb[0], ham_params[0], times)
    print(f"Trajectory shape: {trajectory.shape}")

    # Test physics-informed loss
    loss_fn = PhysicsInformedLoss()
    C_true = torch.rand(batch_size, 1) * 10

    loss, losses = loss_fn(model, code_emb, ham_params, t, C_true)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss components: MSE={losses['mse'].item():.4f}, Mono={losses['monotonicity'].item():.4f}")

    # Test enhanced DeepONet
    enhanced_model = EnhancedDeepONet(code_dim=64, hamiltonian_dim=8, hidden_dim=64)
    enhanced_out = enhanced_model(code_emb, ham_params, t)
    print(f"Enhanced DeepONet output shape: {enhanced_out.shape}")
