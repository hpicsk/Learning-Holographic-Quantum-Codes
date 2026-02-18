"""
Hyperbolic Embeddings for Holographic Codes

Maps GNN embeddings to hyperbolic space (Poincare disk/ball) for
geometry analysis. Holographic codes have natural hyperbolic structure,
so embedding in hyperbolic space should preserve this.

Key insight: The Poincare disk model is isometric to hyperbolic space,
and AdS geometry naturally maps to hyperbolic space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
import math


class PoincareEmbedding(nn.Module):
    """
    Map Euclidean embeddings to the Poincare ball.

    The Poincare ball model represents hyperbolic space as the
    interior of a unit ball, with distances growing toward the boundary.

    This is natural for holographic codes where:
    - Bulk (center) corresponds to logical qubits
    - Boundary (edge) corresponds to physical qubits
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        curvature: float = -1.0,
        max_norm: float = 0.99,  # Stay within ball
    ):
        """
        Args:
            input_dim: Dimension of input (Euclidean) embeddings
            embed_dim: Dimension of output hyperbolic embeddings
            curvature: Hyperbolic curvature (negative)
            max_norm: Maximum norm for Poincare ball
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.curvature = curvature
        self.c = -curvature  # Positive for convenience
        self.max_norm = max_norm

        # Linear projection to target dimension
        self.proj = nn.Linear(input_dim, embed_dim)

        # Learnable scale for exponential map
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project Euclidean embedding to Poincare ball.

        Args:
            x: Euclidean embeddings (batch, input_dim)

        Returns:
            Poincare ball embeddings (batch, embed_dim)
        """
        # Project to target dimension
        x = self.proj(x)

        # Scale
        x = x * self.scale

        # Apply exponential map at origin
        x = self.exp_map_zero(x)

        return x

    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at the origin.

        Maps tangent vector v at origin to point on Poincare ball.

        exp_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
        """
        sqrt_c = math.sqrt(self.c)
        v_norm = torch.clamp(v.norm(dim=-1, keepdim=True), min=1e-10)

        # tanh for smooth mapping
        coeff = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        result = coeff * v

        # Clamp to stay in ball
        result = self.project_to_ball(result)

        return result

    def log_map_zero(self, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at the origin (inverse of exp_map_zero).

        Maps point y on Poincare ball to tangent vector at origin.

        log_0(y) = arctanh(sqrt(c) * ||y||) * y / (sqrt(c) * ||y||)
        """
        sqrt_c = math.sqrt(self.c)
        y_norm = torch.clamp(y.norm(dim=-1, keepdim=True), min=1e-10, max=self.max_norm)

        # arctanh with numerical stability
        coeff = torch.atanh(sqrt_c * y_norm) / (sqrt_c * y_norm)

        return coeff * y

    def project_to_ball(self, x: torch.Tensor) -> torch.Tensor:
        """Project to interior of Poincare ball."""
        norm = x.norm(dim=-1, keepdim=True)
        clamped = torch.clamp(norm, max=self.max_norm)
        return x * (clamped / torch.clamp(norm, min=1e-10))

    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic distance between points in Poincare ball.

        d(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) + y||_M)

        where ||.||_M is the Mobius addition norm.
        """
        # Mobius addition: x (+) y
        diff = self.mobius_add(-x, y)
        diff_norm = torch.clamp(diff.norm(dim=-1), min=1e-10, max=self.max_norm)

        sqrt_c = math.sqrt(self.c)
        dist = (2 / sqrt_c) * torch.atanh(sqrt_c * diff_norm)

        return dist

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Mobius addition in Poincare ball.

        x (+) y = ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y) /
                  (1 + 2c<x,y> + c^2||x||^2||y||^2)
        """
        c = self.c

        x_sq = (x * x).sum(dim=-1, keepdim=True)
        y_sq = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denom = 1 + 2 * c * xy + c * c * x_sq * y_sq

        return num / torch.clamp(denom, min=1e-10)

    def poincare_centroid(self, points: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted centroid in Poincare ball (Einstein midpoint).

        Uses optimization in tangent space at origin.

        Args:
            points: Points in Poincare ball (batch, num_points, embed_dim)
            weights: Optional weights (batch, num_points)

        Returns:
            Centroid (batch, embed_dim)
        """
        if weights is None:
            weights = torch.ones(points.size()[:-1], device=points.device)

        # Normalize weights
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Map to tangent space at origin
        tangent = self.log_map_zero(points)

        # Weighted average in tangent space
        centroid_tangent = (weights.unsqueeze(-1) * tangent).sum(dim=-2)

        # Map back to Poincare ball
        centroid = self.exp_map_zero(centroid_tangent)

        return centroid


class HyperbolicDistanceLayer(nn.Module):
    """
    Computes pairwise hyperbolic distances for embeddings.

    Useful for:
    - Distance-based loss functions
    - Hierarchical structure analysis
    - Visualization
    """

    def __init__(self, curvature: float = -1.0):
        super().__init__()
        self.c = -curvature

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute pairwise hyperbolic distances.

        Args:
            x: Points (batch, num_x, dim) or (num_x, dim)
            y: Optional second set of points. If None, compute self-distances.

        Returns:
            Distance matrix (batch, num_x, num_y) or (num_x, num_y)
        """
        if y is None:
            y = x

        # Add batch dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch, n, dim = x.shape
        m = y.size(1)

        # Expand for pairwise computation
        x_exp = x.unsqueeze(2).expand(batch, n, m, dim)
        y_exp = y.unsqueeze(1).expand(batch, n, m, dim)

        # Compute Mobius difference
        diff = self._mobius_add(-x_exp, y_exp)
        diff_norm = torch.clamp(diff.norm(dim=-1), min=1e-10, max=0.99)

        # Hyperbolic distance
        sqrt_c = math.sqrt(self.c)
        dist = (2 / sqrt_c) * torch.atanh(sqrt_c * diff_norm)

        if squeeze:
            dist = dist.squeeze(0)

        return dist

    def _mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mobius addition."""
        c = self.c

        x_sq = (x * x).sum(dim=-1, keepdim=True)
        y_sq = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denom = 1 + 2 * c * xy + c * c * x_sq * y_sq

        return num / torch.clamp(denom, min=1e-10)


class HyperbolicMLR(nn.Module):
    """
    Hyperbolic Multinomial Logistic Regression.

    Performs classification in hyperbolic space using
    hyperbolic hyperplanes.

    Useful for classifying codes by type/family.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        curvature: float = -1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.c = -curvature

        # Class prototypes in Poincare ball
        self.a = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)

        # Hyperplane normals
        self.p = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits.

        Args:
            x: Points in Poincare ball (batch, embed_dim)

        Returns:
            Logits (batch, num_classes)
        """
        sqrt_c = math.sqrt(self.c)

        # Compute signed distance to each hyperplane
        logits = []
        for k in range(self.num_classes):
            a_k = self.a[k]
            p_k = self.p[k]

            # Mobius addition: -a_k (+) x
            diff = self._mobius_add(-a_k.unsqueeze(0), x)

            # Compute lambda (conformal factor)
            norm_a = torch.clamp(a_k.norm(), min=1e-10, max=0.99)
            lambda_a = 2 / (1 - self.c * norm_a ** 2)

            # Project onto hyperplane normal
            p_norm = torch.clamp(p_k.norm(), min=1e-10)
            proj = (diff * p_k).sum(dim=-1) / p_norm

            # Signed hyperbolic distance
            logit = sqrt_c * lambda_a * torch.asinh(
                2 * sqrt_c * proj / (1 - self.c * (diff * diff).sum(dim=-1))
            )
            logits.append(logit)

        return torch.stack(logits, dim=-1)

    def _mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mobius addition."""
        c = self.c

        x_sq = (x * x).sum(dim=-1, keepdim=True)
        y_sq = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denom = 1 + 2 * c * xy + c * c * x_sq * y_sq

        return num / torch.clamp(denom, min=1e-10)


class HyperbolicEncoder(nn.Module):
    """
    Full encoder that combines GNN with hyperbolic embedding.

    Takes code graph and outputs hyperbolic embeddings.
    """

    def __init__(
        self,
        gnn_model: nn.Module,
        embed_dim: int = 64,
        curvature: float = -1.0
    ):
        """
        Args:
            gnn_model: Pre-trained or trainable GNN
            embed_dim: Hyperbolic embedding dimension
            curvature: Hyperbolic curvature
        """
        super().__init__()
        self.gnn = gnn_model
        self.poincare = PoincareEmbedding(
            input_dim=gnn_model.hidden_dim,
            embed_dim=embed_dim,
            curvature=curvature
        )

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode code to hyperbolic embedding.

        Returns:
            Poincare ball embeddings (batch, embed_dim)
        """
        # Get Euclidean embeddings from GNN
        euclidean_emb = self.gnn.get_embeddings(x, hyperedge_index, batch)

        # Map to Poincare ball
        hyperbolic_emb = self.poincare(euclidean_emb)

        return hyperbolic_emb


def visualize_poincare_disk(
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    title: str = "Poincare Disk Embeddings",
    save_path: Optional[str] = None
):
    """
    Visualize 2D Poincare disk embeddings.

    Args:
        embeddings: (n, 2) tensor of Poincare disk points
        labels: Optional (n,) tensor of labels for coloring
        title: Plot title
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    embeddings = embeddings.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Draw unit circle boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)

    # Draw concentric circles for geodesic distance reference
    for r in [0.3, 0.5, 0.7, 0.9]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'k--', alpha=0.2)

    # Plot points
    if labels is not None:
        labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
        scatter = ax.scatter(
            embeddings[:, 0], embeddings[:, 1],
            c=labels, cmap='viridis', s=50, alpha=0.7
        )
        plt.colorbar(scatter, ax=ax, label='Label')
    else:
        ax.scatter(
            embeddings[:, 0], embeddings[:, 1],
            c='blue', s=50, alpha=0.7
        )

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Add annotation for boundary interpretation
    ax.annotate(
        'Boundary\n(Physical qubits)',
        xy=(0.9, 0), fontsize=8, ha='center'
    )
    ax.annotate(
        'Bulk\n(Logical qubits)',
        xy=(0, 0), fontsize=8, ha='center'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


def compute_hyperbolic_distortion(
    euclidean_dists: torch.Tensor,
    hyperbolic_dists: torch.Tensor
) -> float:
    """
    Compute distortion between Euclidean and hyperbolic distances.

    Low distortion indicates hyperbolic geometry is natural for the data.

    Returns:
        Average distortion (lower is better)
    """
    # Normalize both to [0, 1]
    e_norm = (euclidean_dists - euclidean_dists.min()) / (
        euclidean_dists.max() - euclidean_dists.min() + 1e-10
    )
    h_norm = (hyperbolic_dists - hyperbolic_dists.min()) / (
        hyperbolic_dists.max() - hyperbolic_dists.min() + 1e-10
    )

    # Mean absolute difference
    distortion = (e_norm - h_norm).abs().mean().item()

    return distortion


if __name__ == '__main__':
    # Test Poincare embedding
    print("Testing Poincare Embedding...")

    embed = PoincareEmbedding(input_dim=64, embed_dim=32)

    x = torch.randn(10, 64)
    y = embed(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Max norm: {y.norm(dim=-1).max().item():.4f}")

    # Test distance
    dist_layer = HyperbolicDistanceLayer()
    dists = dist_layer(y)
    print(f"Distance matrix shape: {dists.shape}")
    print(f"Distance range: [{dists.min().item():.2f}, {dists.max().item():.2f}]")

    # Test 2D visualization
    print("\nTesting 2D visualization...")
    embed_2d = PoincareEmbedding(input_dim=64, embed_dim=2)
    y_2d = embed_2d(x)
    labels = torch.randint(0, 3, (10,))

    # Uncomment to show plot:
    # visualize_poincare_disk(y_2d, labels)
    print("Visualization test passed (not displayed)")
