"""
Full Holographic Code GNN Architecture

Advanced architecture for code property prediction and embedding generation:
- 6 hypergraph convolution layers with residual connections
- Multi-head attention for stabilizer interactions
- Separate heads for: distance, rate, threshold prediction
- Embedding extraction for downstream analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import math

from .hypergraph_conv import HypergraphConv


class MultiHeadHypergraphAttention(nn.Module):
    """
    Multi-head attention for hypergraph message passing.

    Computes attention weights between nodes within each hyperedge,
    allowing the model to learn which qubits are most important
    for each stabilizer constraint.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: List[List[int]]
    ) -> torch.Tensor:
        """
        Apply multi-head attention within hyperedges.

        Args:
            x: Node features (num_nodes, hidden_dim)
            hyperedge_index: List of node indices for each hyperedge

        Returns:
            Updated node features (num_nodes, hidden_dim)
        """
        num_nodes = x.size(0)
        device = x.device

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)
        K = K.view(num_nodes, self.num_heads, self.head_dim)
        V = V.view(num_nodes, self.num_heads, self.head_dim)

        # Aggregate attention within each hyperedge
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=device)
        counts = torch.zeros(num_nodes, device=device)

        for hedge_nodes in hyperedge_index:
            if len(hedge_nodes) < 2:
                continue

            hedge_nodes = torch.tensor(hedge_nodes, device=device)

            # Get Q, K, V for this hyperedge
            q = Q[hedge_nodes]  # (k, num_heads, head_dim)
            k = K[hedge_nodes]
            v = V[hedge_nodes]

            # Compute attention scores
            # (k, num_heads, head_dim) x (k, num_heads, head_dim)^T -> (k, num_heads, k)
            scores = torch.einsum('ihd,jhd->ijh', q, k) / self.scale

            # Softmax over keys
            attn = F.softmax(scores, dim=1)  # (k, k, num_heads)
            attn = self.dropout(attn)

            # Weighted sum of values
            # (k, k, num_heads) x (k, num_heads, head_dim) -> (k, num_heads, head_dim)
            attended = torch.einsum('ijh,jhd->ihd', attn, v)

            # Accumulate
            out[hedge_nodes] += attended
            counts[hedge_nodes] += 1

        # Normalize and reshape
        counts = counts.clamp(min=1).view(-1, 1, 1)
        out = out / counts
        out = out.view(num_nodes, self.hidden_dim)

        # Output projection
        out = self.out_proj(out)

        return out


class HypergraphConvBlock(nn.Module):
    """
    A single hypergraph convolution block with attention and residual.

    Components:
    - Hypergraph convolution
    - Multi-head attention
    - Layer normalization
    - Residual connection
    - Dropout
    """

    def __init__(
        self,
        hidden_dim: int,
        hyperedge_dim: int = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hyperedge_dim = hyperedge_dim or hidden_dim
        self.use_attention = use_attention

        # Hypergraph convolution
        self.conv = HypergraphConv(
            hidden_dim, hidden_dim,
            hyperedge_dim=self.hyperedge_dim,
            aggr='mean'
        )

        # Multi-head attention
        if use_attention:
            self.attention = MultiHeadHypergraphAttention(
                hidden_dim, num_heads, dropout
            )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: List[List[int]]
    ) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Hypergraph convolution with residual
        x_conv = self.conv(x, hyperedge_index)
        x = self.norm1(x + self.dropout(x_conv))

        # Attention with residual (if enabled)
        if self.use_attention:
            x_attn = self.attention(x, hyperedge_index)
            x = x + self.dropout(x_attn)

        # Feed-forward with residual
        x = self.norm2(x + self.ff(x))

        return x


class HolographicCodeGNN(nn.Module):
    """
    Full architecture for holographic code property prediction.

    Architecture:
    - Input projection with positional encoding
    - 6 hypergraph convolution blocks with attention
    - Global pooling with multiple strategies
    - Separate prediction heads for different properties

    Properties predicted:
    - Distance: Code distance d (regression, d >= 1)
    - Rate: Encoding rate k/n (sigmoid, 0 to 1)
    - Threshold: Decoder threshold (regression)

    Args:
        node_dim: Input node feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of convolution blocks
        num_heads: Attention heads per block
        dropout: Dropout rate
        use_attention: Whether to use attention in blocks
    """

    def __init__(
        self,
        node_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Positional encoding (for qubit indices)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=200)

        # Hypergraph convolution blocks
        self.blocks = nn.ModuleList([
            HypergraphConvBlock(
                hidden_dim,
                hyperedge_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_attention=use_attention
            )
            for _ in range(num_layers)
        ])

        # Global pooling strategies
        self.pool_strategies = ['mean', 'max', 'sum']

        # Prediction heads
        self.distance_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.threshold_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Embedding projection (for downstream tasks)
        self.embedding_proj = nn.Linear(hidden_dim * 3, hidden_dim)

    def encode(
        self,
        x: torch.Tensor,
        hyperedge_index: List[List[int]],
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode code to node embeddings.

        Args:
            x: Node features (num_nodes, node_dim)
            hyperedge_index: List of node indices for each hyperedge
            batch: Batch assignment (for batched graphs)

        Returns:
            Node embeddings (num_nodes, hidden_dim)
        """
        # Input projection
        x = self.input_proj(x)

        # Add positional encoding based on node indices
        x = self.pos_encoding(x)

        # Pass through hypergraph blocks
        for block in self.blocks:
            x = block(x, hyperedge_index)

        return x

    def global_pool(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multiple global pooling strategies.

        Returns concatenated pool outputs.
        """
        if batch is None:
            # Single graph
            mean_pool = x.mean(dim=0, keepdim=True)
            max_pool = x.max(dim=0, keepdim=True)[0]
            sum_pool = x.sum(dim=0, keepdim=True)
        else:
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            sum_pool = global_add_pool(x, batch)

        return torch.cat([mean_pool, max_pool, sum_pool], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: List[List[int]],
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass with all predictions.

        Args:
            x: Node features (num_nodes, node_dim)
            hyperedge_index: List of node indices for each hyperedge
            batch: Batch assignment

        Returns:
            Tuple of (distance, rate, threshold, embeddings)
        """
        # Encode nodes
        node_embeddings = self.encode(x, hyperedge_index, batch)

        # Global pooling
        graph_embedding = self.global_pool(node_embeddings, batch)

        # Predictions
        distance = F.softplus(self.distance_head(graph_embedding)) + 1  # d >= 1
        rate = self.rate_head(graph_embedding)  # 0 to 1
        threshold = F.softplus(self.threshold_head(graph_embedding))

        # Embedding for downstream
        embeddings = self.embedding_proj(graph_embedding)

        return distance, rate, threshold, embeddings

    def predict_distance(
        self,
        x: torch.Tensor,
        hyperedge_index: List[List[int]],
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict only distance (for inference efficiency)."""
        node_embeddings = self.encode(x, hyperedge_index, batch)
        graph_embedding = self.global_pool(node_embeddings, batch)
        distance = F.softplus(self.distance_head(graph_embedding)) + 1
        return distance

    def get_embeddings(
        self,
        x: torch.Tensor,
        hyperedge_index: List[List[int]],
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get graph-level embeddings for downstream tasks."""
        node_embeddings = self.encode(x, hyperedge_index, batch)
        graph_embedding = self.global_pool(node_embeddings, batch)
        return self.embedding_proj(graph_embedding)

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        hyperedge_index: List[List[int]],
    ) -> torch.Tensor:
        """Get node-level embeddings (for visualization/analysis)."""
        return self.encode(x, hyperedge_index)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for node indices.

    Helps the model distinguish between different qubit positions.
    """

    def __init__(self, hidden_dim: int, max_len: int = 200):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to node features."""
        num_nodes = x.size(0)
        return x + self.pe[:num_nodes]


class ConditionalCodeGenerator(nn.Module):
    """
    Conditional code generator using VAE architecture.

    Given target properties (distance, rate), generate stabilizer
    code structure. This is the inverse of prediction.

    Note: This is an experimental feature for code design.
    """

    def __init__(
        self,
        condition_dim: int = 8,  # Target properties
        latent_dim: int = 64,
        hidden_dim: int = 128,
        max_qubits: int = 50,
        max_stabilizers: int = 49
    ):
        super().__init__()
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_qubits = max_qubits
        self.max_stabilizers = max_stabilizers

        # Encoder (for training with real codes)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log-var
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # Output heads
        # Predict number of qubits
        self.n_qubits_head = nn.Linear(hidden_dim, max_qubits)

        # Predict stabilizer structure
        # Output: (max_stabilizers, max_qubits, 4) for I/X/Y/Z logits
        self.stabilizer_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, max_stabilizers * max_qubits * 4)
        )

    def encode(self, code_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode code to latent distribution."""
        h = self.encoder(code_embedding)
        mean, log_var = h.chunk(2, dim=-1)
        return mean, log_var

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(
        self,
        z: torch.Tensor,
        conditions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent + conditions to code structure.

        Args:
            z: Latent vector (batch, latent_dim)
            conditions: Target properties (batch, condition_dim)

        Returns:
            n_qubits_logits: (batch, max_qubits)
            stabilizer_logits: (batch, max_stabilizers, max_qubits, 4)
        """
        h = torch.cat([z, conditions], dim=-1)
        h = self.decoder(h)

        n_qubits_logits = self.n_qubits_head(h)

        stab_logits = self.stabilizer_head(h)
        stab_logits = stab_logits.view(
            -1, self.max_stabilizers, self.max_qubits, 4
        )

        return n_qubits_logits, stab_logits

    def forward(
        self,
        code_embedding: torch.Tensor,
        conditions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass (with encoding)."""
        mean, log_var = self.encode(code_embedding)
        z = self.reparameterize(mean, log_var)
        n_logits, stab_logits = self.decode(z, conditions)

        return {
            'mean': mean,
            'log_var': log_var,
            'z': z,
            'n_qubits_logits': n_logits,
            'stabilizer_logits': stab_logits
        }

    def generate(
        self,
        conditions: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Generate code from conditions only (inference)."""
        batch_size = conditions.size(0)
        device = conditions.device

        # Sample from prior
        z = torch.randn(batch_size, self.latent_dim, device=device)

        # Decode
        n_logits, stab_logits = self.decode(z, conditions)

        # Sample n_qubits
        n_qubits = F.softmax(n_logits / temperature, dim=-1)

        # Sample stabilizers
        stab_probs = F.softmax(stab_logits / temperature, dim=-1)

        return {
            'z': z,
            'n_qubits_probs': n_qubits,
            'stabilizer_probs': stab_probs
        }


def create_model(
    config: Dict[str, Any],
    model_type: str = 'predictor'
) -> nn.Module:
    """
    Factory function to create model from config.

    Args:
        config: Model configuration dict
        model_type: 'predictor' or 'generator'

    Returns:
        Initialized model
    """
    if model_type == 'predictor':
        return HolographicCodeGNN(
            node_dim=config.get('node_dim', 8),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.1),
            use_attention=config.get('use_attention', True)
        )
    elif model_type == 'generator':
        return ConditionalCodeGenerator(
            condition_dim=config.get('condition_dim', 8),
            latent_dim=config.get('latent_dim', 64),
            hidden_dim=config.get('hidden_dim', 128),
            max_qubits=config.get('max_qubits', 50),
            max_stabilizers=config.get('max_stabilizers', 49)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for simultaneous prediction of distance, rate, threshold.

    Uses learnable loss weights (uncertainty weighting).
    """

    def __init__(self, num_tasks: int = 3):
        super().__init__()
        self.num_tasks = num_tasks
        # Log of task-specific uncertainties
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self,
        distance_pred: torch.Tensor,
        distance_true: torch.Tensor,
        rate_pred: torch.Tensor,
        rate_true: torch.Tensor,
        threshold_pred: Optional[torch.Tensor] = None,
        threshold_true: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss with uncertainty weighting.

        Returns:
            Total loss and dict of individual losses
        """
        losses = {}

        # Squeeze predictions to match target shape [batch] vs [batch, 1]
        distance_pred = distance_pred.squeeze(-1)
        rate_pred = rate_pred.squeeze(-1)

        # Distance loss (MSE)
        loss_d = F.mse_loss(distance_pred, distance_true)
        losses['distance'] = loss_d

        # Rate loss (BCE for 0-1 target)
        loss_r = F.binary_cross_entropy(rate_pred, rate_true)
        losses['rate'] = loss_r

        # Threshold loss (if provided)
        if threshold_pred is not None and threshold_true is not None:
            threshold_pred = threshold_pred.squeeze(-1)
            loss_t = F.mse_loss(threshold_pred, threshold_true)
            losses['threshold'] = loss_t
        else:
            loss_t = torch.tensor(0.0, device=distance_pred.device)

        # Uncertainty-weighted combination
        precision = torch.exp(-self.log_vars)

        total_loss = (
            precision[0] * loss_d + self.log_vars[0] +
            precision[1] * loss_r + self.log_vars[1] +
            precision[2] * loss_t + self.log_vars[2]
        )

        losses['total'] = total_loss
        losses['log_vars'] = self.log_vars.detach()

        return total_loss, losses


if __name__ == '__main__':
    # Test the models
    print("Testing HolographicCodeGNN...")

    model = HolographicCodeGNN(
        node_dim=4,
        hidden_dim=64,
        num_layers=3,
        num_heads=4
    )

    # Dummy data
    x = torch.randn(10, 4)  # 10 nodes, 4 features
    hyperedges = [[0, 1, 2], [2, 3, 4], [4, 5, 6, 7], [7, 8, 9]]

    distance, rate, threshold, embeddings = model(x, hyperedges)

    print(f"Distance: {distance.item():.3f}")
    print(f"Rate: {rate.item():.3f}")
    print(f"Threshold: {threshold.item():.3f}")
    print(f"Embedding shape: {embeddings.shape}")

    # Test multi-task loss
    print("\nTesting MultiTaskLoss...")
    loss_fn = MultiTaskLoss()

    d_true = torch.tensor([[3.0]])
    r_true = torch.tensor([[0.2]])

    total_loss, losses = loss_fn(distance, d_true, rate, r_true)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Distance loss: {losses['distance'].item():.4f}")
    print(f"Rate loss: {losses['rate'].item():.4f}")
