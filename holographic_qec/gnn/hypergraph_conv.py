"""
Hypergraph Convolution Layers for Quantum Code Learning

Implements message passing on hypergraphs where:
- Nodes represent qubits
- Hyperedges represent stabilizer constraints

Reference architecture for learning holographic code structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Optional, Dict
import numpy as np


class HypergraphConv(nn.Module):
    """
    Hypergraph convolution layer.

    Message passing: nodes <-> hyperedges
    1. Aggregate node features to hyperedges
    2. Update hyperedge features
    3. Aggregate hyperedge features back to nodes
    4. Update node features

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        hyperedge_dim: Hyperedge feature dimension
        aggr: Aggregation method ('mean', 'sum', 'max')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hyperedge_dim: int = 32,
        aggr: str = 'mean'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hyperedge_dim = hyperedge_dim
        self.aggr = aggr

        # Node -> hyperedge transformation
        self.node_to_hedge = nn.Linear(in_channels, hyperedge_dim)

        # Hyperedge update
        self.hedge_update = nn.Sequential(
            nn.Linear(hyperedge_dim, hyperedge_dim),
            nn.ReLU(),
            nn.Linear(hyperedge_dim, hyperedge_dim),
        )

        # Hyperedge -> node transformation
        self.hedge_to_node = nn.Linear(hyperedge_dim, out_channels)

        # Node update with residual
        self.node_update = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

        # Layer normalization
        self.node_norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: List[List[int]],
        hyperedge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, in_channels)
            hyperedge_index: List of node indices for each hyperedge
            hyperedge_attr: Optional hyperedge features

        Returns:
            Updated node features (num_nodes, out_channels)
        """
        num_nodes = x.size(0)
        num_hyperedges = len(hyperedge_index)

        # Transform node features
        x_transformed = self.node_to_hedge(x)  # (num_nodes, hyperedge_dim)

        # Aggregate nodes to hyperedges
        hedge_features = []
        for hedge_nodes in hyperedge_index:
            if len(hedge_nodes) > 0:
                hedge_x = x_transformed[hedge_nodes]  # (k, hyperedge_dim)
                if self.aggr == 'mean':
                    hedge_feat = hedge_x.mean(dim=0)
                elif self.aggr == 'sum':
                    hedge_feat = hedge_x.sum(dim=0)
                elif self.aggr == 'max':
                    hedge_feat = hedge_x.max(dim=0)[0]
                else:
                    hedge_feat = hedge_x.mean(dim=0)
            else:
                hedge_feat = torch.zeros(self.hyperedge_dim, device=x.device)
            hedge_features.append(hedge_feat)

        if num_hyperedges > 0:
            hedge_features = torch.stack(hedge_features)  # (num_hyperedges, hyperedge_dim)

            # Update hyperedge features
            hedge_features = self.hedge_update(hedge_features)

            # Aggregate hyperedges back to nodes
            node_from_hedge = torch.zeros(num_nodes, self.hyperedge_dim, device=x.device)
            node_counts = torch.zeros(num_nodes, device=x.device)

            for hedge_idx, hedge_nodes in enumerate(hyperedge_index):
                for node_idx in hedge_nodes:
                    node_from_hedge[node_idx] += hedge_features[hedge_idx]
                    node_counts[node_idx] += 1

            # Normalize by count
            node_counts = node_counts.clamp(min=1).unsqueeze(1)
            node_from_hedge = node_from_hedge / node_counts

            # Transform back to node space
            node_from_hedge = self.hedge_to_node(node_from_hedge)
        else:
            node_from_hedge = torch.zeros(num_nodes, self.out_channels, device=x.device)

        # Residual connection and update
        if self.in_channels == self.out_channels:
            x_combined = torch.cat([x, node_from_hedge], dim=-1)
        else:
            x_combined = torch.cat([
                F.pad(x, (0, self.out_channels - self.in_channels)),
                node_from_hedge
            ], dim=-1)

        x_out = self.node_update(x_combined)
        x_out = self.node_norm(x_out)

        return x_out


class HypergraphNN(nn.Module):
    """
    Full hypergraph neural network for quantum code property prediction.

    Architecture:
    - Multiple hypergraph convolution layers
    - Global pooling
    - Prediction heads for code properties

    Args:
        node_features: Number of input node features
        hidden_dim: Hidden dimension
        num_layers: Number of hypergraph conv layers
        output_dim: Output dimension (e.g., 1 for distance prediction)
    """

    def __init__(
        self,
        node_features: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)

        # Hypergraph convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                HypergraphConv(
                    hidden_dim, hidden_dim,
                    hyperedge_dim=hidden_dim,
                    aggr='mean'
                )
            )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Global pooling
        self.global_pool = global_mean_pool

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: List[List[int]],
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, node_features)
            hyperedge_index: List of node indices for each hyperedge
            batch: Batch assignment for each node (for batched graphs)

        Returns:
            Predictions (batch_size, output_dim)
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # Hypergraph convolutions
        for conv in self.convs:
            x_new = conv(x, hyperedge_index)
            x = x + x_new  # Residual
            x = self.dropout(x)

        # Global pooling
        if batch is None:
            # Single graph
            x = x.mean(dim=0, keepdim=True)
        else:
            x = self.global_pool(x, batch)

        # Prediction
        out = self.predictor(x)

        return out

    def get_embeddings(
        self,
        x: torch.Tensor,
        hyperedge_index: List[List[int]],
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get node embeddings (for analysis/visualization).

        Returns embeddings before final prediction head.
        """
        x = self.input_proj(x)
        x = F.relu(x)

        for conv in self.convs:
            x_new = conv(x, hyperedge_index)
            x = x + x_new

        return x


class CodeDistancePredictor(nn.Module):
    """
    Specialized model for predicting code distance.

    Uses hypergraph GNN backbone with distance-specific
    output processing (ensures integer output >= 1).
    """

    def __init__(
        self,
        node_features: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3
    ):
        super().__init__()

        self.backbone = HypergraphNN(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1
        )

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: List[List[int]],
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict code distance.

        Returns:
            Predicted distance (continuous for training, can be rounded for inference)
        """
        out = self.backbone(x, hyperedge_index, batch)
        # Ensure positive output
        out = F.softplus(out) + 1  # Distance >= 1
        return out

    def predict_integer(
        self,
        x: torch.Tensor,
        hyperedge_index: List[List[int]],
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict distance as integer (for inference)."""
        out = self.forward(x, hyperedge_index, batch)
        return torch.round(out).long()


def code_to_pyg_data(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    hyperedges: List[List[int]],
    target: Optional[float] = None
) -> Dict:
    """
    Convert code representation to PyG-compatible format.

    Args:
        node_features: (n, d) array of node features
        edge_index: (2, e) array of edges
        hyperedges: List of node lists for each hyperedge
        target: Target value (e.g., code distance)

    Returns:
        Dictionary with tensors ready for the model
    """
    data = {
        'x': torch.tensor(node_features, dtype=torch.float32),
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'hyperedge_index': hyperedges,
    }

    if target is not None:
        data['y'] = torch.tensor([target], dtype=torch.float32)

    return data


class CodeDataset:
    """
    Dataset wrapper for quantum codes.

    Handles batching of codes with variable sizes.
    """

    def __init__(self, codes: List, compute_distance: bool = False):
        """
        Args:
            codes: List of HaPPYCode or StabilizerCode instances
            compute_distance: Whether to compute distance for targets
        """
        self.codes = codes
        self.data_list = []

        for code in codes:
            if hasattr(code, 'stabilizer_code'):
                sc = code.stabilizer_code
            else:
                sc = code

            node_feat, edge_idx, hyperedges = sc.to_graph()
            target = sc.distance if sc.distance else None

            self.data_list.append(
                code_to_pyg_data(node_feat, edge_idx, hyperedges, target)
            )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def get_batch(self, indices: List[int]) -> Dict:
        """
        Create a batch from specified indices.

        Note: For simplicity, this returns a list (not batched tensor)
        since hyperedges have variable structure.
        """
        return [self.data_list[i] for i in indices]
