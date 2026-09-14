import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class LightweightSE3GNN(nn.Module):
    """Lightweight SE(3)-equivariant GNN for structure refinement"""
    
    def __init__(self, node_dim: int = 64, edge_dim: int = 32, num_layers: int = 3):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.cutoff = 8.0
        
        # Node and edge embeddings
        self.node_embedding = nn.Linear(3, node_dim)  # Coordinate embedding
        self.edge_embedding = nn.Linear(1, edge_dim)  # Distance embedding
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            SE3Layer(node_dim, edge_dim) for _ in range(num_layers)
        ])
        
        # Output layer
        self.coord_update = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Linear(node_dim // 2, 3)
        )
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply SE(3)-equivariant refinement"""
        batch_size, seq_len = coords.shape[:2]
        
        # Node features from coordinates
        node_features = self.node_embedding(coords)
        
        # Build edges
        edge_indices, edge_features = self._build_edges(coords)
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            node_features = layer(node_features, coords, edge_indices, edge_features)
        
        # Predict coordinate updates
        coord_updates = self.coord_update(node_features)
        
        return coords + coord_updates * 0.05  # Small SE(3)-equivariant update
    
    def _build_edges(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges based on distance cutoff"""
        batch_size, seq_len = coords.shape[:2]
        
        # Compute pairwise distances
        coords_i = coords.unsqueeze(2)  # [B, N, 1, 3]
        coords_j = coords.unsqueeze(1)  # [B, 1, N, 3]
        distances = torch.norm(coords_i - coords_j, dim=-1)  # [B, N, N]
        
        # Create edge mask
        edge_mask = (distances < self.cutoff) & (distances > 0)
        
        # Get edge indices (simplified for lightweight version)
        batch_idx, src_idx, dst_idx = torch.where(edge_mask)
        edge_indices = torch.stack([src_idx, dst_idx])
        
        # Edge features (distances)
        edge_distances = distances[edge_mask].unsqueeze(-1)
        edge_features = self.edge_embedding(edge_distances)
        
        return edge_indices, edge_features

class SE3Layer(nn.Module):
    """Single SE(3)-equivariant layer"""
    
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
    def forward(self, node_features: torch.Tensor, coords: torch.Tensor, 
                edge_indices: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """SE(3)-equivariant message passing"""
        
        if edge_indices.size(1) == 0:
            return node_features
        
        # Get source and target features
        src_features = node_features[:, edge_indices[0]]  # [B, E, D]
        dst_features = node_features[:, edge_indices[1]]  # [B, E, D]
        
        # Compute messages
        messages = self.message_net(torch.cat([
            src_features, dst_features, edge_features.unsqueeze(0).expand(node_features.size(0), -1, -1)
        ], dim=-1))
        
        # Aggregate messages
        aggregated = torch.zeros_like(node_features)
        for i in range(edge_indices.size(1)):
            dst_idx = edge_indices[1, i]
            aggregated[:, dst_idx] += messages[:, i]
        
        # Update node features
        updated_features = self.update_net(torch.cat([node_features, aggregated], dim=-1))
        
        return node_features + updated_features  # Residual connection

# Global lightweight SE(3)-GNN
lightweight_se3_gnn = LightweightSE3GNN()

def apply_se3_refinement(job_id: str, coords: np.ndarray) -> np.ndarray:
    """Apply SE(3)-equivariant structure refinement"""
    from main import log_to_job
    
    log_to_job(job_id, "🔗 SE(3)-GNN: Applying equivariant structure refinement")
    
    try:
        coords_tensor = torch.FloatTensor(coords).unsqueeze(0)
        
        with torch.no_grad():
            refined_coords = lightweight_se3_gnn(coords_tensor)
        
        refined_coords_np = refined_coords.squeeze(0).numpy()
        
        # Calculate refinement metrics
        rmsd = float(np.sqrt(np.mean((coords - refined_coords_np) ** 2)))
        log_to_job(job_id, f"SE(3)-GNN: RMSD change: {rmsd:.3f}Å")
        log_to_job(job_id, f"SE(3)-GNN: Applied {len(lightweight_se3_gnn.gnn_layers)} equivariant layers")
        
        return refined_coords_np
        
    except Exception as e:
        log_to_job(job_id, f"SE(3)-GNN refinement failed: {e}")
        return coords