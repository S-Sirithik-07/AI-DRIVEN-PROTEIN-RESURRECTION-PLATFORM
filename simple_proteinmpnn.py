import torch
import numpy as np
import sys
import os

# Add ProteinMPNN to path - fix the path resolution
proteinmpnn_path = os.path.join(os.path.dirname(__file__), '..', 'ProteinMPNN')
proteinmpnn_path = os.path.abspath(proteinmpnn_path)
if proteinmpnn_path not in sys.path:
    sys.path.append(proteinmpnn_path)
    
print(f"Added ProteinMPNN path: {proteinmpnn_path}")
print(f"ProteinMPNN path exists: {os.path.exists(proteinmpnn_path)}")
print(f"protein_mpnn_utils.py exists: {os.path.exists(os.path.join(proteinmpnn_path, 'protein_mpnn_utils.py'))}")

def load_proteinmpnn_model():
    """Load ProteinMPNN model"""
    try:
        print("Loading ProteinMPNN neural network...")
        
        # Import ProteinMPNN modules
        from protein_mpnn_utils import ProteinMPNN
        print("Imported ProteinMPNN modules successfully")
        
        # Load model weights - fix the path resolution
        model_path = os.path.join(proteinmpnn_path, 'vanilla_model_weights', 'v_48_020.pt')
        if not os.path.exists(model_path):
            print(f"Model weights not found at {model_path}")
            return None
            
        print(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Initialize model with correct parameters
        model = ProteinMPNN(
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            vocab=21,
            k_neighbors=32,
            augment_eps=0.0,
            dropout=0.1
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("ProteinMPNN model loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
        
    except Exception as e:
        print(f"Failed to load ProteinMPNN: {e}")
        print("This will trigger fallback to heuristics")
        return None

def run_proteinmpnn_inference(model, coords, length):
    """Run ProteinMPNN inference on coordinates"""
    try:
        print(f"PROTEINMPNN: Running neural network inference on {length} residues")
        
        # Pad or trim coordinates to match length
        if len(coords) < length:
            coords = coords + [coords[-1]] * (length - len(coords))
        coords = coords[:length]
        
        # Convert coordinates to proper format for ProteinMPNN (full atom)
        # Create dummy N, CA, C, O coordinates from CA positions
        full_coords = []
        for i, ca_coord in enumerate(coords):
            # Generate dummy N, CA, C, O from CA position
            ca = ca_coord
            n = [ca[0] - 1.0, ca[1], ca[2]]  # Dummy N position
            c = [ca[0] + 1.0, ca[1], ca[2]]  # Dummy C position  
            o = [ca[0] + 1.5, ca[1] + 1.0, ca[2]]  # Dummy O position
            full_coords.append([n, ca, c, o])
        
        coords_tensor = torch.tensor(full_coords, dtype=torch.float32).unsqueeze(0)  # [1, L, 4, 3]
        
        # Create required tensors
        mask = torch.ones(1, length, dtype=torch.float32)
        chain_mask = torch.ones(1, length, dtype=torch.float32)
        residue_idx = torch.arange(length).unsqueeze(0)
        chain_encoding = torch.zeros(1, length, dtype=torch.long)
        
        # Create dummy sequence (will be ignored in forward pass)
        S = torch.zeros(1, length, dtype=torch.long)
        
        # Random tensor for decoding order
        randn = torch.randn(1, length)
        
        print("Running forward pass through ProteinMPNN neural network...")
        
        # Run inference through ProteinMPNN
        with torch.no_grad():
            log_probs = model(
                X=coords_tensor,
                S=S,
                mask=mask,
                chain_M=chain_mask,
                residue_idx=residue_idx,
                chain_encoding_all=chain_encoding,
                randn=randn
            )
            
            probs = torch.softmax(log_probs, dim=-1)
            
            # Sample sequence from probability distribution
            sampled_indices = torch.multinomial(probs.squeeze(), 1).squeeze()
            
            # Convert to amino acids (ProteinMPNN alphabet)
            aa_alphabet = "ACDEFGHIKLMNPQRSTVWYX"  # ProteinMPNN alphabet
            sequence = "".join([aa_alphabet[min(idx, len(aa_alphabet)-1)] for idx in sampled_indices])
            
            print(f"PROTEINMPNN: Generated sequence: {sequence[:20]}...")
            print(f"PROTEINMPNN: Output tensor shape: {log_probs.shape}")
            print(f"PROTEINMPNN: Max probability: {torch.max(probs):.4f}")
            print(f"PROTEINMPNN: Mean confidence: {torch.mean(torch.max(probs, dim=-1)[0]):.4f}")
            
            return {
                "sequence": sequence,
                "log_probs": log_probs.squeeze().numpy(),
                "probabilities": probs.squeeze().numpy(),
                "confidence_scores": torch.max(probs, dim=-1)[0].squeeze().numpy()
            }
            
    except Exception as e:
        print(f"ProteinMPNN inference failed: {e}")
        print("This will trigger fallback to heuristics")
        return None

if __name__ == "__main__":
    # Test the model
    print("Testing ProteinMPNN model...")
    model = load_proteinmpnn_model()
    if model:
        test_coords = [[i*3.8, 0, 0] for i in range(10)]
        result = run_proteinmpnn_inference(model, test_coords, 10)
        if result:
            print(f"Test successful: {result['sequence']}")
        else:
            print("Test failed")
    else:
        print("Model loading failed")