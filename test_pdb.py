#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append('.')

# Import the PDB generation function
from main import generate_complete_pdb

def test_pdb_generation():
    """Test PDB generation function"""
    
    # Test sequence and coordinates
    test_sequence = "MKLLVLGLGAGVGK"
    test_coords = []
    
    # Generate some test coordinates
    for i in range(len(test_sequence)):
        x = i * 3.8
        y = np.sin(i * 0.1) * 10
        z = np.cos(i * 0.1) * 8
        test_coords.append([x, y, z])
    
    print("Testing PDB generation...")
    print(f"Sequence: {test_sequence}")
    print(f"Length: {len(test_sequence)}")
    print(f"Coordinates: {len(test_coords)} points")
    
    # Generate PDB
    pdb_content = generate_complete_pdb(test_sequence, test_coords, 1)
    
    print("\nGenerated PDB content:")
    print("=" * 50)
    print(pdb_content)
    print("=" * 50)
    
    # Save to file for testing
    with open("test_protein.pdb", "w") as f:
        f.write(pdb_content)
    
    print(f"\nPDB saved to test_protein.pdb")
    print(f"Total lines: {len(pdb_content.split())}")
    
    # Count atoms
    atom_lines = [line for line in pdb_content.split('\n') if line.startswith('ATOM')]
    print(f"ATOM lines: {len(atom_lines)}")
    
    if len(atom_lines) > 0:
        print("First few ATOM lines:")
        for i, line in enumerate(atom_lines[:5]):
            print(f"  {i+1}: {line}")
    
    return pdb_content

if __name__ == "__main__":
    test_pdb_generation()