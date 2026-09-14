import numpy as np
from typing import Dict, List, Tuple
import math

class ChemistrySimulator:
    """Lightweight chemistry simulation for protein stability analysis"""
    
    def __init__(self):
        # Amino acid properties
        self.aa_properties = {
            'A': {'hydrophobic': 0.62, 'charge': 0, 'size': 67, 'flexibility': 0.36},
            'C': {'hydrophobic': 0.29, 'charge': 0, 'size': 86, 'flexibility': 0.35},
            'D': {'hydrophobic': -0.90, 'charge': -1, 'size': 91, 'flexibility': 0.51},
            'E': {'hydrophobic': -0.74, 'charge': -1, 'size': 109, 'flexibility': 0.50},
            'F': {'hydrophobic': 1.19, 'charge': 0, 'size': 135, 'flexibility': 0.31},
            'G': {'hydrophobic': 0.48, 'charge': 0, 'size': 48, 'flexibility': 0.54},
            'H': {'hydrophobic': -0.40, 'charge': 0.5, 'size': 118, 'flexibility': 0.32},
            'I': {'hydrophobic': 1.38, 'charge': 0, 'size': 124, 'flexibility': 0.18},
            'K': {'hydrophobic': -1.50, 'charge': 1, 'size': 135, 'flexibility': 0.47},
            'L': {'hydrophobic': 1.06, 'charge': 0, 'size': 124, 'flexibility': 0.36},
            'M': {'hydrophobic': 0.64, 'charge': 0, 'size': 124, 'flexibility': 0.30},
            'N': {'hydrophobic': -0.78, 'charge': 0, 'size': 96, 'flexibility': 0.46},
            'P': {'hydrophobic': 0.12, 'charge': 0, 'size': 90, 'flexibility': 0.51},
            'Q': {'hydrophobic': -0.85, 'charge': 0, 'size': 114, 'flexibility': 0.49},
            'R': {'hydrophobic': -2.53, 'charge': 1, 'size': 148, 'flexibility': 0.95},
            'S': {'hydrophobic': -0.18, 'charge': 0, 'size': 73, 'flexibility': 0.51},
            'T': {'hydrophobic': -0.05, 'charge': 0, 'size': 93, 'flexibility': 0.44},
            'V': {'hydrophobic': 1.08, 'charge': 0, 'size': 105, 'flexibility': 0.13},
            'W': {'hydrophobic': 0.81, 'charge': 0, 'size': 163, 'flexibility': 0.31},
            'Y': {'hydrophobic': 0.26, 'charge': 0, 'size': 141, 'flexibility': 0.42}
        }
    
    def calculate_stability_metrics(self, sequence: str, coords: np.ndarray) -> Dict:
        """Calculate comprehensive stability metrics"""
        
        # 1. Hydrophobic core analysis
        hydrophobic_score = self._calculate_hydrophobic_core(sequence, coords)
        
        # 2. Electrostatic interactions
        electrostatic_score = self._calculate_electrostatic_energy(sequence, coords)
        
        # 3. Secondary structure propensity
        secondary_structure_score = self._predict_secondary_structure(sequence)
        
        # 4. Ramachandran analysis (simplified)
        ramachandran_score = self._analyze_backbone_geometry(coords)
        
        # 5. Solvent accessibility
        sasa_score = self._calculate_sasa(sequence, coords)
        
        # 6. Overall stability prediction
        stability_score = self._calculate_overall_stability([
            hydrophobic_score, electrostatic_score, 
            secondary_structure_score, ramachandran_score, sasa_score
        ])
        
        return {
            'stability_score': stability_score,
            'hydrophobic_core': hydrophobic_score,
            'electrostatic_energy': electrostatic_score,
            'secondary_structure': secondary_structure_score,
            'ramachandran_quality': ramachandran_score,
            'solvent_accessibility': sasa_score,
            'thermodynamic_feasibility': 'Favorable' if stability_score > 0.7 else 'Moderate' if stability_score > 0.5 else 'Unfavorable'
        }
    
    def _calculate_hydrophobic_core(self, sequence: str, coords: np.ndarray) -> float:
        """Calculate hydrophobic core formation"""
        hydrophobic_residues = []
        
        for i, aa in enumerate(sequence):
            if i < len(coords) and aa in self.aa_properties:
                if self.aa_properties[aa]['hydrophobic'] > 0.5:
                    hydrophobic_residues.append(i)
        
        if len(hydrophobic_residues) < 2:
            return 0.5
        
        # Calculate average distance between hydrophobic residues
        total_distance = 0
        pairs = 0
        
        for i in range(len(hydrophobic_residues)):
            for j in range(i + 1, len(hydrophobic_residues)):
                idx1, idx2 = hydrophobic_residues[i], hydrophobic_residues[j]
                if idx1 < len(coords) and idx2 < len(coords):
                    distance = np.linalg.norm(coords[idx1] - coords[idx2])
                    total_distance += distance
                    pairs += 1
        
        if pairs == 0:
            return 0.5
        
        avg_distance = total_distance / pairs
        # Optimal hydrophobic core distance is around 6-8 Å
        core_score = max(0, 1 - abs(avg_distance - 7) / 10)
        
        return float(min(1.0, core_score))
    
    def _calculate_electrostatic_energy(self, sequence: str, coords: np.ndarray) -> float:
        """Calculate electrostatic interaction energy"""
        charged_residues = []
        
        for i, aa in enumerate(sequence):
            if i < len(coords) and aa in self.aa_properties:
                charge = self.aa_properties[aa]['charge']
                if charge != 0:
                    charged_residues.append((i, charge))
        
        if len(charged_residues) < 2:
            return 0.7
        
        total_energy = 0
        for i in range(len(charged_residues)):
            for j in range(i + 1, len(charged_residues)):
                idx1, charge1 = charged_residues[i]
                idx2, charge2 = charged_residues[j]
                
                if idx1 < len(coords) and idx2 < len(coords):
                    distance = np.linalg.norm(coords[idx1] - coords[idx2])
                    # Coulomb's law (simplified)
                    energy = (charge1 * charge2) / max(distance, 1.0)
                    total_energy += energy
        
        # Convert to favorable score (lower energy = higher score)
        electrostatic_score = 1 / (1 + abs(total_energy))
        return float(min(1.0, electrostatic_score))
    
    def _predict_secondary_structure(self, sequence: str) -> float:
        """Predict secondary structure propensity"""
        # Simplified Chou-Fasman method
        helix_propensity = {'A': 1.42, 'E': 1.51, 'L': 1.21, 'M': 1.45}
        sheet_propensity = {'V': 1.70, 'I': 1.60, 'Y': 1.47, 'F': 1.38}
        
        helix_score = 0
        sheet_score = 0
        
        for aa in sequence:
            helix_score += helix_propensity.get(aa, 1.0)
            sheet_score += sheet_propensity.get(aa, 1.0)
        
        # Normalize by sequence length
        helix_score /= len(sequence)
        sheet_score /= len(sequence)
        
        # Balance between helix and sheet formation
        structure_score = (helix_score + sheet_score) / 2.4  # Normalize
        return float(min(1.0, structure_score))
    
    def _analyze_backbone_geometry(self, coords: np.ndarray) -> float:
        """Analyze backbone geometry (simplified Ramachandran)"""
        if len(coords) < 4:
            return 0.7
        
        valid_angles = 0
        total_angles = 0
        
        for i in range(1, len(coords) - 2):
            # Calculate phi and psi angles (simplified)
            v1 = coords[i] - coords[i-1]
            v2 = coords[i+1] - coords[i]
            v3 = coords[i+2] - coords[i+1]
            
            # Simplified angle calculation
            angle1 = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
            angle2 = np.arccos(np.clip(np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3)), -1, 1))
            
            # Check if angles are in reasonable range
            if 1.5 < angle1 < 2.5 and 1.5 < angle2 < 2.5:
                valid_angles += 1
            total_angles += 1
        
        return float(valid_angles / max(total_angles, 1))
    
    def _calculate_sasa(self, sequence: str, coords: np.ndarray) -> float:
        """Calculate solvent accessible surface area (simplified)"""
        if len(coords) < 2:
            return 0.5
        
        # Calculate average distance to neighbors
        neighbor_distances = []
        
        for i in range(len(coords)):
            distances = []
            for j in range(len(coords)):
                if i != j:
                    distance = np.linalg.norm(coords[i] - coords[j])
                    if distance < 10:  # Within interaction range
                        distances.append(distance)
            
            if distances:
                neighbor_distances.append(np.mean(distances))
        
        if not neighbor_distances:
            return 0.5
        
        avg_neighbor_distance = np.mean(neighbor_distances)
        # Optimal packing distance is around 4-6 Å
        sasa_score = max(0, 1 - abs(avg_neighbor_distance - 5) / 8)
        
        return float(min(1.0, sasa_score))
    
    def _calculate_overall_stability(self, scores: List[float]) -> float:
        """Calculate weighted overall stability score"""
        weights = [0.25, 0.20, 0.20, 0.15, 0.20]  # Weights for each component
        
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        return float(min(1.0, weighted_score))

# Global chemistry simulator
chemistry_simulator = ChemistrySimulator()

def run_chemistry_simulation(job_id: str, sequence: str, coords: np.ndarray) -> Dict:
    """Run comprehensive chemistry simulation"""
    from main import log_to_job
    
    log_to_job(job_id, "Chemistry: Running thermodynamic stability analysis")
    
    try:
        stability_metrics = chemistry_simulator.calculate_stability_metrics(sequence, coords)
        
        log_to_job(job_id, f"Chemistry: Overall stability score: {stability_metrics['stability_score']:.3f}")
        log_to_job(job_id, f"Chemistry: Hydrophobic core: {stability_metrics['hydrophobic_core']:.3f}")
        log_to_job(job_id, f"Chemistry: Electrostatic energy: {stability_metrics['electrostatic_energy']:.3f}")
        log_to_job(job_id, f"Chemistry: Thermodynamic feasibility: {stability_metrics['thermodynamic_feasibility']}")
        
        return stability_metrics
        
    except Exception as e:
        log_to_job(job_id, f"Chemistry simulation failed: {e}")
        return {
            'stability_score': 0.7,
            'thermodynamic_feasibility': 'Moderate',
            'error': str(e)
        }