"""
QEC decoder implementations using MWPM and BP-OSD methods.
"""

import numpy as np
import stim
from typing import List, Dict, Optional, Tuple, Union
from pymatching import Matching
from ldpc import bposd_decoder


class QECDecoder:
    """Quantum Error Correction decoder with MWPM and BP-OSD support."""
    
    def __init__(self, circuit: stim.Circuit, method: str = 'mwpm'):
        """
        Initialize QEC decoder.
        
        Args:
            circuit: Stim circuit with detectors and observables
            method: Decoding method ('mwpm' or 'bp_osd')
        """
        self.circuit = circuit
        self.method = method.lower()
        
        # Initialize decoder
        self._setup_decoder()
    
    def _setup_decoder(self):
        """Setup the appropriate decoder based on method."""
        if self.method == 'mwpm':
            self._setup_mwpm_decoder()
        elif self.method == 'bp_osd':
            self._setup_bp_osd_decoder()
        else:
            raise ValueError(f"Unknown decoding method: {self.method}")
    
    def _setup_mwpm_decoder(self):
        """Setup Minimum Weight Perfect Matching decoder."""
        # Compile detector sampler
        self.detector_sampler = self.circuit.compile_detector_sampler()
        
        # Setup PyMatching decoder
        self.matching_decoder = Matching.from_detector_error_model(
            self.circuit.detector_error_model(decompose_errors=True)
        )
    
    def _setup_bp_osd_decoder(self):
        """Setup BP-OSD decoder."""
        try:
            self.detector_sampler = self.circuit.compile_detector_sampler()
            
            # Get parity check matrix from circuit
            dem = self.circuit.detector_error_model(decompose_errors=True)
            self.parity_check_matrix = self._dem_to_check_matrix(dem)
            
            # Validate matrix dimensions
            if self.parity_check_matrix.size == 0:
                raise RuntimeError("Empty parity check matrix")
            
            # Setup BP-OSD decoder with error handling
            self.bp_decoder = bposd_decoder(
                self.parity_check_matrix,
                error_rate=0.01,
                max_iter=50,
                bp_method="ms",
                osd_method="osd_cs",
                osd_order=4
            )
        except Exception as e:
            raise RuntimeError(f"Failed to setup BP-OSD decoder: {e}") from e
    
    def _dem_to_check_matrix(self, dem) -> np.ndarray:
        """Convert detector error model to parity check matrix."""
        # Extract check matrix from detector error model
        num_detectors = dem.num_detectors
        num_errors = 0
        
        # Count number of error mechanisms
        for instruction in dem:
            if instruction.type == "error":
                num_errors += 1
        
        if num_errors == 0 or num_detectors == 0:
            # Fallback: create identity matrix
            return np.eye(max(1, num_detectors), dtype=int)
        
        # Create parity check matrix
        H = np.zeros((num_detectors, num_errors), dtype=int)
        
        error_idx = 0
        for instruction in dem:
            if instruction.type == "error":
                for target in instruction.targets_copy():
                    if target.is_detector_id():
                        detector_id = target.value
                        if detector_id < num_detectors:
                            H[detector_id, error_idx] = 1
                error_idx += 1
        
        return H
    
    def decode_single_shot(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a single syndrome measurement."""
        if self.method == 'mwpm':
            return self.matching_decoder.decode(syndrome)
        elif self.method == 'bp_osd':
            return self.bp_decoder.decode(syndrome)
    
    def simulate_errors(self, physical_error_rate: float, num_shots: int) -> Dict:
        """
        Simulate quantum errors and decode them.
        
        Args:
            physical_error_rate: Physical error rate for gates/measurements
            num_shots: Number of shots to simulate
            
        Returns:
            Dictionary with simulation results
        """
        # Add noise to circuit
        noisy_circuit = self._add_noise_to_circuit(physical_error_rate)
        
        # Sample detection events
        sampler = noisy_circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(
            shots=num_shots,
            separate_observables=True
        )
        
        # Decode each shot
        corrections = []
        logical_errors = 0
        
        for shot_idx in range(num_shots):
            syndrome = detection_events[shot_idx]
            
            if self.method == 'mwpm':
                correction = self.matching_decoder.decode(syndrome)
            elif self.method == 'bp_osd':
                correction = self.bp_decoder.decode(syndrome.astype(int))
            
            corrections.append(correction)
            
            # Check if logical error occurred (simplified)
            if len(observable_flips[shot_idx]) > 0:
                predicted_obs = 0 if correction is None else np.sum(correction) % 2
                actual_obs = np.sum(observable_flips[shot_idx]) % 2
                
                if predicted_obs != actual_obs:
                    logical_errors += 1
        
        logical_error_rate = logical_errors / num_shots
        
        return {
            'physical_error_rate': physical_error_rate,
            'logical_error_rate': logical_error_rate,
            'num_shots': num_shots,
            'num_logical_errors': logical_errors,
            'detection_events': detection_events,
            'observable_flips': observable_flips,
            'corrections': np.array(corrections) if corrections else np.array([])
        }
    
    def _add_noise_to_circuit(self, error_rate: float) -> stim.Circuit:
        """Add noise to the circuit."""
        noisy_circuit = stim.Circuit()
        
        for instruction in self.circuit:
            noisy_circuit.append(instruction)
            
            if instruction.name in ['H', 'CNOT', 'CZ']:
                # Add depolarizing noise
                if instruction.name == 'H':
                    noisy_circuit.append("DEPOLARIZE1", instruction.targets_copy(), error_rate)
                else:
                    noisy_circuit.append("DEPOLARIZE2", instruction.targets_copy(), error_rate)
            elif instruction.name == 'M':
                # Add measurement noise
                noisy_circuit.append("X_ERROR", instruction.targets_copy(), error_rate)
        
        return noisy_circuit