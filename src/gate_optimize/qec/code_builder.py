"""
QEC code builder for creating syndrome measurement circuits from stabilizers.
"""

import stim
from typing import List

from importlib import resources
import sys
import gate_optimize
project_root = resources.files(gate_optimize).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# At the top of your test file
import logging
logging.basicConfig(
    filename=str(project_root / "data" / "test.log"),      # Name of the log file
    filemode='a',            # 'a' for append, 'w' for overwrite
    level=logging.INFO,      # Minimum level of messages to log
    format='%(asctime)s - %(levelname)s - %(message)s' # Log message format
)
logger = logging.getLogger(__name__)


class QECCodeBuilder:
    """Builds QEC circuits from stabilizer definitions."""
    
    def __init__(self, stabilizers: List[str], rounds: int = 3, logical_Z_operators: List[str] = None):
        """
        Initialize code builder.
        
        Args:
            stabilizers: List of stabilizer strings (e.g., ['+ZZ_____', '+_ZZ____'])
            rounds: Number of syndrome measurement rounds
            logical_Z_operators: Optional list of logical operator strings (e.g., ['XXXXXXX', 'ZZZZZZZ'])
        """
        self.stabilizers = stabilizers
        self.rounds = rounds
        self.num_qubits = len(stabilizers[0].lstrip('+-').replace('_', 'I'))
        self.explicit_logical_Z_operators = logical_Z_operators
        
        # Clean stabilizers
        self.clean_stabilizers = [s.lstrip('+-').replace('_', 'I') for s in stabilizers]
        self.stim_stabilizers = [stim.PauliString(s) for s in self.clean_stabilizers]
        
        # Clean logical operators if provided
        self.clean_logical_Z_operators = None
        if logical_Z_operators:
            self.clean_logical_Z_operators = [s.lstrip('+-').replace('_', 'I') for s in logical_Z_operators]
        
    def build_syndrome_circuit(self, include_logical_Z_operators: bool = True) -> stim.Circuit:
        """Build the QEC syndrome measurement circuit following proper Stim patterns."""
        circuit = stim.Circuit()
        
        # Data qubits: 0 to num_qubits-1
        # Ancilla qubits: num_qubits to num_qubits+len(stabilizers)-1
        num_ancillas = len(self.clean_stabilizers)
        total_qubits = self.num_qubits + num_ancillas
        
        # Step 1: Initial reset of all qubits
        circuit.append("R", list(range(total_qubits)))
        
        # Initialize data qubits in appropriate state if needed
        # for i in range(self.num_qubits):
            # # Check if any stabilizer has X on this qubit
            # has_x = any('X' in stab or 'Y' in stab for stab in self.clean_stabilizers)
            # if has_x:
            #     circuit.append("H", [i])
        
        # Step 2: First round - apply stabilizer circuits and use MR
        self._apply_stabilizer_round(circuit)
        
        # Use MR (measure and reset) for ancilla qubits
        ancilla_qubits = list(range(self.num_qubits, total_qubits))
        circuit.append("MR", ancilla_qubits)
        
        # Add detectors for first round - only for Z stabilizer measurements
        for stab_idx in range(num_ancillas):
            stab = self.clean_stabilizers[stab_idx]
            # Only add detector if this is a Z-only stabilizer (no X or Y)
            # WARNING ONLY CSS
            has_z = any(pauli == 'Z' for pauli in stab)
            has_x = any(pauli == 'X' for pauli in stab)
            has_y = any(pauli == 'Y' for pauli in stab)
            
            if has_z and not has_x and not has_y:
                # First round detectors check that Z measurements are deterministic
                current_rec = -(num_ancillas - stab_idx)
                circuit.append("DETECTOR", [stim.target_rec(current_rec)])
        
        # Step 3: Rounds 2 to final - use REPEAT block
        if self.rounds > 1:
            repeat_circuit = stim.Circuit()
            
            # Apply stabilizer measurements
            self._apply_stabilizer_round(repeat_circuit)
            
            # Use MR for ancilla qubits
            repeat_circuit.append("MR", ancilla_qubits)
            
            # Add detectors comparing current with previous round
            for stab_idx in range(num_ancillas):
                current_rec = -(num_ancillas - stab_idx)
                previous_rec = current_rec - num_ancillas  # Previous round
                repeat_circuit.append("DETECTOR", [stim.target_rec(current_rec), stim.target_rec(previous_rec)])
            
            # Build the REPEAT block using string construction
            repeat_count = self.rounds - 1
            # Indent the repeat circuit content
            repeat_content = str(repeat_circuit)
            indented_content = '\n'.join('    ' + line for line in repeat_content.split('\n') if line.strip())
            repeat_str = f"REPEAT {repeat_count} {{\n{indented_content}\n}}"
            circuit += stim.Circuit(repeat_str)
        
        # Step 4: Final logical measurements
        # Measure data qubits for logical operators
        circuit.append("M", list(range(self.num_qubits)))
        
        # Add final detectors comparing data measurements with last syndrome round
        for stab_idx in range(num_ancillas):
            stab = self.clean_stabilizers[stab_idx]
            # Only add detector if this is a Z-only stabilizer (no X or Y)
            # WARNING ONLY CSS
            has_z = any(pauli == 'Z' for pauli in stab)
            has_x = any(pauli == 'X' for pauli in stab)
            has_y = any(pauli == 'Y' for pauli in stab)
            
            if has_z and not has_x and not has_y:
                # Compare final syndrome with data qubit measurements
                syndrome_rec = -(self.num_qubits + num_ancillas - stab_idx)  # Last syndrome measurement
                # Find data qubits involved in this stabilizer
                stab = self.clean_stabilizers[stab_idx]
                data_targets = []
                for qubit_idx, pauli in enumerate(stab):
                    if pauli != 'I':  # Any non-identity Pauli
                        data_targets.append(stim.target_rec(-(self.num_qubits - qubit_idx)))
                
                if data_targets:
                    detector_targets = [stim.target_rec(syndrome_rec)] + data_targets
                    circuit.append("DETECTOR", detector_targets)
        
        # Step 5: Add logical observables if requested
        if include_logical_Z_operators:
            logical_ops = self._get_logical_Z_operators()
            for obs_idx, logical_op in enumerate(logical_ops):
                if logical_op:
                    circuit.append("OBSERVABLE_INCLUDE", logical_op, obs_idx)
        
        return circuit
    
    def _apply_stabilizer_round(self, circuit: stim.Circuit):
        """Apply one round of stabilizer measurements to the circuit."""
        num_ancillas = len(self.clean_stabilizers)
        
        # Apply stabilizer measurements
        for stab_idx, stab in enumerate(self.clean_stabilizers):
            ancilla = self.num_qubits + stab_idx
            
            # Determine stabilizer type and apply appropriate measurement structure
            has_x = any(pauli == 'X' for pauli in stab)
            has_z = any(pauli == 'Z' for pauli in stab)
            has_y = any(pauli == 'Y' for pauli in stab)
            
            # For stabilizers with X or Y components, we need different handling
            if has_x or has_y:
                # X stabilizer measurement: H_ancilla CX_{ancilla,data} H_ancilla
                circuit.append("H", [ancilla])
                
                for qubit_idx, pauli in enumerate(stab):
                    if pauli == 'X':
                        circuit.append("CX", [ancilla, qubit_idx])
                    elif pauli == 'Y':
                        # Y measurement: apply both X and Z components
                        circuit.append("CX", [ancilla, qubit_idx])
                        # For Y, we also need the Z component after X
                
                circuit.append("H", [ancilla])
                
                # Handle Y stabilizers: need additional Z gates after X measurement
                if has_y:
                    circuit.append("H", [ancilla])  # Prepare for Z measurement
                    for qubit_idx, pauli in enumerate(stab):
                        if pauli == 'Y':
                            circuit.append("CZ", [ancilla, qubit_idx])
                    circuit.append("H", [ancilla])  # Complete Z measurement
                        
            elif has_z:
                # Z stabilizer measurement: H_ancilla CZ_{ancilla,data} H_ancilla  
                circuit.append("H", [ancilla])
                
                for qubit_idx, pauli in enumerate(stab):
                    if pauli == 'Z':
                        circuit.append("CZ", [ancilla, qubit_idx])
                
                circuit.append("H", [ancilla])
    
    def _detect_code_type(self) -> str:
        """
        Detect the type of QEC code based on stabilizer patterns.
        Returns: 'repetition', 'steane', 'surface', 'toric', 'unknown'
        """
        # Repetition codes: n-1 stabilizers for n qubits, each with exactly 2 Z operators
        if len(self.clean_stabilizers) == self.num_qubits - 1:
            if all(stab.count('Z') == 2 and stab.count('X') == 0 for stab in self.clean_stabilizers):
                return 'repetition'
        
        # Steane code: 7 qubits, 6 stabilizers with specific patterns
        if self.num_qubits == 7 and len(self.clean_stabilizers) == 6:
            return 'steane'
            
        # Toric code patterns (simplified detection based on stabilizer structure)
        # Look for mix of X and Z stabilizers with specific patterns
        x_stabs = [s for s in self.clean_stabilizers if 'X' in s and 'Z' not in s]
        z_stabs = [s for s in self.clean_stabilizers if 'Z' in s and 'X' not in s]
        
        if len(x_stabs) > 0 and len(z_stabs) > 0:
            # Check for toric-like patterns (4-qubit plaquette/star operators)
            if any(s.count('X') == 4 for s in x_stabs) or any(s.count('Z') == 4 for s in z_stabs):
                return 'toric'
            # Check for surface code patterns  
            if any(s.count('X') <= 4 for s in x_stabs) and any(s.count('Z') <= 4 for s in z_stabs):
                return 'surface'
        
        return 'unknown'
    
    def _get_logical_Z_operators(self) -> list:
        """
        Get logical operators for different code types.
        Returns list of measurement record targets for each logical observable.
        """
        # If explicit logical operators are provided, use them
        if self.clean_logical_Z_operators:
            return self._convert_explicit_logical_Z_operators()
        
        # Otherwise, auto-detect based on code type
        code_type = self._detect_code_type()
        logical_ops = []
        
        if code_type == 'repetition':
            # Logical X operator: all qubits
            logical_x = [stim.target_rec(-(self.num_qubits - i)) for i in range(self.num_qubits)]
            logical_ops.append(logical_x)
            
        elif code_type == 'steane':
            # Logical Z operator for Steane code: ZIIIZZZ (qubits 0,4,5,6)
            logical_z_qubits = [0, 4, 5, 6]  # These qubits form a logical Z operator
            logical_z = [stim.target_rec(-(self.num_qubits - i)) for i in logical_z_qubits]
            logical_ops.append(logical_z)
            
        elif code_type == 'toric':
            # For toric codes, define logical operators based on geometry
            logical_ops.extend(self._get_toric_logical_Z_operators())
            
        elif code_type == 'surface':
            # For surface codes, define logical operators at boundaries
            logical_ops.extend(self._get_surface_logical_Z_operators())
            
        # If no logical operators found, return empty list
        # The circuit will still work for syndrome-based error analysis
        return logical_ops
    
    def _get_toric_logical_Z_operators(self) -> list:
        """
        Get logical operators for toric codes.
        Defines logical X and Z operators as non-contractible loops.
        """
        logical_ops = []
        
        # For simplified 3x3 toric code (9 qubits)
        # This is a heuristic approach - real toric codes need proper geometry
        if self.num_qubits >= 8:
            try:
                # Logical X operator (horizontal loop)
                # Target qubits that form a logical X loop
                logical_x_qubits = []
                for i in range(min(4, self.num_qubits)):  # First few qubits as logical X
                    logical_x_qubits.append(stim.target_rec(-(self.num_qubits - i)))
                
                if len(logical_x_qubits) >= 2:
                    logical_ops.append(logical_x_qubits)
                
                # Logical Z operator (vertical loop) 
                # Target different qubits that form a logical Z loop
                logical_z_qubits = []
                start_idx = min(4, self.num_qubits // 2)
                for i in range(start_idx, min(start_idx + 4, self.num_qubits)):
                    logical_z_qubits.append(stim.target_rec(-(self.num_qubits - i)))
                
                if len(logical_z_qubits) >= 2:
                    logical_ops.append(logical_z_qubits)
                    
            except Exception:
                # If logical operator construction fails, return empty list
                pass
        
        return logical_ops
    
    def _get_surface_logical_Z_operators(self) -> list:
        """
        Get logical operators for surface codes.
        Defines logical operators along boundaries.
        """
        logical_ops = []
        
        # For surface codes, logical operators are typically along boundaries
        # This is a simplified approach
        if self.num_qubits >= 4:
            try:
                # Logical operator along one boundary
                boundary_qubits = []
                step = max(1, self.num_qubits // 4)  # Sample qubits across the boundary
                for i in range(0, self.num_qubits, step):
                    if len(boundary_qubits) < 4:  # Limit to 4 qubits
                        boundary_qubits.append(stim.target_rec(-(self.num_qubits - i)))
                
                if len(boundary_qubits) >= 2:
                    logical_ops.append(boundary_qubits)
                    
            except Exception:
                # If logical operator construction fails, return empty list
                pass
        
        return logical_ops
    
    def _is_simple_code(self) -> bool:
        """
        Check if this is a simple code where logical operators can be easily determined.
        Returns True for repetition codes and Steane code.
        """
        code_type = self._detect_code_type()
        return code_type in ['repetition', 'steane']
    
    def _convert_explicit_logical_Z_operators(self) -> list:
        """
        Convert explicit logical operator strings to measurement record targets.
        For proper QEC codes, we need to ensure logical operators commute with stabilizers.
        """
        logical_ops = []
        
        for logical_op_str in self.clean_logical_Z_operators:
            # Find qubits where this logical operator acts as Z (for logical Z measurement)
            qubit_targets = []
            for qubit_idx, pauli in enumerate(logical_op_str):
                if pauli == 'Z':  # Only Z operators for logical Z measurement
                    # Target the measurement record for this qubit
                    qubit_targets.append(stim.target_rec(-(self.num_qubits - qubit_idx)))
            
            if qubit_targets:
                logical_ops.append(qubit_targets)
        
        return logical_ops
    
    def _get_logical_operator_targets(self) -> list:
        """Get logical operator targets for simple codes (backward compatibility)."""
        logical_ops = self._get_logical_Z_operators()
        return logical_ops[0] if logical_ops else []