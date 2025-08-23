"""
Circuit fidelity simulation with realistic noise models.

This module provides functions to evaluate quantum circuit fidelity and error rates
using realistic physical noise models for benchmarking RL and classical synthesis methods.
"""

import numpy as np


def simulate_circuit_fidelity(qiskit_circuit, error_model='physical', num_shots=1000, seed=42):
    """Simulate the fidelity and error rates of a quantum circuit with realistic noise.
    
    Args:
        qiskit_circuit: Qiskit QuantumCircuit object to evaluate
        error_model: Type of error model to use ('physical', 'depolarizing', 'thermal')
        num_shots: Number of simulation shots for statistical sampling
        seed: Random seed for reproducible results
        
    Returns:
        dict: Contains fidelity, error_rate, gate_errors, and measurement_errors
    """

    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
    from qiskit_aer.noise import amplitude_damping_error, phase_damping_error
    from qiskit import transpile
    import numpy as np

    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create noise model based on the specified error model
    noise_model = _create_noise_model(error_model)
    
    # Create ideal circuit (no measurements) for fidelity calculation
    ideal_circuit = qiskit_circuit.copy()
    # Remove measurements if they exist
    if any(instr.name == 'measure' for instr, _, _ in ideal_circuit.data):
        ideal_circuit = ideal_circuit.remove_final_measurements(inplace=False)
    
    # Transpile both circuits with the same backend to ensure they're equivalent
    ideal_simulator = AerSimulator(method='statevector')
    noisy_simulator = AerSimulator(noise_model=noise_model, method='density_matrix')
    
    # Transpile both circuits identically 
    transpiled_ideal = transpile(ideal_circuit, ideal_simulator)
    transpiled_noisy = transpile(ideal_circuit, noisy_simulator)  # Use same base circuit
    
    # Debug: Check if circuits are different after transpilation
    print(f"Original circuit gates: {len(ideal_circuit.data)}")
    print(f"Transpiled ideal gates: {len(transpiled_ideal.data)}")
    print(f"Transpiled noisy gates: {len(transpiled_noisy.data)}")
    
    # Simulate ideal circuit
    transpiled_ideal.save_statevector()  # Save statevector for retrieval
    ideal_result = ideal_simulator.run(transpiled_ideal, shots=1).result()
    ideal_statevector = ideal_result.data(0)['statevector']
    
    # Simulate noisy circuit  
    transpiled_noisy.save_density_matrix()  # Save density matrix for retrieval
    noisy_result = noisy_simulator.run(transpiled_noisy, shots=1).result()
    noisy_density_matrix = noisy_result.data(0)['density_matrix']
    
    # Calculate fidelity between ideal and noisy states
    fidelity = _calculate_state_fidelity(ideal_statevector, noisy_density_matrix)
    error_rate = 1.0 - fidelity
    
    # Analyze gate-specific errors
    gate_errors = _analyze_gate_errors(qiskit_circuit, error_model)
    
    # Estimate measurement errors (if circuit has measurements)
    measurement_errors = _estimate_measurement_errors(qiskit_circuit, error_model)
    
    return {
        'fidelity': float(fidelity),
        'error_rate': float(error_rate),
        'gate_errors': gate_errors,
        'measurement_errors': measurement_errors,
        'total_gates': len(qiskit_circuit.data),
        'circuit_depth': qiskit_circuit.depth(),
        'error_model': error_model
    }


def _create_noise_model(error_model='physical'):
    """Create a noise model based on the specified error type."""
    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
        from qiskit_aer.noise import amplitude_damping_error, phase_damping_error
    except ImportError:
        return None
        
    noise_model = NoiseModel()
    
    if error_model == 'physical':
        # Realistic physical error rates for trapped ion systems
        # Single-qubit gate errors
        single_qubit_error = 1e-4  # 0.01% error rate
        single_qubit_time = 20e-6  # 20 microseconds
        
        # Two-qubit gate errors  
        two_qubit_error = 1e-3     # 0.1% error rate
        two_qubit_time = 100e-6    # 100 microseconds
        
        # Relaxation times
        T1 = 50e-3  # 50 ms T1 time
        T2 = 30e-3  # 30 ms T2 time
        
        # Add single-qubit gate errors
        single_qubit_thermal = thermal_relaxation_error(T1, T2, single_qubit_time)
        single_qubit_depol = depolarizing_error(single_qubit_error, 1)
        single_qubit_combined = single_qubit_thermal.compose(single_qubit_depol)
        
        noise_model.add_all_qubit_quantum_error(single_qubit_combined, ['h', 's', 'sdg', 'x', 'y', 'z', 'sx'])
        
        # Add two-qubit gate errors
        two_qubit_thermal = thermal_relaxation_error(T1, T2, two_qubit_time)
        two_qubit_depol = depolarizing_error(two_qubit_error, 2)
        two_qubit_combined = two_qubit_thermal.compose(two_qubit_depol)
        
        noise_model.add_all_qubit_quantum_error(two_qubit_combined, ['cx', 'cz', 'cnot'])
        
        # Add measurement errors
        measurement_error = 0.02  # 2% measurement error
        from qiskit_aer.noise import ReadoutError
        readout_error = ReadoutError([[1-measurement_error, measurement_error], 
                                    [measurement_error, 1-measurement_error]])
        noise_model.add_all_qubit_readout_error(readout_error)
        
    elif error_model == 'depolarizing':
        # Simple depolarizing channel
        single_qubit_error = depolarizing_error(0.001, 1)
        two_qubit_error = depolarizing_error(0.01, 2)
        
        noise_model.add_all_qubit_quantum_error(single_qubit_error, ['h', 's', 'sdg', 'x', 'y', 'z', 'sx'])
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx', 'cz', 'cnot'])
        
    elif error_model == 'thermal':
        # Thermal relaxation only
        T1, T2 = 50e-3, 30e-3
        single_qubit_thermal = thermal_relaxation_error(T1, T2, 20e-6)
        # For 2-qubit gates, apply thermal error to each qubit individually
        two_qubit_thermal = thermal_relaxation_error(T1, T2, 100e-6).expand(thermal_relaxation_error(T1, T2, 100e-6))
        
        noise_model.add_all_qubit_quantum_error(single_qubit_thermal, ['h', 's', 'sdg', 'x', 'y', 'z', 'sx'])
        noise_model.add_all_qubit_quantum_error(two_qubit_thermal, ['cx', 'cz', 'cnot'])
    
    return noise_model


def _calculate_state_fidelity(ideal_state, noisy_density_matrix):
    """Calculate fidelity between ideal statevector and noisy density matrix."""
    try:
        from qiskit.quantum_info import state_fidelity
        return state_fidelity(ideal_state, noisy_density_matrix)
    except ImportError:
        # Fallback calculation
        import numpy as np
        # Convert statevector to density matrix
        ideal_dm = np.outer(ideal_state, np.conj(ideal_state))
        # Calculate fidelity: Tr(sqrt(sqrt(rho1) * rho2 * sqrt(rho1)))
        sqrt_ideal = np.sqrt(ideal_dm + 1e-12)  # Add small epsilon for numerical stability
        temp = sqrt_ideal @ noisy_density_matrix @ sqrt_ideal
        sqrt_temp = np.sqrt(temp + 1e-12)
        fidelity = np.trace(sqrt_temp).real
        return max(0.0, min(1.0, fidelity))  # Clamp to [0,1]


def _analyze_gate_errors(qiskit_circuit, error_model):
    """Analyze expected errors for each gate type in the circuit."""
    gate_counts = {}
    gate_errors = {}
    
    # Count gates by type
    for instruction, qubits, clbits in qiskit_circuit.data:
        gate_name = instruction.name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
    
    # Estimate error rates by gate type based on error model
    if error_model == 'physical':
        error_rates = {
            'h': 1e-4, 's': 1e-4, 'sdg': 1e-4, 'x': 1e-4, 'y': 1e-4, 'z': 1e-4, 'sx': 1e-4,
            'cx': 1e-3, 'cz': 1e-3, 'cnot': 1e-3,
            'measure': 0.02
        }
    elif error_model == 'depolarizing':
        error_rates = {
            'h': 0.001, 's': 0.001, 'sdg': 0.001, 'x': 0.001, 'y': 0.001, 'z': 0.001, 'sx': 0.001,
            'cx': 0.01, 'cz': 0.01, 'cnot': 0.01,
            'measure': 0.01
        }
    else:  # thermal
        error_rates = {
            'h': 5e-4, 's': 5e-4, 'sdg': 5e-4, 'x': 5e-4, 'y': 5e-4, 'z': 5e-4, 'sx': 5e-4,
            'cx': 5e-3, 'cz': 5e-3, 'cnot': 5e-3,
            'measure': 0.015
        }
    
    for gate_name, count in gate_counts.items():
        error_rate = error_rates.get(gate_name, 0.001)  # Default 0.1% for unknown gates
        gate_errors[gate_name] = {
            'count': count,
            'error_rate_per_gate': error_rate,
            'total_error_contribution': 1.0 - (1.0 - error_rate) ** count
        }
    
    return gate_errors


def _estimate_measurement_errors(qiskit_circuit, error_model):
    """Estimate measurement errors in the circuit."""
    measurement_count = sum(1 for instr, _, _ in qiskit_circuit.data if instr.name == 'measure')
    
    if measurement_count == 0:
        return {'count': 0, 'error_rate': 0.0}
    
    # Measurement error rates by model
    measurement_error_rates = {
        'physical': 0.02,    # 2% for realistic systems
        'depolarizing': 0.01, # 1% for simplified model
        'thermal': 0.015     # 1.5% for thermal model
    }
    
    error_rate = measurement_error_rates.get(error_model, 0.01)
    
    return {
        'count': measurement_count,
        'error_rate_per_measurement': error_rate,
        'total_measurement_error': 1.0 - (1.0 - error_rate) ** measurement_count
    }


def _analytical_fidelity_estimate(qiskit_circuit, error_model):
    """Fallback analytical estimation when qiskit_aer is not available."""
    import numpy as np
    
    # Count different gate types
    single_qubit_gates = 0
    two_qubit_gates = 0
    measurements = 0
    
    for instruction, qubits, clbits in qiskit_circuit.data:
        if instruction.name == 'measure':
            measurements += 1
        elif len(qubits) == 1:
            single_qubit_gates += 1
        else:
            two_qubit_gates += 1
    
    # Error rates by model
    if error_model == 'physical':
        single_error, two_error, meas_error = 1e-4, 1e-3, 0.02
    elif error_model == 'depolarizing':
        single_error, two_error, meas_error = 0.001, 0.01, 0.01
    else:  # thermal
        single_error, two_error, meas_error = 5e-4, 5e-3, 0.015
    
    # Estimate overall fidelity (assuming independent errors)
    fidelity = (1.0 - single_error) ** single_qubit_gates
    fidelity *= (1.0 - two_error) ** two_qubit_gates
    fidelity *= (1.0 - meas_error) ** measurements
    
    return {
        'fidelity': float(fidelity),
        'error_rate': float(1.0 - fidelity),
        'gate_errors': {
            'single_qubit': {'count': single_qubit_gates, 'error_rate_per_gate': single_error},
            'two_qubit': {'count': two_qubit_gates, 'error_rate_per_gate': two_error}
        },
        'measurement_errors': {'count': measurements, 'error_rate_per_measurement': meas_error},
        'total_gates': len(qiskit_circuit.data),
        'circuit_depth': qiskit_circuit.depth(),
        'error_model': error_model,
        'note': 'Analytical estimate (qiskit_aer not available)'
    }


# Example usage function for testing
def example_usage():
    """Example of how to use the fidelity simulation function."""
    from qiskit import QuantumCircuit
    
    # Create a simple test circuit
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.s(0)
    qc.measure_all()
    
    # Evaluate with different error models
    for model in ['physical', 'depolarizing', 'thermal']:
        result = simulate_circuit_fidelity(qc, error_model=model)
        print(f"\n{model.upper()} Error Model:")
        print(f"  Fidelity: {result['fidelity']:.6f}")
        print(f"  Error Rate: {result['error_rate']:.6f}")
        print(f"  Total Gates: {result['total_gates']}")
        print(f"  Circuit Depth: {result['circuit_depth']}")
        
        if 'gate_errors' in result:
            print("  Gate-specific errors:")
            for gate, error_info in result['gate_errors'].items():
                if isinstance(error_info, dict) and 'count' in error_info:
                    print(f"    {gate}: {error_info['count']} gates, "
                          f"{error_info.get('error_rate_per_gate', 0):.6f} error rate")


def _create_custom_noise_model(single_q_error_rate, two_q_error_rate, measurement_error_rate):
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

    noise_model = NoiseModel()

    p_1q = single_q_error_rate
    error_1q = depolarizing_error(p_1q, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['h', 's', 'sdg', 'x', 'y', 'z', 'sx'])

    p_2q = two_q_error_rate
    error_2q = depolarizing_error(p_2q, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'cnot'])

    p_meas = measurement_error_rate
    error_meas = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
    noise_model.add_all_qubit_readout_error(error_meas)

    return noise_model


def simulate_circuit_fidelity_custom(qiskit_circuit, custom_errors, num_shots=1000, seed=42):

    from qiskit_aer import AerSimulator
    from qiskit import transpile

    p_1q = custom_errors.get('single_qubit', 1e-3)
    p_2q = custom_errors.get('two_qubit', 5e-2)
    p_meas = custom_errors.get('measurement', 0.02)

    noise_model = _create_custom_noise_model(p_1q, p_2q, p_meas)
    
    ideal_circuit = qiskit_circuit.copy()
    if any(instr.name == 'measure' for instr, _, _ in ideal_circuit.data):
        ideal_circuit = ideal_circuit.remove_final_measurements(inplace=False)
    
    ideal_simulator = AerSimulator(method='statevector')
    ideal_circuit.save_statevector()
    ideal_result = ideal_simulator.run(ideal_circuit, shots=1).result()
    ideal_statevector = ideal_result.data(0)['statevector']

    noisy_simulator = AerSimulator(noise_model=noise_model, method='density_matrix')
    noisy_circuit = qiskit_circuit.copy()
    if any(instr.name == 'measure' for instr, _, _ in noisy_circuit.data):
        noisy_circuit = noisy_circuit.remove_final_measurements(inplace=False)
    
    transpiled_circuit = transpile(noisy_circuit, noisy_simulator)
    transpiled_circuit.save_density_matrix()
    noisy_result = noisy_simulator.run(transpiled_circuit, shots=1).result()
    noisy_density_matrix = noisy_result.data(0)['density_matrix']
    
    fidelity = _calculate_state_fidelity(ideal_statevector, noisy_density_matrix)
    
    return {
        'fidelity': float(fidelity),
        'error_rate': float(1.0 - fidelity),
        'used_errors': {'single_qubit': p_1q, 'two_qubit': p_2q, 'measurement': p_meas}
    }


if __name__ == "__main__":
    example_usage()
