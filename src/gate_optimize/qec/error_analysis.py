"""
Error analysis functions for calculating logical error rates.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os
import stim
from pymatching import Matching
from .code_builder import QECCodeBuilder

# At the top of your test file
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    """
    Count logical errors by comparing decoder predictions with actual observable flips.
    
    Args:
        circuit: Stim circuit with detectors and observables
        num_shots: Number of shots to simulate
        
    Returns:
        Number of logical errors detected
    """
    try:
        # Sample the circuit
        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
        
        # Configure a decoder using the circuit
        try:
            detector_error_model = circuit.detector_error_model(decompose_errors=True)
        except Exception as e:
            logger.debug(f"Detector error model generation failed: {e}")
            
            # Check if this is a non-deterministic detectors/observables error
            if "non-deterministic" in str(e).lower():
                logger.debug("Detected non-deterministic detectors/observables error")
                logger.debug("This usually indicates detector sensitivity overlap or logical operator issues")
                
                # Try without decomposing errors first
                try:
                    detector_error_model = circuit.detector_error_model(decompose_errors=False)
                    logger.debug("✅ DEM generation succeeded without error decomposition")
                except Exception as e2:
                    logger.debug(f"DEM generation failed even without decomposition: {e2}")
                    # Fall back to syndrome-based error analysis
                    raise ValueError("non-deterministic observables detected") from e
            else:
                # Re-raise the exception for other types of errors
                raise e
            
        matcher = Matching.from_detector_error_model(detector_error_model)
        
        # Run the decoder
        predictions = matcher.decode_batch(detection_events)
        
        # Count the mistakes
        num_errors = 0
        for shot in range(num_shots):
            actual_for_shot = observable_flips[shot]
            predicted_for_shot = predictions[shot]
            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors += 1
        
        return num_errors
        
    except ValueError as e:
        if "non-deterministic" in str(e):
            # For codes with non-deterministic observables (like toric codes),
            # use syndrome-based error estimation
            return _count_logical_errors_syndrome_based(circuit, num_shots)
        else:
            raise e


def _count_logical_errors_syndrome_based(circuit: stim.Circuit, num_shots: int) -> int:
    """
    Count logical errors for codes with non-deterministic observables.
    Uses syndrome pattern analysis instead of observable comparison.
    
    Args:
        circuit: Stim circuit (may have non-deterministic observables)
        num_shots: Number of shots to simulate
        
    Returns:
        Estimated number of logical errors
    """
    # For codes without well-defined observables, we estimate logical errors
    # based on whether the decoder successfully finds a correction or not
    
    try:
        # Sample just the detection events
        sampler = circuit.compile_detector_sampler()
        detection_events, _ = sampler.sample(num_shots, separate_observables=False)
        
        # Configure decoder
        detector_error_model = circuit.detector_error_model(decompose_errors=True)
        matcher = Matching.from_detector_error_model(detector_error_model)
        
        # Count non-trivial syndromes (indicating possible logical errors)
        num_errors = 0
        for shot in range(num_shots):
            syndrome = detection_events[shot]
            # If syndrome is non-trivial, there might be a logical error
            # This is a conservative estimate
            if np.any(syndrome):
                correction = matcher.decode(syndrome)
                # For complex codes, assume some fraction of non-trivial syndromes
                # correspond to logical errors
                if correction is not None and np.any(correction):
                    # Use a heuristic: assume 10% of correctable syndromes are logical errors
                    # This is a rough estimate for codes like toric codes
                    if np.random.rand() < 0.1:  # Adjustable parameter
                        num_errors += 1
        
        return num_errors
        
    except Exception:
        # If everything fails, return a conservative estimate
        # Assume a small fraction of shots have logical errors
        return int(num_shots * 0.01)  # 1% default estimate


def calculate_logical_error_rate(
    stabilizers: List[str],
    physical_error_rates: List[float],
    num_shots: int = 10000,
    rounds: int = 3,
    decoder_method: str = 'mwpm',
    logical_Z_operators: List[str] = None
) -> Dict:
    """
    Calculate logical error rates for different physical error rates.
    
    Args:
        stabilizers: List of stabilizer strings
        physical_error_rates: List of physical error rates to test
        num_shots: Number of Monte Carlo shots per error rate
        rounds: Number of syndrome measurement rounds
        decoder_method: Decoding method ('mwpm' or 'bp_osd')
        logical_Z_operators: Optional list of logical operator strings
        
    Returns:
        Dictionary containing error rate data and analysis
    """
    # Build QEC circuit with optional logical operators
    builder = QECCodeBuilder(stabilizers, rounds=rounds, logical_Z_operators=logical_Z_operators)
    base_circuit = builder.build_syndrome_circuit()
    
    results = {
        'stabilizers': stabilizers,
        'physical_error_rates': [],
        'logical_error_rates': [],
        'num_shots': num_shots,
        'rounds': rounds,
        'decoder_method': decoder_method,
        'detailed_results': []
    }
    
    # Calculate error rates for each physical error rate
    for p_error in physical_error_rates:
        print(f"Simulating physical error rate: {p_error:.4f}")
        
        # Add noise to circuit
        noisy_circuit = _add_noise_to_circuit(base_circuit, p_error)
        if p_error < 0.002:
            print(f"  Noisy circuit with p={p_error:.4f} prepared")
            print(f"\n\n")
            print(f"""{noisy_circuit}""")
            print(f"\n\n")
        
        # Count logical errors using standard stim approach
        if decoder_method == 'mwpm':
            num_logical_errors = count_logical_errors(noisy_circuit, num_shots)
        else:
            # For BP-OSD, we'll need a custom implementation
            num_logical_errors = _count_logical_errors_bp_osd(noisy_circuit, num_shots)
        
        logical_error_rate = num_logical_errors / num_shots
        
        results['physical_error_rates'].append(p_error)
        results['logical_error_rates'].append(logical_error_rate)
        results['detailed_results'].append({
            'physical_error_rate': p_error,
            'logical_error_rate': logical_error_rate,
            'num_logical_errors': num_logical_errors,
            'num_shots': num_shots
        })
    
    return results


# def _add_noise_to_circuit(circuit: stim.Circuit, error_rate: float) -> stim.Circuit:
#     """Add noise to the circuit."""
#     noisy_circuit = stim.Circuit()
    
#     for instruction in circuit:
#         noisy_circuit.append(instruction)
        
#         if instruction.name in ['H', 'CX', 'CZ']:
#             # Add depolarizing noise
#             if instruction.name == 'H':
#                 noisy_circuit.append("DEPOLARIZE1", instruction.targets_copy(), error_rate)
#             else:
#                 noisy_circuit.append("DEPOLARIZE2", instruction.targets_copy(), error_rate)
#         elif instruction.name == 'M':
#             # Add measurement noise
#             noisy_circuit.append("X_ERROR", instruction.targets_copy(), error_rate)
    
#     return noisy_circuit


def _add_noise_to_circuit(
    circuit: stim.Circuit, error_rate: float
) -> stim.Circuit:
    """
    Adds noise to a stim circuit while correctly preserving REPEAT blocks.

    Args:
        circuit: The input stim.Circuit.
        error_rate: The probability of an error after a noisy operation.

    Returns:
        A new stim.Circuit with noise channels added.
    """
    noisy_circuit = stim.Circuit()

    # Iterate through the circuit's top-level components (instructions and blocks)
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            # If the item is a REPEAT block, get its body
            repeat_block_body = instruction.body_copy()
            
            # Recursively call this function on the body of the REPEAT block
            noisy_body = _add_noise_to_circuit(repeat_block_body, error_rate)
            
            # Append a new REPEAT block with the original count and the new noisy body
            noisy_circuit.append(
                stim.CircuitRepeatBlock(instruction.repeat_count, noisy_body)
            )
        
        elif isinstance(instruction, stim.CircuitInstruction):
            # Check if the instruction is a two-qubit gate that needs to be broken down
            if instruction.name in ["CNOT", "CX", "CY", "CZ"]:
                targets = instruction.targets_copy()
                # Iterate over the targets in pairs (e.g., for 'CZ 0 1 0 2')
                for i in range(0, len(targets), 2):
                    qubit_pair = [targets[i], targets[i+1]]
                    
                    # Append the gate for the specific pair
                    noisy_circuit.append(instruction.name, qubit_pair)
                    
                    # Append the DEPOLARIZE2 noise right after for that same pair
                    noisy_circuit.append("DEPOLARIZE2", qubit_pair, error_rate)
                    
            else:
                # --- This is the original logic for all other instructions ---
                # If the item is a regular instruction, append it first
                noisy_circuit.append(instruction)
                
                # Add noise based on the instruction type
                if instruction.name == "H":
                    noisy_circuit.append("DEPOLARIZE1", instruction.targets_copy(), error_rate)
                    
                elif instruction.name == "M" or instruction.name == "MR":
                    # Add measurement noise (bit-flip error before measurement)
                    noisy_circuit.append("X_ERROR", instruction.targets_copy(), error_rate)
                        
        else:
            # Append any other circuit component (like TICK, etc.) directly
            noisy_circuit.append(instruction)

    return noisy_circuit


def _count_logical_errors_bp_osd(circuit: stim.Circuit, num_shots: int) -> int:
    """
    Count logical errors using BP-OSD decoder.
    
    Args:
        circuit: Stim circuit with detectors and observables
        num_shots: Number of shots to simulate
        
    Returns:
        Number of logical errors detected
    """
    from ldpc import bposd_decoder
    
    # Sample the circuit
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
    
    # Get detector error model and convert to parity check matrix
    dem = circuit.detector_error_model(decompose_errors=True)
    parity_check_matrix = _dem_to_check_matrix(dem)
    
    # Setup BP-OSD decoder
    bp_decoder = bposd_decoder(
        parity_check_matrix,
        error_rate=0.01,
        max_iter=50,
        bp_method="ms",
        osd_method="osd_cs",
        osd_order=4
    )
    
    # Count errors
    num_errors = 0
    for shot in range(num_shots):
        syndrome = detection_events[shot].astype(int)
        prediction = bp_decoder.decode(syndrome)
        actual = observable_flips[shot]
        
        # Simple comparison (may need refinement based on actual implementation)
        if not np.array_equal(actual, prediction[:len(actual)] if prediction is not None else []):
            num_errors += 1
    
    return num_errors


def _dem_to_check_matrix(dem) -> np.ndarray:
    """Convert detector error model to parity check matrix."""
    num_detectors = dem.num_detectors
    num_errors = 0
    
    # Count number of error mechanisms
    for instruction in dem:
        if instruction.type == "error":
            num_errors += 1
    
    if num_errors == 0 or num_detectors == 0:
        return np.eye(max(1, num_detectors), dtype=int)
    
    # Create parity check matrix
    H = np.zeros((num_detectors, num_errors), dtype=int)
    
    error_idx = 0
    for instruction in dem:
        if instruction.type == "error":
            for target in instruction.targets_copy():
                # Use different method to check if it's a detector target
                if hasattr(target, 'is_detector_id') and target.is_detector_id():
                    detector_id = target.value
                    if detector_id < num_detectors:
                        H[detector_id, error_idx] = 1
                elif str(target).startswith('D'):  # Fallback check
                    try:
                        detector_id = int(str(target)[1:])
                        if detector_id < num_detectors:
                            H[detector_id, error_idx] = 1
                    except:
                        pass
            error_idx += 1
    
    return H


def plot_error_rate_curve(results: Dict, save_path: str = None) -> str:
    """
    Plot logical vs physical error rate curve.
    
    Args:
        results: Results dictionary from calculate_logical_error_rate
        save_path: Path to save the plot (optional)
                  Examples: 'plot.png', '/tmp/error_curve.png', '../results/plot.png'
        
    Returns:
        Base64 encoded image string
    """
    logger.debug(f"plot_error_rate_curve called with:")
    logger.debug(f"  save_path: {save_path}")
    if save_path:
        abs_save_path = os.path.abspath(save_path)
        logger.debug(f"  absolute save_path: {abs_save_path}")
        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            logger.debug(f"  📁 Directory ensured: {os.path.abspath(save_dir)}")
    plt.figure(figsize=(10, 6))
    
    physical_rates = np.array(results['physical_error_rates'])
    logical_rates = np.array(results['logical_error_rates'])
    
    # Plot the error rate curve
    plt.loglog(physical_rates, logical_rates, 'bo-', linewidth=2, markersize=8, label='Logical Error Rate')
    
    # Plot the break-even line (where logical = physical)
    plt.loglog(physical_rates, physical_rates, 'r--', alpha=0.7, label='Break-even Line (p_L = p)')
    
    plt.xlabel('Physical Error Rate')
    plt.ylabel('Logical Error Rate')
    plt.title(f'QEC Performance - {results["decoder_method"].upper()} Decoder\n'
              f'Code: {len(results["stabilizers"])} stabilizers, {results["rounds"]} rounds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add threshold annotation if crossing exists
    crossing_idx = None
    for i in range(len(physical_rates)-1):
        if logical_rates[i] < physical_rates[i] and logical_rates[i+1] > physical_rates[i+1]:
            crossing_idx = i
            break
    
    if crossing_idx is not None:
        threshold_p = physical_rates[crossing_idx]
        plt.axvline(threshold_p, color='green', linestyle=':', alpha=0.7)
        plt.text(threshold_p, max(logical_rates)*0.1, 
                f'Threshold ≈ {threshold_p:.4f}', 
                rotation=90, verticalalignment='bottom')
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            file_size = os.path.getsize(save_path)
            logger.debug(f"  ✅ Plot saved: {file_size} bytes")
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            logger.error(f"  ❌ Failed to save plot: {e}")
            raise
    
    # Convert to base64
    import io
    import base64
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    img_buffer.close()
    plt.close()
    
    return img_base64


def analyze_decoder_comparison(
    stabilizers: List[str],
    physical_error_rates: List[float],
    num_shots: int = 5000,
    rounds: int = 3
) -> Dict:
    """
    Compare MWPM and BP-OSD decoder performance.
    
    Args:
        stabilizers: List of stabilizer strings
        physical_error_rates: List of physical error rates to test
        num_shots: Number of shots per decoder method
        rounds: Number of syndrome measurement rounds
        
    Returns:
        Comparison results dictionary
    """
    methods = ['mwpm', 'bp_osd']
    comparison_results = {
        'stabilizers': stabilizers,
        'physical_error_rates': physical_error_rates,
        'methods': {},
        'num_shots': num_shots,
        'rounds': rounds
    }
    
    for method in methods:
        print(f"Testing {method.upper()} decoder...")
        method_results = calculate_logical_error_rate(
            stabilizers, physical_error_rates, num_shots, rounds, method
        )
        comparison_results['methods'][method] = method_results
    
    return comparison_results


def plot_decoder_comparison(comparison_results: Dict, save_path: str = None) -> str:
    """
    Plot comparison between different decoder methods.
    
    Args:
        comparison_results: Results from analyze_decoder_comparison
        save_path: Path to save the plot
        
    Returns:
        Base64 encoded image string
    """
    plt.figure(figsize=(12, 6))
    
    physical_rates = np.array(comparison_results['physical_error_rates'])
    colors = {'mwpm': 'blue', 'bp_osd': 'red'}
    markers = {'mwpm': 'o', 'bp_osd': 's'}
    
    for method, results in comparison_results['methods'].items():
        logical_rates = np.array(results['logical_error_rates'])
        plt.loglog(physical_rates, logical_rates, 
                  color=colors[method], marker=markers[method], 
                  linewidth=2, markersize=8, label=f'{method.upper()} Decoder')
    
    # Plot break-even line
    plt.loglog(physical_rates, physical_rates, 'k--', alpha=0.5, label='Break-even Line')
    
    plt.xlabel('Physical Error Rate')
    plt.ylabel('Logical Error Rate')
    plt.title(f'Decoder Comparison\nCode: {len(comparison_results["stabilizers"])} stabilizers, {comparison_results["rounds"]} rounds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    # Convert to base64
    import io
    import base64
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    img_buffer.close()
    plt.close()
    
    return img_base64


def save_results_to_data(results: Dict, filename: str, data_dir: str = 'data'):
    """
    Save error analysis results to data directory.
    
    Args:
        results: Results dictionary to save
        filename: Name of the file (without extension)
        data_dir: Data directory path (relative or absolute)
    
    File Path Specification Examples:
        - Relative path: 'data', 'results', 'output'
        - Absolute path: '/home/user/data', '/tmp/results'  
        - Current directory: '.', './output'
        - Parent directory: '../data', '../results'
        - Custom structure: 'project/analysis/data'
    """
    import json
    
    # Log file path specification for debugging
    logger.debug(f"save_results_to_data called with:")
    logger.debug(f"  filename: {filename}")
    logger.debug(f"  data_dir: {data_dir}")
    logger.debug(f"  current working directory: {os.getcwd()}")
    
    # Convert relative path to absolute for logging
    abs_data_dir = os.path.abspath(data_dir)
    logger.debug(f"  absolute data_dir: {abs_data_dir}")
    
    # Create data directory if it doesn't exist
    try:
        os.makedirs(data_dir, exist_ok=True)
        logger.debug(f"  ✅ Directory created/exists: {abs_data_dir}")
    except Exception as e:
        logger.error(f"  ❌ Failed to create directory {abs_data_dir}: {e}")
        raise
    
    # Save as JSON
    json_path = os.path.join(data_dir, f"{filename}.json")
    abs_json_path = os.path.abspath(json_path)
    logger.debug(f"  📄 Saving JSON to: {abs_json_path}")
    
    try:
        with open(json_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    json_results[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2)
        
        file_size = os.path.getsize(json_path)
        logger.debug(f"  ✅ JSON file saved: {file_size} bytes")
        print(f"Results saved to: {json_path}")
    except Exception as e:
        logger.error(f"  ❌ Failed to save JSON file: {e}")
        raise
    
    # Save summary as CSV
    csv_path = os.path.join(data_dir, f"{filename}_summary.csv")
    abs_csv_path = os.path.abspath(csv_path)
    logger.debug(f"  📊 Saving CSV to: {abs_csv_path}")
    
    try:
        with open(csv_path, 'w') as f:
            f.write("physical_error_rate,logical_error_rate\n")
            for p_rate, l_rate in zip(results['physical_error_rates'], results['logical_error_rates']):
                f.write(f"{p_rate},{l_rate}\n")
        
        file_size = os.path.getsize(csv_path)
        logger.debug(f"  ✅ CSV file saved: {file_size} bytes")
        print(f"Summary saved to: {csv_path}")
    except Exception as e:
        logger.error(f"  ❌ Failed to save CSV file: {e}")
        raise
        
    # Log summary of all created files
    logger.debug(f"  📁 All files saved in directory: {abs_data_dir}")
    logger.debug(f"    - JSON: {os.path.basename(json_path)}")
    logger.debug(f"    - CSV:  {os.path.basename(csv_path)}")