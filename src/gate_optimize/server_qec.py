"""
QEC-specific MCP tools for quantum error correction analysis.
"""

import base64
from typing import Annotated, List
import numpy as np
import matplotlib.pyplot as plt
import os
from mcp.types import ImageContent, TextContent
from .qec import calculate_logical_error_rate, plot_error_rate_curve, analyze_decoder_comparison, plot_decoder_comparison, save_results_to_data, plot_stim_circuit 
from .qec.code_builder import QECCodeBuilder
from .qec.circuit_visualization import plot_noisy_circuit
from .qec.error_analysis import _add_noise_to_circuit


def send_to_gui(**kwargs):
    """Send updates to GUI - placeholder function"""
    try:
        import requests
        kwargs.setdefault("status", "finished")
        requests.post("http://127.0.0.1:12345/update", json=kwargs, timeout=2)
    except:
        pass


async def analyze_qec_logical_error_rate(
    stabilizers: Annotated[List[str], "List of stabilizer strings for the quantum error correction code (e.g., ['+ZZ_____', '+_ZZ____', '+XXXXXXX'] for 7-qubit Steane code)"],
    logical_Z_operators: Annotated[List[str], "List of logical Z operator strings for the code (e.g., ['ZZZZZZZ'] for Steane code logical X and Z operators). Optional for auto-detection."] = None,
    rounds: Annotated[int, "Number of syndrome measurement rounds"] = 3,
    decoder_method: Annotated[str, "Decoding method: 'mwpm', 'bp_osd', or 'both' for comparison"] = 'mwpm',
) -> list[ImageContent | TextContent]:
    """
    Analyze quantum error correction code performance by calculating logical error rates.
    
    Args:
        stabilizers: List of stabilizer strings defining the QEC code
        logical_Z_operators: List of logical operator strings (optional, auto-detected if None)
        physical_error_rates: Physical error rates to test (default: logarithmic range)
        num_shots: Number of Monte Carlo shots per error rate
        rounds: Number of syndrome measurement rounds
        decoder_method: Decoder to use ('mwpm', 'bp_osd', or 'both')
        
    Returns:
        List containing plots and analysis of logical error rates
    """

    # "List of physical error rates to test (e.g., [0.001, 0.005, 0.01, 0.02])"
    physical_error_rates:List[float] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    # "Number of Monte Carlo shots per error rate"
    num_shots:int = 5000

    # Clean the input
    stabilizers = [s.replace(' ', '').replace('\t', '') for s in stabilizers]
    logical_Z_operators= [s.replace(' ', '').replace('\t', '') for s in logical_Z_operators]


    if not stabilizers or not isinstance(stabilizers, list):
        error_text = "Error: stabilizers must be a non-empty list of strings"
        send_to_gui(status="error", text_result=error_text)
        return [TextContent(type="text", text=error_text)]
    
    # Set default error rates if not provided
    
    images = []
    results = []
    results.append("QUANTUM ERROR CORRECTION ANALYSIS")
    results.append("=" * 50)
    results.append(f"Stabilizer Code: {stabilizers}")
    results.append(f"Code qubits: {len(stabilizers[0].lstrip('+-').replace('_', 'I'))}")
    results.append(f"Stabilizers: {len(stabilizers)}")
    
    # Report logical operators
    if logical_Z_operators:
        results.append(f"Logical Operators (provided): {logical_Z_operators}")
        results.append(f"Number of logical operators: {len(logical_Z_operators)}")
    else:
        results.append("Logical Operators: Auto-detected from stabilizer code")
    results.append(f"Syndrome rounds: {rounds}")
    results.append(f"Shots per error rate: {num_shots}")
    results.append(f"Physical error rates: {physical_error_rates}")
    results.append("")
    
    # Create data directory for saving results (use absolute path from project root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Go up to project root
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Build and save the QEC circuit
    
    builder = QECCodeBuilder(stabilizers, rounds=rounds, logical_Z_operators=logical_Z_operators)
    circuit = builder.build_syndrome_circuit(include_logical_Z_operators=True)
    
    # Save circuit as .stim file
    code_type = builder._detect_code_type()
    circuit_filename = f"qec_circuit_{code_type}_{len(stabilizers)}stabs_{rounds}rounds.stim"
    circuit_path = os.path.join(data_dir, circuit_filename)
    
    with open(circuit_path, 'w') as f:
        f.write(str(circuit))
    
    # Generate circuit visualization plots
    circuit_plot_filename = f"circuit_diagram_{code_type}_{len(stabilizers)}stabs_{rounds}rounds.svg"
    circuit_plot_path = os.path.join(data_dir, circuit_plot_filename)
    
    # Create circuit diagram and get SVG file path
    circuit_svg_path = plot_stim_circuit(
        circuit,
        save_path=circuit_plot_path,
        title=f"{code_type.title()} Code Circuit ({len(stabilizers)} stabilizers, {rounds} rounds)"
    )
    
    # Generate noisy circuit and save as stim file and SVG (only once per analysis)
    noisy_circuit_results = plot_noisy_circuit(
        base_circuit=circuit,
        error_rates=[0.001],  # Generate for only one error rate
        data_dir=data_dir,
        save_stim=True,
        save_svg=True
    )
    
    # Generate PNG images for MCP transfer
    print("Generating circuit images for MCP transfer...")
    base_circuit_png = circuit_to_png_base64(circuit, dpi=300)
    noisy_circuit = _add_noise_to_circuit(circuit, 0.001)
    noisy_circuit_png = circuit_to_png_base64(noisy_circuit, dpi=300)
    
    # Add circuit information to results
    results.append(f"CIRCUIT INFORMATION")
    results.append("-" * 30)
    results.append(f"Code type: {code_type}")
    results.append(f"Circuit instructions: {len(circuit)}")
    results.append(f"Detectors: {circuit.num_detectors}")
    results.append(f"Observables: {circuit.num_observables}")
    results.append(f"Circuit saved: {circuit_path}")
    results.append(f"Circuit diagram (SVG): {circuit_svg_path}")
    
    # Add noisy circuit information
    noisy_info = noisy_circuit_results['noisy_circuits'][0.001]
    if noisy_info['stim_path']:
        results.append(f"Noisy circuit (p=0.001): {noisy_info['stim_path']}")
    if noisy_info['svg_path']:
        results.append(f"Noisy circuit diagram: {noisy_info['svg_path']}")
    results.append("")
    
    if decoder_method == 'both':
        # Compare both decoders
        results.append("DECODER COMPARISON MODE")
        results.append("-" * 30)
        
        send_to_gui(
            status="running",
            tool_name="analyze_qec_logical_error_rate",
            parameters={"stabilizers": stabilizers, "decoder_method": decoder_method},
            live_metrics={"phase": "Running decoder comparison"}
        )
        
        comparison_results = analyze_decoder_comparison(
            stabilizers, physical_error_rates, num_shots, rounds
        )
        
        # Plot comparison
        comparison_img_base64 = plot_decoder_comparison(
            comparison_results, 
            save_path=os.path.join(data_dir, "decoder_comparison.png")
        )
        images.append(ImageContent(type="image", data=comparison_img_base64, mimeType="image/png"))
        
        # Save comparison data
        save_results_to_data(comparison_results, "decoder_comparison", data_dir)
        
        # Add results summary
        results.append("Decoder Performance Summary:")
        for method, method_results in comparison_results['methods'].items():
            results.append(f"\n{method.upper()} Decoder:")
            for p_rate, l_rate in zip(method_results['physical_error_rates'], 
                                    method_results['logical_error_rates']):
                results.append(f"  p={p_rate:.4f} -> p_L={l_rate:.6f}")
        
    else:
        # Single decoder analysis
        results.append(f"{decoder_method.upper()} DECODER ANALYSIS")
        results.append("-" * 30)
        
        send_to_gui(
            status="running",
            tool_name="analyze_qec_logical_error_rate",
            parameters={"stabilizers": stabilizers, "decoder_method": decoder_method},
            live_metrics={"phase": f"Running {decoder_method.upper()} analysis"}
        )
        
        error_analysis_results = calculate_logical_error_rate(
            stabilizers, physical_error_rates, num_shots, rounds, decoder_method, logical_Z_operators
        )
        
        # Plot error rate curve
        curve_img_base64 = plot_error_rate_curve(
            error_analysis_results,
            save_path=os.path.join(data_dir, f"error_curve_{decoder_method}.png")
        )
        images.append(ImageContent(type="image", data=curve_img_base64, mimeType="image/png"))
        
        # Save single decoder data
        save_results_to_data(error_analysis_results, f"error_analysis_{decoder_method}", data_dir)
        
        # Add results summary
        results.append("Error Rate Analysis:")
        for p_rate, l_rate in zip(error_analysis_results['physical_error_rates'], 
                                error_analysis_results['logical_error_rates']):
            improvement = p_rate / l_rate if l_rate > 0 else float('inf')
            results.append(f"  p={p_rate:.4f} -> p_L={l_rate:.6f} (improvement: {improvement:.2f}x)")
        
        # Find error correction threshold (if exists)
        physical_rates = np.array(error_analysis_results['physical_error_rates'])
        logical_rates = np.array(error_analysis_results['logical_error_rates'])
        
        below_threshold = logical_rates < physical_rates
        if any(below_threshold):
            threshold_idx = np.where(below_threshold)[0][-1]  # Last point below threshold
            if threshold_idx < len(physical_rates) - 1:
                threshold_estimate = physical_rates[threshold_idx]
                results.append(f"\nEstimated error correction threshold: ~{threshold_estimate:.4f}")
            else:
                results.append(f"\nCode shows error correction benefit up to p={physical_rates[-1]:.4f}")
        else:
            results.append(f"\nNo error correction benefit observed in tested range")
    
    results.append(f"\nData saved to: {data_dir}/")
    results.append("Files created:")
    results.append(f"- {circuit_filename}: Stim circuit file")
    results.append(f"- {circuit_plot_filename}: Circuit diagram SVG")
    results.append("- error_curve_*.png: Error rate analysis plots")
    results.append("- *.json: Full analysis results")
    results.append("- *_summary.csv: Error rates in CSV format")
    
    final_text = "\n".join(results)
    images_base64 = [img.data for img in images]
    
    send_to_gui(
        status="finished",
        tool_name="analyze_qec_logical_error_rate",
        parameters={"stabilizers": stabilizers, "decoder_method": decoder_method},
        text_result=final_text,
        main_images=images_base64
    )
    
    # Read circuit file content for inclusion in output
    with open(circuit_path, 'r') as f:
        circuit_content = f.read()
    
    # Create circuit content with header
    circuit_text = f"STIM CIRCUIT FILE: {circuit_filename}\n"
    circuit_text += "=" * 50 + "\n"
    circuit_text += f"Instructions: {len(circuit)}\n"
    circuit_text += f"Detectors: {circuit.num_detectors}\n" 
    circuit_text += f"Observables: {circuit.num_observables}\n"
    circuit_text += "-" * 50 + "\n"
    circuit_text += circuit_content
    
    # Add circuit images to the response
    circuit_images = []
    if base_circuit_png:
        circuit_images.append(ImageContent(
            type="image", 
            data=base_circuit_png, 
            mimeType="image/png"
        ))
    if noisy_circuit_png:
        circuit_images.append(ImageContent(
            type="image", 
            data=noisy_circuit_png, 
            mimeType="image/png"
        ))
    
    content_list = [
        TextContent(type="text", text=final_text),
        TextContent(type="text", text=circuit_text)
    ]
    content_list.extend(circuit_images)  # Add circuit images first
    content_list.extend(images)  # Then add error rate plots
    
    return content_list