"""
Circuit visualization functions for QEC circuits.
"""

import stim
from typing import Optional, List
import os
import base64
from .error_analysis import _add_noise_to_circuit

 
def plot_stim_circuit(circuit: stim.Circuit, save_path: str = None, title: str = "QEC Circuit") -> str:
    """
    Plot a Stim circuit using Stim's native timeline-svg diagram and save as SVG.
    
    Args:
        circuit: Stim circuit to visualize
        save_path: Optional path to save the SVG file (if None, auto-generated)
        title: Title for the plot (currently unused, kept for compatibility)
        
    Returns:
        Path to the saved SVG file
    """
    
    # Use Stim's built-in timeline diagram
    svg_content = circuit.diagram('timeline-svg')
    
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save SVG file directly
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(str(svg_content))
    
    print(f"Circuit diagram SVG saved to: {save_path}")
    return save_path


def circuit_to_png_base64(circuit: stim.Circuit, dpi: int = 300) -> str:
    """
    Convert a Stim circuit to PNG format and return as base64 string for MCP transfer.
    
    Args:
        circuit: Stim circuit to visualize
        dpi: DPI for PNG output (default: 300 for high quality)
        
    Returns:
        Base64 encoded PNG image string
    """
    import cairosvg
    import re
    
    # Generate SVG content
    svg_content = str(circuit.diagram('timeline-svg'))
    
    # Add white background to SVG
    # Find the opening <svg> tag and add a white rectangle background
    svg_match = re.search(r'<svg[^>]*>', svg_content)
    if svg_match:
        svg_tag = svg_match.group()
        # Extract viewBox dimensions if available
        viewbox_match = re.search(r'viewBox="([^"]*)"', svg_tag)
        if viewbox_match:
            viewbox = viewbox_match.group(1)
            x, y, width, height = viewbox.split()
            # Insert white background rectangle after the opening svg tag
            background_rect = f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="white"/>'
            svg_content = svg_content.replace(svg_tag, svg_tag + '\n' + background_rect, 1)
        else:
            # Fallback: add a large white background rectangle
            background_rect = '<rect x="0" y="0" width="100%" height="100%" fill="white"/>'
            svg_content = svg_content.replace(svg_tag, svg_tag + '\n' + background_rect, 1)
    
    # Convert SVG to PNG using cairosvg with high DPI
    png_bytes = cairosvg.svg2png(
        bytestring=svg_content.encode('utf-8'),
        dpi=dpi
    )
    
    # Encode as base64
    png_base64 = base64.b64encode(png_bytes).decode('utf-8')
    
    return png_base64


def plot_noisy_circuit(
    base_circuit: stim.Circuit, 
    error_rates: List[float] = [0.001, 0.01, 0.05],
    data_dir: str = "data",
    save_stim: bool = True,
    save_svg: bool = True
) -> dict:
    """
    Create noisy circuits with different error rates and save them as stim files and SVG diagrams.
    
    Args:
        base_circuit: Base stim circuit to add noise to
        error_rates: List of physical error rates to test
        data_dir: Directory to save files (will be created if doesn't exist)
        save_stim: Whether to save .stim circuit files
        save_svg: Whether to save SVG diagram files
        
    Returns:
        Dictionary with paths to created files organized by error rate
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    results = {
        'base_circuit_path': None,
        'noisy_circuits': {}
    }
    
    # Save base circuit
    if save_stim:
        base_path = os.path.join(data_dir, "base_circuit.stim")
        with open(base_path, 'w') as f:
            f.write(str(base_circuit))
        results['base_circuit_path'] = base_path
        print(f"Base circuit saved to: {base_path}")
    
    # Create noisy circuits for each error rate
    for error_rate in error_rates:
        print(f"Processing error rate: {error_rate}")
        
        # Generate noisy circuit using the imported function
        noisy_circuit = _add_noise_to_circuit(base_circuit, error_rate)
        
        circuit_info = {
            'error_rate': error_rate,
            'stim_path': None,
            'svg_path': None
        }
        
        # Save as .stim file
        if save_stim:
            stim_filename = f"noisy_circuit_p{error_rate:.3f}.stim"
            stim_path = os.path.join(data_dir, stim_filename)
            with open(stim_path, 'w') as f:
                f.write(str(noisy_circuit))
            circuit_info['stim_path'] = stim_path
            print(f"  Stim file saved to: {stim_path}")
        
        # Save SVG diagram (only for first error rate to avoid clutter, or if explicitly requested)
        if save_svg and (error_rate == error_rates[0] or len(error_rates) <= 3):
            try:
                svg_filename = f"noisy_circuit_p{error_rate:.3f}.svg"
                svg_path = os.path.join(data_dir, svg_filename)
                
                # Generate SVG using stim's built-in diagram functionality
                diagram = noisy_circuit.diagram("timeline-svg")
                with open(svg_path, 'w') as f:
                    f.write(str(diagram))
                circuit_info['svg_path'] = svg_path
                print(f"  SVG diagram saved to: {svg_path}")
            except Exception as e:
                print(f"  Warning: Could not generate SVG diagram for p={error_rate}: {e}")
        
        results['noisy_circuits'][error_rate] = circuit_info
    
    # Summary
    abs_data_dir = os.path.abspath(data_dir)
    print(f"\nAll files saved in directory: {abs_data_dir}")
    print("Circuit files created:")
    if results['base_circuit_path']:
        print(f"  - {os.path.basename(results['base_circuit_path'])}")
    
    for error_rate, info in results['noisy_circuits'].items():
        if info['stim_path']:
            print(f"  - {os.path.basename(info['stim_path'])}")
        if info['svg_path']:
            print(f"  - {os.path.basename(info['svg_path'])}")
    
    return results


def demo_noisy_circuit_from_stabilizers(
    stabilizers: List[str] = ["ZZI", "IZZ"],
    rounds: int = 2,
    error_rates: List[float] = [0.001, 0.01, 0.05],
    data_dir: str = "data"
) -> dict:
    """
    Convenience function to create and plot noisy circuits from stabilizer codes.
    
    Args:
        stabilizers: List of stabilizer strings (e.g., ["ZZI", "IZZ"])
        rounds: Number of syndrome measurement rounds
        error_rates: List of physical error rates to test
        data_dir: Directory to save files
        
    Returns:
        Dictionary with circuit information and file paths
    """
    from .code_builder import QECCodeBuilder
    
    print(f"Creating demo circuit from stabilizers: {stabilizers}")
    print(f"Rounds: {rounds}, Error rates: {error_rates}")
    
    # Create base circuit from stabilizers
    builder = QECCodeBuilder(stabilizers, rounds=rounds)
    base_circuit = builder.build_syndrome_circuit()
    
    print(f"Base circuit created with {len(str(base_circuit).splitlines())} instructions")
    
    # Generate noisy circuits and save files
    return plot_noisy_circuit(
        base_circuit=base_circuit,
        error_rates=error_rates,
        data_dir=data_dir,
        save_stim=True,
        save_svg=True
    )