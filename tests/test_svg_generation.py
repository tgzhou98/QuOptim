#!/usr/bin/env python3
"""
Test script to debug SVG generation issues.
"""

import stim
import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from gate_optimize.qec.code_builder import QECCodeBuilder
from gate_optimize.qec.circuit_visualization import plot_stim_circuit, plot_noisy_circuit

def test_svg_generation():
    """Test SVG generation for circuits."""
    print("Testing SVG generation...")
    
    # Create data directory
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create a simple circuit
    print("1. Creating simple circuit...")
    try:
        stabilizers = ["ZZI", "IZZ"]
        builder = QECCodeBuilder(stabilizers, rounds=1)
        circuit = builder.build_syndrome_circuit()
        print(f"   Circuit created with {len(circuit)} instructions")
    except Exception as e:
        print(f"   ERROR: Failed to create circuit: {e}")
        return
    
    # Test basic SVG generation using stim directly
    print("2. Testing stim SVG generation...")
    try:
        svg_content = circuit.diagram('timeline-svg')
        print(f"   SVG content generated: {len(str(svg_content))} characters")
        print(f"   SVG content type: {type(svg_content)}")
        print(f"   SVG preview (first 100 chars): {str(svg_content)[:100]}")
    except Exception as e:
        print(f"   ERROR: Stim SVG generation failed: {e}")
        return
    
    # Test our plot_stim_circuit function
    print("3. Testing plot_stim_circuit function...")
    try:
        svg_path = data_dir / "test_circuit.svg"
        result_path = plot_stim_circuit(circuit, save_path=str(svg_path))
        
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path)
            print(f"   SUCCESS: SVG saved to {result_path} ({file_size} bytes)")
            
            # Read first few lines to verify content
            with open(result_path, 'r') as f:
                first_lines = [f.readline().strip() for _ in range(3)]
            print(f"   SVG file starts with: {first_lines}")
        else:
            print(f"   ERROR: SVG file was not created at {result_path}")
    except Exception as e:
        print(f"   ERROR: plot_stim_circuit failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test noisy circuit generation
    print("4. Testing noisy circuit generation...")
    try:
        results = plot_noisy_circuit(
            base_circuit=circuit,
            error_rates=[0.001],
            data_dir=str(data_dir),
            save_stim=True,
            save_svg=True
        )
        
        noisy_info = results['noisy_circuits'][0.001]
        if noisy_info['svg_path'] and os.path.exists(noisy_info['svg_path']):
            file_size = os.path.getsize(noisy_info['svg_path'])
            print(f"   SUCCESS: Noisy circuit SVG saved ({file_size} bytes)")
        else:
            print(f"   ERROR: Noisy circuit SVG not created")
            
        if noisy_info['stim_path'] and os.path.exists(noisy_info['stim_path']):
            file_size = os.path.getsize(noisy_info['stim_path'])
            print(f"   SUCCESS: Noisy circuit stim file saved ({file_size} bytes)")
        else:
            print(f"   ERROR: Noisy circuit stim file not created")
            
    except Exception as e:
        print(f"   ERROR: Noisy circuit generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_svg_generation()