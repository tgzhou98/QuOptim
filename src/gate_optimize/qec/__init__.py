"""
Quantum Error Correction (QEC) submodule for gate optimization.

This module provides tools for:
- Creating circuits from stabilizer codes
- Decoding with MWPM and BP-OSD methods
- Calculating logical error rates vs physical error rates
"""

from .code_builder import QECCodeBuilder
from .decoder import QECDecoder
from .error_analysis import (
    calculate_logical_error_rate, 
    count_logical_errors,
    plot_error_rate_curve,
    analyze_decoder_comparison,
    plot_decoder_comparison,
    save_results_to_data
)
from .circuit_visualization import plot_stim_circuit


from importlib import resources
import sys
import gate_optimize
project_root = resources.files(gate_optimize).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

__all__ = [
    'QECCodeBuilder', 
    'QECDecoder', 
    'calculate_logical_error_rate', 
    'count_logical_errors',
    'plot_error_rate_curve',
    'analyze_decoder_comparison',
    'plot_decoder_comparison',
    'save_results_to_data',
    'plot_stim_circuit',
]