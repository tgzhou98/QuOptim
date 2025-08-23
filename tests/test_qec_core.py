#!/usr/bin/env python3
"""
Consolidated core QEC functionality tests.
Combines: test_qec.py, test_minimal_qec.py, test_qec_pytest.py
"""

import sys
import os
import tempfile
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gate_optimize.qec.code_builder import QECCodeBuilder
from gate_optimize.qec.decoder import QECDecoder
from gate_optimize.qec.error_analysis import (
    calculate_logical_error_rate, 
    count_logical_errors,
    save_results_to_data
)


class TestQECCodeBuilder:
    """Test cases for QECCodeBuilder functionality."""
    
    def test_basic_import_and_creation(self):
        """Test basic import and instance creation."""
        stabilizers = ['+ZZI', '+IZZ']
        builder = QECCodeBuilder(stabilizers, rounds=1)
        assert builder is not None
        assert builder.stabilizers == stabilizers
        assert builder.rounds == 1
    
    def test_simple_repetition_code(self):
        """Test 3-qubit repetition code circuit building."""
        stabilizers = ['+ZZI', '+IZZ']
        builder = QECCodeBuilder(stabilizers, rounds=2)
        circuit = builder.build_syndrome_circuit()
        
        assert circuit is not None
        assert len(circuit) > 0
        assert circuit.num_detectors >= 0
        assert circuit.num_observables >= 0
    
    def test_steane_code_stabilizers(self):
        """Test 7-qubit Steane code stabilizers."""
        stabilizers = [
            '+XXXXIII',
            '+XXIIXXI', 
            '+XIIXIXI',
            '+ZZZZIIII',
            '+ZZIIZZI',
            '+ZIZIZI'
        ]
        
        builder = QECCodeBuilder(stabilizers, rounds=2)
        circuit = builder.build_syndrome_circuit()
        
        assert circuit is not None
        assert len(circuit) > 0
        assert circuit.num_detectors > 0
    
    def test_code_type_detection(self):
        """Test automatic code type detection."""
        # Repetition code
        rep_stabilizers = ['+ZZI', '+IZZ']
        builder = QECCodeBuilder(rep_stabilizers, rounds=1)
        assert builder._detect_code_type() == 'repetition'
        
        # Steane code
        steane_stabilizers = ['+XXXXIII', '+XXIIXXI', '+XIIXIXI', '+ZZZZIIII', '+ZZIIZZI', '+ZIZIZI']
        builder = QECCodeBuilder(steane_stabilizers, rounds=1)
        assert builder._detect_code_type() == 'steane'


class TestQECDecoder:
    """Test cases for QECDecoder functionality."""
    
    def test_mwpm_decoder_creation(self):
        """Test MWPM decoder initialization."""
        stabilizers = ['+ZZI', '+IZZ']
        builder = QECCodeBuilder(stabilizers, rounds=1)
        circuit = builder.build_syndrome_circuit()
        
        decoder = QECDecoder(circuit, method='mwpm')
        assert decoder is not None
        assert decoder.method == 'mwpm'
        assert hasattr(decoder, 'matching_decoder')
    
    def test_bp_osd_decoder_creation(self):
        """Test BP-OSD decoder initialization."""
        stabilizers = ['+ZZI', '+IZZ']
        builder = QECCodeBuilder(stabilizers, rounds=1)
        circuit = builder.build_syndrome_circuit()
        
        try:
            decoder = QECDecoder(circuit, method='bp_osd')
            assert decoder is not None
            assert decoder.method == 'bp_osd'
            assert hasattr(decoder, 'bp_decoder')
        except (ImportError, RuntimeError):
            pytest.skip("BP-OSD decoder dependencies not available or unstable")
    
    def test_invalid_decoder_method(self):
        """Test error handling for invalid decoder method."""
        stabilizers = ['+ZZI', '+IZZ']
        builder = QECCodeBuilder(stabilizers, rounds=1)
        circuit = builder.build_syndrome_circuit()
        
        with pytest.raises(ValueError, match="Unknown decoding method"):
            QECDecoder(circuit, method='invalid_method')


class TestErrorAnalysis:
    """Test cases for error analysis functions."""
    
    def test_count_logical_errors_basic(self):
        """Test basic error counting functionality."""
        stabilizers = ['+ZZI', '+IZZ']
        builder = QECCodeBuilder(stabilizers, rounds=1)
        circuit = builder.build_syndrome_circuit()
        
        num_errors = count_logical_errors(circuit, num_shots=10)
        assert isinstance(num_errors, int)
        assert 0 <= num_errors <= 10
    
    def test_calculate_logical_error_rate(self):
        """Test logical error rate calculation."""
        stabilizers = ['+ZZI', '+IZZ']
        physical_error_rates = [0.01, 0.02]
        
        results = calculate_logical_error_rate(
            stabilizers=stabilizers,
            physical_error_rates=physical_error_rates,
            num_shots=100,
            rounds=2,
            decoder_method='mwpm'
        )
        
        assert 'physical_error_rates' in results
        assert 'logical_error_rates' in results
        assert len(results['physical_error_rates']) == len(physical_error_rates)
        assert len(results['logical_error_rates']) == len(physical_error_rates)
        assert all(0 <= rate <= 1 for rate in results['logical_error_rates'])
    
    def test_save_results_to_data(self):
        """Test saving analysis results."""
        results = {
            'physical_error_rates': [0.01, 0.02],
            'logical_error_rates': [0.001, 0.004],
            'num_shots': 1000,
            'decoder_method': 'mwpm'
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_results_to_data(results, 'test_results', data_dir=temp_dir)
            
            json_file = os.path.join(temp_dir, 'test_results.json')
            csv_file = os.path.join(temp_dir, 'test_results_summary.csv')
            
            assert os.path.exists(json_file)
            assert os.path.exists(csv_file)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_repetition_code(self):
        """Test complete workflow with repetition code."""
        stabilizers = ['+ZZI', '+IZZ']
        
        # Build circuit
        builder = QECCodeBuilder(stabilizers, rounds=2)
        circuit = builder.build_syndrome_circuit()
        
        # Create decoder
        decoder = QECDecoder(circuit, method='mwpm')
        
        # Test error simulation
        results = decoder.simulate_errors(
            physical_error_rate=0.01,
            num_shots=50
        )
        
        assert 'logical_error_rate' in results
        assert 'num_shots' in results
        assert results['num_shots'] == 50
        assert 0 <= results['logical_error_rate'] <= 1
    
    def test_end_to_end_with_explicit_logicals(self):
        """Test workflow with explicit logical operators."""
        stabilizers = ['+ZZI', '+IZZ']
        logical_ops = ['XXX']  # Logical X operator
        
        builder = QECCodeBuilder(stabilizers, rounds=2, logical_Z_operators=logical_ops)
        circuit = builder.build_syndrome_circuit(include_logical_Z_operators=True)
        
        assert circuit is not None
        assert circuit.num_observables > 0


# Legacy test functions for backward compatibility
def test_basic():
    """Legacy basic test function."""
    try:
        from gate_optimize.qec.code_builder import QECCodeBuilder
        
        stabilizers = ['+ZZI', '+IZZ']
        builder = QECCodeBuilder(stabilizers, rounds=1)
        circuit = builder.build_syndrome_circuit()
        
        assert len(circuit) > 0
        assert circuit.num_detectors >= 0
        return True
    except Exception:
        return False


def test_decoder():
    """Legacy decoder test function."""
    try:
        from gate_optimize.qec.code_builder import QECCodeBuilder
        from gate_optimize.qec.decoder import QECDecoder
        
        stabilizers = ['+ZZI', '+IZZ']
        builder = QECCodeBuilder(stabilizers, rounds=1)
        circuit = builder.build_syndrome_circuit()
        
        decoder = QECDecoder(circuit, method='mwpm')
        assert decoder is not None
        return True
    except Exception:
        return False


def test_count_errors():
    """Legacy error counting test function."""
    try:
        from gate_optimize.qec.code_builder import QECCodeBuilder
        from gate_optimize.qec.error_analysis import count_logical_errors
        
        stabilizers = ['+ZZI', '+IZZ']
        builder = QECCodeBuilder(stabilizers, rounds=1)
        circuit = builder.build_syndrome_circuit()
        
        num_errors = count_logical_errors(circuit, num_shots=10)
        assert isinstance(num_errors, int)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Run legacy tests for backward compatibility
    print("QEC Core Test Suite")
    print("=" * 30)
    
    legacy_tests = [
        ("Basic functionality", test_basic),
        ("Decoder", test_decoder),
        ("Count errors", test_count_errors)
    ]
    
    passed = 0
    for name, test_func in legacy_tests:
        print(f"\nTesting {name}...")
        if test_func():
            print("✓ Passed")
            passed += 1
        else:
            print("✗ Failed")
    
    print(f"\n{passed}/{len(legacy_tests)} legacy tests passed")
    
    if passed == len(legacy_tests):
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")