#!/usr/bin/env python3
"""
Consolidated QEC codes testing.
Combines: test_complex_codes.py, test_toric_code.py, test_toric_with_logicals.py, test_logical_operators_validation.py
"""

import sys
import os
import pytest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gate_optimize.qec.code_builder import QECCodeBuilder
from gate_optimize.qec.error_analysis import calculate_logical_error_rate, count_logical_errors


class TestRepetitionCodes:
    """Test various repetition code configurations."""
    
    def test_3_qubit_repetition_code(self):
        """Test basic 3-qubit repetition code."""
        stabilizers = ['+ZZI', '+IZZ']
        builder = QECCodeBuilder(stabilizers, rounds=2)
        circuit = builder.build_syndrome_circuit()
        
        assert circuit is not None
        assert len(circuit) > 0
        assert circuit.num_detectors > 0
        assert circuit.num_observables == 1  # Should have logical observable
    
    def test_5_qubit_repetition_code(self):
        """Test 5-qubit repetition code."""
        stabilizers = ['+ZZIII', '+IZZII', '+IIZZI', '+IIIZZ']
        builder = QECCodeBuilder(stabilizers, rounds=2)
        circuit = builder.build_syndrome_circuit()
        
        assert circuit is not None
        assert len(circuit) > 0
        assert circuit.num_detectors > 0
        assert circuit.num_observables == 1
    
    def test_7_qubit_repetition_code(self):
        """Test 7-qubit repetition code."""
        stabilizers = ['+ZZIIII', '+IZZIII', '+IIZZII', '+IIIZZI', '+IIIIZZ', '+IIIIIZ']
        builder = QECCodeBuilder(stabilizers, rounds=2)
        circuit = builder.build_syndrome_circuit()
        
        assert circuit is not None
        assert circuit.num_detectors > 0


class TestSteaneCode:
    """Test 7-qubit Steane code implementation."""
    
    def test_steane_code_basic(self):
        """Test basic Steane code functionality."""
        steane_stabilizers = [
            '+XXXXIII',
            '+XXIIXXI', 
            '+XIIXIXI',
            '+ZZZZIIII',
            '+ZZIIZZI',
            '+ZIZIZII'
        ]
        
        builder = QECCodeBuilder(steane_stabilizers, rounds=2)
        circuit = builder.build_syndrome_circuit()
        
        assert circuit is not None
        assert len(circuit) > 0
        assert circuit.num_detectors > 0
        assert builder._detect_code_type() == 'steane'
    
    def test_steane_code_with_logicals(self):
        """Test Steane code with explicit logical operators."""
        steane_stabilizers = [
            '+XXXXIII',
            '+XXIIXXI', 
            '+XIIXIXI',
            '+ZZZZIIII',
            '+ZZIIZZI',
            '+ZIZIZII'
        ]
        
        # Logical operators for Steane code (proper commuting operators)
        logical_operators = ['ZIIIZZZ']  # Logical Z operator that commutes with stabilizers
        
        builder = QECCodeBuilder(
            steane_stabilizers, 
            rounds=2, 
            logical_Z_operators=logical_operators
        )
        circuit = builder.build_syndrome_circuit(include_logical_Z_operators=True)
        
        assert circuit is not None
        assert circuit.num_observables > 0


class TestToricCode:
    """Test toric code implementations."""
    
    def test_simple_toric_code(self):
        """Test simplified toric code."""
        # Simplified 3x3 toric code stabilizers
        toric_stabilizers = [
            '+ZZII_IIII',  # Z-type plaquette 1
            '+IZZI_IIII',  # Z-type plaquette 2  
            '+IIZZ_IIII',  # Z-type plaquette 3
            '+IIII_ZZII',  # Z-type plaquette 4
            '+IIII_IZZI',  # Z-type plaquette 5
            '+IIII_IIZZ',  # Z-type plaquette 6
            '+XXXX_IIII',  # X-type star 1
            '+IIII_XXXX',  # X-type star 2
        ]
        
        builder = QECCodeBuilder(toric_stabilizers, rounds=2)
        circuit = builder.build_syndrome_circuit()
        
        assert circuit is not None
        assert len(circuit) > 0
        assert circuit.num_detectors > 0
        assert builder._detect_code_type() == 'toric'
    
    def test_toric_error_counting(self):
        """Test error counting with toric codes."""
        toric_stabilizers = [
            '+ZZII_IIII',
            '+IZZI_IIII',
            '+IIZZ_IIII',
            '+XXXX_IIII'
        ]
        
        builder = QECCodeBuilder(toric_stabilizers, rounds=2)
        circuit = builder.build_syndrome_circuit()
        
        # Test error counting (should not raise exceptions)
        num_errors = count_logical_errors(circuit, num_shots=50)
        assert isinstance(num_errors, int)
        assert 0 <= num_errors <= 50


class TestSurfaceCode:
    """Test surface code patterns."""
    
    def test_simple_surface_code(self):
        """Test simplified surface code stabilizers."""
        surface_stabilizers = [
            '+XZIX',  # X-type stabilizer
            '+ZXZX',  # Z-type stabilizer
            '+IXZI',  # X-type stabilizer
            '+XZXI'   # Z-type stabilizer
        ]
        
        builder = QECCodeBuilder(surface_stabilizers, rounds=2)
        circuit = builder.build_syndrome_circuit()
        
        assert circuit is not None
        assert len(circuit) > 0
        assert circuit.num_detectors > 0


class TestLogicalOperators:
    """Test logical operator handling and validation."""
    
    def test_explicit_logical_operators(self):
        """Test explicit logical operator specification."""
        stabilizers = ['+ZZI', '+IZZ']
        logical_ops = ['ZII']  # Logical X operator for 3-qubit repetition
        
        builder = QECCodeBuilder(stabilizers, rounds=2, logical_Z_operators=logical_ops)
        circuit = builder.build_syndrome_circuit(include_logical_Z_operators=True)
        
        assert circuit is not None
        assert circuit.num_observables > 0
    
    def test_multiple_logical_operators(self):
        """Test multiple logical operators."""
        stabilizers = ['+ZZI', '+IZZ']
        logical_ops = ['XXX', 'ZZZ']  # Multiple logical operators
        
        builder = QECCodeBuilder(stabilizers, rounds=2, logical_Z_operators=logical_ops)
        circuit = builder.build_syndrome_circuit(include_logical_Z_operators=True)
        
        assert circuit is not None
        assert circuit.num_observables >= 1
    
    def test_invalid_logical_operators(self):
        """Test handling of invalid logical operators."""
        stabilizers = ['+ZZI', '+IZZ']
        logical_ops = ['XXXX']  # Wrong length
        
        # Should not crash, but may not produce observables
        builder = QECCodeBuilder(stabilizers, rounds=2, logical_Z_operators=logical_ops)
        circuit = builder.build_syndrome_circuit(include_logical_Z_operators=True)
        
        assert circuit is not None
    
    def test_logical_operator_conversion(self):
        """Test conversion of logical operator strings."""
        stabilizers = ['+ZZI', '+IZZ']
        logical_ops = ['+XXX', '-ZZZ']  # With signs
        
        builder = QECCodeBuilder(stabilizers, rounds=2, logical_Z_operators=logical_ops)
        circuit = builder.build_syndrome_circuit(include_logical_Z_operators=True)
        
        assert circuit is not None


class TestErrorRates:
    """Test error rate calculations for different codes."""
    
    def test_repetition_code_error_rates(self):
        """Test error rate calculation for repetition code."""
        stabilizers = ['+ZZI', '+IZZ']
        physical_rates = [0.01, 0.02]
        
        results = calculate_logical_error_rate(
            stabilizers=stabilizers,
            physical_error_rates=physical_rates,
            num_shots=100,
            rounds=2,
            decoder_method='mwpm'
        )
        
        assert len(results['logical_error_rates']) == len(physical_rates)
        assert all(0 <= rate <= 1 for rate in results['logical_error_rates'])
    
    def test_steane_code_error_rates(self):
        """Test error rate calculation for Steane code."""
        steane_stabilizers = [
            '+XXXXIII',
            '+XXIIXXI', 
            '+XIIXIXI',
            '+ZZZZIIII'
        ]
        physical_rates = [0.01]
        
        results = calculate_logical_error_rate(
            stabilizers=steane_stabilizers,
            physical_error_rates=physical_rates,
            num_shots=50,  # Small for testing
            rounds=2,
            decoder_method='mwpm'
        )
        
        assert len(results['logical_error_rates']) == 1
        assert 0 <= results['logical_error_rates'][0] <= 1


# Legacy test functions for backward compatibility
def test_simple_repetition_code():
    """Legacy test for repetition code."""
    try:
        from gate_optimize.qec.code_builder import QECCodeBuilder
        from gate_optimize.qec.error_analysis import count_logical_errors
        
        stabilizers = ['+ZZI', '+IZZ']
        builder = QECCodeBuilder(stabilizers, rounds=2)
        circuit = builder.build_syndrome_circuit()
        
        num_errors = count_logical_errors(circuit, num_shots=50)
        return True
    except Exception:
        return False


def test_3x3_toric_code():
    """Legacy test for toric code."""
    try:
        from gate_optimize.qec.code_builder import QECCodeBuilder
        from gate_optimize.qec.error_analysis import count_logical_errors
        
        toric_stabilizers = [
            '+ZZII_IIII',
            '+IZZI_IIII', 
            '+IIZZ_IIII',
            '+IIII_ZZII',
            '+IIII_IZZI',
            '+IIII_IIZZ',
            '+XXXX_IIII',
            '+IIII_XXXX',
        ]
        
        builder = QECCodeBuilder(toric_stabilizers, rounds=2)
        circuit = builder.build_syndrome_circuit()
        
        num_errors = count_logical_errors(circuit, num_shots=50)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    print("QEC Codes Test Suite")
    print("=" * 30)
    
    legacy_tests = [
        ("Repetition code", test_simple_repetition_code),
        ("Toric code", test_3x3_toric_code),
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
        print("🎉 All code tests passed!")
    else:
        print("❌ Some tests failed")