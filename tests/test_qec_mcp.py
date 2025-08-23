#!/usr/bin/env python3
"""
Consolidated MCP tool testing and integration tests.
Combines: test_mcp_qec_tool.py, test_qec_standalone.py, test_enhanced_output.py
"""

import sys
import os
import asyncio
import pytest
import json
import tempfile
from unittest.mock import patch, AsyncMock

import logging

# At the top of your test file
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gate_optimize.qec.error_analysis import plot_error_rate_curve, plot_decoder_comparison


class TestMCPToolIntegration:
    """Test MCP tool integration with QEC functionality."""
    
    @pytest.mark.asyncio
    async def test_qec_mcp_tool_basic(self):
        """Test basic MCP tool functionality."""
        try:
            from gate_optimize.server_qec import analyze_qec_logical_error_rate
            
            stabilizers = ['+ZZI', '+IZZ']
            physical_error_rates = [0.01, 0.02]
            
            result = await analyze_qec_logical_error_rate(
                stabilizers=stabilizers,
                logical_Z_operators=['ZII'],  # Single qubit logical Z  # Logical Z for repetition code
                rounds=2,
                decoder_method='mwpm'
            )
            
            assert len(result) > 0
            assert any(item.type == 'text' for item in result)
            
        except ImportError:
            pytest.skip("MCP server not available")
        except Exception as e:
            pytest.fail(f"MCP tool test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_qec_mcp_tool_with_toric_code(self):
        """Test MCP tool with toric code."""
        try:
            from gate_optimize.server_qec import analyze_qec_logical_error_rate
            
            toric_stabilizers = [
                '+ZZII_IIII',
                '+IZZI_IIII', 
                '+IIZZ_IIII',
                '+XXXX_IIII'
            ]
            
            result = await analyze_qec_logical_error_rate(
                stabilizers=toric_stabilizers,
                logical_Z_operators=['ZIIIIIIII', 'IIIIZIIIII'],
                rounds=2,
                decoder_method='mwpm'
            )
            
            assert len(result) > 0
            
        except ImportError:
            pytest.skip("MCP server not available")
        except Exception as e:
            pytest.fail(f"MCP toric code test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_qec_mcp_tool_steane_code(self):
        """Test MCP tool with Steane code."""
        try:
            from gate_optimize.server_qec import analyze_qec_logical_error_rate
            print("Testing Steane code...")
            
            steane_stabilizers = [
                "IIIXXXX",
                "IXXIIXX",
                "XIXIXIX",
                "IIIZZZZ",
                "IZZIIZZ",
                "ZIZIZIZ",
            ]
            
            result = await analyze_qec_logical_error_rate(
                stabilizers=steane_stabilizers,
                logical_Z_operators=['ZZZZZZZ'],  # Proper logical Z for Steane
                rounds=4,
                decoder_method='mwpm'
            )
            print("result", result)
            
            assert len(result) > 0
            
        except ImportError:
            pytest.skip("MCP server not available")
        except Exception as e:
            pytest.fail(f"MCP Steane code test failed: {e}")

    @pytest.mark.asyncio
    async def test_qec_mcp_tool_decoder_comparison(self):
        """Test MCP tool with decoder comparison."""
        try:
            from gate_optimize.server_qec import analyze_qec_logical_error_rate
            
            stabilizers = ['+ZZI', '+IZZ']
            
            # Test with different decoders
            for decoder in ['mwpm']:  # Only test MWPM as BP-OSD may not be available
                result = await analyze_qec_logical_error_rate(
                    stabilizers=stabilizers,
                    logical_Z_operators=['ZII'],  # Single qubit logical Z
                    rounds=2,
                    decoder_method=decoder
                )
                
                assert len(result) > 0
                
        except ImportError:
            pytest.skip("MCP server not available")


class TestStandaloneExecution:
    """Test standalone execution of QEC modules."""
    
    def test_qec_module_import(self):
        """Test that QEC modules can be imported standalone."""
        from gate_optimize.qec import QECCodeBuilder, QECDecoder
        from gate_optimize.qec import calculate_logical_error_rate, count_logical_errors
        from gate_optimize.qec import plot_error_rate_curve, plot_stim_circuit
        
        # Basic smoke test
        stabilizers = ['+ZZI', '+IZZ']
        builder = QECCodeBuilder(stabilizers, rounds=1)
        circuit = builder.build_syndrome_circuit()
        
        assert circuit is not None
    
    def test_error_analysis_plotting(self):
        """Test error analysis plotting functions."""
        # Mock results data
        mock_results = {
            'physical_error_rates': [0.01, 0.02, 0.03],
            'logical_error_rates': [0.001, 0.004, 0.009],
            'decoder_method': 'mwpm',
            'stabilizers': ['+ZZI', '+IZZ'],
            'rounds': 2
        }
        
        # Test plot generation
        plot_b64 = plot_error_rate_curve(mock_results)
        assert isinstance(plot_b64, str)
        assert len(plot_b64) > 0
    
    def test_decoder_comparison_plotting(self):
        """Test decoder comparison plotting."""
        # Mock comparison results
        mock_comparison = {
            'physical_error_rates': [0.01, 0.02],
            'stabilizers': ['+ZZI', '+IZZ'],
            'rounds': 2,
            'methods': {
                'mwpm': {
                    'logical_error_rates': [0.001, 0.004]
                }
            }
        }
        
        plot_b64 = plot_decoder_comparison(mock_comparison)
        assert isinstance(plot_b64, str)
        assert len(plot_b64) > 0


class TestEnhancedOutput:
    """Test enhanced output formatting and data handling."""
    
    def test_results_json_serialization(self):
        """Test JSON serialization of results."""
        from gate_optimize.qec.error_analysis import save_results_to_data
        import numpy as np
        
        # Mock results with numpy arrays
        results = {
            'physical_error_rates': np.array([0.01, 0.02]),
            'logical_error_rates': np.array([0.001, 0.004]),
            'num_shots': 1000,
            'decoder_method': 'mwpm',
            'stabilizers': ['+ZZI', '+IZZ']
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_results_to_data(results, 'test_output', data_dir=temp_dir)
            
            json_file = os.path.join(temp_dir, 'test_output.json')
            assert os.path.exists(json_file)
            
            # Verify JSON is valid
            with open(json_file, 'r') as f:
                loaded_data = json.load(f)
                assert 'physical_error_rates' in loaded_data
                assert isinstance(loaded_data['physical_error_rates'], list)
    
    def test_csv_output_format(self):
        """Test CSV output formatting."""
        from gate_optimize.qec.error_analysis import save_results_to_data
        
        results = {
            'physical_error_rates': [0.01, 0.02, 0.03],
            'logical_error_rates': [0.001, 0.004, 0.009],
            'num_shots': 1000
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_results_to_data(results, 'test_csv', data_dir=temp_dir)
            
            csv_file = os.path.join(temp_dir, 'test_csv_summary.csv')
            assert os.path.exists(csv_file)
            
            # Verify CSV format
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 4  # Header + 3 data rows
                assert 'physical_error_rate,logical_error_rate' in lines[0]
    
    def test_plot_base64_encoding(self):
        """Test base64 encoding of plots."""
        import base64
        
        mock_results = {
            'physical_error_rates': [0.01, 0.02],
            'logical_error_rates': [0.001, 0.004],
            'decoder_method': 'mwpm',
            'stabilizers': ['+ZZI', '+IZZ'],
            'rounds': 2
        }
        
        plot_b64 = plot_error_rate_curve(mock_results)
        
        # Verify it's valid base64
        try:
            decoded = base64.b64decode(plot_b64)
            assert len(decoded) > 0
        except Exception:
            pytest.fail("Invalid base64 encoding")


class TestPerformanceAndStress:
    """Test performance and stress scenarios."""
    
    def test_large_code_performance(self):
        """Test performance with larger codes."""
        # 9-qubit repetition code
        stabilizers = ['+ZZI' + 'I'*6, '+IZZ' + 'I'*6, '+IIZ' + 'Z' + 'I'*5]
        
        from gate_optimize.qec.code_builder import QECCodeBuilder
        builder = QECCodeBuilder(stabilizers, rounds=1)
        
        # This should complete reasonably quickly
        circuit = builder.build_syndrome_circuit()
        assert circuit is not None
        assert len(circuit) > 0
    
    def test_many_shots_simulation(self):
        """Test simulation with many shots."""
        from gate_optimize.qec.code_builder import QECCodeBuilder
        from gate_optimize.qec.error_analysis import count_logical_errors
        
        stabilizers = ['+ZZI', '+IZZ']
        builder = QECCodeBuilder(stabilizers, rounds=1)
        circuit = builder.build_syndrome_circuit()
        
        # Test with more shots (but still reasonable for testing)
        num_errors = count_logical_errors(circuit, num_shots=500)
        assert isinstance(num_errors, int)
        assert 0 <= num_errors <= 500


# Legacy test functions for backward compatibility
def test_qec_mcp_tool():
    """Legacy MCP tool test."""
    async def run_test():
        try:
            from gate_optimize.server_qec import analyze_qec_logical_error_rate
            
            result = await analyze_qec_logical_error_rate(
                stabilizers=['+ZZI', '+IZZ'],
                logical_Z_operators=['ZII'],  # Single qubit logical Z
                rounds=2,
                decoder_method='mwpm'
            )
            return len(result) > 0
        except Exception:
            return False
    
    try:
        return asyncio.run(run_test())
    except Exception:
        return False


def test_mcp_tool_with_toric():
    """Legacy toric MCP test."""
    async def run_test():
        try:
            from gate_optimize.server_qec import analyze_qec_logical_error_rate
            
            toric_stabilizers = [
                '+ZZII_IIII',
                '+IZZI_IIII', 
                '+IIZZ_IIII',
                '+XXXX_IIII'
            ]
            
            result = await analyze_qec_logical_error_rate(
                stabilizers=toric_stabilizers,
                logical_Z_operators=['ZIIIIIIII', 'IIIIZIIIII'],
                rounds=2,
                decoder_method='mwpm'
            )
            return len(result) > 0
        except Exception:
            return False
    
    try:
        return asyncio.run(run_test())
    except Exception:
        return False


if __name__ == "__main__":
    print("QEC MCP Integration Test Suite")
    print("=" * 40)
    
    legacy_tests = [
        ("MCP tool basic", test_qec_mcp_tool),
        ("MCP tool with toric", test_mcp_tool_with_toric),
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
        print("🎉 All MCP tests passed!")
    else:
        print("❌ Some tests failed")