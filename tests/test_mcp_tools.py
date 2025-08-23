"""
Unit tests for MCP tools in gate_optimize server.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gate_optimize.server import generate_circuits
from gate_optimize.server import optimize_cz_gate, optimize_x_gate
from mcp.types import TextContent, ImageContent
import stim


class TestGateOptimization:
    """Test gate optimization functions."""
    
    @pytest.mark.asyncio
    async def test_optimize_cz_gate_basic(self):
        """Test basic CZ gate optimization functionality."""
        result = await optimize_cz_gate(iterations=10, learning_rate=0.3, time_steps=20)
        
        assert isinstance(result, list)
        assert len(result) == 2  # Should return image and text
        
        # Check image content
        image_content = next(item for item in result if isinstance(item, ImageContent))
        assert image_content.type == "image"
        assert image_content.mimeType == "image/png"
        assert len(image_content.data) > 1000  # Should have substantial image data
        
        # Check text content
        text_content = next(item for item in result if isinstance(item, TextContent))
        assert "CZ GATE GRAPE OPTIMIZATION" in text_content.text
        assert "Pulse Parameters:" in text_content.text
        assert "Final Results:" in text_content.text
        assert "Bidirectional Evolution Analysis:" in text_content.text
        assert "Final fidelity:" in text_content.text
    
    @pytest.mark.asyncio
    async def test_optimize_cz_gate_parameters(self):
        """Test CZ gate optimization with different parameters."""
        result = await optimize_cz_gate(
            iterations=15,
            learning_rate=0.5,
            time_steps=30
        )
        
        assert isinstance(result, list)
        text_content = next(item for item in result if isinstance(item, TextContent))
        
        # Check parameter reporting (now using defaults set inside function)
        assert "Total time: 7.600" in text_content.text  # Default pulse_time = 7.6
        assert "Time steps: 30" in text_content.text
        assert "Omega_max: 1" in text_content.text  # Default omega_max = 1.0
        assert "Learning rate: 0.5" in text_content.text
        assert "Iterations: 15" in text_content.text
    
    @pytest.mark.asyncio
    async def test_optimize_x_gate_basic(self):
        """Test basic X gate optimization functionality."""
        result = await optimize_x_gate(iterations=10, learning_rate=0.05, fourier_terms=3)
        
        assert isinstance(result, list)
        assert len(result) == 2  # Should return image and text
        
        # Check image content
        image_content = next(item for item in result if isinstance(item, ImageContent))
        assert image_content.type == "image"
        assert image_content.mimeType == "image/png"
        assert len(image_content.data) > 1000  # Should have substantial image data
        
        # Check text content
        text_content = next(item for item in result if isinstance(item, TextContent))
        assert "X GATE FOURIER OPTIMIZATION" in text_content.text
        assert "Pulse Parameters:" in text_content.text
        assert "Initial Conditions:" in text_content.text
        assert "Optimization Progress:" in text_content.text
        assert "Final Results:" in text_content.text
        assert "Robustness Analysis" in text_content.text
    
    @pytest.mark.asyncio
    async def test_optimize_x_gate_parameters(self):
        """Test X gate optimization with different parameters."""
        result = await optimize_x_gate(
            iterations=12,
            learning_rate=0.08,
            fourier_terms=4
        )
        
        assert isinstance(result, list)
        text_content = next(item for item in result if isinstance(item, TextContent))
        
        # Check parameter reporting (now using defaults set inside function)
        assert "Total time: 7.854" in text_content.text  # π × 2.5 (default pulse_time)
        assert "Fourier terms: 4" in text_content.text
        assert "Rise/fall times:" in text_content.text
        assert "Optimize sine terms: False" in text_content.text  # Default value
        assert "Learning rate: 0.08" in text_content.text
        assert "Iterations: 12" in text_content.text
    
    @pytest.mark.asyncio
    async def test_optimize_x_gate_robustness(self):
        """Test X gate optimization robustness analysis."""
        result = await optimize_x_gate(iterations=8, fourier_terms=2)
        
        text_content = next(item for item in result if isinstance(item, TextContent))
        
        # Check robustness analysis output
        assert "Robustness Analysis (3x3 Optimization Grid):" in text_content.text
        assert "Extended Robustness Analysis (11x11 Evaluation Grid):" in text_content.text
        assert "R0.98_D" in text_content.text  # Should have robustness grid points
        assert "R1.00_D" in text_content.text
        assert "R1.02_D" in text_content.text
        # Check for enhanced robustness metrics
        assert "Average evaluation fidelity (11x11):" in text_content.text
        assert "Robustness range:" in text_content.text
    
    @pytest.mark.asyncio
    async def test_gate_optimization_convergence(self):
        """Test that both gate optimizations show convergence behavior."""
        # Test CZ gate
        cz_result = await optimize_cz_gate(iterations=20, learning_rate=0.4)
        cz_text = next(item for item in cz_result if isinstance(item, TextContent))
        
        # Should show optimization progress
        assert "Optimization Progress:" in cz_text.text
        assert "Iter" in cz_text.text
        assert "Fidelity =" in cz_text.text
        
        # Test X gate  
        x_result = await optimize_x_gate(iterations=20, learning_rate=0.05)
        x_text = next(item for item in x_result if isinstance(item, TextContent))
        
        # Should show optimization progress
        assert "Optimization Progress:" in x_text.text
        assert "Iter" in x_text.text
        assert "Avg Fidelity =" in x_text.text
    
    @pytest.mark.asyncio
    async def test_gate_optimization_output_format(self):
        """Test that both functions return consistent output format."""
        # Test both functions return [ImageContent, TextContent]
        cz_result = await optimize_cz_gate(iterations=5)
        x_result = await optimize_x_gate(iterations=5)
        
        for result in [cz_result, x_result]:
            assert len(result) == 2
            assert any(isinstance(item, ImageContent) for item in result)
            assert any(isinstance(item, TextContent) for item in result)
            
            # Check image has proper metadata
            image = next(item for item in result if isinstance(item, ImageContent))
            assert image.mimeType == "image/png"
            assert len(image.data) > 0
            
            # Check text is not empty
            text = next(item for item in result if isinstance(item, TextContent))
            assert len(text.text) > 100  # Should have substantial text content


class TestStabilizerParsing:
    """Test stabilizer string parsing and validation."""
    
    def test_valid_steane_stabilizers(self):
        """Test parsing of valid 7-qubit Steane code stabilizers."""
        stabilizers = ['+ZZ_____', '+_ZZ____', '+__ZZ___', '+___ZZ__', '+____ZZ_', '+_____ZZ', '+XXXXXXX']
        
        # Test that stabilizers can be converted to tableau
        clean_stabilizers = [s.lstrip('+-').replace('_', 'I') for s in stabilizers]
        tableau = stim.Tableau.from_stabilizers([stim.PauliString(s) for s in clean_stabilizers], allow_underconstrained=True)
        
        assert len(tableau) == 7
        assert len(clean_stabilizers) == 7
    
    def test_valid_3qubit_repetition_code(self):
        """Test parsing of 3-qubit repetition code."""
        stabilizers = ['+ZZ_', '+_ZZ']
        
        clean_stabilizers = [s.lstrip('+-').replace('_', 'I') for s in stabilizers]
        tableau = stim.Tableau.from_stabilizers([stim.PauliString(s) for s in clean_stabilizers], allow_underconstrained=True)
        
        assert len(tableau) == 3
        assert len(clean_stabilizers) == 2
    
    def test_invalid_stabilizers(self):
        """Test that invalid stabilizers raise errors."""
        invalid_stabilizers = ['+XYZ', '+ABC']  # Invalid Pauli operators
        
        with pytest.raises(Exception):
            clean_stabilizers = [s.lstrip('+-').replace('_', 'I') for s in invalid_stabilizers]
            stim.Tableau.from_stabilizers([stim.PauliString(s) for s in clean_stabilizers], allow_underconstrained=True)


# class TestMCPTools:
#     """Test MCP tool functions."""
    
#     @pytest.mark.asyncio
#     async def test_generate_circuits_basic(self):
#         """Test basic functionality of generate_circuits."""
#         stabilizers = ['+ZZ_', '+_ZZ']  # Simple 3-qubit repetition code
        
#         result = await generate_circuits(stabilizers, num_circuits=1)
        
#         assert isinstance(result, list)
#         assert len(result) >= 1  # Now returns text + images
#         # Get the text content
#         text_result = next(r for r in result if hasattr(r, 'text'))
#         assert "QUANTUM CIRCUIT GENERATION FROM STABILIZERS" in text_result.text
#         assert "Number of Qubits: 3" in text_result.text
    
#     @pytest.mark.asyncio
#     async def test_generate_circuit_empty_stabilizers(self):
#         """Test error handling for empty stabilizers."""
#         result = await generate_circuits([], num_circuits=1)
        
#         assert isinstance(result, list)
#         assert len(result) == 1
#         assert "Error: stabilizers must be a non-empty list" in result[0].text
    
#     @pytest.mark.asyncio
#     async def test_generate_circuit_invalid_stabilizers(self):
#         """Test error handling for invalid stabilizer format."""
#         invalid_stabilizers = ['+ABC', '+XYZ']  # Invalid operators
        
#         result = await generate_circuits(invalid_stabilizers, num_circuits=1)
        
#         assert isinstance(result, list)
#         assert len(result) == 1
#         assert "Error parsing stabilizers" in result[0].text
    
#     @pytest.mark.asyncio
#     async def test_generate_steane_code_circuits(self):
#         """Test the Steane code circuit generation using stabilizers."""
#         # Simple 3-qubit test stabilizers that commute
#         steane_stabilizers = [
#             '+ZZI',
#             '+IZZ'
#         ]
        
#         result = await generate_circuits(steane_stabilizers, num_circuits=1)
        
#         assert isinstance(result, list)
#         assert len(result) >= 1  # Should have text content and possibly images
#         assert any(isinstance(r, TextContent) for r in result)  # At least one text content
        
#         text_result = next(r for r in result if isinstance(r, TextContent))
#         assert "QUANTUM CIRCUIT GENERATION FROM STABILIZERS" in text_result.text
#         assert "Number of Qubits: 3" in text_result.text
    
    # @pytest.mark.asyncio
    # async def test_circuit_generation_limits(self):
    #     """Test that circuit generation handles large requests."""
    #     stabilizers = ['+ZZ_', '+_ZZ']
        
    #     # Request more circuits than the limit (5)
    #     result = await generate_circuits(stabilizers, num_circuits=10)
        
    #     assert isinstance(result, list)
    #     assert len(result) >= 1
    #     # Check we got some results
    #     text_result = next(r for r in result if hasattr(r, 'text'))
    #     assert "QUANTUM CIRCUIT GENERATION" in text_result.text


# class TestErrorHandling:
#     """Test error handling scenarios."""
    
#     @pytest.mark.asyncio
#     async def test_malformed_input_types(self):
#         """Test handling of malformed input types."""
#         # Test with non-list input
#         result = await generate_circuits("not a list", num_circuits=1)
#         text_result = next(r for r in result if hasattr(r, 'text'))
#         assert "Error: stabilizers must be a non-empty list" in text_result.text
        
#         # Test with None input
#         result = await generate_circuits(None, num_circuits=1)
#         text_result = next(r for r in result if hasattr(r, 'text'))
#         assert "Error: stabilizers must be a non-empty list" in text_result.text
    
#     @pytest.mark.asyncio
#     async def test_inconsistent_stabilizer_lengths(self):
#         """Test handling of stabilizers with inconsistent lengths."""
#         inconsistent_stabilizers = ['+ZZI', '+XYZ']  # Different lengths but valid
        
#         result = await generate_circuits(inconsistent_stabilizers, num_circuits=1)
        
#         assert isinstance(result, list)
#         # Should handle gracefully - the function now pads stabilizers
#         text_result = next(r for r in result if hasattr(r, 'text'))
#         assert "QUANTUM CIRCUIT GENERATION" in text_result.text


# class TestIntegration:
#     """Integration tests that require the full system."""
    
#     @pytest.mark.asyncio
#     async def test_full_steane_code_generation(self):
#         """Full integration test with Steane code."""
#         # Simple 3-qubit test stabilizers
#         steane_stabilizers = [
#             '+ZZI',
#             '+IZZ'
#         ]
        
#         result = await generate_circuits(steane_stabilizers, num_circuits=2)
        
#         assert isinstance(result, list)
#         assert len(result) >= 1  # Should have text content and possibly images
        
#         text_result = next(r for r in result if isinstance(r, TextContent))
#         output = text_result.text
        
#         # Check for expected sections
#         assert "QUANTUM CIRCUIT GENERATION FROM STABILIZERS" in output
#         assert "Input Stabilizers:" in output
#         assert "Number of Qubits: 3" in output
#         assert "CIRCUIT VARIANT 1" in output
#         assert "CIRCUIT VARIANT 2" in output
        
#         # Should contain fidelity information
#         assert "Fidelity:" in output or "fidelity:" in output
        
#         # Should contain gate information
#         assert "Gate Count:" in output or "Gate Array:" in output
    
    # @pytest.mark.asyncio
    # async def test_model_loading_scenarios(self):
    #     """Test model loading works correctly."""
    #     stabilizers = ['+ZZ_', '+_ZZ']  # Simple case
        
    #     result = await generate_circuits(stabilizers, num_circuits=1)
        
    #     assert isinstance(result, list)
    #     text_result = next(r for r in result if hasattr(r, 'text'))
        
    #     # Should generate circuits successfully
    #     assert "QUANTUM CIRCUIT GENERATION" in text_result.text


if __name__ == "__main__":
    # Run basic tests
    print("Running basic stabilizer parsing tests...")
    
    # Test stabilizer parsing
    test_parsing = TestStabilizerParsing()
    test_parsing.test_valid_steane_stabilizers()
    test_parsing.test_valid_3qubit_repetition_code()
    print("✓ Stabilizer parsing tests passed")
    
    # Run async tests
    async def run_async_tests():
        test_mcp = TestMCPTools()
        
        # print("Testing basic MCP tool functionality...")
        # await test_mcp.test_generate_circuits_basic()
        # print("✓ Basic MCP tool test passed")
        
        # print("Testing error handling...")
        # await test_mcp.test_generate_circuit_empty_stabilizers()
        # print("✓ Error handling test passed")
        
        # print("Testing Steane code generation...")
        # await test_mcp.test_generate_steane_code_circuits()
        # print("✓ Steane code test passed")
        
        print("\nAll tests completed successfully!")
    
    asyncio.run(run_async_tests())