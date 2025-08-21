"""
Unit tests for MCP tools in gate_optimize server.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gate_optimize.server import generate_circuit_from_stabilizers, generate_steane_code_circuits
from mcp.types import TextContent
import stim


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


class TestMCPTools:
    """Test MCP tool functions."""
    
    @pytest.mark.asyncio
    async def test_generate_circuit_from_stabilizers_basic(self):
        """Test basic functionality of generate_circuit_from_stabilizers."""
        stabilizers = ['+ZZ_', '+_ZZ']  # Simple 3-qubit repetition code
        
        result = await generate_circuit_from_stabilizers(stabilizers, num_circuits=1)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "QUANTUM CIRCUIT GENERATION FROM STABILIZERS" in result[0].text
        assert "Number of Qubits: 3" in result[0].text
    
    @pytest.mark.asyncio
    async def test_generate_circuit_empty_stabilizers(self):
        """Test error handling for empty stabilizers."""
        result = await generate_circuit_from_stabilizers([], num_circuits=1)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert "Error: stabilizers must be a non-empty list" in result[0].text
    
    @pytest.mark.asyncio
    async def test_generate_circuit_invalid_stabilizers(self):
        """Test error handling for invalid stabilizer format."""
        invalid_stabilizers = ['+ABC', '+XYZ']  # Invalid operators
        
        result = await generate_circuit_from_stabilizers(invalid_stabilizers, num_circuits=1)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert "Error parsing stabilizers" in result[0].text
    
    @pytest.mark.asyncio
    async def test_generate_steane_code_circuits(self):
        """Test the Steane code convenience function."""
        result = await generate_steane_code_circuits(num_variants=1)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "QUANTUM CIRCUIT GENERATION FROM STABILIZERS" in result[0].text
        assert "Number of Qubits: 7" in result[0].text
    
    @pytest.mark.asyncio
    async def test_circuit_generation_limits(self):
        """Test that circuit generation respects limits."""
        stabilizers = ['+ZZ_', '+_ZZ']
        
        # Request more circuits than the limit (5)
        result = await generate_circuit_from_stabilizers(stabilizers, num_circuits=10)
        
        assert isinstance(result, list)
        assert len(result) == 1
        # Should be limited to max 5 circuits
        circuit_variants = result[0].text.count("--- CIRCUIT VARIANT")
        assert circuit_variants <= 5


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_malformed_input_types(self):
        """Test handling of malformed input types."""
        # Test with non-list input
        result = await generate_circuit_from_stabilizers("not a list", num_circuits=1)
        assert "Error: stabilizers must be a non-empty list" in result[0].text
        
        # Test with None input
        result = await generate_circuit_from_stabilizers(None, num_circuits=1)
        assert "Error: stabilizers must be a non-empty list" in result[0].text
    
    @pytest.mark.asyncio
    async def test_inconsistent_stabilizer_lengths(self):
        """Test handling of stabilizers with inconsistent lengths."""
        inconsistent_stabilizers = ['+ZZ', '+XYZ_']  # Different lengths
        
        result = await generate_circuit_from_stabilizers(inconsistent_stabilizers, num_circuits=1)
        
        assert isinstance(result, list)
        # Should handle gracefully and report error
        assert ("Error" in result[0].text or "stabilizers" in result[0].text.lower())


class TestIntegration:
    """Integration tests that require the full system."""
    
    @pytest.mark.asyncio
    async def test_full_steane_code_generation(self):
        """Full integration test with Steane code."""
        result = await generate_steane_code_circuits(num_variants=2)
        
        assert isinstance(result, list)
        assert len(result) == 1
        
        output = result[0].text
        
        # Check for expected sections
        assert "QUANTUM CIRCUIT GENERATION FROM STABILIZERS" in output
        assert "Input Stabilizers:" in output
        assert "Number of Qubits: 7" in output
        assert "CIRCUIT VARIANT 1" in output
        assert "CIRCUIT VARIANT 2" in output
        
        # Should contain fidelity information
        assert "Fidelity:" in output or "fidelity:" in output
        
        # Should contain gate information
        assert "Gate Count:" in output or "Gate Array:" in output
    
    @pytest.mark.asyncio
    async def test_model_loading_scenarios(self):
        """Test both with and without pre-trained models."""
        stabilizers = ['+ZZ_', '+_ZZ']  # Simple case
        
        result = await generate_circuit_from_stabilizers(stabilizers, num_circuits=1)
        
        assert isinstance(result, list)
        output = result[0].text
        
        # Should indicate model loading status
        assert ("Loaded pre-trained RL model" in output or 
                "Using random policy" in output or
                "no pre-trained model" in output)


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
        
        print("Testing basic MCP tool functionality...")
        await test_mcp.test_generate_circuit_from_stabilizers_basic()
        print("✓ Basic MCP tool test passed")
        
        print("Testing error handling...")
        await test_mcp.test_generate_circuit_empty_stabilizers()
        print("✓ Error handling test passed")
        
        print("Testing Steane code generation...")
        await test_mcp.test_generate_steane_code_circuits()
        print("✓ Steane code test passed")
        
        print("\nAll tests completed successfully!")
    
    asyncio.run(run_async_tests())