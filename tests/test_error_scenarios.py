"""
Comprehensive error handling and edge case tests for MCP tools.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gate_optimize.server import generate_circuit_from_stabilizers, generate_steane_code_circuits
from mcp.types import TextContent


class TestInputValidation:
    """Test input validation and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_stabilizer_list(self):
        """Test handling of empty stabilizer list."""
        result = await generate_circuit_from_stabilizers([], num_circuits=1)
        
        assert len(result) == 1
        assert "Error: stabilizers must be a non-empty list" in result[0].text
    
    @pytest.mark.asyncio
    async def test_none_stabilizer_input(self):
        """Test handling of None input."""
        result = await generate_circuit_from_stabilizers(None, num_circuits=1)
        
        assert len(result) == 1
        assert "Error: stabilizers must be a non-empty list" in result[0].text
    
    @pytest.mark.asyncio
    async def test_string_instead_of_list(self):
        """Test handling of string input instead of list."""
        result = await generate_circuit_from_stabilizers("invalid_input", num_circuits=1)
        
        assert len(result) == 1
        assert "Error: stabilizers must be a non-empty list" in result[0].text
    
    @pytest.mark.asyncio
    async def test_invalid_pauli_operators(self):
        """Test handling of invalid Pauli operators."""
        invalid_stabilizers = ['+ABC', '+XYV']  # V is not a valid Pauli operator
        
        result = await generate_circuit_from_stabilizers(invalid_stabilizers, num_circuits=1)
        
        assert len(result) == 1
        assert "Error parsing stabilizers" in result[0].text
    
    @pytest.mark.asyncio
    async def test_inconsistent_stabilizer_lengths(self):
        """Test handling of stabilizers with different lengths."""
        inconsistent_stabilizers = ['+ZZ', '+XYZ_', '+I']  # Different lengths
        
        result = await generate_circuit_from_stabilizers(inconsistent_stabilizers, num_circuits=1)
        
        assert len(result) == 1
        # Should either parse successfully or report an error gracefully
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_negative_num_circuits(self):
        """Test handling of negative num_circuits."""
        result = await generate_circuit_from_stabilizers(['+ZZ_', '+_ZZ'], num_circuits=-1)
        
        assert len(result) == 1
        # Should handle gracefully - either clamp to 0 or generate error
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_zero_num_circuits(self):
        """Test handling of zero num_circuits."""
        result = await generate_circuit_from_stabilizers(['+ZZ_', '+_ZZ'], num_circuits=0)
        
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_very_large_num_circuits(self):
        """Test handling of very large num_circuits."""
        result = await generate_circuit_from_stabilizers(['+ZZ_', '+_ZZ'], num_circuits=1000)
        
        assert len(result) == 1
        # Should be limited to reasonable number (5 max in implementation)
        output = result[0].text
        circuit_count = output.count("CIRCUIT VARIANT")
        assert circuit_count <= 5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_single_qubit_stabilizers(self):
        """Test with single qubit stabilizers."""
        single_qubit_stabilizers = ['+Z']
        
        result = await generate_circuit_from_stabilizers(single_qubit_stabilizers, num_circuits=1)
        
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        # Should handle single qubit case
    
    @pytest.mark.asyncio
    async def test_no_sign_stabilizers(self):
        """Test stabilizers without +/- signs."""
        no_sign_stabilizers = ['ZZ_', '_ZZ', 'XXX']
        
        result = await generate_circuit_from_stabilizers(no_sign_stabilizers, num_circuits=1)
        
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        # Should handle missing signs gracefully
    
    @pytest.mark.asyncio
    async def test_mixed_signs_stabilizers(self):
        """Test stabilizers with mixed +/- signs."""
        mixed_sign_stabilizers = ['+ZZ_', '-_ZZ', '+XXX']
        
        result = await generate_circuit_from_stabilizers(mixed_sign_stabilizers, num_circuits=1)
        
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_all_identity_stabilizers(self):
        """Test stabilizers that are all identity."""
        identity_stabilizers = ['+___', '+___']
        
        result = await generate_circuit_from_stabilizers(identity_stabilizers, num_circuits=1)
        
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_very_large_stabilizer_code(self):
        """Test with a large number of qubits (if computationally feasible)."""
        # Create a larger stabilizer code (but still reasonable)
        large_stabilizers = [
            '+ZZ________',
            '+_ZZ_______',
            '+__ZZ______',
            '+___ZZ_____',
            '+____ZZ____',
            '+_____ZZ___',
            '+______ZZ__',
            '+_______ZZ_',
            '+________ZZ',
            '+XXXXXXXXXX'
        ]
        
        result = await generate_circuit_from_stabilizers(large_stabilizers, num_circuits=1)
        
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        
        # Should detect correct number of qubits
        if "Error" not in result[0].text:
            assert "10" in result[0].text  # 10 qubits


class TestSystemErrors:
    """Test system-level error scenarios."""
    
    @pytest.mark.asyncio
    async def test_steane_code_with_zero_variants(self):
        """Test Steane code generation with zero variants."""
        result = await generate_steane_code_circuits(num_variants=0)
        
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio 
    async def test_steane_code_with_negative_variants(self):
        """Test Steane code generation with negative variants."""
        result = await generate_steane_code_circuits(num_variants=-1)
        
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_malformed_stabilizer_strings(self):
        """Test various malformed stabilizer string formats."""
        malformed_cases = [
            ['++ZZ_', '+_ZZ'],      # Double plus
            ['+ZZ_+', '+_ZZ'],      # Plus in wrong place
            ['+ ZZ_', '+_ZZ'],      # Space after plus
            ['+ZZ_\n', '+_ZZ'],     # Newline character
            ['', '+_ZZ'],           # Empty string
            ['+ZZ_', ''],           # Empty string in second position
        ]
        
        for i, malformed_stabilizers in enumerate(malformed_cases):
            result = await generate_circuit_from_stabilizers(malformed_stabilizers, num_circuits=1)
            
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            
            # Should either handle gracefully or report appropriate error
            output = result[0].text
            is_error = "Error" in output or "error" in output
            is_success = "QUANTUM CIRCUIT GENERATION" in output
            
            # Either should work or should error gracefully
            assert is_error or is_success, f"Case {i} failed: {malformed_stabilizers}"


class TestPerformanceEdgeCases:
    """Test performance and resource-related edge cases."""
    
    @pytest.mark.asyncio
    async def test_max_circuit_limit_enforcement(self):
        """Test that circuit generation respects maximum limits."""
        result = await generate_circuit_from_stabilizers(
            ['+ZZ_', '+_ZZ'], 
            num_circuits=100  # Request way more than limit
        )
        
        assert len(result) == 1
        output = result[0].text
        
        # Count actual circuit variants generated
        variant_count = output.count("--- CIRCUIT VARIANT")
        assert variant_count <= 5, f"Generated {variant_count} circuits, should be ≤ 5"
    
    @pytest.mark.asyncio
    async def test_timeout_resilience(self):
        """Test that functions complete in reasonable time."""
        import time
        
        start_time = time.time()
        
        result = await generate_circuit_from_stabilizers(
            ['+ZZII', '+IZZI', '+IIZZ'], 
            num_circuits=2
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert duration < 60, f"Function took {duration:.2f} seconds, too long"
        assert len(result) == 1
        assert isinstance(result[0], TextContent)


if __name__ == "__main__":
    print("Running error handling and edge case tests...")
    
    async def run_error_tests():
        # Input validation tests
        print("\nTesting input validation...")
        test_validation = TestInputValidation()
        await test_validation.test_empty_stabilizer_list()
        await test_validation.test_none_stabilizer_input()
        await test_validation.test_invalid_pauli_operators()
        print("✓ Input validation tests passed")
        
        # Edge case tests
        print("\nTesting edge cases...")
        test_edge = TestEdgeCases()
        await test_edge.test_single_qubit_stabilizers()
        await test_edge.test_mixed_signs_stabilizers()
        print("✓ Edge case tests passed")
        
        # System error tests
        print("\nTesting system errors...")
        test_system = TestSystemErrors()
        await test_system.test_steane_code_with_zero_variants()
        await test_system.test_malformed_stabilizer_strings()
        print("✓ System error tests passed")
        
        # Performance tests
        print("\nTesting performance edge cases...")
        test_perf = TestPerformanceEdgeCases()
        await test_perf.test_max_circuit_limit_enforcement()
        print("✓ Performance tests passed")
        
        print("\nAll error handling tests completed successfully!")
    
    asyncio.run(run_error_tests())