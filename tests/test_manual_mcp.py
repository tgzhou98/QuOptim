"""
Manual testing script for MCP tools - can be run directly to test functionality.
"""

import asyncio
import sys
from pathlib import Path
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gate_optimize.server import generate_circuit_from_stabilizers, generate_steane_code_circuits


@pytest.mark.asyncio
async def test_simple_repetition_code():
    """Test with a simple 3-qubit repetition code."""
    print("\n" + "="*60)
    print("TEST 1: 3-qubit repetition code")
    print("="*60)
    
    stabilizers = ['+ZZ_', '+_ZZ']
    print(f"Input stabilizers: {stabilizers}")
    
    result = await generate_circuit_from_stabilizers(stabilizers, num_circuits=1)
    print("\nResult:")
    print(result[0].text)
    

@pytest.mark.asyncio
async def test_steane_code():
    """Test with the 7-qubit Steane code."""
    print("\n" + "="*60)
    print("TEST 2: 7-qubit Steane code")
    print("="*60)
    
    result = await generate_steane_code_circuits(num_variants=2)
    print("Result:")
    print(result[0].text)


@pytest.mark.asyncio
async def test_5qubit_code():
    """Test with 5-qubit stabilizer code."""
    print("\n" + "="*60)
    print("TEST 3: 5-qubit stabilizer code")
    print("="*60)
    
    stabilizers = ['+XZZXI', '+IXZZX', '+XIXZZ', '+ZXIXZ']
    print(f"Input stabilizers: {stabilizers}")
    
    result = await generate_circuit_from_stabilizers(stabilizers, num_circuits=1)
    print("\nResult:")
    print(result[0].text)


@pytest.mark.asyncio
async def test_error_cases():
    """Test error handling."""
    print("\n" + "="*60)
    print("TEST 4: Error handling")
    print("="*60)
    
    # Test empty stabilizers
    print("Testing empty stabilizers...")
    result = await generate_circuit_from_stabilizers([], num_circuits=1)
    print(f"Result: {result[0].text[:100]}...")
    
    # Test invalid stabilizers
    print("\nTesting invalid stabilizers...")
    result = await generate_circuit_from_stabilizers(['+ABC', '+XYZ'], num_circuits=1)
    print(f"Result: {result[0].text[:100]}...")


@pytest.mark.asyncio
async def test_different_formats():
    """Test different stabilizer formats."""
    print("\n" + "="*60)
    print("TEST 5: Different stabilizer formats")
    print("="*60)
    
    formats_to_test = [
        ['+ZZ_', '+_ZZ'],           # With + signs
        ['-ZZ_', '+_ZZ'],           # Mixed signs  
        ['ZZ_', '_ZZ'],             # No signs
        ['+ZZIII', '+IZZII']        # Explicit identity
    ]
    
    for i, stabilizers in enumerate(formats_to_test):
        print(f"\nFormat {i+1}: {stabilizers}")
        try:
            result = await generate_circuit_from_stabilizers(stabilizers, num_circuits=1)
            success = "Error" not in result[0].text[:200]
            print(f"Success: {success}")
            if success:
                lines = result[0].text.split('\n')[:10]  # First 10 lines
                print("Output preview:")
                for line in lines:
                    print(f"  {line}")
        except Exception as e:
            print(f"Exception: {e}")


async def main():
    """Run all manual tests."""
    print("Starting manual MCP tool tests...")
    print("This will test the circuit generation tools with various inputs.")
    
    try:
        # Quick test first
        await test_simple_repetition_code()
        
        # More comprehensive tests
        await test_steane_code()
        await test_5qubit_code() 
        await test_error_cases()
        await test_different_formats()
        
        print("\n" + "="*60)
        print("ALL MANUAL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())