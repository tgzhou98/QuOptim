"""
End-to-end MCP server tests using the FastMCP testing framework.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gate_optimize.server import mcp
from mcp.types import TextContent, ImageContent


class TestMCPServer:
    """Test the MCP server functionality."""
    
    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test that the MCP server initializes correctly."""
        assert mcp is not None
        assert mcp.name == "mcp-gate-optimize"
        
        # Check that tools are registered
        tools = await mcp.list_tools()
        tool_names = [tool.name for tool in tools]
        expected_tools = [
            "optimize_cz_gate",
            "optimize_x_gate", 
            "generate_circuit_from_stabilizers",
            "generate_steane_code_circuits"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Tool {expected_tool} not found in registered tools"
    
    @pytest.mark.asyncio
    async def test_tool_descriptions(self):
        """Test that tools have proper descriptions."""
        tools_list = await mcp.list_tools()
        tools = {tool.name: tool for tool in tools_list}
        
        # Check circuit generation tools
        circuit_tool = tools["generate_circuit_from_stabilizers"]
        assert "stabilizer" in circuit_tool.description.lower()
        assert "circuit" in circuit_tool.description.lower()
        
        steane_tool = tools["generate_steane_code_circuits"]
        assert "steane" in steane_tool.description.lower()
        assert "7-qubit" in steane_tool.description.lower()
    
    @pytest.mark.asyncio
    async def test_tool_execution_basic(self):
        """Test basic tool execution through MCP framework."""
        # Test the Steane code tool
        result = await mcp.call_tool(
            "generate_steane_code_circuits", 
            {"num_variants": 1}
        )
        
        assert isinstance(result, list)
        assert len(result) >= 1
        assert isinstance(result[0], TextContent)
        assert "QUANTUM CIRCUIT GENERATION" in result[0].text
    
    @pytest.mark.asyncio 
    async def test_tool_parameter_validation(self):
        """Test tool parameter validation."""
        # Test with invalid parameter types
        with pytest.raises(Exception):
            await mcp.call_tool(
                "generate_steane_code_circuits",
                {"num_variants": "invalid"}  # Should be int
            )
    
    @pytest.mark.asyncio
    async def test_custom_stabilizers_tool(self):
        """Test custom stabilizer tool through MCP."""
        result = await mcp.call_tool(
            "generate_circuit_from_stabilizers",
            {
                "stabilizers": ["+ZZ_", "+_ZZ"],
                "num_circuits": 1
            }
        )
        
        assert isinstance(result, list)
        assert len(result) >= 1
        assert isinstance(result[0], TextContent)
        assert "3" in result[0].text  # Should detect 3 qubits


class TestPulseOptimizationTools:
    """Test the pulse optimization tools."""
    
    @pytest.mark.asyncio
    async def test_cz_gate_optimization(self):
        """Test CZ gate optimization tool."""
        # Mock matplotlib to avoid display issues in tests
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.plot'), \
             patch('matplotlib.pyplot.xlabel'), \
             patch('matplotlib.pyplot.ylabel'), \
             patch('matplotlib.pyplot.title'), \
             patch('matplotlib.pyplot.ylim'), \
             patch('matplotlib.pyplot.clf'):
            
            # Mock the figure's to_image method
            mock_fig = MagicMock()
            mock_fig.to_image.return_value = b'fake_image_data'
            
            with patch('matplotlib.pyplot.figure', return_value=mock_fig):
                result = await mcp.call_tool(
                    "optimize_cz_gate",
                    {"iterations": 5, "learning_rate": 0.1}  # Small values for testing
                )
                
                assert isinstance(result, list)
                assert len(result) == 2
                assert isinstance(result[0], ImageContent)
                assert isinstance(result[1], TextContent)
                assert "fidelity" in result[1].text.lower()
    
    @pytest.mark.asyncio
    async def test_x_gate_optimization(self):
        """Test X gate optimization tool."""
        # Mock matplotlib components
        with patch('matplotlib.pyplot.figure'), \
             patch('numpy.linspace'), \
             patch('matplotlib.pyplot.plot'):
            
            mock_fig = MagicMock()
            mock_fig.to_image.return_value = b'fake_image_data'
            mock_fig.add_subplot.return_value = MagicMock()
            
            with patch('matplotlib.pyplot.figure', return_value=mock_fig):
                result = await mcp.call_tool(
                    "optimize_x_gate", 
                    {
                        "iterations": 3, 
                        "learning_rate": 0.05,
                        "fourier_terms": 3
                    }
                )
                
                assert isinstance(result, list)
                assert len(result) == 2
                assert isinstance(result[0], ImageContent)
                assert isinstance(result[1], TextContent)
                assert "fidelity" in result[1].text.lower()


class TestMCPServerIntegration:
    """Full integration tests for MCP server."""
    
    @pytest.mark.asyncio
    async def test_server_tool_listing(self):
        """Test that server can list all tools correctly."""
        tools = await mcp.list_tools()
        
        assert len(tools) >= 4  # We have at least 4 tools
        
        tool_names = [tool.name for tool in tools]
        required_tools = [
            "optimize_cz_gate",
            "optimize_x_gate",
            "generate_circuit_from_stabilizers", 
            "generate_steane_code_circuits"
        ]
        
        for tool_name in required_tools:
            assert tool_name in tool_names
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self):
        """Test that multiple tools can be called concurrently."""
        # Create multiple concurrent tool calls
        tasks = [
            mcp.call_tool("generate_steane_code_circuits", {"num_variants": 1}),
            mcp.call_tool("generate_circuit_from_stabilizers", {
                "stabilizers": ["+ZZ_", "+_ZZ"], 
                "num_circuits": 1
            })
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 2
        for result in results:
            assert isinstance(result, list)
            assert len(result) >= 1
            assert isinstance(result[0], TextContent)


# Test configuration
pytest_plugins = ["pytest_asyncio"]


if __name__ == "__main__":
    # Run basic server tests
    print("Testing MCP server...")
    
    async def run_async_tests():
        test_server = TestMCPServer()
        test_integration = TestMCPServerIntegration()
        
        print("Testing server initialization...")
        await test_server.test_server_initialization()
        print("✓ Server initialization test passed")
        
        print("Testing tool descriptions...")
        await test_server.test_tool_descriptions()
        print("✓ Tool descriptions test passed")
        
        print("Testing tool listing...")
        await test_integration.test_server_tool_listing()
        print("✓ Tool listing test passed")
        
        print("Testing basic tool execution...")
        await test_server.test_tool_execution_basic()
        print("✓ Basic tool execution test passed")
        
        print("Testing custom stabilizer tool...")
        await test_server.test_custom_stabilizers_tool()
        print("✓ Custom stabilizer tool test passed")
        
        print("\nAll MCP server tests completed successfully!")
    
    asyncio.run(run_async_tests())