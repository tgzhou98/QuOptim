"""
Simplified MCP server tests - focusing on core functionality.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path for imports  
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gate_optimize.server import mcp


class TestMCPServerBasics:
    """Basic MCP server functionality tests."""
    
    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test that MCP server initializes correctly."""
        tools = await mcp.list_tools()
        assert len(tools) >= 3  # Should have at least 3 tools
        
        tool_names = [tool.name for tool in tools]
        expected_tools = [
            "optimize_cz_gate",
            "optimize_x_gate", 
            "generate_circuit_from_stabilizers"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Tool {expected_tool} not found"
    
    # Circuit generation functionality is tested in test_mcp_tools.py
    # MCP framework call structure is complex and already covered by direct function tests