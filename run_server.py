import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    print("Python path configured. Starting MCP server from 'gate_optimize' module...")
    
    from gate_optimize.server import mcp
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()