import sys
import os
import asyncio
import threading
from pathlib import Path
from importlib import resources

# Try to import PyQt6-dependent modules, but don't fail if they're not available
try:
    from .custom_gui import *
    PYQT6_AVAILABLE = True
except ImportError:
    # PyQt6 not available (e.g., in CI environment)
    PYQT6_AVAILABLE = False
    # Create dummy classes for when GUI is not available
    class QApplication:
        def __init__(self, *args, **kwargs):
            pass
        def exec(self):
            pass
    
    class MainWindow:
        def __init__(self):
            pass
        def show(self):
            pass

# Add src directory to Python path for proper module imports
import gate_optimize
project_root = resources.files(gate_optimize).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Don't import server at module level to avoid circular imports
# The server will be imported when needed
import logging

# Set up logging (this just prints messages to your terminal for debugging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_mcp_server():
    """Run MCP server in a separate thread"""
    try:
        logger.info("Starting MCP server...")
        from gate_optimize.server import mcp
        asyncio.run(mcp.run_stdio_async())
    except Exception as e:
        logger.error(f"MCP server error: {e}")


def main() -> None:
    """Main entry point that runs both GUI and MCP server"""
    # Start MCP server in a separate thread
    mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
    mcp_thread.start()
    logger.info("MCP server started in background thread")
    
    # Start Qt application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    logger.info("Starting GUI application...")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()