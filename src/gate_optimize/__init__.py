import sys
import os
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

from gate_optimize.server import mcp
import logging

# Set up logging (this just prints messages to your terminal for debugging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main() -> None:
    if PYQT6_AVAILABLE:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    else:
        # Fallback when GUI is not available
        logger.info("GUI not available, starting MCP server directly...")
        mcp.run(transport="stdio")

    # logger.info("starting gpaw computation server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()