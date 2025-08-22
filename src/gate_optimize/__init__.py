import sys
import os
from pathlib import Path
from importlib import resources
from .custom_gui import *

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
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

    # logger.info("starting gpaw computation server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()