import sys
import base64
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QTextEdit, QVBoxLayout, QHBoxLayout,
                             QWidget, QScrollArea, QSplitter, QFormLayout, QFrame, QSlider)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QSize
from PyQt6.QtGui import QPixmap, QImage, QMouseEvent

from flask import Flask, request, jsonify

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

dark_stylesheet = """
    QWidget { background-color: #2b2b2b; color: #f0f0f0; font-family: 'Segoe UI', Arial, sans-serif; }
    QMainWindow { border: 1px solid #3c3c3c; }
    QLabel { font-size: 14px; }
    QTextEdit { background-color: #3c3c3c; border: 1px solid #4f4f4f; font-size: 14px; border-radius: 4px; }
    QScrollArea { border: none; }
    QSplitter::handle { background-color: #4f4f4f; }
    QFrame { border: 1px solid #4f4f4f; border-radius: 5px; }
    QLabel#title { font-size: 18px; font-weight: bold; padding: 5px; border-bottom: 2px solid #007acc; }
    QLabel#statusLabel { font-size: 16px; font-weight: bold; padding: 5px; border-radius: 4px; }
    QSlider::groove:horizontal { border: 1px solid #4f4f4f; background: #3c3c3c; height: 8px; border-radius: 4px; }
    QSlider::handle:horizontal { background: #007acc; border: 1px solid #007acc; width: 18px; margin: -2px 0; border-radius: 9px; }
"""

class FlaskApp(QObject):
    new_data = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self.app = Flask(__name__)
        @self.app.route('/update', methods=['POST'])
        def update_content():
            self.new_data.emit(request.json)
            return jsonify({"status": "success"}), 200
    def run(self):
        self.app.run(host='127.0.0.1', port=12345, debug=False, use_reloader=False)

class ImageViewer(QWidget):
    def __init__(self, pixmap: QPixmap):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout(self)
        scroll_area = QScrollArea(self)
        layout.addWidget(scroll_area)
        
        self.image_label = QLabel()
        self.image_label.setPixmap(pixmap)
        
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_pixmap = None

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        self.clicked.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIQuCompiler Monitoring Center")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(dark_stylesheet)
        self.open_viewers = []

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.setCentralWidget(self.splitter)

        self.top_frame = QFrame()
        self.top_layout = QVBoxLayout(self.top_frame)
        self.top_layout.addWidget(QLabel("Real-time Metrics", objectName="title"))
        self.info_layout = QFormLayout()
        self.status_label = QLabel("Idle", objectName="statusLabel")
        self.tool_label = QLabel("N/A")
        self.params_label = QLabel("N/A")
        self.current_fidelity_label = QLabel("N/A")
        self.info_layout.addRow("Status:", self.status_label)
        self.info_layout.addRow("Tool:", self.tool_label)
        self.info_layout.addRow("Parameters:", self.params_label)
        self.info_layout.addRow("Current Fidelity:", self.current_fidelity_label)
        self.top_layout.addLayout(self.info_layout)
        self.fidelity_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.fidelity_canvas.figure.patch.set_facecolor("#2b2b2b")
        self.fidelity_ax = self.fidelity_canvas.figure.add_subplot(111)
        self.top_layout.addWidget(self.fidelity_canvas)
        
        self.bottom_frame = QFrame()
        self.bottom_layout = QVBoxLayout(self.bottom_frame)
        self.bottom_layout.addWidget(QLabel("Result Visualization", objectName="title"))

        zoom_control_layout = QHBoxLayout()
        zoom_control_layout.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(20, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(200)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(50)
        
        self.zoom_slider.valueChanged.connect(self.update_image_sizes)
        
        zoom_control_layout.addWidget(self.zoom_slider)
        zoom_control_layout.addWidget(self.zoom_label)
        zoom_control_layout.addStretch()
        self.bottom_layout.addLayout(zoom_control_layout)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_container = QWidget()
        self.image_layout = QVBoxLayout(self.image_container)
        self.scroll_area.setWidget(self.image_container)
        self.bottom_layout.addWidget(self.scroll_area)
        
        self.splitter.addWidget(self.top_frame)
        self.splitter.addWidget(self.bottom_frame)
        self.splitter.setSizes([400, 400])

        self.flask_app = FlaskApp()
        self.flask_app.new_data.connect(self.update_ui)
        self.flask_thread = QThread()
        self.flask_app.moveToThread(self.flask_thread)
        self.flask_thread.started.connect(self.flask_app.run)
        self.flask_thread.start()

    def update_image_sizes(self, value):
        self.zoom_label.setText(f"{value}%")
        for i in range(self.image_layout.count()):
            widget = self.image_layout.itemAt(i).widget()
            if isinstance(widget, ClickableLabel):
                original_pixmap = widget.original_pixmap
                if original_pixmap:
                    new_width = int(original_pixmap.width() * value / 100)
                    scaled_pixmap = original_pixmap.scaledToWidth(new_width, 
                                                                  Qt.TransformationMode.SmoothTransformation)
                    widget.setPixmap(scaled_pixmap)

    def show_image_in_new_window(self, pixmap):
        viewer = ImageViewer(pixmap)
        self.open_viewers.append(viewer)
        viewer.show()

    def update_ui(self, data):
        status = data.get("status", "unknown")

        # --- Part 1: Update labels and top plot (Always happens) ---
        if status == "running": self.status_label.setText("Running..."); self.status_label.setStyleSheet("background-color: #b8860b; color: white;")
        elif status == "finished": self.status_label.setText("Finished"); self.status_label.setStyleSheet("background-color: #228b22; color: white;")
        elif status == "error": self.status_label.setText("Error"); self.status_label.setStyleSheet("background-color: #b22222; color: white;")
        self.tool_label.setText(data.get("tool_name", "N/A"))
        params_str = ", ".join(f"{k}={v}" for k, v in data.get("parameters", {}).items())
        self.params_label.setText(params_str if params_str else "N/A")
        fidelity = data.get("live_metrics", {}).get("fidelity")
        if fidelity is not None: self.current_fidelity_label.setText(f"<b>{fidelity:.6f}</b>")
        
        fidelity_history = data.get("fidelity_history", [])
        if fidelity_history:
            self.fidelity_ax.cla()
            self.fidelity_ax.plot(fidelity_history, marker='o', linestyle='-', markersize=3, color='#007acc')
            self.fidelity_ax.set_title("Fidelity vs. Iterations", color="white")
            self.fidelity_ax.set_xlabel("Update Step", color="white")
            self.fidelity_ax.set_ylabel("Fidelity", color="white")
            self.fidelity_ax.grid(True, linestyle='--', alpha=0.3)
            self.fidelity_ax.tick_params(axis='x', colors='white')
            self.fidelity_ax.tick_params(axis='y', colors='white')
            self.fidelity_ax.spines['bottom'].set_color('white'); self.fidelity_ax.spines['top'].set_color('white')
            self.fidelity_ax.spines['right'].set_color('white'); self.fidelity_ax.spines['left'].set_color('white')
            self.fidelity_canvas.draw()
        
        # --- Part 2: Conditionally update the bottom panel ---
        images_base64 = data.get("main_images", [])
        text_result = data.get("text_result", "")

        # Only clear the bottom panel if new content (images or text) is being sent.
        if images_base64 or text_result:
            while self.image_layout.count():
                child = self.image_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        
        # FIX: Only reset the zoom slider if the task is finished or has errored.
        # This prevents the slider from resetting during live "running" updates.
        # if status != 'running':
        #     self.zoom_slider.setValue(100)

        # --- Part 3: Populate the bottom panel with new content (Always happens if content exists) ---
        for img_b64 in images_base64:
            try:
                img_bytes = base64.b64decode(img_b64)
                original_pixmap = QPixmap()
                original_pixmap.loadFromData(img_bytes)
                
                image_label = ClickableLabel()
                image_label.original_pixmap = original_pixmap
                image_label.setCursor(Qt.CursorShape.PointingHandCursor)
                
                image_label.clicked.connect(lambda pix=original_pixmap: self.show_image_in_new_window(pix))
                
                self.image_layout.addWidget(image_label)
            except Exception as e:
                # print(f"Error displaying image: {e}")
                pass
        
        # Apply the current zoom level to the newly added images
        self.update_image_sizes(self.zoom_slider.value())
        
        if text_result:
            text_label = QLabel(text_result)
            text_label.setWordWrap(True)
            text_label.setTextFormat(Qt.TextFormat.PlainText)
            text_label.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")
            self.image_layout.addWidget(text_label)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())