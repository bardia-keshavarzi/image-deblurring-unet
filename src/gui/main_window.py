# src/gui/main_window.py
"""
Main GUI Window - FIXED VERSION
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QStatusBar
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import numpy as np
from pathlib import Path


class DeblurWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.input_image = None
        self.output_image = None
        self.predictor = None
        
        self._init_ui()
        self._load_model()
    
    def _init_ui(self):
        """Initialize user interface"""
        
        # Window settings
        self.setWindowTitle("Image Deblurring - Week 4 Demo")
        self.setGeometry(100, 100, 1000, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Buttons at top
        button_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("📁 Load Image")
        self.btn_load.clicked.connect(self._on_load_clicked)
        button_layout.addWidget(self.btn_load)
        
        self.btn_deblur = QPushButton("✨ Deblur!")
        self.btn_deblur.clicked.connect(self._on_deblur_clicked)
        self.btn_deblur.setEnabled(False)
        button_layout.addWidget(self.btn_deblur)
        
        self.btn_save = QPushButton("💾 Save Result")
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_save.setEnabled(False)
        button_layout.addWidget(self.btn_save)
        
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        # Image display area
        image_layout = QHBoxLayout()
        
        # Left: Input image
        input_container = QVBoxLayout()
        input_label = QLabel("Input (Blurred)")
        input_label.setAlignment(Qt.AlignCenter)
        input_container.addWidget(input_label)
        
        self.input_display = QLabel()
        self.input_display.setAlignment(Qt.AlignCenter)
        self.input_display.setMinimumSize(400, 400)
        self.input_display.setStyleSheet("border: 2px solid #ccc; background: #f0f0f0;")
        self.input_display.setText("No image loaded")
        input_container.addWidget(self.input_display)
        
        image_layout.addLayout(input_container)
        
        # Right: Output image
        output_container = QVBoxLayout()
        output_label = QLabel("Output (Deblurred)")
        output_label.setAlignment(Qt.AlignCenter)
        output_container.addWidget(output_label)
        
        self.output_display = QLabel()
        self.output_display.setAlignment(Qt.AlignCenter)
        self.output_display.setMinimumSize(400, 400)
        self.output_display.setStyleSheet("border: 2px solid #ccc; background: #f0f0f0;")
        self.output_display.setText("Deblur an image to see result")
        output_container.addWidget(self.output_display)
        
        image_layout.addLayout(output_container)
        
        main_layout.addLayout(image_layout)
        
        # Status bar at bottom
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _load_model(self):
        """Load trained U-Net model - FIXED VERSION"""
        try:
            from src.inference.predictor import DeblurPredictor
            
            # CRITICAL FIX: Use absolute path
            gui_dir = Path(__file__).parent
            project_root = gui_dir.parent.parent
            model_path = project_root / 'checkpoints' / 'best_model.pth'
            
            print(f"🔍 Looking for model at: {model_path.absolute()}")
            print(f"   File exists: {model_path.exists()}")
            
            if not model_path.exists():
                error_msg = (
                    f"Model not found!\n\n"
                    f"Looking at:\n{model_path.absolute()}\n\n"
                    f"Please ensure:\n"
                    f"1. You have trained a model (Week 3)\n"
                    f"2. checkpoints/best_model.pth exists"
                )
                self.status_bar.showMessage("⚠️ Model not found! Train model first.")
                QMessageBox.warning(self, "Model Not Found", error_msg)
                return
            
            print(f"✓ Model file found! Loading...")
            self.predictor = DeblurPredictor(str(model_path))
            self.status_bar.showMessage("✓ Model loaded successfully")
            print(f"✓ Model loaded successfully!")
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ Error loading model:")
            print(error_detail)
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
            self.status_bar.showMessage("❌ Model loading failed")
    
    def _on_load_clicked(self):
        """Handle Load Image button click"""
        
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Blurred Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if not file_path:
            return
        
        # Load image
        try:
            import cv2
            
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                raise ValueError("Cannot load image")
            
            # Convert BGR to RGB
            self.input_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Display
            self._display_image(self.input_image, self.input_display)
            
            # Enable deblur button
            self.btn_deblur.setEnabled(True)
            self.output_display.clear()
            self.output_display.setText("Click 'Deblur!' to process")
            self.btn_save.setEnabled(False)
            
            self.status_bar.showMessage(f"✓ Loaded: {Path(file_path).name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
    
    def _on_deblur_clicked(self):
        """Handle Deblur button click"""
        
        if self.input_image is None:
            return
        
        if self.predictor is None:
            QMessageBox.warning(self, "Warning", "Model not loaded!")
            return
        
        try:
            self.status_bar.showMessage("⏳ Deblurring...")
            self.btn_deblur.setEnabled(False)
            
            # Process image
            print("Processing image...")
            self.output_image = self.predictor.predict(self.input_image)
            print(f"Output range: [{self.output_image.min()}, {self.output_image.max()}]")
            
            # Display result
            self._display_image(self.output_image, self.output_display)
            
            # Enable save button
            self.btn_save.setEnabled(True)
            self.btn_deblur.setEnabled(True)
            
            self.status_bar.showMessage("✓ Deblurring complete!")
            
        except Exception as e:
            import traceback
            print(f"❌ Deblurring error:")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Deblurring failed:\n{str(e)}")
            self.btn_deblur.setEnabled(True)
            self.status_bar.showMessage("❌ Deblurring failed")
    
    def _on_save_clicked(self):
        """Handle Save button click"""
        
        if self.output_image is None:
            return
        
        # Save file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Deblurred Image",
            "deblurred.png",
            "PNG (*.png);;JPEG (*.jpg)"
        )
        
        if not file_path:
            return
        
        try:
            import cv2
            
            # Convert RGB to BGR for saving
            img_bgr = cv2.cvtColor(self.output_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, img_bgr)
            
            self.status_bar.showMessage(f"✓ Saved: {Path(file_path).name}")
            QMessageBox.information(self, "Success", f"Saved to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")
    
    def _display_image(self, image_rgb, label_widget):
        """Display numpy image in QLabel"""
        
        # Ensure uint8 type
        if image_rgb.dtype != np.uint8:
            if image_rgb.max() <= 1.0:
                image_rgb = (image_rgb * 255).astype(np.uint8)
            else:
                image_rgb = image_rgb.astype(np.uint8)
        
        # Make C-contiguous
        image_rgb = np.ascontiguousarray(image_rgb)
        
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        
        # Create QImage
        q_image = QImage(
            image_rgb.data,
            w, h,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        # Copy to prevent memory issues
        q_image = q_image.copy()
        
        # Convert to pixmap and scale
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            label_widget.width() - 20,
            label_widget.height() - 20,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        label_widget.setPixmap(scaled_pixmap)


def main():
    """Run the application"""
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = DeblurWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
