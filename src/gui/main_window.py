# gui_app.py
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QThread, Signal
import numpy as np
import cv2
from src.inference.predictor import DeblurPredictor

class DeblurGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize predictor ONCE when GUI starts
        self.predictor = DeblurPredictor('checkpoints/best_model.pth')
        
        self.setup_ui()
    
    def setup_ui(self):
        # Your UI elements
        self.btn_load = QPushButton("Load Blurry Image", self)
        self.btn_deblur = QPushButton("Deblur Image", self)
        self.label_before = QLabel("Before", self)
        self.label_after = QLabel("After", self)
        
        self.btn_load.clicked.connect(self.load_image)
        self.btn_deblur.clicked.connect(self.deblur_image)
    
    def load_image(self):
        # File dialog to select image
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Blurry Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        
        if file_path:
            self.blurred_image = cv2.imread(file_path)
            self.blurred_rgb = cv2.cvtColor(self.blurred_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.blurred_rgb, self.label_before)
    
    def deblur_image(self):
        if hasattr(self, 'blurred_rgb'):
            # USE PREDICTOR HERE - Simple one-liner!
            deblurred_rgb = self.predictor.predict(self.blurred_rgb)
            
            self.display_image(deblurred_rgb, self.label_after)
    
    def display_image(self, image_rgb, label):
        # Convert numpy array to QPixmap for display
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_image))

if __name__ == '__main__':
    app = QApplication([])
    window = DeblurGUI()
    window.show()
    app.exec()
