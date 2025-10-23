# app.py
"""
Launch Image Deblurring GUI Application

Usage:
    python app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import QApplication
from src.gui.main_window import DeblurWindow


def main():
    """Launch the application"""
    
    print("=" * 60)
    print("Image Deblurring Application")
    print("=" * 60)
    print("\nStarting GUI...")
    
    app = QApplication(sys.argv)
    
    # Set application info
    app.setApplicationName("Image Deblurring")
    app.setOrganizationName("Week 4 Project")
    
    # Create and show main window
    window = DeblurWindow()
    window.show()
    
    print("✓ GUI started successfully")
    print("\nClose the window to exit.")
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
