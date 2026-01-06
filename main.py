"""
Main entry point for Fashion-MNIST Image Retrieval System
"""

import sys
import warnings
import threading
import tkinter as tk
from retrieval import ImageRetrievalSystem
from gui import ImageRetrievalGUI

warnings.filterwarnings('ignore')

def main():
    print("Starting Fashion-MNIST Image Retrieval System...")
    print("This may take a few minutes on first run (downloading & extracting features)\n")
    
    # Create system with smaller dataset for faster demo
    system = ImageRetrievalSystem(dataset_limit=1000)
    
    # Initialize in separate thread to show progress
    def init_system():
        system.initialize()
        print("\nSystem ready! Launching GUI...")
        
    init_thread = threading.Thread(target=init_system)
    init_thread.start()
    init_thread.join()  # Wait for initialization
    
    # Create GUI
    root = tk.Tk()
    app = ImageRetrievalGUI(root, system)
    
    print("Application launched successfully!")
    print("\nHow to use:")
    print("  1. Text Search: Enter keywords like 'dress', 'shoes', 'trouser'")
    print("  2. Upload Image: Select any color/grayscale image (auto-resized)")
    print("  3. Mark relevant (✓) / irrelevant (✗) images")
    print("  4. Click 'Refine Search' to apply feedback")
    
    root.mainloop()


if __name__ == "__main__":
    main()
