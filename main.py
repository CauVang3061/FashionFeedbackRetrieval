"""
Main entry point for Fashion-MNIST Image Retrieval System
Run this file to start the application
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Check Python version
if sys.version_info < (3, 7):
    print("Error: Python 3.7 or higher is required")
    sys.exit(1)

# Check required packages
required_packages = {
    'numpy': 'numpy',
    'PIL': 'Pillow',
    'tensorflow': 'tensorflow',
    'scipy': 'scipy'
}

missing_packages = []
for module, package in required_packages.items():
    try:
        __import__(module)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("Error: Missing required packages. Please install them using:")
    print(f"pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Import GUI
from gui import main

if __name__ == "__main__":
    print("=" * 70)
    print("Fashion-MNIST Image Retrieval System with Relevance Feedback")
    print("=" * 70)
    print("\nFeatures:")
    print("- Fashion-MNIST dataset (10 clothing categories)")
    print("- CNN-based feature extraction (ResNet50)")
    print("- Text and image-based queries")
    print("- Rocchio relevance feedback algorithm")
    print("- Interactive GUI with Tkinter")
    print("\n" + "=" * 70 + "\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
