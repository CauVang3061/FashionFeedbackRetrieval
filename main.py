"""
Main Pre-processing Script for Fashion Image Retrieval
Run this first to extract features for the entire dataset.
"""

import warnings
from retrieval import ImageRetrievalSystem

warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("FASHION PRODUCT IMAGES - PRE-PROCESSING")
    print("=" * 60)
    
    # IMPORTANT: This value must match the dataset_limit in gui.py
    # If you change this, also update gui.py line 12
    dataset_size = 1000
    
    system = ImageRetrievalSystem(dataset_limit=dataset_size)
    
    print(f"\n[*] Initializing and extracting features for {dataset_size} images...")
    
    system.initialize(force_recompute=True)
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS!")
    print(f"📁 Features cached at: {system.feature_cache_path}")
    print(f"📊 Dataset size: {len(system.image_ids)} images")
    print(f"📐 Feature dimension: {system.features.shape[1]}")
    print(f"🏷️  Unique categories: {len(system.dataset.get_unique_labels())}")
    print("\n🚀 Run the GUI:")
    print("    streamlit run gui.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
