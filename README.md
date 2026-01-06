# Fashion-MNIST Image Retrieval System with Relevance Feedback

A complete image retrieval system built with Python, featuring CNN-based feature extraction and Rocchio relevance feedback algorithm for iterative search refinement.

## 🎯 Features

- **Fashion-MNIST Dataset**: 10 categories of fashion items (T-shirts, Trousers, Dresses, etc.)
- **CNN Feature Extraction**: Pre-trained ResNet50 for deep feature extraction
- **Multiple Query Types**:
  - Text-based search (e.g., "dress", "shoes")
  - Image-based search (from dataset)
  - Upload custom images (28x28 - grayscale, 28x28x3 - RGB)
- **Relevance Feedback**: Rocchio algorithm for iterative search refinement
- **Interactive GUI**: Built with Tkinter for easy interaction

## 📁 Project Structure

```
fashion_retrieval/
├── dataset.py          # Fashion-MNIST dataset loader
├── features.py         # CNN feature extraction (ResNet50)
├── similarity.py       # Cosine similarity metrics
├── feedback.py         # Rocchio relevance feedback algorithm
├── retrieval.py        # Main retrieval engine
├── gui.py              # Tkinter GUI interface
├── main.py             # Entry point
├── requirements.txt    # Python dependencies
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Run the Application

```bash
python main.py
```

On first run, the system will:

1. Download Fashion-MNIST dataset (~30MB)
2. Load pre-trained ResNet50 model
3. Extract features from all images (this may take 2-5 minutes)
4. Cache features for future use

### Using the GUI

1. **Text Search**:

   - Enter keywords like "dress", "shoes", "jacket"
   - Click "Search by Text"

2. **Random Image Search**:

   - Click "Search by Random Image" to query with a random image from dataset

3. **Upload Image**:

   - Click "Upload Image (28x28)" to search with your own 28x28 grayscale image

4. **Relevance Feedback**:
   - Mark relevant images with ✓ (green button)
   - Mark irrelevant images with ✗ (red button)
   - Click "🔄 Refine Search" to apply feedback
   - System uses Rocchio algorithm to update query and re-rank results

## 🔬 Technical Details

### Feature Extraction

- Uses pre-trained ResNet50 (ImageNet weights)
- Extracts 2048-dimensional feature vectors
- L2 normalization applied to all features

### Similarity Metric

- Cosine similarity (dot product of normalized vectors)
- Higher scores indicate greater similarity

### Rocchio Algorithm

Classic relevance feedback formula:

```
Q_new = α × Q_original + β × (1/|Dr|) × Σ(relevant) - γ × (1/|Dn|) × Σ(irrelevant)
```

Default popular parameters:

- α = 1.0 (original query weight)
- β = 0.75 (relevant documents weight)
- γ = 0.25 (irrelevant documents weight)

## 📊 Dataset Information

**Fashion-MNIST**:

- 70,000 grayscale images (28×28 pixels)
- 10 categories:
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

Default: System uses 1000 images for faster demo (configurable)

## 🎓 Academic Background

This system implements techniques from:

- **Content-Based Image Retrieval (CBIR)**
- **Deep Learning for Computer Vision**
- **Information Retrieval with Relevance Feedback**

Key papers/concepts:

- Rocchio, J.J. (1971), "Relevance feedback in information retrieval"
- He et al. (2016), "Deep Residual Learning for Image Recognition" (ResNet)
- Xiao et al. (2017), "Fashion-MNIST: a Novel Image Dataset"
- Lee et al. (CVPR 2021), “CoSMo: Content-Style Modulation for Image Retrieval with Text Feedback”
- Sonam et al. (CVPR 2022), “FashionVLP: Vision Language Transformer for Fashion Retrieval with Feedback”
- Alberto et al. (MM Asia 2021), “Conditoned Image Retrieval for Fashion using Contras ve Learning and CLIP-based Features”, MM Asia 2021
- Tian et al. (WACV 2023), “Fashion Image Retrieval with Text Feedback by Additive Attention Compositional Learning”
- Yuan et al. (SIGIR 2021), “Conversational Fashion Image Retrieval via Multiturn Natural Language Feedback”

## 🛠️ Customization

### Change Dataset Size

Edit in `main.py`:

```python
system = ImageRetrievalSystem(dataset_limit=1000)  # Change to desired size
```

### Adjust Rocchio Parameters

Edit in `retrieval.py`:

```python
self.rocchio = RocchioFeedback(alpha=1.0, beta=0.8, gamma=0.3)
```

## 📈 Performance

- Feature extraction: ~2-5 minutes for 500 images (first run only)
- Search query: <1 second
- Relevance feedback: <1 second
- Memory usage: ~1-2 GB (depends on dataset size)

## 🐛 Troubleshooting

**Issue**: Slow feature extraction

- Reduce dataset size
- Or install GPU version: `pip install tensorflow-gpu` (requires CUDA)

**Issue**: GUI not displaying

- Ensure Tkinter is installed (usually comes with Python)
- On Linux: `sudo apt-get install python3-tk`

## 📝 License

This project is for educational purposes. Fashion-MNIST dataset is under MIT License.
