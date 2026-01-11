# Fashion Product Image (FPI) Retrieval System with Relevance Feedback

A complete image retrieval system built with Python, featuring CNN-based feature extraction and Rocchio relevance feedback algorithm for iterative search refinement.

## 🎯 Features

- **FPI Dataset**: Real-world fashion products from Kaggle with multiple categories (Shirts, Dresses, Shoes, Jeans, etc.)
- **CNN Feature Extraction**: Pre-trained ResNet50 for deep feature extraction
- **Multiple Query Types**:
  - Text-based search (e.g., "dress", "shoes")
  - Image-based search (from dataset)
  - Upload custom images ()
- **Relevance Feedback**: Rocchio algorithm for iterative search refinement
- **Interactive GUI**: Built with Streamlit for intuitive interaction

## 📁 Project Structure

```
fashion_retrieval/
├── dataset.py          # Fashion Product Image dataset loader
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

### Dataset Setup

1. Download **Fashion Product Images (Small)** dataset from Kaggle:
   - Dataset: [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
   
2. Extract to project directory:
```
   fashion_retrieval/
   └── data/
       ├── images/          # Contains .jpg files
       └── styles.csv       # Metadata file
```

## 💻 Usage

### Step 1: Pre-process Dataset (Extract Features)

Run this **once** to extract and cache features:
```bash
python main.py
```

This will:
1. Load Fashion Product dataset (1000 images by default)
2. Extract ResNet50 features for all images (~5-10 minutes)
3. Cache features to `features_cache_1000.npy`

**Note:** You only need to run this once. Features are cached for future use.

### Step 2: Launch Web Interface
```bash
streamlit run gui.py
```

The web interface will open in your browser at `http://localhost:8501`

### Using the Web Interface

1. **Text Search**:
   - Enter product category in sidebar (e.g., "Shirts", "Shoes", "Dresses")
   - Click "Search by Text"
   - View top 20 similar results

2. **Image Search**:
   - Upload any image (color or grayscale, any size)
   - System automatically resizes and processes it
   - Click "Search by Image"

3. **Relevance Feedback**:
   - Select feedback for each result using radio buttons:
     - **None**: No feedback
     - **Relevant ✓**: Mark as relevant
     - **Irrelevant ✗**: Mark as irrelevant
   - Click **"🔄 Refine Results"** to apply Rocchio feedback
   - System re-ranks results based on your selections
   - Repeat process iteratively for better results

## 🔬 Technical Details

### Feature Extraction

- Uses pre-trained ResNet50 (ImageNet weights)
- Extracts **2048-dimensional** feature vectors
- Images resized to 224×224 for ResNet50 input
- L2 normalization applied to all features

### Similarity Metric

- **Cosine similarity** (dot product of normalized vectors)
- Range: [-1, 1], higher scores = more similar
- Optimal for L2-normalized deep learning features

### Rocchio Algorithm

Classic Rocchio relevance feedback formula:

$$
\mathbf{q}_{new} = \alpha \mathbf{q}_{0} + \frac{\beta}{|\mathcal{D}_r|} \sum_{\mathbf{d}_i \in \mathcal{D}_r} \mathbf{d}_i - \frac{\gamma}{|\mathcal{D}_n|} \sum_{\mathbf{d}_j \in \mathcal{D}_n} \mathbf{d}_j
$$

**Where:**

- $\mathbf{q}_{new}$: The modified query vector.
- $\mathbf{q}_{0}$: The original query vector.
- $\mathcal{D}_r$: Set of relevant documents.
- $\mathcal{D}_n$: Set of non-relevant documents.
- $\alpha, \beta, \gamma$: Weights for each component.

Default popular parameters:

- $\alpha$ = 1.0 (original query weight)
- $\beta$ = 0.75 (relevant documents weight)
- $\gamma$ = 0.25 (irrelevant documents weight)

## 📊 Dataset Information

**Fashion Product Images (Small)**:
- Source: Kaggle dataset
- ~44,000+ product images (RGB, various sizes)
- Multiple fashion categories:
  - Shirts, T-shirts, Casual Shoes, Watches
  - Jeans, Dresses, Heels, Handbags
  - Jackets, Tops, Sandals, Flats
  - And many more...
- Each image has metadata: ID, gender, category, color, season, etc.

**System Configuration:**
- Default: 1000 images for faster processing
- Configurable via `dataset_limit` parameter
- Features cached for instant subsequent loads

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

- **Feature extraction**: ~5-10 minutes for 1000 color images (first run only)
- **Search query**: <1 second
- **Relevance feedback**: <1 second
- **Memory usage**: ~1-3 GB (depends on dataset size)
- **Feature cache**: ~8MB per 1000 images (2048-dim features)

## 🐛 Troubleshooting

**Issue**: "FileNotFoundError: styles.csv not found"
- **Solution**: Download dataset from Kaggle and extract to `data/` folder

**Issue**: "No valid images found!"
- **Solution**: Check that `data/images/` contains .jpg files

**Issue**: Slow feature extraction
- **Solution 1**: Reduce dataset size in `main.py`
- **Solution 2**: Install GPU version: `pip install tensorflow-gpu` (requires CUDA)
- **Solution 3**: Increase `batch_size` (requires more RAM)

## 📝 License

This project is for educational purposes. Fashion-MNIST dataset is under MIT License.
**Dataset License:** Fashion Product Images dataset follows Kaggle's terms of use.

## 🙏 Acknowledgments

- **Dataset**: Fashion Product Images (Small) from Kaggle
- **Model**: ResNet50 pre-trained on ImageNet
- **Framework**: TensorFlow/Keras, Streamlit
