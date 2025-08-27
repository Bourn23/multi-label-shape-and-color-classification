# Shapes and Colors Prediction Challenge

A comprehensive computer vision solution for multi-label shape and color classification in synthetic images using hybrid OpenCV-CNN architectures.

## 🎯 Challenge Overview

This project addresses the **Shapes and Colors Prediction Challenge**, which involves predicting geometric shapes (circles, squares, triangles) and their corresponding colors (red, blue, green) in synthetic images. Each image contains randomly placed, rotated geometric shapes without overlaps, presenting a multi-label image recognition task.

### Problem Statement
- **Input**: Synthetic images with geometric shapes
- **Output**: List of (shape, color) tuples for each detected object
- **Evaluation**: Jaccard similarity between predicted and ground truth sets
- **Shapes**: Circle, Square, Triangle
- **Colors**: Red, Blue, Green

## 📊 Key Results

| Model | Exact Match Ratio (EMR) | Jaccard Score | Architecture |
|-------|------------------------|---------------|--------------|
| OpenCV Baseline | 68.30% | 66.19% | HSV + Contours |
| Simple CNN | 88.00% | 81.51% | Basic ConvNet |
| Hybrid CNN | 98.00% | 98.68% | OpenCV + CNN |
| **ResNet Hybrid** | **99.50%** | **99.22%** | ResNet + OpenCV |

## 🔬 Methodology

### 1. Exploratory Data Analysis
- **Dataset Balance**: Analyzed distribution of shapes, colors, and object counts
- **Data Characteristics**: Identified noise patterns, overlapping objects, and edge cases
- **Class Imbalance**: Images with 5 objects are significantly underrepresented

### 2. Baseline Model (OpenCV)
- **Color Detection**: HSV color space thresholding
- **Shape Classification**: Contour analysis with vertex counting
- **Performance**: 66.19% Jaccard score
- **Issues**: Struggles with overlapping objects and complex scenes

### 3. CNN Models
- **Simple CNN**: Basic convolutional architecture with multi-label classification
- **Hybrid Approach**: OpenCV for color masking + CNN for shape classification
- **ResNet Enhancement**: Residual connections for better feature learning

### 4. Data Preprocessing Innovations
- **Color Masking**: HSV-based color isolation
- **Noise Reduction**: Morphological operations for clean segmentation
- **Multi-channel Input**: 4-channel input (RGB + mask) for enhanced feature extraction

## 🛠️ Technical Implementation

### Architecture Components

1. **Color Detection Module**
   ```python
   HSV_RANGES = {
       'red': [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
       'green': [((40, 100, 100), (80, 255, 255))],
       'blue': [((90, 100, 100), (130, 255, 255))]
   }
   ```

2. **ResNet-based Shape Classifier**
   - 4-channel input (RGB + binary mask)
   - Residual blocks for deep feature learning
   - Multi-label output with BCEWithLogitsLoss

3. **Hybrid Pipeline**
   - Color-specific masking using OpenCV
   - Shape classification using trained CNN
   - Ensemble predictions across color channels

### Model Architecture
```
Input (4-channel: RGB + Mask) → 
Initial Conv + BN + ReLU → 
Residual Blocks (32→64→128→256) → 
Global Average Pooling → 
Classifier (Linear + Dropout)
```

## 🔍 Key Insights

### Model Performance Analysis

1. **OpenCV Limitations**:
   - Severe under-prediction on complex images (3+ objects)
   - Missed 1,585 circles out of 2,880 total
   - All 465 false positives were misclassified as squares

2. **CNN Improvements**:
   - Reduced false negatives from 2,072 to 129 objects
   - Dramatically decreased hallucinations from 465 to 32 objects
   - Better handling of overlapping and noisy objects

3. **Hybrid Success Factors**:
   - Color masking isolates objects effectively
   - CNN focuses on shape-specific features
   - Residual connections enable deeper feature learning

### Failure Case Analysis
- **Hidden Objects**: Objects of same color overlapping (e.g., red triangle inside red circle)
- **Spurious Correlations**: Model learning spatial arrangements rather than geometric features
- **Edge Cases**: Subtle features competing with dominant patterns

## 📁 Project Structure

```
├── 00.BRF_report.ipynb          # Main analysis notebook
├── Archive/                     # Dataset files
│   ├── train.csv
│   ├── test.csv
│   ├── train_dataset/          # Training images
└── └── test_dataset/           # Test images
```

## 🎯 Model Training Strategy

### Cross-Validation
- **Strategy**: Stratified 5-fold cross-validation
- **Stratification**: Based on number of objects per image
- **Reason**: Address class imbalance (especially 5-object images)

### Training Configuration
- **Image Size**: 128×128 pixels
- **Batch Size**: 32
- **Epochs**: 30
- **Optimizer**: AdamW with learning rate 0.001
- **Loss Function**: BCEWithLogitsLoss (multi-label)

### Data Augmentation
- Random horizontal flips
- Random rotations (±15°)
- Random resized crops
- D4 dihedral transformations

## 🏆 Achievements

1. **99.22% Jaccard Score**: Near-perfect object detection and classification
2. **Robust Architecture**: Handles overlapping objects and noise effectively
3. **Comprehensive Analysis**: Deep insights into model failures and improvements
4. **Scalable Pipeline**: Modular design for easy extension and modification

## 🔧 Usage

### Dependencies
```bash
pip install torch torchvision opencv-python pandas numpy matplotlib seaborn pillow albumentations scikit-learn tqdm
```

### Running the Analysis
```bash
jupyter notebook 00.BRF_report.ipynb
```

## 📈 Future Improvements

1. **Object Detection Integration**: YOLO/R-CNN for instance segmentation
2. **Attention Mechanisms**: Focus on relevant image regions
3. **Synthetic Data Augmentation**: Generate more complex training scenarios
4. **Real-world Adaptation**: Transfer learning for natural images

## 🤝 Contributing

This project demonstrates advanced computer vision techniques combining classical image processing with deep learning for robust multi-label classification.

---

*This project was developed as part of a Kaggle competition focused on shape and color recognition in synthetic images. The solution showcases a progression from basic OpenCV methods to sophisticated hybrid architectures achieving near-perfect performance.*


