# M-CET: An ensemble learning-based model for porcine enteric virus identification

## Description
To address the limitations of traditional identification methods for porcine enteric viruses, such as high costs and poor timeliness, this study proposes a novel porcine enteric virus identification model, M-CET. This model integrates three complementary feature extraction strategies based on the MLP deep learning framework: AAC, DPC, and PseAAC, which comprehensively capture global amino acid proportions, local sequence dependencies of adjacent residues, and physicochemical properties incorporating sequence positional information, respectively. Furthermore, the model incorporates three advanced attention modules: CBAM for identifying key regions through synchronous optimization of channel and spatial features, ECA for efficient channel feature recalibration, and Triplet Attention for capturing multidimensional long-range dependencies. The model is fused using a Stacking ensemble learning strategy to generate stable and reliable prediction results. Extensive experiments demonstrate that M-CET exhibits outstanding performance in the binary classification task of porcine enteric virus sequences, achieving an accuracy of 98.48% and an AUC value of 99.72%, significantly outperforming baseline machine learning models, stacked ensemble methods, and voting ensemble methods. Additionally, compared to publicly available virus identification models, M-CET demonstrates superior robustness in identifying porcine enteric virus sequences, with significantly reduced error variance. This highly accurate and efficient tool provides a reliable solution for timely intervention in viral transmission, effectively reducing piglet mortality and associated economic losses, thereby offering a novel approach for epidemic prevention and control.

Key features:
- Feature extraction using AAC, DPC, and PseAAC methods
- Three attention-based neural network architectures (CBAM, ECA, and Triplet Attention)
- Stacking ensemble methods
- Comprehensive evaluation with 10-fold cross-validation

## Dataset Information
### Dataset source
NCBI: https://www.ncbi.nlm.nih.gov/
UniProt: https://www.uniprot.org/
VirusDIP: https://db.cngb.org/virusdip/
<img width="2000" height="1508" alt="Fig2" src="https://github.com/user-attachments/assets/677de531-e72c-4cab-a270-5c66d22e6e70" />

The dataset consists of:
- **Positive samples**: 7,355 porcine enteric viruses (50.0%)
- **Negative samples**: 7,355 porcine enteric bacteria and non-porcine enteric viruses (50.0%)
- File: `All.csv` containing sequences with labels (0=porcine enteric viruses, 1=porcine enteric bacteria and non-porcine enteric viruses)

## Code Information
Main implementation file:
- `M-CET.py`: Contains all core functionality including:
  - Feature extraction pipelines (AAC, DPC, PseAAC)
  - Attention mechanism implementations (CBAM, ECA, Triplet)
  - Model training and evaluation workflows
  - Ensemble methods (weighted average, voting, stacking)
  - Performance metric calculations

Key components:
- Data preprocessing and standardization
- Custom Keras layers for attention mechanisms
- Model training with early stopping and learning rate reduction
- Cross-validation framework
- Result analysis and reporting

## Usage Instructions
1. Install required dependencies (see below)
2. Prepare your dataset as `All.csv` with columns: 'Sequence', 'Label'
3. Run the main script:
   ```bash
   python M-CET.py

## Requirements

### Core Dependencies
| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.8+ | Base programming language |
| TensorFlow | 2.8.0 | Deep learning framework |
| scikit-learn | 1.0.2 | Machine learning utilities |
| pandas | 1.4.2 | Data processing and analysis |
| numpy | 1.22.3 | Numerical computations |
| matplotlib | 3.5.1 | Visualization (for reproducing charts) |

### Optional Dependencies
| Library | Version | Purpose |
|---------|---------|---------|
| seaborn | 0.11.2 | Enhanced visualizations |
| imbalanced-learn | 0.8.1 | Handling class imbalance (if needed) |
| tqdm | 4.62.3 | Progress bars for long operations |

### Installation
1. **Base installation** (required):
   ```bash
   pip install tensorflow==2.8.0 scikit-learn==1.0.2 pandas==1.4.2 numpy==1.22.3 matplotlib==3.5.1

## Methodology
### For the model building part, please read the diagram below
<img width="2250" height="2682" alt="Fig1" src="https://github.com/user-attachments/assets/fb36e2c0-eb03-4377-9269-a230e4f0a428" />
(A) This section presents the comprehensive architecture integrating both machine learning and deep learning approaches, along with the employed feature extraction methodologies.
(B) Focusing on model ensemble and optimization strategies, it comparatively demonstrates three distinct integration approaches—weighted averaging, voting, and stacking—with explicit annotation of binary classification performance metrics to visually contrast different strategies.
(C) It provides a detailed technical elaboration of three core attention modules: CBAM, ECA and Triplet Attention, delineating their respective mechanisms and implementations.

# License & Contribution Guidelines
- ✅ Free to use, modify, and distribute  
- ✅ No liability or warranty provided  
