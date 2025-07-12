# M-CET: An ensemble learning-based model for porcine enteric virus identification

## Description
M-CET is an advanced deep learning framework that combines multiple attention mechanisms (CBAM, ECA, and Triplet Attention) with stacking ensemble learning techniques for accurate identification of porcine enteric viruses from protein sequences. The system extracts comprehensive sequence features and employs rigorous evaluation methods to achieve state-of-the-art performance.

Key features:
- Feature extraction using AAC, DPC, and PseAAC methods
- Three attention-based neural network architectures (CBAM, ECA, and Triplet Attention)
- Stacking ensemble methods
- Comprehensive evaluation with 10-fold cross-validation

## Dataset Information
<img width="2000" height="1518" alt="Fig2" src="https://github.com/user-attachments/assets/677de531-e72c-4cab-a270-5c66d22e6e70" />

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

###Methodology
