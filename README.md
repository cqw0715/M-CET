# M-CET: An ensemble learning-based model for porcine enteric virus identification

## Description
M-CET is an advanced deep learning framework that combines multiple attention mechanisms (CBAM, ECA, and Triplet Attention) with ensemble learning techniques for accurate identification of porcine enteric viruses from protein sequences. The system extracts comprehensive sequence features and employs rigorous evaluation methods to achieve state-of-the-art performance.

Key features:
- Feature extraction using AAC, DPC, and PseAAC methods
- Three attention-based neural network architectures
- Weighted ensemble and stacking ensemble methods
- Comprehensive evaluation with 10-fold cross-validation

## Dataset Information
<img width="2000" height="1518" alt="Fig2" src="https://github.com/user-attachments/assets/677de531-e72c-4cab-a270-5c66d22e6e70" />

The dataset consists of:
- Positive samples: Verified porcine enteric virus sequences
- Negative samples: Non-viral protein sequences
- Sequence characteristics:
  - Length: Variable (processed to fixed feature vectors)
  - Balanced class distribution
- File: `All.csv` containing sequences with labels (1=virus, 0=non-virus)

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
