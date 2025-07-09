# EEG Seizure Prediction with RNN

## Project Overview
This project implements a Recurrent Neural Network (RNN) for binary classification of EEG segments as preictal (pre-seizure) or interictal (non-seizure). The data consists of multi-channel scalp EEG recordings, segmented and preprocessed for deep learning.

## Data Source
- EEG data in EDF format, stored in the `siena scalp eeg data` folder.
- Each file contains labeled seizure and baseline intervals.

## Preprocessing Steps
- **Bandpass filtering** (1–50 Hz)
- **Resampling** to 256 Hz
- **Z-score normalization** per channel, per segment
- **Segmentation** into 30-second windows
- **Class balance**: ~900 interictal, ~730 preictal windows
- **Sequence reduction**: Only the last 1024 time steps per segment used for RNN input

## Model Architecture
- **Type:** Bidirectional RNN (`nn.RNN` in PyTorch)
- **Input:** (batch, 1024, 30) — 30 channels, 1024 time steps
- **Hidden size:** 128
- **Layers:** 3 RNN layers
- **Dropout:** 0.5
- **BatchNorm:** 1D after RNN output
- **Fully connected output:** 1 neuron (binary classification)
- **Bidirectional:** Yes
- **Loss:** Weighted BCEWithLogitsLoss (class weights: [0.91, 1.12])
- **Optimizer:** AdamW

### Approximate Number of Parameters
- RNN: ~132,608 parameters (for 3 layers, 128 hidden, 30 input, bidirectional)
- BatchNorm: 256
- FC: 257
- **Total:** ~133,121 parameters

## Training & Results
- **Early stopping:** after 7 epochs
- **Best validation accuracy:** 0.55

### Confusion Matrix (Test Set)

|        | Pred 0 | Pred 1 |
|--------|--------|--------|
| True 0 |   180  |   0    |
| True 1 |   147  |   0    |

**Confusion Matrix Plot:**

![Confusion Matrix](confusion_matrix_rnn.png)

*Replace `confusion_matrix_rnn.png` with your actual confusion matrix plot if available.*

- **Classification report:**
  - Interictal: Precision 0.55, Recall 1.00, F1 0.71
  - Preictal: Precision 0.00, Recall 0.00, F1 0.00
  - Accuracy: 0.55

### Example Loss/Accuracy Graph Code
```python
import matplotlib.pyplot as plt
# Suppose you store losses/accuracies in lists: train_losses, val_losses, train_accs, val_accs
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## How to Run
1. Place EEG EDF files in the `siena scalp eeg data` folder.
2. Run the notebook cell by cell (see `model_RNN.ipynb`).
3. Follow the EDA, preprocessing, and model training steps.
4. Evaluate results and confusion matrix.

## Discussion & Limitations
- The RNN model struggled to detect preictal events, predicting only the majority class (interictal).
- Even with class weighting, bidirectionality, and sequence reduction, recall for preictal was 0.
- For EEG, LSTM/GRU or CNN+RNN architectures are recommended for better performance.
- Further feature engineering, data augmentation, or advanced models may be needed.

## Requirements
- Python 3.7+
- PyTorch
- numpy, matplotlib, seaborn, scikit-learn, mne, scipy

## Author / Credits
- Notebook and code: [Your Name]
- Data: Provided EEG EDF files
- Guidance: AI assistant 