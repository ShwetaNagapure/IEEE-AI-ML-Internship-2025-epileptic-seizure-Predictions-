
# 🧠 Seizure Prediction using LSTM

This project implements a deep learning pipeline to predict epileptic seizures **30 minutes before onset** using EEG data stored in `.edf` format.

---

## 📊 Objective

To detect seizure onset **30 minutes prior**, using multi-channel EEG data processed and trained on an LSTM-based neural network.

---

## ✅ Model Performance

- **Best Validation Accuracy:** `96.88%`

### 📉 Confusion Matrix

```
[[733  25]
 [ 23 730]]
```

### 📋 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Interictal) | 0.9696 | 0.9670 | 0.9683 | 758 |
| 1 (Preictal)   | 0.9669 | 0.9695 | 0.9682 | 753 |

- **Overall Accuracy:** `0.9682`
- **Macro Avg F1-Score:** `0.9682`
- **Weighted Avg F1-Score:** `0.9682`

---

## ⚙️ Preprocessing Pipeline

The raw `.edf` EEG files are:

1. **Bandpass filtered** (0.5 – 40 Hz)
2. **Resampled** to 256 Hz (from varying native sampling rates)
3. **Segmented** into 30-second windows
4. **Labeled** as:
   - `1`: Preictal (within 30 minutes before seizure)
   - `0`: Interictal (random 30-minute baseline)

---

## 🧠 Model Architecture (LSTM)

```python
class SeizureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(SeizureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.sigmoid(self.fc(out))
```

---

## 🔍 Evaluation Strategy

- **Stratified train-validation split**
- **Binary Cross Entropy Loss**
- **Adam Optimizer**
- **Accuracy, Confusion Matrix, Classification Report** for performance metrics

---

## 📈 Future Improvements

- Early stopping and learning rate scheduling
- Ensemble models (LSTM)
- Attention mechanisms
- Cross-patient generalization

---

## 🧪 Requirements

- Python 
- PyTorch
- MNE
- NumPy, Pandas
- Matplotlib, Seaborn, scikit-learn
