#  EEG Seizure Prediction using LSTM 

This project focuses on predicting epileptic seizures using EEG data in **EDF format** with a **Long Short-Term Memory (LSTM)** model built using **PyTorch**.

---

##  Dataset

We use EDF recordings containing annotated seizure events. For this prototype, we only use data from patient PN00, extracted from the Siena Scalp EEG Database 1.0.0.

- `PN00-1.edf`
- `PN00-2.edf`
- `PN00-3.edf`
- `PN00-4.edf`
- `PN00-5.edf`

Each file has a known:

- **Seizure start and end time**
- **Baseline (interictal) start time**

The EDF files are located in:

```
data/
├── PN00-1.edf
├── PN00-2.edf
├── ...
```

---

##  Data Preprocessing

###  1. Channel Selection

Only the **first 18 EEG channels** are used for training, consistent across all samples.

```python
channels_to_use = list(range(18))
```

---

###  2. Time Conversion

We convert **wall-clock seizure and baseline timestamps** to seconds relative to recording start time using:

```python
def relative_seconds(timestr, meas_date):
    ...
```

---

###  3. Window Extraction

Each EDF file is segmented into two 30-minute chunks:

- **Preictal**: 30 minutes before seizure
- **Interictal**: 30 minutes from baseline start

Each chunk is further divided into **30-second non-overlapping windows** (e.g., 60 windows per 30 mins).

---

###  4. Data Cleaning

Files that have insufficient duration for preictal windowing (i.e., seizure occurred too early) are skipped with a warning:

```
[SKIP] Preictal start is before beginning of recording.
```

---

###  5. Normalization

Each 30-second EEG window is **z-score normalized (standardized)** channel-wise using:

```python
X = (X - X.mean()) / X.std()
```

This ensures consistent input scaling to the LSTM model.

---

##  Model Architecture

We use a **1-layer LSTM network** with:

- **Input**: 30 seconds of EEG data → shape `(time_steps, channels)`
- **Hidden size**: 64
- **Output**: Binary classification — preictal (1) or interictal (0)

```python
class SeizureLSTM(nn.Module):
    def __init__(self, input_size=18, hidden_size=64):
        ...
```

---

##  Model Training

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
```

- Input shape: `(batch_size, time_steps, 18)`
- Output: Scalar probability ∈ [0, 1]
- Loss: Binary Cross Entropy (BCE)
- Epochs: \~10-20 (user-defined)
- Evaluation done on validation accuracy or manually inspected

---

##  Saving & Loading Model

```python
# Save
torch.save(model.state_dict(), "seizure_lstm_model.pth")

# Load
model.load_state_dict(torch.load("seizure_lstm_model.pth"))
model.eval()
```

---

##  Prediction on New EEG Signal

1. Load new EDF file using `mne`
2. Extract the first 30 seconds from the first 18 channels
3. Normalize the signal
4. Convert to shape `(1, time_steps, 18)` as `torch.Tensor`
5. Pass to the model for prediction

```python
with torch.no_grad():
    output = model(new_tensor).item()
    print("Seizure Probability:", output)
```

---

##  Requirements

- `mne`
- `numpy`
- `torch`
- `matplotlib` (optional)

Install via:

```bash
pip install mne torch numpy
```

---



