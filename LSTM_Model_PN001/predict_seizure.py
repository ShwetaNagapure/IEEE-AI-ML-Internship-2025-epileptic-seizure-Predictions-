import mne
import torch
import numpy as np
from datetime import datetime, timedelta

# === User Inputs ===
edf_file = r"C:\Users\Admin\Desktop\Data_Preprocessing\data\PN05-2.edf"
seizure_start_time = "08:45:25"  # Seizure onset (HH:MM:SS)
registration_start_time = "06:46:02"  # Recording start time (HH:MM:SS)
window_offset = 60  # seconds into preictal window to extract (0 to 1800 - 30)

# === Load model ===
from model import SeizureLSTM  # assuming your LSTM class is in model.py

model = SeizureLSTM(input_size=18, hidden_size=64)
model.load_state_dict(torch.load("seizure_lstm_model.pth"))
model.eval()

# === Load EDF File ===
raw = mne.io.read_raw_edf(edf_file, preload=True)
raw.pick_channels(raw.ch_names[:18])  # select first 18 channels

sfreq = int(raw.info['sfreq'])
window_sec = 30
window_samples = window_sec * sfreq

# === Time Alignment ===
def to_seconds(t):
    return (datetime.strptime(t, "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")).total_seconds()

seizure_start_sec = to_seconds(seizure_start_time)
recording_start_sec = to_seconds(registration_start_time)
relative_seizure_time = seizure_start_sec - recording_start_sec

# Define preictal window
preictal_start = max(0, relative_seizure_time - 1800)
window_start = preictal_start + window_offset
window_end = window_start + 30

start_sample = int(window_start * sfreq)
end_sample = int(window_end * sfreq)

# === Extract and Preprocess ===
data, _ = raw[:, start_sample:end_sample]  # shape: (channels, time)
data = data.T  # shape: (time_steps, channels)
data = (data - data.mean()) / data.std()  # z-score normalization
data = data[np.newaxis, ...]  # shape: (1, time_steps, channels)

# === Predict ===
input_tensor = torch.tensor(data, dtype=torch.float32)
with torch.no_grad():
    output = model(input_tensor).item()
    print(f"\nPredicted Seizure Probability: {output:.4f}")
    if output >= 0.5:
        print(" Seizure likely within 30 minutes (Preictal)")
    else:
        print(" Normal brain activity (Interictal)")
