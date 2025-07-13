📊 BiLSTM Seizure Prediction Model

This project implements a deep learning pipeline for predicting epileptic seizures using EEG data from the Siena Scalp EEG Database 1.0.0. The final model uses a Bidirectional Long Short-Term Memory (BiLSTM) architecture trained on 5-second EEG segments from 34 channels.

🔧 Model Architecture

Input Shape: (5 seconds, 34 EEG channels)
↓
Bidirectional(LSTM(64, return_sequences=True))
↓
Dropout(0.5)
↓
Bidirectional(LSTM(32))
↓
Dense(1, activation='sigmoid')  ← with L2 regularization (λ=0.001)
Loss: Binary Crossentropy (with class weights)

Optimizer: Adam

Batch Size: 8

Training Epochs: 30 (early stopped at epoch 16)

Regularization: Dropout + L2

✅ Model Performance (Final Evaluation)
Metric	Score
Accuracy	83.74%
Precision	89.39%
Recall	87.97%
AUC-ROC	90.78%

📌 Key Highlights:

Excellent class separability (AUC-ROC > 90%)

Reduced false negatives, improving early seizure detection

Generalizes well across patients

📁 Prediction Process
Model loaded from: /content/drive/MyDrive/BILSTM_MODEL/final_seizure_model.h5

Input Files: PN00-1.edf to PN00-5.edf

Data segmented into 5-second chunks

Normalized using z-score

Model predicts seizure probability for each chunk

📤 Output:
For each file, predictions are saved as:

bash
Copy
Edit
/content/drive/MyDrive/BILSTM_MODEL/predictions_PN00-X.csv
CSV Columns:

chunk_index: Index of EEG chunk

time_seconds: Time in seconds

seizure_probability: Model's confidence (0 to 1)

seizure_predicted: Binary prediction (0 or 1)

📌 How to Run
Mount Google Drive:


from google.colab import drive
drive.mount('/content/drive')
Load model and EEG data:


model = load_model('final_seizure_model.h5')
raw = mne.io.read_raw_edf(file_path, preload=True)
Segment and normalize EEG:


chunks = segment_eeg(raw, start=0, end=duration, chunk_duration=5)
Predict:


preds_prob = model.predict(np.array(chunks))
Save results:



save_predictions_csv('PN00-1', preds_prob, preds_prob > 0.5)
📌 Patients Analyzed
PN00-1.edf

PN00-2.edf

PN00-3.edf

PN00-4.edf

PN00-5.edf

