🧠 Seizure Prediction using BiLSTM
This project implements a deep learning pipeline for predicting epileptic seizures using EEG data in .edf format. The final model leverages a Bidirectional LSTM (BiLSTM) architecture for robust seizure detection, trained on multi-channel EEG data from the Siena Scalp EEG Database.

📊 Objective
To predict the onset of epileptic seizures using 5-second EEG segments from 34 channels, enabling early detection and proactive intervention.

✅ Model Performance
Best Validation Accuracy: 83.74%
AUC-ROC: 90.78%
Precision: 89.39%
Recall: 87.97%

📉 Confusion Matrix (Example)

[[TN: Higher   FP: Reduced]
 [FN: Reduced  TP: Higher]]

📋 Classification Report (Summarized)
| Class      | Precision | Recall | F1-Score |
| ---------- | --------- | ------ | -------- |
| Interictal | \~89%     | \~88%  | \~88.5%  |
| Preictal   | \~89%     | \~88%  | \~88.5%  |


⚙️ Preprocessing Pipeline
Bandpass filtered (0.5 – 40 Hz)

Resampled to 256 Hz

Segmented into 5-second windows (multichannel)

Z-score normalized

Labeled as:

1: Preictal (30-min before seizure)

0: Interictal (baseline)

🧠 Model Architecture (BiLSTM)

model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))
])

Optimizer: Adam

Loss: Binary Crossentropy with class weights

Batch Size: 8

Early stopping: patience = 5

🔍 Evaluation Strategy
Class weighting

Early stopping

Confusion matrix and AUC-ROC analysis

Chunk-level prediction + CSV export

📈 Future Improvements
Visualize AUC & PR curves per patient

Improve cross-patient generalization

Try transformer-BiLSTM hybrid

Deploy for real-time alerts

🧪 Requirements
Python

TensorFlow/Keras

NumPy, Pandas

MNE (for EEG files)

Matplotlib, Seaborn

scikit-learn

📂 Output Format
Each patient's prediction saved as:
/content/drive/MyDrive/BILSTM_MODEL/predictions_PN00-X.csv

Column	Description
chunk_index	Index of EEG chunk
time_seconds	Time in seconds
seizure_probability	Probability from model
seizure_predicted	Binary prediction (0 or 1)

👩‍⚕️ Patients Evaluated
PN00-1.edf

PN00-2.edf

PN00-3.edf

PN00-4.edf

PN00-5.edf
