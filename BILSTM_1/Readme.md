BiLSTM Seizure Detection Model (PN00)
A lightweight, bidirectional LSTM-based neural network for classifying preictal vs. interictal EEG patterns.

🧩 Model Architecture
Type: Bidirectional LSTM (BiLSTM)

Input: 5-second EEG clips across 34 electrode channels

Architecture:

python

Sequential([
    Bidirectional(LSTM(64)),       # 64 forward + 64 backward = 128 units
    Dense(1, activation='sigmoid') # Output: seizure probability
])
Total parameters: 239,881

⚙️ Training Setup
Optimizer: Adam, learning rate = 0.001

Loss function: Binary cross-entropy

Batch size: 32

Epochs: 50 (early stopping triggered at epoch 20)

Class distribution:

Preictal: 1006 samples

Interictal: 923 samples

📊 Performance Metrics (stopped at epoch 20)

| Metric        | Score  | Interpretation                         |
| ------------- | ------ | -------------------------------------- |
|   Accuracy    | 67.88% | Correct predictions overall            |
|   Precision   | 71.91% | True positive rate                     |
|   Recall      | 63.37% | Percentage of preictal seizures caught |
|   AUC-ROC     | 76.61% | Capability to distinguish classes      |

Confusion Matrix:

| Actual \ Predicted  | Normal | Preictal |
| ------------------- | ------ | -------- |
| Normal (Interictal) | 67     | 25       |
| Preictal            | 37     | 64       |


False negatives in preictal = 37 / (37 + 64) = 36.63%

🧠 Key Observations
Contextual Sequence Modeling: Captures both past and future EEG context using bidirectional LSTM.
Moderate Accuracy: Good performance but with room for improvement—false negatives remain significant.
Comparison Point: Previous plain LSTM reached ∼94.4% accuracy—PN00 underperforms, possibly due to overfitting or channel noise.

🛠 Improvements Needed

Channel Optimization:
Select top 18 most informative EEG channels.
Reduce input size, improve signal-to-noise ratio.

Class Imbalance Handling:
Use class weights or weighted loss to prioritize recall on preictal samples.

Regularization:
Add dropout (e.g., 0.5) and possibly recurrent dropout within LSTM layers.

Use L2 regularization or batch normalization.
Tune Sequence Length:

Experiment ith shorter (3-sec) or longer EEG input windows to balance temporal context.

Advanced Architectures:
Try stacked LSTMs, CNN-LSTM hybrids, attention modules, or Transformer-based models.

🚀 Usage
Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Train the Model
bash
Copy
Edit
python src/train.py \
  --data_dir ./data \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001 \
  --early_stopping
Evaluate Model
bash
Copy
Edit
python src/evaluate.py \
  --model_path ./models/pn00_best.h5 \
  --data_dir ./data
Results include accuracy, precision, recall, AUC-ROC, and confusion matrix.

🧬 Reproducibility
Random seed set to 42 for consistent results.

Ensure identical data splits by fixing train/validation/test indices.

 References
Core dataset: Siena Scalp EEG Database 1.0.0.

