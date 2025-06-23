# EEG Seizure Detection - Multi-Channel EDA and Analysis

This project explores EEG signal data from the Siena Scalp EEG Database (multi-channel) to analyze seizure vs non-seizure brain activity using Exploratory Data Analysis (EDA) and advanced preprocessing. The analysis supports robust feature extraction and model development for seizure prediction.

---

## Dataset
- **Source:** Siena Scalp EEG Database ([PhysioNet link](https://www.physionet.org/content/siena-scalp-eeg/1.0.0/))
- **Sampling Rate:** 256 Hz (varies by recording, see dataset documentation)
- **Channels:** Multi-channel scalp EEG
- **Files:**
  - `.edf` EEG recordings (e.g., PN00-1.edf, PN00-2.edf)
  - Seizure event annotations (`Seizures-list-PN00.txt`)

---

## Analysis Overview

### Preprocessing & Feature Extraction:
- Data loaded from raw EDF files
- Channel selection and signal alignment
- Seizure event segmentation
- Features extracted per segment:
  - **Statistical:** Mean, Std, Skew, Kurtosis
  - **Signal:** Line Length, Hjorth Parameters
  - **Frequency:** Bandpower (delta, theta, alpha, beta, gamma)

### EDA:
- Signal distribution across seizure and non-seizure segments
- Visualization of EEG traces for different channels
- Class-wise feature statistics and boxplots
- Seizure event duration and distribution analysis

### Spectrograms (if performed):
- Time–Frequency domain analysis for selected channels
- Computed using `scipy.signal.spectrogram`
- Visual comparison between seizure and non-seizure states

---

## Citation
If you use this dataset, please cite the following:

> Lodato, R., Romagnoli, G., Varanini, M., Tassi, L., & Avanzini, G. (2020). Siena Scalp EEG Database. PhysioNet. https://doi.org/10.13026/8esn-sz67

---

## Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- mne (for EDF file handling)

**Install with:**
```sh
pip install pandas numpy matplotlib seaborn scipy mne
```

---

## Folder Contents
- `EDA_Multichannel2.ipynb`: Jupyter notebook with EDA and feature extraction
- `siena data/`: Raw EEG data and annotations (not tracked in git)
- `X_PN00-1.npy`, `y_PN00-1.npy`: Preprocessed data arrays
- `.gitignore`: Excludes large data files from version control

---

## Contact
For questions or collaboration, please open an issue or contact the project contributors via GitHub. 