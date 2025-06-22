# EEG Seizure Detection - EDA and Spectrogram Analysis

This project explores EEG signal data from the **Bonn University Single-Channel Dataset** to analyze seizure vs non-seizure brain activity using Exploratory Data Analysis (EDA) and frequency-domain representations (spectrograms). It also includes advanced preprocessing and feature extraction for seizure prediction.

---

##  Dataset

- **Source**: Andrzejak et al. (2001), *Phys. Rev. E, 64, 061907*
- **Sampling Rate**: 173.61 Hz
- **Sets**:
  - `Z`: Healthy (eyes open)
  - `O`: Healthy (eyes closed)
  - `N`: Interictal (non-seizure, epileptogenic zone)
  - `F`: Interictal (non-seizure, opposite hemisphere)
  - `S`: Seizure

---

## Analysis Overview

###  Preprocessing & Feature Extraction:
- Extracted from each EEG time series (4096 samples)
- Features:
  - Statistical: Mean, Std, Skew, Kurtosis
  - Signal: Line Length, Hjorth Parameters
  - Frequency: Bandpower (delta to gamma bands)

###  EDA:
- Signal distribution across classes
- PCA and t-SNE plots for feature separability
- Boxplots for class-wise feature variation

### Spectrograms:
- Time–Frequency domain analysis
- Computed using `scipy.signal.spectrogram`
- Visual comparison between seizure and non-seizure states

---

## Citation

If you use this dataset, **please cite the following**:

> Andrzejak RG, Lehnertz K, Rieke C, Mormann F, David P, Elger CE (2001)  
> *Indications of nonlinear deterministic and finite dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state*,  
> Physical Review E, 64, 061907

---

##  Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scipy

Install with:

```bash
pip install numpy pandas matplotlib seaborn scipy
