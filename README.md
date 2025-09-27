# ECG Lead Misplacement Detection - Machine Learning Application

## Overview
This repository contains a modular implementation for detecting ECG lead misplacement using synthetic data, feature extraction, and machine learning classification.

## Structure

- `ecg_simulation/`: Generate synthetic ECG signals and simulate lead misplacement.
- `feature_extraction/`: Extract diagnostic features from ECG signals.
- `models/`: Machine learning models for classification.
- `utils/`: Data preprocessing utilities.
- `main.py`: Main script for dataset generation, preprocessing, training, and evaluation.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

## Dataset

The code generates synthetic data; you can replace it with real ECG datasets by modifying the simulation module.

## License

MIT (or your choice)
