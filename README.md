# Speech Signal Analysis and Vowel Classification System

## Overview
This project implements a comprehensive system for acoustic signal analysis and vowel sound classification using Python. It combines traditional speech processing techniques with machine learning to analyze, visualize, and classify vowel sounds from recorded audio samples.

## Key Features
- **Audio Recording & Preprocessing:** Self-spoken audio samples are recorded and processed to extract amplitude, pitch, frequency, and RMS energy using Librosa and NumPy.
- **Feature Extraction:** Zero-Crossing Rate, Short-Time Energy, and MFCC features are computed and compared across diverse speech tones and speakers.
- **Vowel Classification Pipeline:** Utilizes Linear Predictive Coding (LPC) for formant extraction and trains multiple classifiers (KNN, Gaussian Mixture Models, threshold-based) to distinguish vowel sounds.
- **Model Evaluation:** Classifier performance is evaluated using ROC curves, F1-scores, and confusion matrices, with GMM achieving up to 58% accuracy. Detailed metrics and visualizations are provided to study vowel-space separability.
- **Data Organization:** Includes structured directories for male/female vowel samples, historical audio, and analysis outputs (CSV, PNG).

## Technologies Used
- Python (Librosa, NumPy, scikit-learn, matplotlib)
- Jupyter Notebook for interactive analysis

## Results
- Achieved 58% accuracy in vowel classification using Gaussian Mixture Models.
- Generated comprehensive visualizations and reports for feature comparison and classifier performance.

## Repository Structure
- `question_1.py`, `question_2.py`, `question_3.py`, `question_3.ipynb`: Main scripts and notebook for analysis and classification.
- `vowel_samples/`, `historical_audio/`: Audio datasets for training and evaluation.
- `question_1/`, `question_2/`, `question_3/`: Output directories containing analysis results, metrics, and visualizations.

## How to Run
1. Install required Python packages (see code for dependencies).
2. Place audio samples in the appropriate directories.
3. Run the scripts or notebook to perform analysis and classification.

## Author
Academic coursework, Semester Six, November 2025.

---
For more details, see the code and output files in this repository.