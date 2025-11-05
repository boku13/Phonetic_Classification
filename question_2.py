"""
Question 2: Historical Audio Analysis

This script analyzes historical audio recordings to extract speech features
and compare emotional tones between different speech styles.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import pandas as pd
from scipy.stats import zscore
import requests
import zipfile
import io

# Constants
SAMPLE_RATE = 22050  # Default sample rate for librosa
FRAME_SIZE = 2048    # Frame size for feature extraction
HOP_LENGTH = 512     # Hop length for feature extraction
N_MFCC = 13          # Number of MFCC coefficients to extract

def return_dataset(extract_to='.'):
    """
    Placeholder folder for the dataset.
    """
    dataset_dir = 'historical_audio'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created directory {dataset_dir}. Please place the downloaded audio files there.")
    
    return dataset_dir

def load_audio(file_path):
    """
    Load an audio file using librosa
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def extract_features(y, sr):
    """
    Extract speech features from an audio signal
    """
    if y is None:
        return None
    
    # Ensure the audio is not empty
    if len(y) == 0:
        print("Warning: Empty audio signal")
        return None
    
    # Zero-Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    
    # Short-Time Energy (STE)
    ste = np.sum(np.square(librosa.util.frame(y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)), axis=0)
    
    # Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
    
    # Calculate statistics for each feature
    feature_stats = {
        'zcr_mean': np.mean(zcr),
        'zcr_std': np.std(zcr),
        'zcr_max': np.max(zcr),
        'zcr_min': np.min(zcr),
        
        'ste_mean': np.mean(ste),
        'ste_std': np.std(ste),
        'ste_max': np.max(ste),
        'ste_min': np.min(ste),
    }
    
    # Add MFCC statistics
    for i in range(N_MFCC):
        feature_stats[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
        feature_stats[f'mfcc{i+1}_std'] = np.std(mfccs[i])
    
    return {
        'zcr': zcr,
        'ste': ste,
        'mfccs': mfccs,
        'stats': feature_stats,
        'frames': len(zcr),
        'y': y,  # Store the audio data
        'sr': sr  # Store the sample rate
    }

def plot_features(features, title, filename):
    """
    Plot extracted features
    """
    if features is None:
        print(f"Cannot plot features for {title}: features are None")
        return
    
    plt.figure(figsize=(15, 12))
    
    # Plot Zero-Crossing Rate
    plt.subplot(3, 1, 1)
    plt.plot(features['zcr'])
    plt.title(f'Zero-Crossing Rate - {title}')
    plt.xlabel('Frame')
    plt.ylabel('ZCR')
    plt.grid(True)
    
    # Plot Short-Time Energy
    plt.subplot(3, 1, 2)
    plt.plot(features['ste'])
    plt.title(f'Short-Time Energy - {title}')
    plt.xlabel('Frame')
    plt.ylabel('Energy')
    plt.grid(True)
    
    # Plot MFCCs
    plt.subplot(3, 1, 3)
    librosa.display.specshow(features['mfccs'], x_axis='time')
    plt.colorbar(format='%+2.0f')
    plt.title(f'MFCCs - {title}')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def compare_features(calm_features, energetic_features):
    """
    Compare features between calm and energetic speech
    """
    if calm_features is None or energetic_features is None:
        print("Cannot compare features: one or both feature sets are None")
        return None
    
    # Create a comparison dataframe
    comparison = pd.DataFrame({
        'Calm': pd.Series(calm_features['stats']),
        'Energetic': pd.Series(energetic_features['stats'])
    })
    
    # Calculate percentage difference
    comparison['Difference (%)'] = ((comparison['Energetic'] - comparison['Calm']) / comparison['Calm'] * 100).round(2)
    
    return comparison

def plot_comparison(calm_features, energetic_features, filename):
    """
    Plot comparison between calm and energetic speech features
    """
    if calm_features is None or energetic_features is None:
        print("Cannot plot comparison: one or both feature sets are None")
        return
    
    plt.figure(figsize=(15, 15))
    
    # Plot ZCR comparison
    plt.subplot(3, 1, 1)
    plt.plot(calm_features['zcr'], label='Calm', alpha=0.7)
    plt.plot(energetic_features['zcr'], label='Energetic', alpha=0.7)
    plt.title('Zero-Crossing Rate Comparison')
    plt.xlabel('Frame')
    plt.ylabel('ZCR')
    plt.legend()
    plt.grid(True)
    
    # Plot STE comparison
    plt.subplot(3, 1, 2)
    plt.plot(calm_features['ste'], label='Calm', alpha=0.7)
    plt.plot(energetic_features['ste'], label='Energetic', alpha=0.7)
    plt.title('Short-Time Energy Comparison')
    plt.xlabel('Frame')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    
    # Plot MFCC means comparison
    plt.subplot(3, 1, 3)
    
    calm_mfcc_means = [np.mean(calm_features['mfccs'][i]) for i in range(N_MFCC)]
    energetic_mfcc_means = [np.mean(energetic_features['mfccs'][i]) for i in range(N_MFCC)]
    
    x = np.arange(N_MFCC)
    width = 0.35
    
    plt.bar(x - width/2, calm_mfcc_means, width, label='Calm')
    plt.bar(x + width/2, energetic_mfcc_means, width, label='Energetic')
    
    plt.title('MFCC Coefficients Comparison (Mean Values)')
    plt.xlabel('MFCC Coefficient')
    plt.ylabel('Mean Value')
    plt.xticks(x, [f'MFCC {i+1}' for i in range(N_MFCC)])
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def analyze_emotional_tone(dataset_dir):
    """
    Analyze audio files to determine which ones best represent calm/formal vs passionate/energetic tones
    """
    output_dir = 'question_2'
    
    # Dictionary to store features relevant to emotional tone
    tone_features = {}
    
    for file in os.listdir(dataset_dir):
        if file.endswith(('.wav', '.mp3')):
            file_path = os.path.join(dataset_dir, file)
            print(f"Analyzing tone of {file}...")
            
            # Load audio
            y, sr = load_audio(file_path)
            if y is None:
                continue
                
            # Extract features
            features = extract_features(y, sr)
            if features is None:
                continue
            
            # Calculate metrics relevant to emotional tone
            # 1. Energy variation (high variation suggests more emotional speech)
            energy_variation = features['stats']['ste_std'] / (features['stats']['ste_mean'] + 1e-10)
            
            # 2. Average energy (higher energy often correlates with passionate speech)
            avg_energy = features['stats']['ste_mean']
            
            # 3. ZCR variation (more variation in frequency content suggests more emotional speech)
            zcr_variation = features['stats']['zcr_std'] / (features['stats']['zcr_mean'] + 1e-10)
            
            # 4. MFCC dynamics (calculate variance of first few MFCCs)
            mfcc_dynamics = np.mean([features['stats'][f'mfcc{i+1}_std'] for i in range(3)])
            
            # 5. Calculate speech rate (approximated by zero-crossing rate)
            speech_rate = features['stats']['zcr_mean']
            
            # Combine metrics into an "emotional intensity score"
            # Higher score suggests more passionate/energetic speech
            emotional_score = (
                0.3 * (avg_energy / 100) +  # Normalized energy
                0.2 * energy_variation +    # Energy variation
                0.2 * zcr_variation +       # Frequency variation
                0.2 * mfcc_dynamics +       # Vocal tract dynamics
                0.1 * speech_rate / 100     # Speech rate
            )
            
            tone_features[file] = {
                'emotional_score': emotional_score,
                'avg_energy': avg_energy,
                'energy_variation': energy_variation,
                'zcr_variation': zcr_variation,
                'mfcc_dynamics': mfcc_dynamics,
                'speech_rate': speech_rate
            }
    
    # Sort files by emotional intensity score
    sorted_files = sorted(tone_features.keys(), key=lambda x: tone_features[x]['emotional_score'])
    
    # Create a dataframe for better visualization
    tone_df = pd.DataFrame.from_dict(tone_features, orient='index')
    tone_df.sort_values('emotional_score', inplace=True)
    tone_df.to_csv(f'{output_dir}/emotional_tone_analysis.csv')
    
    # Visualize the emotional scores
    plt.figure(figsize=(12, 6))
    plt.bar(tone_df.index, tone_df['emotional_score'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Emotional Intensity Score by Audio File')
    plt.ylabel('Emotional Intensity Score')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/emotional_tone_scores.png')
    plt.close()
    
    # Suggest files for each tone category
    if len(sorted_files) >= 2:
        calm_file = sorted_files[0]  # File with lowest emotional score
        energetic_file = sorted_files[-1]  # File with highest emotional score
        
        print("\nBased on quantitative analysis:")
        print(f"Suggested calm/formal file: {calm_file} (score: {tone_features[calm_file]['emotional_score']:.2f})")
        print(f"Suggested passionate/energetic file: {energetic_file} (score: {tone_features[energetic_file]['emotional_score']:.2f})")
        print(f"\nFull analysis saved to {output_dir}/emotional_tone_analysis.csv")
        
        return calm_file, energetic_file
    else:
        print("Not enough files to make a recommendation.")
        return None, None

def plot_additional_comparisons(calm_features, energetic_features, output_dir):
    """
    Generate additional comparative plots to better visualize differences
    between calm and energetic speech
    """
    if calm_features is None or energetic_features is None:
        print("Cannot create additional plots: one or both feature sets are None")
        return
    
    # 1. Energy distribution histogram
    plt.figure(figsize=(12, 6))
    plt.hist(calm_features['ste'], bins=50, alpha=0.5, label='Calm', density=True)
    plt.hist(energetic_features['ste'], bins=50, alpha=0.5, label='Energetic', density=True)
    plt.title('Energy Distribution Comparison')
    plt.xlabel('Energy')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/energy_distribution_comparison.png")
    plt.close()
    
    # 2. MFCC heatmap comparison (side by side)
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    librosa.display.specshow(calm_features['mfccs'], x_axis='time')
    plt.colorbar(format='%+2.0f')
    plt.title('MFCCs - Calm Speech')
    
    plt.subplot(1, 2, 2)
    librosa.display.specshow(energetic_features['mfccs'], x_axis='time')
    plt.colorbar(format='%+2.0f')
    plt.title('MFCCs - Energetic Speech')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mfcc_heatmap_comparison.png")
    plt.close()
    
    # 3. Feature variability comparison (standard deviations)
    plt.figure(figsize=(12, 6))
    
    # Extract standard deviations for key features
    features = ['zcr', 'ste']
    calm_stds = [np.std(calm_features[feat]) for feat in features]
    energetic_stds = [np.std(energetic_features[feat]) for feat in features]
    
    # Add MFCC standard deviations
    for i in range(min(5, N_MFCC)):  # First 5 MFCCs
        features.append(f'mfcc{i+1}')
        calm_stds.append(np.std(calm_features['mfccs'][i]))
        energetic_stds.append(np.std(energetic_features['mfccs'][i]))
    
    x = np.arange(len(features))
    width = 0.35
    
    plt.bar(x - width/2, calm_stds, width, label='Calm')
    plt.bar(x + width/2, energetic_stds, width, label='Energetic')
    
    plt.title('Feature Variability Comparison')
    plt.xlabel('Feature')
    plt.ylabel('Standard Deviation')
    plt.xticks(x, features)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_variability_comparison.png")
    plt.close()
    
    # 4. Spectral contrast comparison
    plt.figure(figsize=(15, 6))
    
    # Calculate spectral contrast for both audio samples
    if 'y' in calm_features and 'sr' in calm_features and 'y' in energetic_features and 'sr' in energetic_features:
        calm_contrast = librosa.feature.spectral_contrast(y=calm_features['y'], sr=calm_features['sr'])
        energetic_contrast = librosa.feature.spectral_contrast(y=energetic_features['y'], sr=energetic_features['sr'])
        
        plt.subplot(1, 2, 1)
        librosa.display.specshow(calm_contrast, x_axis='time')
        plt.colorbar()
        plt.title('Spectral Contrast - Calm Speech')
        
        plt.subplot(1, 2, 2)
        librosa.display.specshow(energetic_contrast, x_axis='time')
        plt.colorbar()
        plt.title('Spectral Contrast - Energetic Speech')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/spectral_contrast_comparison.png")
    else:
        print("Cannot create spectral contrast plot: audio data not available in features")
    
    plt.close()

def main():
    """
    Main function to analyze historical audio recordings
    """
    # Download or locate the dataset
    dataset_dir = return_dataset()
    
    # Create output directory
    output_dir = 'question_2'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process all files in the dataset for feature extraction
    print("\nProcessing all files in the dataset for feature extraction...")
    all_features = {}
    
    for file in os.listdir(dataset_dir):
        if file.endswith(('.wav', '.mp3')):
            file_path = os.path.join(dataset_dir, file)
            print(f"Processing {file}...")
            
            # Load audio
            y, sr = load_audio(file_path)
            if y is None:
                continue
                
            # Extract features
            features = extract_features(y, sr)
            if features is None:
                continue
                
            # Store features
            all_features[file] = features
            
            # Plot individual features
            plot_features(features, f"Speech: {file}", f"{output_dir}/{os.path.splitext(file)[0]}_features.png")
    
    print(f"Processed {len(all_features)} files from the dataset.")
    
    # For comparative analysis, use specific files
    calm_file = os.path.join(dataset_dir, "Queen Victoria.mp3")
    energetic_file = os.path.join(dataset_dir, "Rahul Gandhi.mp3")
    
    # Load audio files for comparative analysis
    print(f"\nLoading calm speech: {calm_file}")
    calm_audio, calm_sr = load_audio(calm_file)
    
    print(f"Loading energetic speech: {energetic_file}")
    energetic_audio, energetic_sr = load_audio(energetic_file)
    
    # Extract features
    print("Extracting features from calm speech...")
    calm_features = extract_features(calm_audio, calm_sr)
    
    print("Extracting features from energetic speech...")
    energetic_features = extract_features(energetic_audio, energetic_sr)
    
    # Plot individual features
    print("Plotting features...")
    plot_features(calm_features, "Calm and Formal Speech", f"{output_dir}/calm_speech_features.png")
    plot_features(energetic_features, "Passionate and Energetic Speech", f"{output_dir}/energetic_speech_features.png")
    
    # Compare features
    print("Comparing features...")
    comparison = compare_features(calm_features, energetic_features)
    
    # Save comparison to CSV
    if comparison is not None:
        comparison.to_csv(f"{output_dir}/feature_comparison.csv")
        print(f"Feature comparison saved to {output_dir}/feature_comparison.csv")
    
    # Plot comparison
    plot_comparison(calm_features, energetic_features, f"{output_dir}/feature_comparison.png")
    
    # Generate additional comparative plots
    print("Generating additional comparative plots...")
    plot_additional_comparisons(calm_features, energetic_features, output_dir)
    
    # Now analyze emotional tone
    print("\nAnalyzing emotional tone of audio files...")
    analyze_emotional_tone(dataset_dir)
    
    print("\nAnalysis complete! Check the generated files for results.")

if __name__ == "__main__":
    main() 