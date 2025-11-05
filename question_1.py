"""
Question 1: Audio Sample Analysis

This script collects and analyzes 10 audio samples of my voice,
and computes the amplitude, pitch, frequency, RMS energy, and generates spectrogram plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd
import soundfile as sf
import os
from scipy.io import wavfile
from scipy.signal import find_peaks

# Create directory for audio samples if it doesn't exist
if not os.path.exists('question_1/audio_samples'):
    os.makedirs('question_1/audio_samples')

def record_audio(filename, duration=5, fs=44100):
    """
    Records audio from microphone
    """
    print(f"Recording {filename}... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, recording, fs)
    print(f"Recording saved to {filename}")

def analyze_audio(filename):
    """
    Analyze audio file and compute various audio features.
    
    Input the audio filepath and returns a dictionary of statistical audio features.
    """
    # Load audio file
    y, sr = librosa.load(filename, sr=None)
    
    # Compute amplitude
    amplitude = np.abs(y)
    max_amplitude = np.max(amplitude)
    
    # Compute pitch (fundamental frequency) using autocorrelation
    f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                fmin=librosa.note_to_hz('C2'), 
                                                fmax=librosa.note_to_hz('C7'),
                                                sr=sr)
    # Remove NaN values for average calculation
    f0_clean = f0[~np.isnan(f0)]
    avg_pitch = np.mean(f0_clean) if len(f0_clean) > 0 else 0
    
    # Compute frequency content using FFT
    n_fft = 2048
    ft = np.abs(librosa.stft(y, n_fft=n_fft))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    dominant_freq_idx = np.argmax(np.mean(ft, axis=1))
    dominant_freq = frequencies[dominant_freq_idx]
    
    # Compute RMS energy
    rms_energy = np.sqrt(np.mean(y**2))
    
    return {
        'amplitude': amplitude,
        'max_amplitude': max_amplitude,
        'pitch': f0,
        'avg_pitch': avg_pitch,
        'frequencies': frequencies,
        'frequency_magnitudes': np.mean(ft, axis=1),
        'dominant_freq': dominant_freq,
        'rms_energy': rms_energy,
        'audio': y,
        'sr': sr
    }

def plot_audio_features(features, filename, index):
    """
    Plot audio features
    
    Parameters:
    -----------
    features : dict
        Dictionary containing audio features
    filename : str
        Name of the audio file
    index : int
        Index of the audio sample
    """
    # Create a figure with subplots
    plt.figure(figsize=(15, 12))
    
    # Plot waveform (amplitude)
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(features['audio'], sr=features['sr'])
    plt.title(f'Waveform (Max Amplitude: {features["max_amplitude"]:.4f})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Plot pitch
    plt.subplot(4, 1, 2)
    times = librosa.times_like(features['pitch'], sr=features['sr'])
    plt.plot(times, features['pitch'])
    plt.title(f'Pitch (Average: {features["avg_pitch"]:.2f} Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    # Plot frequency spectrum
    plt.subplot(4, 1, 3)
    plt.plot(features['frequencies'], features['frequency_magnitudes'])
    plt.title(f'Frequency Spectrum (Dominant: {features["dominant_freq"]:.2f} Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xscale('log')
    
    # Plot spectrogram
    plt.subplot(4, 1, 4)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(features['audio'])), ref=np.max)
    librosa.display.specshow(D, sr=features['sr'], x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram (RMS Energy: {features["rms_energy"]:.4f})')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'question_1/audio_analysis_{index}.png')
    plt.close()

def main():
    """
    Main function to record and analyze audio samples
    """
    # Number of samples to record
    num_samples = 10
    
    # Record audio samples or use existing ones
    record_new = input("Do you want to record new audio samples? (y/n): ").lower() == 'y'
    
    if record_new:
        for i in range(num_samples):
            print(f"\nSample {i+1}/{num_samples}")
            print("Speak a different text with varying pitch and volume")
            record_audio(f'question_1/audio_samples/sample_{i+1}.wav')
    
    # Analyze each audio sample
    for i in range(num_samples):
        filename = f'question_1/audio_samples/sample_{i+1}.wav'
        
        # Check if file exists
        if not os.path.exists(filename):
            print(f"File {filename} does not exist. Skipping...")
            continue
        
        print(f"Analyzing {filename}...")
        features = analyze_audio(filename)
        plot_audio_features(features, filename, i+1)
        
        # Print summary of features
        print(f"  Max Amplitude: {features['max_amplitude']:.4f}")
        print(f"  Average Pitch: {features['avg_pitch']:.2f} Hz")
        print(f"  Dominant Frequency: {features['dominant_freq']:.2f} Hz")
        print(f"  RMS Energy: {features['rms_energy']:.4f}")
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()
