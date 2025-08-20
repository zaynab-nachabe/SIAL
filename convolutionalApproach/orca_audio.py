#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyNN.nest as sim

def parse_args():
    parser = argparse.ArgumentParser(description='Process orca whale audio for SNN simulation')
    
    parser.add_argument('--audio-file', type=str, default='audio/int_orca.wav', 
                        help='Path to the audio file to process')
    parser.add_argument('--threshold', type=float, default=0.2,
                       help='Threshold for spike detection')
    parser.add_argument('--max-time', type=float, default=None,
                       help='Maximum time to process (ms)')
    parser.add_argument('--plot', action='store_true',
                       help='Plot the audio waveform and detected spikes')
    
    return parser.parse_args()

def load_audio_data(file_path, threshold=0.2, max_time=None):
    """Load and preprocess audio data from a WAV file."""
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: Audio file {file_path} not found")
        return []
    
    # Load the audio file
    sample_rate, audio_data = wavfile.read(file_path)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalize audio data to range [-1, 1]
    audio_data = audio_data.astype(float)
    if np.max(np.abs(audio_data)) > 0:
        audio_data /= np.max(np.abs(audio_data))
    
    # Convert to spike times by detecting threshold crossings
    spike_times = []
    last_spike = -10  # Minimum interval between spikes (ms)
    
    # Convert sample rate to ms timebase
    time_ms = np.arange(len(audio_data)) * 1000 / sample_rate
    
    if max_time is not None:
        # Truncate data if max_time is specified
        valid_indices = time_ms < max_time
        time_ms = time_ms[valid_indices]
        audio_data = audio_data[valid_indices]
    
    for i, (time, amplitude) in enumerate(zip(time_ms, audio_data)):
        if amplitude > threshold and time > last_spike + 10:
            spike_times.append(time)
            last_spike = time
    
    return spike_times, audio_data, time_ms, sample_rate

def main():
    args = parse_args()
    
    # Load audio data
    spike_times, audio_data, time_ms, sample_rate = load_audio_data(
        args.audio_file, 
        threshold=args.threshold,
        max_time=args.max_time
    )
    
    # Print statistics
    print(f"Loaded audio file: {args.audio_file}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {time_ms[-1]/1000:.2f} seconds")
    print(f"Detected {len(spike_times)} spikes using threshold {args.threshold}")
    
    # Plot if requested
    if args.plot:
        plt.figure(figsize=(12, 6))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        plt.plot(time_ms/1000, audio_data, 'b-', alpha=0.5)
        plt.axhline(y=args.threshold, color='r', linestyle='--', label=f'Threshold ({args.threshold})')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Orca Audio Waveform')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot detected spikes
        plt.subplot(2, 1, 2)
        if spike_times:
            plt.vlines(np.array(spike_times)/1000, 0, 1, color='r')
        plt.xlabel('Time (s)')
        plt.title(f'Detected Spikes (n={len(spike_times)})')
        plt.ylim(0, 1.2)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('orca_spikes.png', dpi=150)
        plt.close()
        
        print(f"Plot saved to orca_spikes.png")
    
    return spike_times

if __name__ == "__main__":
    main()
