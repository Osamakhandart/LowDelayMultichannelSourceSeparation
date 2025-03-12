import os
import soundfile as sf
import numpy as np

# Base path to the trimmed dataset
base_path = "/Users/usamakhan/Documents/project/withdata/BSS/measurements/trimmed_data"

# Function to calculate SNR
def calculate_snr(audio):
    """
    Calculate the Signal-to-Noise Ratio (SNR) of the audio.

    Parameters:
        audio (numpy.ndarray): The audio signal.

    Returns:
        float: SNR in decibels (dB).
    """
    # Calculate the power of the signal
    signal_power = np.sum(audio**2) / len(audio)

    # Estimate the noise power (assuming the noise is in the first 100 samples)
    noise_samples = audio[:100]  # Use the first 100 samples as noise
    noise_power = np.sum(noise_samples**2) / len(noise_samples)

    # Avoid division by zero
    if noise_power == 0:
        return float('inf')  # Infinite SNR if there is no noise

    # Calculate SNR in decibels
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Iterate through each folder (1, 2, 3, ..., 6)
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    
    # Skip if it's not a directory
    if not os.path.isdir(folder_path):
        continue
    
    # Dictionary to store SNR values and corresponding file paths
    snr_dict = {}

    # Process each audio file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            # Load the trimmed audio file
            file_path = os.path.join(folder_path, file_name)
            audio, sample_rate = sf.read(file_path)

            # Calculate SNR for each channel
            if len(audio.shape) == 1:  # Mono audio
                snr = calculate_snr(audio)
            else:  # Stereo audio
                snr_left = calculate_snr(audio[:, 0])
                snr_right = calculate_snr(audio[:, 1])
                snr = (snr_left + snr_right) / 2  # Average SNR for stereo

            # Store SNR and file path in the dictionary
            snr_dict[file_name] = snr

    # Check if snr_dict is not empty
    if not snr_dict:
        print(f"⚠️ No .wav files found in folder {folder_name}. Skipping...")
        continue

    # Find the file with the highest SNR
    best_file = max(snr_dict, key=snr_dict.get)
    best_snr = snr_dict[best_file]

    print(f"📊 Best SNR in folder {folder_name}: {best_file} (SNR: {best_snr:.2f} dB)")

    # Delete the other two files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav") and file_name != best_file:
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)
            print(f"🗑️ Deleted: {file_path}")

print("🎉 Best SNR files selected and other files deleted!")