import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

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

# Lists to store SNR values and file names
snr_values = []
file_names = []

# Iterate through each folder (1, 2, 3, ..., 6)
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    
    # Skip if it's not a directory
    if not os.path.isdir(folder_path):
        continue
    
    # Process each audio file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            # Load the trimmed audio file
            file_path = os.path.join(folder_path, file_name)
            audio, sample_rate = sf.read(file_path)

            # Calculate SNR for each channel
            if len(audio.shape) == 1:  # Mono audio
                snr = calculate_snr(audio)
                snr_values.append(snr)
                file_names.append(file_name)
                print(f"📊 SNR for {file_name} (Mono): {snr:.2f} dB")
            else:  # Stereo audio
                snr_left = calculate_snr(audio[:, 0])
                snr_right = calculate_snr(audio[:, 1])
                snr_values.append((snr_left + snr_right) / 2)  # Average SNR for stereo
                file_names.append(file_name)
                print(f"📊 SNR for {file_name} (Left Channel): {snr_left:.2f} dB")
                print(f"📊 SNR for {file_name} (Right Channel): {snr_right:.2f} dB")

# Plot the overall SNR graph
plt.figure(figsize=(12, 6))
plt.bar(file_names, snr_values, color='blue')
plt.xlabel("Audio Files")
plt.ylabel("SNR (dB)")
plt.title("Overall SNR for All Audio Files")
plt.xticks(rotation=90)  # Rotate file names for better readability
plt.grid(axis='y')
plt.tight_layout()

# Save the overall SNR graph
output_graph_path = os.path.join(base_path, "overall_snr_graph.png")
plt.savefig(output_graph_path)
plt.close()

print(f"📈 Overall SNR graph saved as {output_graph_path}")
print("🎉 SNR calculation and overall graph plotting complete!")