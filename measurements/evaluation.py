import os
import soundfile as sf
import numpy as np
import librosa
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from sklearn.linear_model import LinearRegression
from scipy.stats import kurtosis
# Base path to the dataset
base_path = "/Users/usamakhan/Documents/project/withdata/BSS/measurements/Audio Dataset"
base_path_trimmed = "/Users/usamakhan/Documents/project/withdata/BSS/measurements/trimmed_data"
snr_values = []
kurtosis_values=[]
predictability_scores=[]
file_names = []
co_relate=[]
# Function to calculate correlation between two mono audio signals
def calculate_correlation(audio1, audio2):
    """
    Calculate the Pearson correlation coefficient between two mono audio signals.
    """
    min_length = min(len(audio1), len(audio2))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]
    correlation = np.corrcoef(audio1, audio2)[0, 1]
    return correlation

# List of audio file combinations to analyze
combinations = [
    {"Left Channel": "speech-female.wav", "Right Channel": "music.wav"},
    {"Left Channel": "male_speech.wav", "Right Channel": "backgroundmusic.wav"},
    {"Left Channel": "singing.wav", "Right Channel": "drums.wav"},
    {"Left Channel": "speech-female.wav", "Right Channel": "drums.wav"},
    {"Left Channel": "male_speech.wav", "Right Channel": "casta.wav"},
    {"Left Channel": "musicSong.wav", "Right Channel": "backgroundmusic.wav"}
]
def linear_predictability(data1, data2):
    # Ensuring both arrays are of the same length
    min_length = min(len(data1), len(data2))
    data1 = data1[:min_length]
    data2 = data2[:min_length]

    # Fit linear model
    model = LinearRegression().fit(data1.reshape(-1, 1), data2)
    score = model.score(data1.reshape(-1, 1), data2)
    
    return score
# Process each combination
for combo in combinations:
    left = combo["Left Channel"]
    right = combo["Right Channel"]
    mono1_path = os.path.join(base_path, left)
    mono2_path = os.path.join(base_path, right)

    # Check if files exist
    if not os.path.exists(mono1_path) and os.path.exists(mono2_path):
        print(f"⚠️ Files not found for {left} and {right}. Skipping...")
        continue

    # Load the audio files
    mono1, sr1 = librosa.load(mono1_path, sr=48000)
    mono2, sr2 = librosa.load(mono2_path, sr=48000)

    # Ensure both audio files have the same sample rate
    if sr1 != sr2:
        print(f"⚠️ Sample rates do not match for {left} and {right}. Skipping...")
        continue

    # Convert stereo to mono if necessary
    if len(mono1.shape) > 1:
        mono1 = np.mean(mono1, axis=1)
    if len(mono2.shape) > 1:
        mono2 = np.mean(mono2, axis=1)

    # Check for NaNs or Infs
    if np.any(np.isnan(mono1)) or np.any(np.isnan(mono2)):
        print(f"⚠️ NaN values detected in {left} or {right}. Skipping...")
        continue
    if np.any(np.isinf(mono1)) or np.any(np.isinf(mono2)):
        print(f"⚠️ Infinite values detected in {left} or {right}. Skipping...")
        continue

    # Calculate the correlation
    correlation = calculate_correlation(mono1, mono2)
    co_relate.append(correlation)
    predictability_score = linear_predictability(mono1, mono2)
    predictability_scores.append(predictability_score)
    print(f"Linear Predictability (R^2 score): {predictability_score}")
  
    print(f"📊 Correlation between {left} and {right}: {correlation:.4f}")

print("🎉 Correlation calculation complete!")



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
def calculate_kurtosis(data):
    # Load the audio file
    # rate, data = wav.read(file_path)
    
    # Convert stereo to mono by averaging if necessary
    if len(data.shape) == 2:
        data = data.mean(axis=1)
    
    # Calculate kurtosis
    kurtosis_value = kurtosis(data, fisher=True)  # Fisher's definition (normal ==> 0)
    return kurtosis_value

# Iterate through each folder (1, 2, 3, ..., 6)
for folder_name in os.listdir(base_path_trimmed):
    folder_path = os.path.join(base_path_trimmed, folder_name)
    
    # Skip if it's not a directory
    if not os.path.isdir(folder_path):
        continue
    
    # Process each audio file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            # Load the trimmed audio file
            file_path = os.path.join(folder_path, file_name)
            audio, sample_rate = sf.read(file_path)
            kurtosis_value = calculate_kurtosis(audio)
            kurtosis_values.append(kurtosis_value)
            print(f"Kurtosis: {kurtosis_value}")
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



fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Adjust size as needed

# Plot 1: Correlation
axs[0, 0].bar(file_names, co_relate, color='blue')
axs[0, 0].set_title('Correlation Coefficients')
axs[0, 0].set_ylabel('Correlation')
axs[0, 0].tick_params(labelrotation=45)

# Plot 2: Linear Predictability
axs[0, 1].bar(file_names, predictability_scores, color='green')
axs[0, 1].set_title('Linear Predictability (R² Scores)')
axs[0, 1].set_ylabel('R² Score')
axs[0, 1].tick_params(labelrotation=45)

# Plot 3: Kurtosis
axs[1, 0].bar(file_names, kurtosis_values, color='red')
axs[1, 0].set_title('Kurtosis Values')
axs[1, 0].set_ylabel('Kurtosis')
axs[1, 0].tick_params(labelrotation=45)

# Plot 4: SNR
axs[1, 1].bar(file_names, snr_values, color='purple')
axs[1, 1].set_title('Signal-to-Noise Ratio (SNR)')
axs[1, 1].set_ylabel('SNR (dB)')
axs[1, 1].tick_params(labelrotation=45)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()