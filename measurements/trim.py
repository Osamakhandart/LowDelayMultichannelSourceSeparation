import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# Base path to the dataset
base_path = "/Users/usamakhan/Documents/project/withdata/BSS/measurements/measured_data"

# Output folder for trimmed audio
output_base_path = "/Users/usamakhan/Documents/project/withdata/BSS/measurements/trimmed_data"

# Create the output base folder if it doesn't exist
if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)

# Function to trim audio
def trim_audio(audio, sample_rate, trim_duration=1):
    """
    Trim the first and last `trim_duration` seconds of the audio.
    """
    trim_samples = int(trim_duration * sample_rate)
    return audio[trim_samples:-trim_samples]

# Iterate through each folder (1, 2, 3, ..., 6)
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    
    # Skip if it's not a directory
    if not os.path.isdir(folder_path):
        continue
    
    # Create a corresponding output folder
    output_folder = os.path.join(output_base_path, folder_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each audio file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            # Load the audio file
            file_path = os.path.join(folder_path, file_name)
            audio, sample_rate = sf.read(file_path)

            # Trim the audio
            trimmed_audio = trim_audio(audio, sample_rate)

            # Save the trimmed audio
            output_path = os.path.join(output_folder, file_name)
            sf.write(output_path, trimmed_audio, sample_rate)

            # Plot the waveform of the trimmed audio
            time = np.linspace(0, len(trimmed_audio) / sample_rate, num=len(trimmed_audio))
            plt.figure(figsize=(10, 5))
            if len(trimmed_audio.shape) == 1:  # Mono audio
                plt.plot(time, trimmed_audio, color='b', label="Mono Audio")
            else:  # Stereo audio
                plt.plot(time, trimmed_audio[:, 0], color='b', label="Left Channel")
                plt.plot(time, trimmed_audio[:, 1], color='r', label="Right Channel")
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")
            plt.title(f"Waveform of {file_name} (Trimmed)")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{file_name}_waveform.png"))
            plt.close()

            print(f"✅ Trimmed and saved: {output_path}")

print("🎉 All files processed!")