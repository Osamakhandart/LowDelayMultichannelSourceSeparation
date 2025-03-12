import os
import soundfile as sf
import numpy as np

# Base path to the dataset
base_path = "/Users/usamakhan/Documents/project/withdata/BSS/measurements/measured_data"

# Output folder for trimmed audio
output_base_path = "/Users/usamakhan/Documents/project/withdata/BSS/measurements/trimmed_data"

# Create the output base folder if it doesn't exist
if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)

# Function to trim only the first 1 second of audio
def trim_first_second(audio, sample_rate, trim_duration=1):
    """
    Trim the first `trim_duration` seconds of the audio.
    """
    trim_samples = int(trim_duration * sample_rate)
    return audio[trim_samples:]

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
    for i, file_name in enumerate(os.listdir(folder_path), start=1):
        if file_name.endswith(".wav"):
            # Load the audio file
            file_path = os.path.join(folder_path, file_name)
            audio, sample_rate = sf.read(file_path)

            # Trim the first 1 second of the audio
            trimmed_audio = trim_first_second(audio, sample_rate)

            # Rename the trimmed audio file with folder name
            new_file_name = f"tda_audio{folder_name}_{i}.wav"
            output_path = os.path.join(output_folder, new_file_name)

            # Save the trimmed audio
            sf.write(output_path, trimmed_audio, sample_rate)

            print(f"✅ Trimmed and saved: {output_path}")

print("🎉 All files processed!")