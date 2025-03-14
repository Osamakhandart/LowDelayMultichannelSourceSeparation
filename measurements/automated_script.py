# import os
# import numpy as np
# import sounddevice as sd
# import soundfile as sf
# import librosa
# import matplotlib.pyplot as plt
# import threading

# # Function to load audio
# def load_audio(file_path, target_sr):
#     audio, _ = librosa.load(file_path, sr=target_sr)
#     return audio

# # Function to play the stereo sources
# def play_audio(stereo_output, target_sr):
#     print("🔊 Playing two audio sources...")
#     sd.play(stereo_output, samplerate=target_sr)
#     sd.wait()

# # Function to record from two-channel microphone
# def record_audio(duration, target_sr):
#     print("🎤 Recording from two microphones...")
#     recording = sd.rec(int(target_sr * duration), samplerate=target_sr, channels=2, dtype='float32')
#     sd.wait()
#     print("✅ Recording complete!")
#     return recording

# # Main function to handle processing
# def process_audio_combinations(base_path, combinations, target_sr=48000):
#     for index, combo in enumerate(combinations):  # Added index with enumerate
#         file_paths = [os.path.join(base_path, combo[0]), os.path.join(base_path, combo[1])]
#         sources = [load_audio(fp, target_sr) for fp in file_paths]

#         # Calculate minimum length to match both sources
#         minlen = min(len(sources[0]), len(sources[1]))
#         sources = [src[:minlen] / (np.max(np.abs(src)) or 1) for src in sources]  # Normalize and prevent divide by zero

#         stereo_output = np.column_stack(sources)
#         duration = minlen / target_sr

#         # Play and record audio
#         play_thread = threading.Thread(target=play_audio, args=(stereo_output, target_sr))
#         record_thread = threading.Thread(target=record_audio, args=(duration, target_sr))

#         play_thread.start()
#         record_thread.start()

#         play_thread.join()
#         recording = record_thread.join()

#         # Pass the index here to uniquely save each file
#         save_and_plot_recorded_audio(recording, duration, target_sr, base_path, index)

# def save_and_plot_recorded_audio(recording, duration, target_sr, base_path, index):
#     # Ensure the RiR_data directory exists
#     output_dir = os.path.join(base_path, 'RiR_data')
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Construct the filename with incrementing index
#     recorded_filename = os.path.join(output_dir, f"rir_output{index}.wav")
#     sf.write(recorded_filename, (recording * 32767).astype(np.int16), target_sr)  # Convert to 16-bit PCM
#     print(f"💾 Recorded audio saved as {recorded_filename}")

#     # Plot the recorded waveforms
#     time = np.linspace(0, duration, num=recording.shape[0])
#     plt.figure(figsize=(10, 5))
#     for i in range(2):
#         plt.subplot(2, 1, i+1)
#         plt.plot(time, recording[:, i], label=f"Microphone {i+1}")
#         plt.xlabel("Time [s]")
#         plt.ylabel("Amplitude")
#         plt.title(f"Waveform of Microphone {i+1}")
#         plt.legend()
#         plt.grid()
#     plt.tight_layout()
#     plt.show()


# base_path = "/home/ZA/Music/Media project/BSS_MP/BSS_self/LowDelayMultichannelSourceSeparation/measurements/Audio Dataset/"
# combinations = [("speech-female.wav", "music.wav"), ("male_speech.wav", "backgroundmusic.wav"), ("singing.wav", "drums.wav"), ("speech-female.wav", "drums.wav"), 
#                 ("male_speech.wav", "casta.wav"), ("musicSong.wav", "backgroundmusic.wav")]
# process_audio_combinations(base_path, combinations)


import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import threading

# Custom output directory
output_dir = "/home/ZA/Music/Media project/BSS_MP/BSS_self/LowDelayMultichannelSourceSeparation/measurements/automated_script_outputs"

# Function to load audio
def load_audio(file_path, target_sr):
    audio, _ = librosa.load(file_path, sr=target_sr)
    return audio

# Function to play the stereo sources
def play_audio(stereo_output, target_sr):
    print("🔊 Playing two audio sources...")
    sd.play(stereo_output, samplerate=target_sr)
    sd.wait()
    print("✅ Playback complete!")

# Function to record from two-channel microphone and save the result in a container
def record_audio(duration, target_sr, result_container):
    print("🎤 Recording from two microphones...")
    recording = sd.rec(int(target_sr * duration), samplerate=target_sr, channels=2, dtype='float32')
    sd.wait()
    print("✅ Recording complete!")
    result_container.append(recording)  # Store the recording in the container

# Function to save and plot the recorded audio
def save_and_plot_recorded_audio(recording, duration, target_sr, index):
    # Ensure the output directory exists
    rir_output_dir = os.path.join(output_dir, 'RiR_data')
    os.makedirs(rir_output_dir, exist_ok=True)

    # Construct the filename with incrementing index
    recorded_filename = os.path.join(rir_output_dir, f"rir_output_{index}.wav")
    
    # Convert and save as 16-bit PCM WAV file
    sf.write(recorded_filename, (recording * 32767).astype(np.int16), target_sr)
    print(f"💾 Recorded audio saved as {recorded_filename}")

    # Plot the recorded waveforms
    time = np.linspace(0, duration, num=recording.shape[0])
    # plt.figure(figsize=(10, 5))
    
    # for i in range(2):  # For each channel
    #     plt.subplot(2, 1, i + 1)
    #     plt.plot(time, recording[:, i], label=f"Microphone {i + 1}")
    #     plt.xlabel("Time [s]")
    #     plt.ylabel("Amplitude")
    #     plt.title(f"Waveform of Microphone {i + 1}")
    #     plt.legend()
    #     plt.grid()

    # plt.tight_layout()

    # Save the plot as an image file
    # plot_filename = os.path.join(rir_output_dir, f"rir_output_{index}_plot.png")
    # plt.savefig(plot_filename)
    # print(f"🖼️ Plot saved as {plot_filename}")

    plt.show()

# Main function to handle processing of audio combinations
def process_audio_combinations(base_path, combinations, target_sr=48000):
    for index, combo in enumerate(combinations, start=1):
        print(f"\n➡️ Processing combination {index}: {combo[0]} + {combo[1]}")
        
        # Get full file paths
        file_paths = [os.path.join(base_path, combo[0]), os.path.join(base_path, combo[1])]
        
        # Load and normalize audio files
        sources = [load_audio(fp, target_sr) for fp in file_paths]

        # Match lengths of both sources
        minlen = min(len(sources[0]), len(sources[1]))
        sources = [src[:minlen] / (np.max(np.abs(src)) or 1) for src in sources]  # Normalize and trim
        
        # Combine into stereo output
        stereo_output = np.column_stack(sources)
        duration = minlen / target_sr

        # Use a shared container to capture the recording from the thread
        recording_container = []

        # Start playback and recording threads
        play_thread = threading.Thread(target=play_audio, args=(stereo_output, target_sr))
        record_thread = threading.Thread(target=record_audio, args=(duration, target_sr, recording_container))

        play_thread.start()
        record_thread.start()

        play_thread.join()
        record_thread.join()

        # Extract the recorded audio from the container
        if recording_container:
            recording = recording_container[0]
            save_and_plot_recorded_audio(recording, duration, target_sr, index)
        else:
            print(f"⚠️ No recording found for combination {index}")

# Base path and combinations
base_path = "/home/ZA/Music/Media project/BSS_MP/BSS_self/LowDelayMultichannelSourceSeparation/measurements/Audio Dataset/"

# List of audio combinations
combinations = [
    ("speech-female.wav", "music.wav"),
    ("male_speech.wav", "backgroundmusic.wav"),
    ("singing.wav", "drums.wav"),
    ("speech-female.wav", "drums.wav"),
    ("male_speech.wav", "casta.wav"),
    ("musicSong.wav", "backgroundmusic.wav")
]

# Run the process
process_audio_combinations(base_path, combinations)
