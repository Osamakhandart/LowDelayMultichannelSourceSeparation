import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import threading

# Function to load audio
def load_audio(file_path, target_sr):
    audio, _ = librosa.load(file_path, sr=target_sr)
    return audio

# Function to play the stereo sources
def play_audio(stereo_output, target_sr):
    print("🔊 Playing two audio sources...")
    sd.play(stereo_output, samplerate=target_sr)
    sd.wait()

# Function to record from two-channel microphone
def record_audio(duration, target_sr):
    print("🎤 Recording from two microphones...")
    recording = sd.rec(int(target_sr * duration), samplerate=target_sr, channels=2, dtype='float32')
    sd.wait()
    print("✅ Recording complete!")
    return recording

# Main function to handle processing
def process_audio_combinations(base_path, combinations, target_sr=48000):
    for combo in combinations:
        file_paths = [os.path.join(base_path, combo[0]), os.path.join(base_path, combo[1])]
        sources = [load_audio(fp, target_sr) for fp in file_paths]

        # Calculate minimum length to match both sources
        minlen = min(len(sources[0]), len(sources[1]))
        sources = [src[:minlen] / (np.max(np.abs(src)) or 1) for src in sources]  # Normalize and prevent divide by zero

        stereo_output = np.column_stack(sources)
        duration = minlen / target_sr

        # Play and record audio
        play_thread = threading.Thread(target=play_audio, args=(stereo_output, target_sr))
        record_thread = threading.Thread(target=record_audio, args=(duration, target_sr))

        play_thread.start()
        record_thread.start()

        play_thread.join()
        recording = record_thread.join()

        # Save and plot the recorded audio
        save_and_plot_recorded_audio(recording, duration, target_sr,base_path)

def save_and_plot_recorded_audio(recording, duration, target_sr, base_path, index):
    # Ensure the RiR_data directory exists
    output_dir = os.path.join(base_path, 'RiR_data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the filename with incrementing index
    recorded_filename = os.path.join(output_dir, f"rir_output{index}.wav")
    sf.write(recorded_filename, (recording * 32767).astype(np.int16), target_sr)  # Convert to 16-bit PCM
    print(f"💾 Recorded audio saved as {recorded_filename}")

    # Plot the recorded waveforms
    time = np.linspace(0, duration, num=recording.shape[0])
    plt.figure(figsize=(10, 5))
    for i in range(2):
        plt.subplot(2, 1, i+1)
        plt.plot(time, recording[:, i], label=f"Microphone {i+1}")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"Waveform of Microphone {i+1}")
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.show()


base_path = "/Users/usamakhan/Documents/project/withdata/BSS/measurements/Audio Dataset"
combinations = [("drums.wav", "pinkish16.wav"), ("another_pair1.wav", "another_pair2.wav")]
process_audio_combinations(base_path, combinations)
