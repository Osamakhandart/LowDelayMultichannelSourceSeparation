import sounddevice as sd
import numpy as np
import soundfile as sf
import scipy.io.wavfile as wav
import librosa
import matplotlib.pyplot as plt
import threading

# File paths
path1 = "/home/ZA/Music/Media project/BSS_MP/BSS_MP/measurements/Audio Dataset/drums.wav"
path2 = "/home/ZA/Music/Media project/BSS_MP/BSS_MP/measurements/Audio Dataset/pinkish16.wav"

# Load and resample both audio sources to 48kHz
target_sr = 48000
source1, _ = librosa.load(path1, sr=target_sr)
source2, _ = librosa.load(path2, sr=target_sr)

# Ensure both sources have the same length (match the shorter one)
minlen = min(len(source1), len(source2))
source1 = source1[:minlen] / (np.max(np.abs(source1)) or 1)  # Normalize & prevent divide by zero
source2 = source2[:minlen] / (np.max(np.abs(source2)) or 1)

# Combine into stereo output (source1 → Left speaker, source2 → Right speaker)
stereo_output = np.column_stack((source1, source2))

# Recording parameters
duration = minlen / target_sr
channels = 2

# Function to play the stereo sources
def play_audio():
    print("🔊 Playing two audio sources...")
    sd.play(stereo_output, samplerate=target_sr)
    sd.wait()

# Function to record from two-channel microphone
def record_audio():
    print("🎤 Recording from two microphones...")
    recording = sd.rec(int(target_sr * duration), samplerate=target_sr, channels=channels, dtype=np.float32)
    sd.wait()
    print("✅ Recording complete!")

    # Save recorded mixture as a WAV file
    recorded_filename = "two_speakers_recording.wav"
    wav.write(recorded_filename, target_sr, (recording * 32767).astype(np.int16))  # Convert to 16-bit PCM
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

# Run playback and recording in parallel
play_thread = threading.Thread(target=play_audio)
record_thread = threading.Thread(target=record_audio)

play_thread.start()
record_thread.start()

play_thread.join()
record_thread.join()
