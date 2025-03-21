import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, convolve
from scipy.io import wavfile
import librosa
# Parameters
sample_rate = 16000  # Sample rate in Hz
duration = 1.0       # Duration of the signals in seconds
nperseg = 1024       # STFT window length
noverlap = 512       # STFT overlap


def load_stereo_audio(file_path,target_sr):
    stereo_audio, sample_rate = librosa.load(file_path, sr=target_sr, mono=False)  # Correctly load and unpack

    # Ensure the audio is in the shape (samples, channels)
    if len(stereo_audio.shape) == 1:
        stereo_audio = np.column_stack((stereo_audio, stereo_audio))
    return sample_rate, stereo_audio

# Load real RIR (2 channels)
def load_rir(file_path,target_sr):
   
    rir, sample_rate = librosa.load(file_path, sr=target_sr, mono=False)  # Correctly load and unpack

    # Ensure the RIR is in the shape (samples, channels)
    if len(rir.shape) == 1:
        rir = np.column_stack((rir, rir))
    return sample_rate, rir
# Generate a test RIR (Room Impulse Response)
def generate_test_rir(sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Simulate a simple RIR with two channels (microphones)
    rir1 = np.exp(-5 * t) * np.sin(2 * np.pi * 500 * t)  # First microphone
    rir2 = np.exp(-5 * t) * np.sin(2 * np.pi * 500 * (t - 0.001))  # Second microphone with delay
    rir = np.column_stack((rir1, rir2))
    return rir

# Generate test stereo audio (two microphones recording a source)
def generate_test_stereo_audio(sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Simulate a source signal (e.g., a sine wave)
    source_signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    # Simulate stereo audio by convolving the source with the RIR
    rir = generate_test_rir(sample_rate, duration)
    stereo_audio = np.column_stack((convolve(source_signal, rir[:, 0], mode='same'),
                                   convolve(source_signal, rir[:, 1], mode='same')))
    return stereo_audio

# Estimate the Relative Transfer Function (RTF)
def estimate_rtf(stereo_audio, sample_rate, nperseg, noverlap):
    # Compute STFT for both channels
    f, t, Z1 = stft(stereo_audio[:, 0], fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    f, t, Z2 = stft(stereo_audio[:, 1], fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    # RTF is the ratio of the STFTs
    RTF = Z2 / Z1
    return f, t, RTF

# Plot the RTF
def plot_rtf(f, t, RTF):
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, np.abs(RTF), shading='gouraud', vmin=0, vmax=2)
    plt.colorbar(label='Magnitude')
    plt.title('Relative Transfer Function (RTF) Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, np.angle(RTF), shading='gouraud', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(label='Phase [radians]')
    plt.title('Relative Transfer Function (RTF) Phase')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

# Main script
if __name__ == "__main__":
    # Generate test RIR and stereo audio
    # rir = generate_test_rir(sample_rate, duration)
    # stereo_audio = generate_test_stereo_audio(sample_rate, duration)

        # Load real stereo audio and RIR
    sample_rate, stereo_audio = load_stereo_audio('/home/ZA/Music/Media project/BSS_MP/BSS_self/LowDelayMultichannelSourceSeparation/measurements/trimmed_data_final_eval/4/tda_audio4_1.wav',target_sr=48000)
    sample_rate_rir, rir = load_rir('real_rir.wav',target_sr=48000)





    # Estimate RTF
    f, t, RTF = estimate_rtf(stereo_audio, sample_rate, nperseg, noverlap)

    # Plot RTF
    plot_rtf(f, t, RTF)

    # Save test signals for verification
    wavfile.write('test_rir.wav', sample_rate, rir.astype(np.float32))
    wavfile.write('test_stereo_audio.wav', sample_rate, stereo_audio.astype(np.float32))
    print("Test RIR and stereo audio saved as 'test_rir.wav' and 'test_stereo_audio.wav'.")