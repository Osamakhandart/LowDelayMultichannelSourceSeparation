import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.linalg import solve

# Configuration
epsilon = 1e-10  # Small value to avoid division by zero
plot_results = True  # Set to False to disable plotting

def preprocess_signal(signal):
    """Remove DC offset and normalize signal"""
    signal = signal - np.mean(signal)
    return signal / (np.linalg.norm(signal) + epsilon)

def compute_rtf(x1, x2, n_fft=4096):
    """Compute Room Transfer Function between two signals"""
    X1 = np.fft.fft(x1, n=n_fft)
    X2 = np.fft.fft(x2, n=n_fft)
    return X2 / (X1 + epsilon)

def compute_delay(x1, x2, fs):
    """Compute time delay between two signals using cross-correlation"""
    corr = scipy.signal.correlate(x1, x2, mode='full')
    lags = scipy.signal.correlation_lags(len(x1), len(x2), mode='full')
    delay_samples = lags[np.argmax(corr)]
    return delay_samples / fs

# Load data
path = "/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/new/rirs"
audio, fs_audio = sf.read('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/new/stereoautomated_script_outputs/RiR_data/stereo_output_3.wav')
audio_left = audio[:, 0]
audio_right = audio[:, 1]

# Load RIRs (Room Impulse Responses)
rir_source1 = sf.read(path+'/ls00-left/RIR_LEFT_RAW_FLOAT.wav')[0]
rir_source2 = sf.read(path+'/ls01-right/RIR_RIGHT_RAW_FLOAT.wav')[0]

# Extract individual channel RIRs
rir_s1_m1 = preprocess_signal(rir_source1[:, 0])  # Source 1 → Mic 1
rir_s1_m2 = preprocess_signal(rir_source1[:, 1])  # Source 1 → Mic 2
rir_s2_m1 = preprocess_signal(rir_source2[:, 0])  # Source 2 → Mic 1
rir_s2_m2 = preprocess_signal(rir_source2[:, 1])  # Source 2 → Mic 2

# ==================================================================
# PART A: Single Source Analysis (Source 1)
# ==================================================================
print("\nPart A: Single Source Analysis (Source 1)")

# Simulate microphone signals for Source 1 only
mic1_s1 = scipy.signal.convolve(audio_left, rir_s1_m1, mode='same')
mic2_s1 = scipy.signal.convolve(audio_left, rir_s1_m2, mode='same')

mic1_s2 = scipy.signal.convolve(audio_right, rir_s2_m1, mode='same')
mic2_s2 = scipy.signal.convolve(audio_right, rir_s2_m2, mode='same')


# Compute time delay
delay_s1 = compute_delay(mic1_s1, mic2_s1, fs_audio)
print(f"Estimated delay between mics for Source 1: {delay_s1*1000:.2f} ms")
delay_s2 = compute_delay(mic1_s2, mic2_s2, fs_audio)
print(f"Estimated delay between mics for Source 2: {delay_s2*1000:.2f} ms")


# Compute RTF between microphones for Source 1
rtf_s1 = compute_rtf(mic1_s1, mic2_s1)
rtf_s2 = compute_rtf(mic1_s2, mic2_s2)
# Plot RTF analysis


if plot_results:
    freq = np.fft.fftfreq(len(rtf_s1), 1/fs_audio)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    plt.semilogy(freq[:len(freq)//2], np.abs(rtf_s1[:len(freq)//2]))
    plt.title('RTF Magnitude (Source 1)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    
    plt.subplot(212)
    plt.plot(freq[:len(freq)//2], np.angle(rtf_s1[:len(freq)//2]))
    plt.title('RTF Phase (Source 1)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (rad)')
    
    plt.tight_layout()
    plt.show()


#for second source 
if plot_results:
    freq = np.fft.fftfreq(len(rtf_s2), 1/fs_audio)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    plt.semilogy(freq[:len(freq)//2], np.abs(rtf_s2[:len(freq)//2]))
    plt.title('RTF Magnitude (Source 2)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    
    plt.subplot(212)
    plt.plot(freq[:len(freq)//2], np.angle(rtf_s2[:len(freq)//2]))
    plt.title('RTF Phase (Source 2)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (rad)')
    
    plt.tight_layout()
    plt.show()



# ==================================================================
# PART B: Crosstalk Cancellation (Two Sources)
# ==================================================================
print("\nPart B: Crosstalk Cancellation (Two Sources)")
x1_total = mic1_s1 + mic1_s2  # Signal at Mic 1: S1 + S2 contributions
x2_total = mic2_s1 + mic2_s2  # Signal at Mic 2: S1 + S2 contributions




