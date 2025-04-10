import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.linalg import solve
from pysofaconventions import SOFAFile

# Configuration
epsilon = 1e-10  # small value to avoid division by zero
plot_results = True
sofa = SOFAFile('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/sofa/HL_-0.25X_0.0Y.sofa', 'r')

# Extract RIRs
rir1 = sofa.getDataIR()[0, 0, :]  # RIR from the first speaker to the first microphone
rir2 = sofa.getDataIR()[0, 1, :]  # RIR from the first speaker to the second microphone
rir3 = sofa.getDataIR()[1, 0, :]  # RIR from the first speaker to the first microphone
rir4 = sofa.getDataIR()[1, 1, :] 

 #Preprocess the signals
rir1 -= np.mean(rir1)
rir2 -= np.mean(rir2)
rir1 /= np.linalg.norm(rir1)
rir2 /= np.linalg.norm(rir2)


def preprocess_signal(signal):
    """Remove DC offset and normalize signal."""
    signal = signal - np.mean(signal)
    return signal / (np.linalg.norm(signal) + epsilon)

def compute_rtf(x1, x2, n_fft=4096):
    """Compute RTF as X2(f)/X1(f)."""
    X1 = np.fft.fft(x1, n=n_fft)
    X2 = np.fft.fft(x2, n=n_fft)
    return X2 / (X1 + epsilon)

def compute_delay(x1, x2, fs):
    """Compute time delay between two signals using cross-correlation."""
    corr = scipy.signal.correlate(x1, x2, mode='full')
    lags = scipy.signal.correlation_lags(len(x1), len(x2), mode='full')
    delay_samples = lags[np.argmax(corr)]
    return delay_samples / fs

# --- LOAD DATA ---
path = "/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/new/rirs"
audio, fs_audio = sf.read('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/trimmed_data_final_eval/2/tda_audio2_1.wav')
audio_left = audio[:, 0]
audio_right = audio[:, 1]
# Load RIRs for the two sources (each file includes responses to both microphones)
rir_source1 = sf.read(path+'/ls00-left/RIR_LEFT_NORM_FLOAT.wav')[0]  # Left speaker to both mics (Source 1)
rir_source2 = sf.read(path+'/ls01-right/RIR_RIGHT_NORM_FLOAT.wav')[0]  # Right speaker to both mics (Source 2)

# Extract individual channel RIRs (these represent the transfer functions)

rir_s1_m1 = preprocess_signal(rir1)  # H_S1→M1
rir_s1_m2 = preprocess_signal(rir2)  # H_S1→M2
rir_s2_m1 = preprocess_signal(rir3)  # H_S2→M1
rir_s2_m2 = preprocess_signal(rir4)  # H_S2→M2

# --- PART A: Compute RTF for Each Source ---
# Convolve source signals with the respective RIRs to simulate microphone recordings.
mic1_s1 = scipy.signal.convolve(audio_left, rir_s1_m1, mode='same')
mic2_s1 = scipy.signal.convolve(audio_left, rir_s1_m2, mode='same')

mic1_s2 = scipy.signal.convolve(audio_right, rir_s2_m1, mode='same')
mic2_s2 = scipy.signal.convolve(audio_right, rir_s2_m2, mode='same')



sf.write('mixed_mic1.wav', mic1_s1, fs_audio)
sf.write('mixed_mic2.wav', mic2_s2, fs_audio)