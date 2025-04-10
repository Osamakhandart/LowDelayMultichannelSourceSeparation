import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.linalg import solve
from pysofaconventions import SOFAFile
# Configuration
epsilon = 1e-10  # small value to avoid division by zero
plot_results = True

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
audio, fs_audio = sf.read('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/trimmed_data_final_eval/5/tda_audio5_1.wav')
audio_left = audio[:, 0]
audio_right = audio[:, 1]
sofa = SOFAFile('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/sofa/HL_-0.25X_0.0Y.sofa', 'r')

# Load RIRs for the two sources (each file includes responses to both microphones)
rir_source1 = sf.read(path+'/ls00-left/RIR_LEFT_NORM_FLOAT.wav')[0]  # Left speaker to both mics (Source 1)
rir_source2 = sf.read(path+'/ls01-right/RIR_RIGHT_NORM_FLOAT.wav')[0]  # Right speaker to both mics (Source 2)

# Extract individual channel RIRs (these represent the transfer functions)
rir1 = sofa.getDataIR()[0, 0, :]  # RIR from the first speaker to the first microphone
rir2 = sofa.getDataIR()[0, 1, :]  # RIR from the first speaker to the second microphone
rir3 = sofa.getDataIR()[1, 0, :]  # RIR from the first speaker to the first microphone
rir4 = sofa.getDataIR()[1, 1, :] 
rir_s1_m1 = preprocess_signal(rir1)  # H_S1→M1
rir_s1_m2 = preprocess_signal(rir2)  # H_S1→M2
rir_s2_m1 = preprocess_signal(rir3)  # H_S2→M1
rir_s2_m2 = preprocess_signal(rir4)  # H_S2→M2

# rir_s1_m1 = preprocess_signal(rir_source1[:, 0])  # H_S1→M1
# rir_s1_m2 = preprocess_signal(rir_source1[:, 1])  # H_S1→M2
# rir_s2_m1 = preprocess_signal(rir_source2[:, 0])  # H_S2→M1
# rir_s2_m2 = preprocess_signal(rir_source2[:, 1])  # H_S2→M2
# --- PART A: Compute RTF for Each Source ---
# Convolve source signals with the respective RIRs to simulate microphone recordings.
mic1_s1 = scipy.signal.convolve(audio_left, rir_s1_m1, mode='same')
mic2_s1 = scipy.signal.convolve(audio_left, rir_s1_m2, mode='same')

mic1_s2 = scipy.signal.convolve(audio_right, rir_s2_m1, mode='same')
mic2_s2 = scipy.signal.convolve(audio_right, rir_s2_m2, mode='same')




# Compute time delay for verification
delay_s1 = compute_delay(mic1_s1, mic2_s1, fs_audio)
print(f"Estimated delay between mics for Source 1: {delay_s1*1000:.2f} ms")
delay_s2 = compute_delay(mic1_s2, mic2_s2, fs_audio)
print(f"Estimated delay between mics for Source 2: {delay_s2*1000:.2f} ms")

# Compute relative RTFs (these are the ratios for each source)
rtf_s1 = compute_rtf(mic1_s1, mic2_s1)
rtf_s2 = compute_rtf(mic1_s2, mic2_s2)


plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(rir_s1_m1)
plt.title('RIR: H_S1→M1')

plt.subplot(2, 2, 2)
plt.plot(rir_s1_m2)
plt.title('RIR: H_S1→M2')

plt.subplot(2, 2, 3)
plt.plot(rir_s2_m1)
plt.title('RIR: H_S2→M1')

plt.subplot(2, 2, 4)
plt.plot(rir_s2_m2)
plt.title('RIR: H_S2→M2')

plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(mic1_s1, label='Mic1 S1')
plt.plot(mic2_s1, label='Mic2 S1')
plt.title('Simulated Signals for Source 1')
plt.xlabel('Samples')
plt.legend()

plt.subplot(212)
plt.plot(mic1_s2, label='Mic1 S2')
plt.plot(mic2_s2, label='Mic2 S2')
plt.title('Simulated Signals for Source 2')
plt.xlabel('Samples')
plt.legend()
plt.tight_layout()
plt.show()




if plot_results:
    freq = np.fft.fftfreq(4096, 1/fs_audio)
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



# --- PART B: Inverse Filtering for Crosstalk Cancellation ---
print("\nPart B: Crosstalk Cancellation (Time Domain)")
# Form the overall mixed signals at each microphone
x1_total = mic1_s1 + mic1_s2   # Mic 1 receives S1 and S2 contributions.
x2_total = mic2_s1 + mic2_s2   # Mic 2 receives S1 and S2 contributions.
sf.write('mixed_mic1.wav', x1_total, fs_audio)
sf.write('mixed_mic2.wav', x2_total, fs_audio)



X1 = np.fft.fft(x1_total)
X2 = np.fft.fft(x2_total)
H_S2_M1 = np.fft.fft(rir_s2_m1, n=len(X1))
H_S1_M2 = np.fft.fft(rir_s1_m2, n=len(X2))

# Apply inverse filtering (regularized)
Y1 = X1 - (X2 * H_S2_M1.conj()) / (np.abs(H_S2_M1)**2 + epsilon)
Y2 = X2 - (X1 * H_S1_M2.conj()) / (np.abs(H_S1_M2)**2 + epsilon)

y1_hat = np.fft.ifft(Y1).real
y2_hat = np.fft.ifft(Y2).real



# --- (Optional) Normalize the outputs before saving ---
def normalize_audio(signal):
    max_val = np.max(np.abs(signal)) + epsilon
    return signal / max_val

y1_hat_norm = normalize_audio(y1_hat)
y2_hat_norm = normalize_audio(y2_hat)

# Save the processed signals to disk
sf.write('ctc_output_time_domain_mic1.wav', y1_hat_norm, fs_audio)
sf.write('ctc_output_time_domain_mic2.wav', y2_hat_norm, fs_audio)
print("Time-domain crosstalk cancellation outputs saved as 'ctc_output_time_domain_mic1.wav' and 'ctc_output_time_domain_mic2.wav'")

# --- (Optional) Plot the results ---
if plot_results:
    t_axis = np.linspace(0, len(x1_total)/fs_audio, len(x1_total))
    
    # Plot for Mic 1
    plt.figure(figsize=(12, 10))
    plt.subplot(311)
    plt.plot(t_axis, x1_total, label='Mic 1 Combined Signal')
    plt.title('Mic 1 Combined Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # plt.subplot(312)
    # plt.plot(t_axis, modeled_S2_at_M1, label='Modeled S2 Contribution at Mic 1')
    # plt.title('Modeled S2 at Mic 1 (via Convolution)')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.legend()
    
    plt.subplot(313)
    plt.plot(t_axis, y1_hat_norm, label='CTC Output for Mic 1 (S1 Isolated)')
    plt.title('Crosstalk Cancellation Output - Mic 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot for Mic 2
    plt.figure(figsize=(12, 10))
    plt.subplot(311)
    plt.plot(t_axis, x2_total, label='Mic 2 Combined Signal')
    plt.title('Mic 2 Combined Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # plt.subplot(312)
    # plt.plot(t_axis, modeled_S1_at_M2, label='Modeled S1 Contribution at Mic 2')
    # plt.title('Modeled S1 at Mic 2 (via Convolution)')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.legend()
    
    plt.subplot(313)
    plt.plot(t_axis, y2_hat_norm, label='CTC Output for Mic 2 (S2 Isolated)')
    plt.title('Crosstalk Cancellation Output - Mic 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
