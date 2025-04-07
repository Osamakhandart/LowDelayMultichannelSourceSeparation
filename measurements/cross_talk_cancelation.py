import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt

# Step 1: Load the stereo Room Impulse Responses (RIRs) from each speaker
rir_speaker1, fs = sf.read('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/measured_data/RIRrecordings/ls01-left01mic60/RIR_LEFT_NORM_FLOAT.wav')  # RIR when playing from Speaker 1
rir_speaker2, fs = sf.read('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/measured_data/RIRrecordings/ls02-left02mic60/RIR_LEFT_NORM_FLOAT.wav')  # RIR when playing from Speaker 2
stereo_audio, fs = sf.read('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/trimmed_data_final_eval/4/tda_audio4_1.wav')

mic1_audio = stereo_audio[:, 0]
mic2_audio = stereo_audio[:, 1]

# Step 4: Preprocess the signals
# Remove DC offset and normalize the signals
mic1_audio -= np.mean(mic1_audio)
mic2_audio -= np.mean(mic2_audio)
mic1_audio /= np.linalg.norm(mic1_audio)
mic2_audio /= np.linalg.norm(mic2_audio)


rir1_mic1 = rir_speaker1[:, 0]
rir1_mic2 = rir_speaker1[:, 1]
rir2_mic1 = rir_speaker2[:, 0]
rir2_mic2 = rir_speaker2[:, 1]

# Placeholder for transfer functions, which would typically be measured or estimated
H_S1_M1 = np.fft.fft(rir1_mic1)
H_S2_M1 = np.fft.fft(rir2_mic1)
H_S1_M2 = np.fft.fft(rir1_mic2)
H_S2_M2 = np.fft.fft(rir2_mic2)
rir_length = len(rir2_mic1)
signal_length = len(mic2_audio)

# If the RIR is shorter than the signal, pad it
if rir_length < signal_length:
    padded_rir2_mic1 = np.pad(rir2_mic1, (0, signal_length - rir_length), mode='constant', constant_values=(0))
else:
    padded_rir2_mic1 = rir2_mic1

# Then calculate the FFT of the padded RIR
H_S2_M1 = np.fft.fft(padded_rir2_mic1)




# Length of the recorded signal from Microphone 1
signal_length_X1 = len(mic1_audio)

# Length of the impulse response from Source 1 to Mic 2
rir_length_H_S1_M2 = len(rir1_mic2)

# Pad the shorter array to match the length of the longer one
if rir_length_H_S1_M2 < signal_length_X1:
    padded_rir1_mic2 = np.pad(rir1_mic2, (0, signal_length_X1 - rir_length_H_S1_M2), mode='constant', constant_values=(0))
else:
    padded_rir1_mic2 = rir1_mic2

# Compute the FFT of the padded RIR
H_S1_M2 = np.fft.fft(padded_rir1_mic2)


signal_length_X1 = len(mic1_audio)

# Length of the impulse response from Source 1 to Mic 2
rir_length_H_S1_M2 = len(rir1_mic2)

# Pad the shorter array to match the length of the longer one
if rir_length_H_S1_M2 < signal_length_X1:
    padded_rir1_mic2 = np.pad(rir1_mic2, (0, signal_length_X1 - rir_length_H_S1_M2), mode='constant', constant_values=(0))
else:
    padded_rir1_mic2 = rir1_mic2

# Compute the FFT of the padded RIR
# Calculate the inverse filters using the pseudo-inverse or other methods
H_S2_M1_inv = np.conj(H_S2_M1) / (np.abs(H_S2_M1)**2 + 1e-10)  # Regularized inverse
H_S1_M2_inv = np.conj(H_S1_M2) / (np.abs(H_S1_M2)**2 + 1e-10)  # Regularized inverse

# Applying the filters: this involves convolution in the time domain
# First, calculate the convolution in frequency domain for efficiency
X1 = np.fft.fft(mic1_audio)
X2 = np.fft.fft(mic2_audio)


# Apply inverse filtering
S2_contrib_M1 = np.fft.ifft(H_S2_M1_inv * X2).real  # Contribution of S2 to M1
S1_contrib_M2 = np.fft.ifft(H_S1_M2_inv * X1).real  # Contribution of S1 to M2

# Subtract the contributions
y1_hat = mic1_audio - S2_contrib_M1  # Output at Mic1 with S2 canceled
y2_hat = mic2_audio - S1_contrib_M2  # Output at Mic2 with S1 canceled
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(mic1_audio, label='Original Mic 1')
plt.plot(y1_hat, label='Crosstalk Removed Mic 1', linestyle='--')
plt.title('Mic 1: Original vs Crosstalk Canceled')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(mic2_audio, label='Original Mic 2')
plt.plot(y2_hat, label='Crosstalk Removed Mic 2', linestyle='--')
plt.title('Mic 2: Original vs Crosstalk Canceled')
plt.legend()

plt.tight_layout()
plt.show()
