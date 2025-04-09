import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import soundfile as sf

# Load the RIRs from WAV files
rir1, fs1 = sf.read('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/new/rirs/ls00-left/RIR_LEFT_RAW_FLOAT.wav')  # Update with actual file path
rir2, fs2 = sf.read('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/new/rirs/ls01-right/RIR_RIGHT_RAW_FLOAT.wav')  # Update with actual file path

# Ensure both RIRs have the same sampling rate
assert fs1 == fs2, "Sampling rates do not match."


rir1 = rir1[:, 0]
rir2 = rir2[:, 0]
# Preprocess the signals: remove DC offset and normalize
rir1 -= np.mean(rir1)
rir2 -= np.mean(rir2)
rir1 /= np.linalg.norm(rir1)
rir2 /= np.linalg.norm(rir2)

# Cross-correlation & Estimate the time delay
# Cross-correlation & Estimate the time delay
cross_corr = scipy.signal.correlate(rir2, rir1, mode='full')
lags = np.arange(-len(rir2) + 1, len(rir1))

# Ensure lags covers all indices in cross_corr
assert len(lags) == len(cross_corr), "Length of lags does not match cross-correlation length."

time_delay = lags[np.argmax(cross_corr)]
print(f"Estimated time delay between the two RIR paths: {time_delay} samples")

# Perform spectral analysis
X1 = np.fft.fft(rir1)
X2 = np.fft.fft(rir2)

# Room Transfer Function
H12 = X2 / X1

# Calculating magnitude and phase
magnitude = np.abs(H12)
phase = np.angle(H12)
frequency_axis = np.fft.fftfreq(len(rir1), 1 / fs1)  # Use the sampling rate from the loaded RIRs
positive_indices = frequency_axis > 0
frequency_axis = frequency_axis[positive_indices]
magnitude = magnitude[positive_indices]
phase = phase[positive_indices]

# Inverse Filtering
epsilon = 1e-10  # Small constant to avoid division by zero
H12_inv = 1 / (H12 + epsilon)
h12_inv_time = np.fft.ifft(H12_inv).real  # Take the real part
recovered_signal = scipy.signal.convolve(rir2, h12_inv_time, mode='same')

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(frequency_axis, magnitude)
plt.title('Magnitude of RTF')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.plot(frequency_axis, phase)
plt.title('Phase of RTF')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (Radians)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(rir1, label='Original Signal at Mic 1', alpha=0.7)
plt.plot(rir2, label='Original Signal at Mic 2', alpha=0.7)
plt.axvline(x=time_delay, color='r', linestyle='--', label='Estimated Delay')
plt.plot(recovered_signal, label='Recovered Signal', linestyle='--')
plt.title('Comparison of Original and Recovered Signals')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
