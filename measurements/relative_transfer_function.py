


import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt

# Step 1: Load the stereo Room Impulse Responses (RIRs) from each speaker
rir_speaker1, fs = sf.read('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/measured_data/RIRrecordings/ls01-left01mic60/RIR_LEFT_NORM_FLOAT.wav')  # RIR when playing from Speaker 1
rir_speaker2, fs = sf.read('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/measured_data/RIRrecordings/ls02-left02mic60/RIR_LEFT_NORM_FLOAT.wav')  # RIR when playing from Speaker 2
stereo_audio, fs = sf.read('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/trimmed_data_final_eval/4/tda_audio4_1.wav')

#Step 2: Separate channels for each RIR
rir1_mic1 = rir_speaker1[:, 0]
rir1_mic2 = rir_speaker1[:, 1]
rir2_mic1 = rir_speaker2[:, 0]
rir2_mic2 = rir_speaker2[:, 1]



# Step 3: Assuming left channel is Microphone 1, right channel is Microphone 2
mic1_audio = stereo_audio[:, 0]
mic2_audio = stereo_audio[:, 1]

# Step 4: Preprocess the signals
# Remove DC offset and normalize the signals
mic1_audio -= np.mean(mic1_audio)
mic2_audio -= np.mean(mic2_audio)
mic1_audio /= np.linalg.norm(mic1_audio)
mic2_audio /= np.linalg.norm(mic2_audio)

# Step 5: Compute cross-correlation and estimate delay
cross_corr = scipy.signal.correlate(mic2_audio, mic1_audio, mode='full')
lags = np.arange(-len(mic2_audio)+1, len(mic1_audio))
time_delay = lags[np.argmax(cross_corr)]
print(f"Estimated time delay between Mic 1 and Mic 2: {time_delay} samples")

# Step 6: Fourier Transform for spectral analysis
X1 = np.fft.fft(mic1_audio)
X2 = np.fft.fft(mic2_audio)

# Step 7: Calculate the Relative Transfer Function (RTF)
H12 = X2 / X1
magnitude = np.abs(H12)
phase = np.angle(H12)

# Generate the frequency axis
frequency_axis = np.fft.fftfreq(len(mic1_audio), 1 / fs)

# Filter to keep only positive frequencies up to the Nyquist frequency
positive_indices = frequency_axis > 0
frequency_axis = frequency_axis[positive_indices]
magnitude = magnitude[positive_indices]
phase = phase[positive_indices]



#Step 8 inverse Filtering

# Compute the inverse of the RTF, adding a small constant to avoid division by zero
epsilon = 1e-10  # A small number to prevent division by zero issues
H12_inv = 1 / (H12 + epsilon)

# Apply inverse filtering to recover the original signal at Microphone 1 from Microphone 2
# Transform H12_inv back to time domain to apply it as a filter
h12_inv_time = np.fft.ifft(H12_inv).real  # Take the real part since the inverse FFT might introduce a small imaginary component due to numerical errors

# Assuming the signal is long enough, or using zero-padding if necessary
recovered_signal = scipy.signal.convolve(mic2_audio, h12_inv_time, mode='same')



# Step 8: Plotting
plt.figure(figsize=(12, 8))

# Subplot for the magnitude of RTF
plt.subplot(2, 1, 1)
plt.plot(frequency_axis, magnitude)
plt.title('Magnitude of RTF')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# Subplot for the phase of RTF
plt.subplot(2, 1, 2)
plt.plot(frequency_axis, phase)
plt.title('Phase of RTF')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (Radians)')
plt.grid(True)

plt.tight_layout()
plt.show()


# Plotting the original and recovered signals for comparison
plt.figure(figsize=(12, 6))
plt.plot(mic1_audio, label='Original Signal from Mic 1')
plt.plot(recovered_signal, label='Recovered Signal', linestyle='--')
plt.title('Comparison of Original and Recovered Signals')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.show()