from pysofaconventions import SOFAFile
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import soundfile as sf

# Load the SOFA file
sofa = SOFAFile('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/sofa/HL_-0.25X_0.0Y.sofa', 'r')



# Extract RIRs
rir1 = sofa.getDataIR()[0, 0, :]  # RIR from the first speaker to the first microphone
rir2 = sofa.getDataIR()[0, 1, :]  # RIR from the first speaker to the second microphone



 #Preprocess the signals
rir1 -= np.mean(rir1)
rir2 -= np.mean(rir2)
rir1 /= np.linalg.norm(rir1)
rir2 /= np.linalg.norm(rir2)



#cross-correlation & Estimate the time delay
cross_corr = scipy.signal.correlate(rir2, rir1, mode='full')
lags = np.arange(-len(rir2) + 1, len(rir1))
time_delay = lags[np.argmax(cross_corr)]
print(f"Estimated time delay between the two RIR paths: {time_delay} samples")


#Perform spectral analysis 
X1 = np.fft.fft(rir1)
X2 = np.fft.fft(rir2)

# Room Transfer Function
H12 = X2 / X1  




#calculating magnitude and phase
magnitude = np.abs(H12)
phase = np.angle(H12)
frequency_axis = np.fft.fftfreq(len(rir1), 1 / sofa.getSamplingRate())
positive_indices = frequency_axis > 0
frequency_axis = frequency_axis[positive_indices]
magnitude = magnitude[positive_indices]
phase = phase[positive_indices]




#Inverse Filtering
epsilon = 1e-10  # Small constant to avoid division by zero
H12_inv = 1 / (H12 + epsilon)
h12_inv_time = np.fft.ifft(H12_inv).real  # Take the real part
recovered_signal = scipy.signal.convolve(rir2, h12_inv_time, mode='same')







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
plt.plot(recovered_signal, label='Recovered Signal', linestyle='--')
plt.title('Comparison of Original and Recovered Signals')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
