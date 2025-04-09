from pysofaconventions import SOFAFile
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Load the SOFA file
sofa_path = '/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/sofa/HL_-0.25X_0.0Y.sofa'
sofa = SOFAFile(sofa_path, 'r')

# Extract RIRs and zero-pad them to match the length of the source audio
source_audio, fs = sf.read('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/new/stereoautomated_script_outputs/RiR_data/stereo_output_3.wav')
S1 = source_audio[:, 0]  # Left channel as Source 1
S2 = source_audio[:, 1]  # Right channel as Source 2
len_source = len(S1)  # Length of the source signals
rirS1_M1 = sofa.getDataIR()[0, 0, :]  # first speaker first microphone
rirS2_M1 = sofa.getDataIR()[0, 1, :]  # first speaker second microphone
rirS1_M2 = sofa.getDataIR()[1, 0, :]  # second speaker first microphone
rirS2_M2 = sofa.getDataIR()[1, 1, :] 

# Extract and pad RIRs
rirS1_M1 = np.pad(sofa.getDataIR()[0, 0, :], (0, len_source - len(rirS1_M1)), 'constant')
rirS2_M1 = np.pad(sofa.getDataIR()[0, 1, :], (0, len_source - len(rirS2_M1)), 'constant')
rirS1_M2 = np.pad(sofa.getDataIR()[1, 0, :], (0, len_source - len(rirS1_M2)), 'constant')
rirS2_M2 = np.pad(sofa.getDataIR()[1, 1, :], (0, len_source - len(rirS2_M2)), 'constant')

# Perform FFT on both the padded RIRs and the source signals
HS1_M1 = np.fft.fft(rirS1_M1)
HS2_M1 = np.fft.fft(rirS2_M1)
HS1_M2 = np.fft.fft(rirS1_M2)
HS2_M2 = np.fft.fft(rirS2_M2)

# Simulate microphone recordings by convolving the source with the RTF
x1 = np.fft.ifft(np.fft.fft(S1) * HS1_M1 + np.fft.fft(S2) * HS2_M1).real  # Signal at Mic 1
x2 = np.fft.ifft(np.fft.fft(S1) * HS1_M2 + np.fft.fft(S2) * HS2_M2).real  # Signal at Mic 2




# Compute inverse filters based on the RTFs
# Use regularized inversion instead of simple reciprocal
epsilon = 1e-3  # Regularization parameter
HS2_M1_inv = np.conj(HS2_M1) / (np.abs(HS2_M1)**2 + epsilon)
HS1_M2_inv = np.conj(HS1_M2) / (np.abs(HS1_M2)**2 + epsilon) # Inverse filter to cancel S1 at Mic 2





# Apply inverse filtering to cancel out S2 from Mic 1 and S1 from Mic 2
# Correct CTC implementation based on your equations
y1_hat = np.fft.ifft(np.fft.fft(x1) - HS2_M1 * np.fft.fft(x2) * HS1_M2_inv).real
y2_hat = np.fft.ifft(np.fft.fft(x2) - HS1_M2 * np.fft.fft(x1) * HS2_M1_inv).real
# y1_hat = y1_hat / np.max(np.abs(y1_hat))
# y2_hat = y2_hat / np.max(np.abs(y2_hat))







import soundfile as sf

# Save the original and processed signals for Mic 1
sf.write('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/cross_talk_output/original_mic1.wav', x1, fs)
sf.write('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/cross_talk_output/processed_mic1.wav', y1_hat, fs)

# Save the original and processed signals for Mic 2
sf.write('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/cross_talk_output/original_mic2.wav', x2, fs)
sf.write('/Users/usamakhan/Documents/project/LowDelayMultichannelSourceSeparation/measurements/cross_talk_output/processed_mic2.wav', y2_hat, fs)

import matplotlib.pyplot as plt

# Plot the original and processed signals
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(x1, label='Original Mic 1 Signal')
plt.plot(y1_hat, label='Processed Signal at Mic 1', linestyle='--')
plt.title('Mic 1: Original vs. Crosstalk-Cancelled Signal')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x2, label='Original Mic 2 Signal')
plt.plot(y2_hat, label='Processed Signal at Mic 2', linestyle='--')
plt.title('Mic 2: Original vs. Crosstalk-Cancelled Signal')
plt.legend()

plt.tight_layout()
plt.show()

