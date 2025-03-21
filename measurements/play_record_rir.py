import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import os

def generate_sine_sweep(duration, sample_rate, f0, f1):
    """
    Generate a sine sweep (chirp) signal.
    
    Parameters:
        duration (float): Duration of the sweep in seconds.
        sample_rate (int): Sampling rate in Hz.
        f0 (float): Start frequency of the sweep in Hz.
        f1 (float): End frequency of the sweep in Hz.
    
    Returns:
        sweep (np.array): Sine sweep signal.
        t (np.array): Time axis.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    phase = 2 * np.pi * f0 * duration / np.log(f1 / f0) * (np.exp(t * np.log(f1 / f0) / duration) - 1)
    sweep = np.sin(phase)
    return sweep, t

# Parameters for the sine sweep
duration = 1.5      # Duration of the sweep in seconds
sample_rate = 44100 # Sampling rate in Hz
f0 = 20             # Start frequency in Hz
f1 = 20000          # End frequency in Hz

# Generate the sine sweep
sine_sweep, time_axis = generate_sine_sweep(duration, sample_rate, f0, f1)

# Normalize the sweep to prevent clipping
sine_sweep = sine_sweep / np.max(np.abs(sine_sweep))

# Generate the inverse sweep (time-reversed version of the sweep)
inverse_sweep = sine_sweep[::-1]  # Reverse the array

# Directory to save the files
save_dir = "/home/ZA/Music/Media project/BSS_MP/BSS_self/LowDelayMultichannelSourceSeparation/measurements/measured_data"

# Ensure directory exists
os.makedirs(save_dir, exist_ok=True)

# File paths
sine_sweep_filename = os.path.join(save_dir, "sine_sweep.wav")
inverse_sweep_filename = os.path.join(save_dir, "inverse_sweep.wav")
sine_sweep_plot_filename = os.path.join(save_dir, "sine_sweep_plot.png")
inverse_sweep_plot_filename = os.path.join(save_dir, "inverse_sweep_plot.png")

# Save the sine sweep as a WAV file (16-bit PCM)
write(sine_sweep_filename, sample_rate, (sine_sweep * 32767).astype(np.int16))
print(f"✅ Sine sweep saved at: {sine_sweep_filename}")

# Save the inverse sine sweep as a WAV file (16-bit PCM)
write(inverse_sweep_filename, sample_rate, (inverse_sweep * 32767).astype(np.int16))
print(f"✅ Inverse sine sweep saved at: {inverse_sweep_filename}")

# Plot and save the sine sweep waveform
plt.figure(figsize=(14, 5))
plt.plot(time_axis, sine_sweep)
plt.title('Sine Sweep Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.savefig(sine_sweep_plot_filename)
plt.close()
print(f"✅ Sine sweep plot saved at: {sine_sweep_plot_filename}")

# Plot and save the inverse sine sweep waveform
plt.figure(figsize=(14, 5))
plt.plot(time_axis, inverse_sweep)
plt.title('Inverse Sine Sweep Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.savefig(inverse_sweep_plot_filename)
plt.close()
print(f"✅ Inverse sine sweep plot saved at: {inverse_sweep_plot_filename}")

# OPTIONAL: Uncomment if you want to display the plots interactively
# plt.show()
