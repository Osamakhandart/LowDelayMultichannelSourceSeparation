import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import os

def sineSweep(fmin, fmax, duration, sampling_rate, peak):
    k = np.exp(np.log(fmax / fmin) / duration)
    data_len = int(duration * sampling_rate)
    sinsw = np.zeros(data_len)
    dt = 1.0 / sampling_rate
    t = 0.0
    p = 2 * np.pi * fmin / np.log(k)

    for i in range(data_len):
        sinsw[i] = peak * np.sin(p * (pow(k, t) - 1))
        t += dt

    return sinsw
def inverseFilt(sinsw, fmin, fmax):
    length = len(sinsw)
    frac = 1
    k = 1 / (np.exp(np.log(fmax / fmin) / length))

    invf = np.zeros(length)

    for i in range(length):
        invf[i] = sinsw[-i] * frac
        frac *= k

    # Normalize the inverse filter to prevent scaling issues
    # invf /= np.max(np.abs(invf))

    return invf

def save_wav(filename, data, sample_rate):
    """Save a numpy array as a WAV file without normalization."""
    # Convert to int16 without normalization
    data_int16 = (data * 32767).astype(np.int16)
    write(filename, sample_rate, data_int16)
    print(f"✅ WAV file saved: {filename}")

def save_plot(filename, time_axis, signal, title, xlabel='Time (samples)', ylabel='Amplitude'):
    """Save a plot as a PNG file."""
    plt.figure(figsize=(14, 5))
    plt.plot(time_axis, signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Plot saved: {filename}")

f1 = 20                 # Start frequency in Hz
f2 = 20000              # End frequency in Hz
fs = 48000              # Sampling rate in Hz
duration = 1.6          # Duration in seconds
peak = 1           # Peak amplitude

save_dir = "/home/ZA/Music/Media project/BSS_MP/BSS_self/LowDelayMultichannelSourceSeparation/measurements/measured_data/sweeps/"
os.makedirs(save_dir, exist_ok=True)

sinsw = sineSweep(f1, f2, duration, fs, peak)
invf = inverseFilt(sinsw, f1, f2)

sine_sweep_wav = os.path.join(save_dir, "sine_sweep.wav")
inverse_sweep_wav = os.path.join(save_dir, "inverse_sweep.wav")

save_wav(sine_sweep_wav, sinsw, fs)
save_wav(inverse_sweep_wav, invf, fs)

time_axis = np.arange(len(sinsw)) / fs

sine_sweep_plot = os.path.join(save_dir, "sine_sweep_plot.png")
inverse_sweep_plot = os.path.join(save_dir, "inverse_sweep_plot.png")

save_plot(sine_sweep_plot, time_axis, sinsw, 'Sine Sweep Signal', xlabel='Time (seconds)')
save_plot(inverse_sweep_plot, time_axis, invf, 'Inverse Filter', xlabel='Time (seconds)')

impulse = np.convolve(sinsw, invf, mode='full')

plt.figure(figsize=(12, 4))
plt.plot(impulse)
plt.title('Impulse from Sweep * Inverse Sweep (Should Look Like a Spike)')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

print("Sweep and Inverse Sweep generation complete.")

