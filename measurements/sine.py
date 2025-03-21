# import numpy as np
# import rir_generator  # Correct import as module name
# from scipy import signal as si
# import matplotlib.pyplot as plt

# def sineSweep(fmin, fmax, duration, sampling_rate, peak):
#     k = np.exp(np.log(fmax / fmin) / duration)
#     data_len = int(duration * sampling_rate)  # Ensure it's an integer
#     sinsw = np.zeros(data_len)  # array of output 
#     dt = 1.0 / sampling_rate  # time between samples
#     t = 0.0  # start time
#     p = 2 * np.pi * fmin / np.log(k)
#     for i in range(data_len):
#         sinsw[i] = peak * np.sin(p * (pow(k, t) - 1))
#         t += dt
#     return sinsw

# def inverseFilt(sinsw, fmin, fmax):
#     length = len(sinsw)
#     frac = 1
#     k = 1 / (np.exp(np.log(fmax / fmin) / length))

#     invf = np.zeros(length)

#     for i in range(length):
#         invf[i] = sinsw[-i] * frac
#         frac *= k

#     return invf

# # Parameters
# f1 = 20
# f2 = 20000
# fs = 48000
# duration = 1  # seconds

# # Generate sine sweep
# sinsw = sineSweep(f1, f2, duration, fs, 1)
# # Generate inverse filter
# invf = inverseFilt(sinsw, f1, f2)

# # Room impulse response generation
# rir = rir_generator.generate(
#     c=340,                  # Sound velocity (m/s)
#     fs=fs,                  # Sample frequency (samples/s)
#     r=[1, 0.75, 0.5],       # Receiver position(s) [x y z] (m)
#     s=[1, 1.75, 1],         # Source position [x y z] (m)
#     L=[8, 8, 3],            # Room dimensions [x y z] (m)
#     reverberation_time=0.4, # Reverberation time (s)
#     nsample=1024,           # Number of output samples
# ).T[0]

# # Plotting the signals
# plt.subplot(2, 1, 1)
# plt.plot(sinsw)
# plt.title('Sine Sweep Signal')
# plt.subplot(2, 1, 2)
# plt.plot(invf)
# plt.title('Inverse Filter')
# plt.show()

# # Applying the RIR to the sine sweep
# output = si.lfilter(rir, 1, sinsw)
# # Convolving output with inverse filter
# ir = si.fftconvolve(output, invf, mode='same')

# # Plotting results
# plt.subplot(2, 1, 1)
# plt.plot(rir)
# plt.title('Room Impulse Response')
# plt.subplot(2, 1, 2)
# plt.plot(ir[int(len(ir)/2): int(len(ir)/2) + 1024])  # Centered part of the IR
# plt.title('Impulse Response by Inverse Filter')
# plt.show()



import numpy as np
import rir_generator  # Correct import as module name
from scipy import signal as si
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import os

def sineSweep(fmin, fmax, duration, sampling_rate, peak):
    k = np.exp(np.log(fmax / fmin) / duration)
    data_len = int(duration * sampling_rate)  # Ensure it's an integer
    sinsw = np.zeros(data_len)  # array of output 
    dt = 1.0 / sampling_rate  # time between samples
    t = 0.0  # start time
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

    return invf

def save_wav(filename, data, sample_rate):
    """Save a numpy array as a WAV file."""
    # Normalize to int16 PCM
    data_int16 = np.int16(data / np.max(np.abs(data)) * 32767)
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

# Parameters
f1 = 20
f2 = 20000
fs = 48000
duration = 1  # seconds
peak = 1.0

# Directory to save the files
save_dir = "/home/ZA/Music/Media project/BSS_MP/BSS_self/LowDelayMultichannelSourceSeparation/measurements/measured_data"
os.makedirs(save_dir, exist_ok=True)

# Generate sine sweep
sinsw = sineSweep(f1, f2, duration, fs, peak)

# Generate inverse filter
invf = inverseFilt(sinsw, f1, f2)

# Save WAV files
sine_sweep_wav = os.path.join(save_dir, "sine_sweep.wav")
inverse_sweep_wav = os.path.join(save_dir, "inverse_sweep.wav")

save_wav(sine_sweep_wav, sinsw, fs)
save_wav(inverse_sweep_wav, invf, fs)

# Save Plots
sine_sweep_plot = os.path.join(save_dir, "sine_sweep_plot.png")
inverse_sweep_plot = os.path.join(save_dir, "inverse_sweep_plot.png")

time_axis = np.arange(len(sinsw)) / fs
save_plot(sine_sweep_plot, time_axis, sinsw, 'Sine Sweep Signal', xlabel='Time (seconds)')
save_plot(inverse_sweep_plot, time_axis, invf, 'Inverse Filter', xlabel='Time (seconds)')

# Room impulse response generation
rir = rir_generator.generate(
    c=340,                  # Sound velocity (m/s)
    fs=fs,                  # Sample frequency (samples/s)
    r=[1, 0.75, 0.5],       # Receiver position(s) [x y z] (m)
    s=[1, 1.75, 1],         # Source position [x y z] (m)
    L=[8, 8, 3],            # Room dimensions [x y z] (m)
    reverberation_time=0.4, # Reverberation time (
)