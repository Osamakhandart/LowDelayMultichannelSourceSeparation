import numpy as np
import rir_generator  # Correct import as module name
from scipy import signal as si
import matplotlib.pyplot as plt

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

# Parameters
f1 = 20
f2 = 20000
fs = 48000
duration = 1  # seconds

# Generate sine sweep
sinsw = sineSweep(f1, f2, duration, fs, 1)
# Generate inverse filter
invf = inverseFilt(sinsw, f1, f2)

# Room impulse response generation
rir = rir_generator.generate(
    c=340,                  # Sound velocity (m/s)
    fs=fs,                  # Sample frequency (samples/s)
    r=[1, 0.75, 0.5],       # Receiver position(s) [x y z] (m)
    s=[1, 1.75, 1],         # Source position [x y z] (m)
    L=[8, 8, 3],            # Room dimensions [x y z] (m)
    reverberation_time=0.4, # Reverberation time (s)
    nsample=1024,           # Number of output samples
).T[0]

# Plotting the signals
plt.subplot(2, 1, 1)
plt.plot(sinsw)
plt.title('Sine Sweep Signal')
plt.subplot(2, 1, 2)
plt.plot(invf)
plt.title('Inverse Filter')
plt.show()

# Applying the RIR to the sine sweep
output = si.lfilter(rir, 1, sinsw)
# Convolving output with inverse filter
ir = si.fftconvolve(output, invf, mode='same')

# Plotting results
plt.subplot(2, 1, 1)
plt.plot(rir)
plt.title('Room Impulse Response')
plt.subplot(2, 1, 2)
plt.plot(ir[int(len(ir)/2): int(len(ir)/2) + 1024])  # Centered part of the IR
plt.title('Impulse Response by Inverse Filter')
plt.show()