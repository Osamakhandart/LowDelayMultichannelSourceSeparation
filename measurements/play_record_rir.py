import numpy as np
import sounddevice as sd
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import os

def play_and_record_rir(sweep_file, inverse_sweep_file, save_dir, speaker_channel='left', sample_rate=48000):
    """
    Plays sine sweep on one speaker and records RIR with two microphones.
    Saves both raw and normalized RIRs and plots.
    """
    sr_sweep, sine_sweep = read(sweep_file)
    sr_inv, inverse_sweep = read(inverse_sweep_file)

    assert sr_sweep == sample_rate, f"Sine sweep sample rate mismatch! {sr_sweep} != {sample_rate}"
    assert sr_inv == sample_rate, f"Inverse sweep sample rate mismatch! {sr_inv} != {sample_rate}"

    # Convert from int16 to float32 for processing
    if sine_sweep.dtype == np.int16:
        sine_sweep = sine_sweep.astype(np.float32) / 32767.0
    if inverse_sweep.dtype == np.int16:
        inverse_sweep = inverse_sweep.astype(np.float32) / 32767.0

    # Normalize Sweep Signals
    sine_sweep /= np.max(np.abs(sine_sweep))
    inverse_sweep /= np.max(np.abs(inverse_sweep))

    duration = len(sine_sweep) / sample_rate

    # Stereo Playback
    stereo_play = np.zeros((len(sine_sweep), 2))
    if speaker_channel == 'left':
        stereo_play[:, 0] = sine_sweep
        print("Playing LEFT speaker...")
    elif speaker_channel == 'right':
        stereo_play[:, 1] = sine_sweep
        print("Playing RIGHT speaker...")

    
    print(f"Recording for {duration:.2f} seconds (2 mic channels)...")
    sd.default.reset()

    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2)
    sd.play(stereo_play, samplerate=sample_rate)
    sd.wait()
    print("RIR recording complete!")

    mic1 = recording[:, 0]
    mic2 = recording[:, 1]

    # Convolving 
    rir_mic1_raw = np.convolve(mic1, inverse_sweep, mode='full')
    rir_mic2_raw = np.convolve(mic2, inverse_sweep, mode='full')

    # Normalizing 
    rir_mic1_norm = rir_mic1_raw / np.max(np.abs(rir_mic1_raw))
    rir_mic2_norm = rir_mic2_raw / np.max(np.abs(rir_mic2_raw))

    # Stack for stereo
    rir_stereo_raw = np.vstack((rir_mic1_raw, rir_mic2_raw)).T
    rir_stereo_norm = np.vstack((rir_mic1_norm, rir_mic2_norm)).T

    os.makedirs(save_dir, exist_ok=True)

    # Plot & Save Subplots 
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axs[0].plot(np.arange(len(rir_mic1_norm)) / sample_rate, rir_mic1_norm)
    axs[0].set_title('RIR Mic 1 (Normalized)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)

    axs[1].plot(np.arange(len(rir_mic2_norm)) / sample_rate, rir_mic2_norm)
    axs[1].set_title('RIR Mic 2 (Normalized)')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True)

    fig.tight_layout()
    stereo_subplot_path = os.path.join(save_dir, f"RIR_{speaker_channel.upper()}_Stereo_Subplots.png")
    plt.savefig(stereo_subplot_path)
    print(f"Saved subplot figure: {stereo_subplot_path}")
    plt.show()

    # Overlay Plot 
    time_axis = np.arange(rir_stereo_norm.shape[0]) / sample_rate
    plt.figure(figsize=(14, 5))
    plt.plot(time_axis, rir_stereo_norm[:, 0], label="Mic 1")
    plt.plot(time_axis, rir_stereo_norm[:, 1], label="Mic 2", alpha=0.7)
    plt.title(f"Stereo RIR Overlay - Speaker: {speaker_channel.upper()}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    stereo_overlay_path = os.path.join(save_dir, f"RIR_{speaker_channel.upper()}_Overlay_Plot.png")
    plt.savefig(stereo_overlay_path)
    print(f"Saved overlay figure: {stereo_overlay_path}")
    plt.show()

    # Raw RIR
    rir_raw_wav = os.path.join(save_dir, f"RIR_{speaker_channel.upper()}_RAW_FLOAT.wav")
    write(rir_raw_wav, sample_rate, rir_stereo_raw.astype(np.float32))
    print(f"Raw RIR saved: {rir_raw_wav}")

    # Normalized RIR 
    rir_norm_wav = os.path.join(save_dir, f"RIR_{speaker_channel.upper()}_NORM_FLOAT.wav")
    write(rir_norm_wav, sample_rate, rir_stereo_norm.astype(np.float32))
    print(f"Normalized RIR saved: {rir_norm_wav}")

sweep_file_path = "/home/ZA/Music/Media project/BSS_MP/BSS_self/LowDelayMultichannelSourceSeparation/measurements/measured_data/sweeps/sine_sweep.wav"
inverse_sweep_file_path = "/home/ZA/Music/Media project/BSS_MP/BSS_self/LowDelayMultichannelSourceSeparation/measurements/measured_data/sweeps/inverse_sweep.wav"

save_rir_directory = "/home/ZA/Music/Media project/BSS_MP/BSS_self/LowDelayMultichannelSourceSeparation/measurements/measured_data/RIRrecordings/ls01 - right/"

speaker_channel = 'right'  # Run once for 'left', then 'right' separately
sample_rate = 48000

play_and_record_rir(
    sweep_file=sweep_file_path,
    inverse_sweep_file=inverse_sweep_file_path,
    save_dir=save_rir_directory,
    speaker_channel=speaker_channel,
    sample_rate=sample_rate
)
