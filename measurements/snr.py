import numpy as np
import soundfile as sf
from mir_eval.separation import bss_eval_sources

# Load ground truth references
ref0, _ = sf.read("/Users/usamakhan/Documents/original/LowDelayMultichannelSourceSeparation_Random-Directions_Demo/espeakwav_16.wav")
ref1, _ = sf.read("/Users/usamakhan/Documents/original/LowDelayMultichannelSourceSeparation_Random-Directions_Demo/pinkish16.wav")

# Ensure same length
min_len = min(len(ref0), len(ref1))
refs = np.vstack([ref0[:min_len], ref1[:min_len]])

def evaluate_sep(path, label):
    est, _ = sf.read(path)
    est = est[:min_len, :].T  # shape (2, samples)
    sdr, sir, sar, perm = bss_eval_sources(refs, est)
    print(f"\n===== {label} =====")
    print("SDR:", sdr)
    print("SIR:", sir)
    print("SAR:", sar)
    print("Permutation:", perm)
    return np.mean(sdr)

# Paths to your separated output WAVs
# Make sure these files exist
sdr_rtf = evaluate_sep("withRtf.wav", "RTF-based Separation")
sdr_opt = evaluate_sep("sepchan_randdir_online.wav", "Optimized Separation")

print("\n📈 SDR Comparison:")
if sdr_rtf > sdr_opt:
    print("✅ RTF-based performed better.")
else:
    print("✅ Optimized (random init + training) performed better.")
