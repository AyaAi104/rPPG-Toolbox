"""
Granular Evaluation by Motion Level for Aya Dataset.

Usage:
    python evaluate_by_motion.py ^
        --test_output  "runs/exp/.../saved_test_outputs/Aya_Aya_TSCAN_1m_v2_outputs.pickle" ^
        --cached_path  "D:/PreprocessedData/Aya_v2_test/Aya_SizeW72_..." ^
        --fs 50
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np
from scipy.stats import pearsonr


def _calculate_hr_fft(bvp_signal, fs):
    """Estimate heart rate from BVP signal using FFT."""
    # Ensure numpy array and flatten
    bvp_signal = np.array(bvp_signal).flatten().astype(np.float64)

    if len(bvp_signal) < 30:
        return 0.0

    N = len(bvp_signal)
    fft_result = np.fft.rfft(bvp_signal)
    fft_mag = np.abs(fft_result)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Both fft_mag and freqs now guaranteed same length
    valid = (freqs >= 0.65) & (freqs <= 3.33)
    if not np.any(valid):
        return 0.0

    peak_freq = freqs[valid][np.argmax(fft_mag[valid])]
    return peak_freq * 60.0  # Hz -> BPM


def evaluate_by_motion(test_output_path, cached_path, fs=50):
    """Main evaluation function."""

    # ---- 1. Load test outputs ----
    print(f"Loading test outputs from: {test_output_path}")
    with open(test_output_path, 'rb') as f:
        data = pickle.load(f)

    predictions = data['predictions']
    labels = data['labels']

    # ---- 2. Load motion metadata ----
    print(f"Loading motion metadata from: {cached_path}")
    chunk_motion_map = {}
    meta_files = glob.glob(os.path.join(cached_path, "*_motion_meta.json"))
    for mf in meta_files:
        with open(mf, 'r') as f:
            chunk_motion_map.update(json.load(f))

    print(f"  Motion map entries: {len(chunk_motion_map)}")

    # ---- 3. Group by motion level ----
    motion_hr_preds = {}
    motion_hr_labels = {}

    for filename, chunks in predictions.items():
        for chunk_id, pred_signal in sorted(chunks.items()):
            chunk_id_int = int(chunk_id)
            label_signal = labels[filename][chunk_id_int]

            # Look up motion level
            key = f"{filename}_input{chunk_id_int}.npy"
            motion = chunk_motion_map.get(key, "Unknown")

            # Compute HR via FFT
            hr_pred = _calculate_hr_fft(pred_signal, fs)
            hr_label = _calculate_hr_fft(label_signal, fs)

            if motion not in motion_hr_preds:
                motion_hr_preds[motion] = []
                motion_hr_labels[motion] = []
            motion_hr_preds[motion].append(hr_pred)
            motion_hr_labels[motion].append(hr_label)

    # ---- 4. Compute metrics per motion level ----
    print("\n" + "=" * 70)
    print("GRANULAR EVALUATION BY MOTION LEVEL")
    print("=" * 70)

    all_preds = []
    all_labels = []

    for motion in sorted(motion_hr_preds.keys()):
        preds = np.array(motion_hr_preds[motion])
        labs = np.array(motion_hr_labels[motion])

        all_preds.extend(preds)
        all_labels.extend(labs)

        n = len(preds)
        mae = np.mean(np.abs(preds - labs))
        rmse = np.sqrt(np.mean((preds - labs) ** 2))

        if n > 2 and np.std(preds) > 0 and np.std(labs) > 0:
            r, _ = pearsonr(preds, labs)
        else:
            r = float('nan')

        mape = np.mean(np.abs((preds - labs) / (labs + 1e-8))) * 100

        print(f"\n  Motion: {motion:15s} | N={n:4d} samples")
        print(f"    MAE:     {mae:.3f} BPM")
        print(f"    RMSE:    {rmse:.3f} BPM")
        print(f"    MAPE:    {mape:.2f} %")
        print(f"    Pearson: {r:.4f}")

    # Overall
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall_mae = np.mean(np.abs(all_preds - all_labels))
    overall_rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    if len(all_preds) > 2:
        overall_r, _ = pearsonr(all_preds, all_labels)
    else:
        overall_r = float('nan')

    print(f"\n{'=' * 70}")
    print(f"  OVERALL          | N={len(all_preds):4d} samples")
    print(f"    MAE:     {overall_mae:.3f} BPM")
    print(f"    RMSE:    {overall_rmse:.3f} BPM")
    print(f"    Pearson: {overall_r:.4f}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate rPPG by motion level")
    parser.add_argument("--test_output", type=str, required=True,
                        help="Path to *_outputs.pickle from rPPG-Toolbox test")
    parser.add_argument("--cached_path", type=str, required=True,
                        help="Path to test CACHED_PATH (contains *_motion_meta.json)")
    parser.add_argument("--fs", type=float, default=50,
                        help="Sampling rate of the dataset")
    args = parser.parse_args()

    evaluate_by_motion(args.test_output, args.cached_path, args.fs)