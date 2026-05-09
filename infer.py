"""
Usage:
python infer.py --model_path ./premodel.pth --input_path ./frames_1764806710387 --output_path ./outputs/ --fps 50 --num_chunks_per_batch 10
"""

# Fix OpenMP duplicate library error on Windows
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from my_models.my_tscan import *


# ==============================================================================
# Preprocessing
# ==============================================================================

def diff_normalize_frames(frames: np.ndarray) -> np.ndarray:
    diff_frames = np.diff(frames, axis=0)
    normalized = np.zeros_like(diff_frames)
    for i in range(len(diff_frames)):
        frame = diff_frames[i]
        mean = np.mean(frame)
        std = np.std(frame)
        normalized[i] = (frame - mean) / (std + 1e-8)
    return normalized


def standardize_frames(frames: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(frames)
    for i in range(len(frames)):
        frame = frames[i]
        mean = np.mean(frame)
        std = np.std(frame)
        normalized[i] = (frame - mean) / (std + 1e-8)
    return normalized


def preprocess_video_frames(frames: np.ndarray, img_size: int = 72) -> np.ndarray:
    T = len(frames)

    print(f"  Resizing {T} frames to {img_size}x{img_size}...")
    resized = np.zeros((T, img_size, img_size, 3), dtype=np.float32)
    for i in range(T):
        resized[i] = cv2.resize(frames[i], (img_size, img_size)).astype(np.float32)

    resized = resized[:, :, :, ::-1].copy()  # BGR -> RGB

    print("  Applying DiffNormalized...")
    diff_norm = diff_normalize_frames(resized)

    print("  Applying Standardized...")
    standard = standardize_frames(resized[1:])

    combined = np.concatenate([diff_norm, standard], axis=3)
    combined = np.transpose(combined, (0, 3, 1, 2))

    print(f"  Preprocessed shape: {combined.shape}")
    return combined.astype(np.float32)


# ==============================================================================
# PNG Loading - Fixed to only load from direct folder (no subfolders)
# ==============================================================================

def natural_sort_key(s: str) -> List:
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def load_png_frames(input_folder: str, face_detector: FaceDetector,
                    max_frames: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Load PNG frames ONLY from direct folder
    """

    print(f"\n{'=' * 60}")
    print(f"Loading PNG frames from: {input_folder}")
    print(f"{'=' * 60}")

    # Only look for files directly in the folder (NOT subfolders)
    image_files = []

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        # Check if it's a file (not directory) and has image extension
        if os.path.isfile(filepath):
            ext = filename.lower()
            if ext.endswith('.png'):
                image_files.append(filepath)

    if len(image_files) == 0:
        raise ValueError(f"No PNG/JPG files found directly in {input_folder}")

    # Natural sort
    image_files = sorted(image_files, key=natural_sort_key)

    print(f"Found {len(image_files)} image files (direct children only)")

    if max_frames and max_frames < len(image_files):
        image_files = image_files[:max_frames]
        print(f"Limited to {max_frames} frames")

    total_frames = len(image_files)
    print(f"Will load {total_frames} frames")

    # Detect face
    print("\nDetecting face...")
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        raise ValueError(f"Failed to read: {image_files[0]}")

    h_orig, w_orig = first_frame.shape[:2]
    print(f"Original resolution: {w_orig}x{h_orig}")

    face_box = None
    for i in range(min(10, total_frames)):
        test_frame = cv2.imread(image_files[i])
        face_box = face_detector.detect(test_frame)
        if face_box is not None:
            print(
                f"Face detected in frame {i}: x={face_box[0]}, y={face_box[1]}, w={face_box[2]}, h={face_box[3]}"
            )
            break

    if face_box is None:
        raise ValueError("No face detected in the first 10 frames!")

    x, y, w, h = face_box

    print(f"\nLoading and cropping {total_frames} frames...")
    frames = []
    filenames = []

    for img_path in tqdm(image_files, desc="Loading", ncols=80):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        face_crop = frame[y:y + h, x:x + w].copy()
        if face_crop.size == 0:
            continue

        frames.append(face_crop)
        filenames.append(os.path.basename(img_path))

    frames = np.array(frames)
    print(f"\nLoaded {len(frames)} frames")
    print(f"  Face crop size: {frames.shape[1]}x{frames.shape[2]}")

    return frames, filenames


# ==============================================================================
# Heart Rate Estimation
# ==============================================================================
from scipy.signal import welch
def estimate_heart_rate(signal: np.ndarray, fps: int,
                        min_hr: float = 40, max_hr: float = 180) -> dict:
    n = len(signal)
    window = np.hanning(n)
    signal_windowed = signal * window

    fft_vals = np.abs(fft(signal_windowed))
    freqs = fftfreq(n, 1 / fps)

    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_vals = fft_vals[pos_mask]

    min_freq, max_freq = min_hr / 60, max_hr / 60
    hr_mask = (freqs >= min_freq) & (freqs <= max_freq)

    if not np.any(hr_mask):
        return {'hr_bpm': 0, 'confidence': 0}

    hr_freqs = freqs[hr_mask]
    hr_fft = fft_vals[hr_mask]

    peak_idx = np.argmax(hr_fft)
    peak_freq = hr_freqs[peak_idx]
    peak_power = hr_fft[peak_idx]

    hr_bpm = peak_freq * 60
    confidence = peak_power / (np.mean(hr_fft) + 1e-8)

    return {
        'hr_bpm': hr_bpm,
        'hr_freq': peak_freq,
        'confidence': confidence,
        'freqs': hr_freqs,
        'power': hr_fft
    }


# ==============================================================================
# Inference Pipeline
# ==============================================================================

class rPPGInference:
    def __init__(self, model_path: str, img_size: int = 72, frame_depth: int = 10,
                 device: str = 'cuda:0', large_box_coef: float = 1.5):

        self.img_size = img_size
        self.frame_depth = frame_depth
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"\n{'=' * 60}")
        print("rPPG Inference Pipeline")
        print(f"{'=' * 60}")
        print(f"Device: {self.device}")
        print(f"Image Size: {img_size}x{img_size}")
        print(f"Frame Depth: {frame_depth}")

        self.face_detector = FaceDetector(large_box_coef=large_box_coef)

        print(f"\nLoading model: {model_path}")
        self.model = TSCAN(
            in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3,
            dropout_rate1=0.25, dropout_rate2=0.5, pool_size=(2, 2),
            nb_dense=128, frame_depth=frame_depth, img_size=img_size
        )

        state_dict = torch.load(model_path, map_location=self.device)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded!")

    def predict(self, input_folder: str, fps: int = 30,
                max_frames: Optional[int] = None,
                num_chunks_per_batch: int = 10) -> Tuple[np.ndarray, int]:
        """
        Args:
            num_chunks_per_batch: Number of frame_depth chunks per batch
                                  Batch size = num_chunks_per_batch * frame_depth
        """
        frames, _ = load_png_frames(input_folder, self.face_detector, max_frames)

        print(f"\n{'=' * 60}")
        print("Preprocessing")
        print(f"{'=' * 60}")
        preprocessed = preprocess_video_frames(frames, self.img_size)

        num_frames = len(preprocessed)
        usable_frames = (num_frames // self.frame_depth) * self.frame_depth

        if usable_frames < self.frame_depth:
            raise ValueError(f"Need at least {self.frame_depth} frames, got {num_frames}")

        preprocessed = preprocessed[:usable_frames]
        print(f"Usable frames: {usable_frames}")

        input_tensor = torch.from_numpy(preprocessed).float()
        batch_size = num_chunks_per_batch * self.frame_depth

        print(f"\n{'=' * 60}")
        print("Running Inference")
        print(f"{'=' * 60}")
        print(f"Total frames: {usable_frames}")
        print(f"Batch size: {num_chunks_per_batch} x {self.frame_depth} = {batch_size}")

        predictions = []

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, usable_frames, batch_size), desc="Inference", ncols=80):
                end_idx = min(i + batch_size, usable_frames)
                batch = input_tensor[i:end_idx]

                # Ensure divisible by frame_depth
                valid_size = (len(batch) // self.frame_depth) * self.frame_depth
                if valid_size == 0:
                    continue

                batch = batch[:valid_size].to(self.device)
                pred = self.model(batch)
                predictions.append(pred.cpu().numpy())

        rppg_signal = np.concatenate(predictions, axis=0).flatten()

        print(f"\nDone. Signal length: {len(rppg_signal)} ({len(rppg_signal) / fps:.1f}s)")
        return rppg_signal, fps

    def postprocess(self, signal: np.ndarray, fps: int,
                    lowcut: float = 0.7, highcut: float = 3.5) -> np.ndarray:
        print(f"\nBandpass filter: {lowcut}-{highcut} Hz")

        signal = signal - np.mean(signal)
        nyquist = fps / 2
        low = max(0.01, lowcut / nyquist)
        high = min(0.99, highcut / nyquist)

        b, a = butter(2, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)

        return filtered


def detect_peaks_simple(signal: np.ndarray, fps: int, min_hr: float = 40, max_hr: float = 180):
    """
    Detect peaks in PPG signal for heartbeat visualization
    """
    from scipy.signal import find_peaks

    # Calculate expected peak distance based on HR range
    min_distance = int(fps * 60 / max_hr)  # minimum samples between peaks
    max_distance = int(fps * 60 / min_hr)  # maximum samples between peaks

    # Find peaks
    peaks, properties = find_peaks(signal, distance=min_distance, prominence=0.01)

    return peaks


def save_results(signal: np.ndarray, filtered_signal: np.ndarray,
                 fps: int, output_path: str):
    """
    Save results and generate detailed PPG waveform plots
    """
    os.makedirs(output_path, exist_ok=True)

    time = np.arange(len(signal)) / fps

    # CSV
    csv_path = os.path.join(output_path, 'rppg_signal.csv')
    data = np.column_stack([time, signal, filtered_signal])
    np.savetxt(csv_path, data, delimiter=',',
               header='time_sec,raw_signal,filtered_signal', comments='')
    print(f"Saved: {csv_path}")

    # HR estimation
    hr_result = estimate_heart_rate(filtered_signal, fps)
    print(f"\nEstimated HR: {hr_result['hr_bpm']:.1f} BPM (confidence: {hr_result['confidence']:.2f})")

    # Detect peaks for visualization
    peaks = detect_peaks_simple(filtered_signal, fps)

    # =========================================================================
    # Plot 1: Complete Analysis (4 subplots)
    # =========================================================================
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))

    # 1.1 Raw signal
    axes[0].plot(time, signal, 'b-', linewidth=0.5, alpha=0.8)
    axes[0].set_title('Raw rPPG Signal', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, time[-1]])

    # 1.2 Filtered signal with peaks
    axes[1].plot(time, filtered_signal, 'r-', linewidth=0.8, label='Filtered PPG')
    if len(peaks) > 0:
        axes[1].plot(time[peaks], filtered_signal[peaks], 'go', markersize=6,
                     label=f'Peaks ({len(peaks)} beats)')
    axes[1].set_title(f'Filtered rPPG Signal with Peak Detection | HR: {hr_result["hr_bpm"]:.1f} BPM',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, time[-1]])

    # 1.3 Zoomed PPG waveform (10 seconds)
    zoom_duration = 10  # seconds
    zoom_samples = int(zoom_duration * fps)
    # Start from middle of signal for better quality
    start_idx = len(filtered_signal) // 2 - zoom_samples // 2
    end_idx = start_idx + zoom_samples

    if end_idx <= len(filtered_signal):
        zoom_time = time[start_idx:end_idx]
        zoom_signal = filtered_signal[start_idx:end_idx]

        # Find peaks in zoomed region
        zoom_peaks = peaks[(peaks >= start_idx) & (peaks < end_idx)] - start_idx

        axes[2].plot(zoom_time, zoom_signal, 'r-', linewidth=1.5, label='PPG Waveform')

        if len(zoom_peaks) > 0:
            axes[2].plot(zoom_time[zoom_peaks], zoom_signal[zoom_peaks], 'go',
                         markersize=8, label='Systolic Peaks')
        axes[2].set_title(f'Zoomed PPG Waveform ({zoom_duration}s window)',
                          fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

    # 1.4 Power spectrum
    if 'freqs' in hr_result:
        axes[3].plot(hr_result['freqs'] * 60, hr_result['power'], 'g-', linewidth=1.5)

        axes[3].axvline(x=hr_result['hr_bpm'], color='r', linestyle='--', linewidth=2,
                        label=f'Peak HR: {hr_result["hr_bpm"]:.1f} BPM')
        axes[3].set_title('Power Spectrum (FFT)', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('Heart Rate (BPM)')
        axes[3].set_ylabel('Power')
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_xlim([40, 180])

    plt.tight_layout()
    plot_path = os.path.join(output_path, 'rppg_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # =========================================================================
    # Plot 2: Detailed PPG Waveform Only (single beats visible)
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 1, figsize=(16, 10))
    temp_axe = axes2
    axes2 = [temp_axe]

    # 2.1 Multiple cardiac cycles (5-6 seconds to show ~5-8 beats)
    short_duration = 6  # seconds
    short_samples = int(short_duration * fps)
    start_idx = len(filtered_signal) // 2 - short_samples // 2
    end_idx = start_idx + short_samples

    if end_idx <= len(filtered_signal):
        short_time = time[start_idx:end_idx] - time[start_idx]  # Start from 0
        short_signal = filtered_signal[start_idx:end_idx]

        # Find peaks
        short_peaks = peaks[(peaks >= start_idx) & (peaks < end_idx)] - start_idx

        axes2[0].plot(short_time, short_signal, 'r-', linewidth=2, label='rPPG Waveform')

        if len(short_peaks) > 0:
            axes2[0].plot(short_time[short_peaks], short_signal[short_peaks], 'g^',
                          markersize=12, markeredgecolor='darkgreen', markeredgewidth=2,
                          label='Systolic Peak')

            # Mark IBI (Inter-Beat Interval)

        axes2[0].set_title(f'rPPG Waveform - {short_duration} Seconds | Heart Rate: {hr_result["hr_bpm"]:.1f} BPM',
                           fontsize=14, fontweight='bold')
        axes2[0].set_xlabel('Time (s)', fontsize=12)
        axes2[0].set_ylabel('Amplitude', fontsize=12)
        axes2[0].legend(loc='upper right', fontsize=10)
        axes2[0].grid(True, alpha=0.3)
        axes2[0].set_xlim([0, short_duration])
    plt.tight_layout()
    waveform_path = os.path.join(output_path, 'ppg_waveform_detail.png')
    plt.savefig(waveform_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {waveform_path}")

    # =========================================================================
    # Plot 3: Simple clean PPG waveform for presentation
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(14, 5))

    # Show 15 seconds of clean waveform
    display_duration = 15
    display_samples = int(display_duration * fps)
    start_idx = len(filtered_signal) // 2 - display_samples // 2
    end_idx = start_idx + display_samples

    if end_idx <= len(filtered_signal):
        display_time = time[start_idx:end_idx] - time[start_idx]
        display_signal = filtered_signal[start_idx:end_idx]

        ax3.plot(display_time, display_signal, 'r-', linewidth=1.5)

        # Add peaks
        display_peaks = peaks[(peaks >= start_idx) & (peaks < end_idx)] - start_idx
        if len(display_peaks) > 0:
            ax3.plot(display_time[display_peaks], display_signal[display_peaks],
                     'go', markersize=8, alpha=0.7)

        ax3.set_title(
            f'Extracted rPPG Waveform | Heart Rate: {hr_result["hr_bpm"]:.1f} BPM',
            fontsize=14, fontweight='bold'
        )
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('Amplitude', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, display_duration])

    plt.tight_layout()
    clean_path = os.path.join(output_path, 'ppg_waveform_clean.png')
    plt.savefig(clean_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {clean_path}")

    # HR text
    hr_path = os.path.join(output_path, 'heart_rate.txt')
    with open(hr_path, 'w', encoding='utf-8') as f:
        f.write("===== rPPG Analysis Results =====\n\n")
        f.write(f"Heart Rate: {hr_result['hr_bpm']:.1f} BPM\n")
        f.write(f"Confidence: {hr_result['confidence']:.2f}\n")
        f.write(f"Duration: {time[-1]:.1f} seconds\n")
        f.write(f"Total Samples: {len(signal)}\n")
        f.write(f"Frame Rate: {fps} FPS\n")
        f.write(f"Detected Beats: {len(peaks)}\n")
        if len(peaks) > 1:
            ibis = np.diff(peaks) / fps * 1000  # Inter-beat intervals in ms
            f.write("\nInter-Beat Interval (IBI) Statistics:\n")
            f.write(f"  Mean IBI: {np.mean(ibis):.1f} ms\n")
            f.write(f"  Std IBI: {np.std(ibis):.1f} ms\n")
            f.write(f"  Min IBI: {np.min(ibis):.1f} ms\n")
            f.write(f"  Max IBI: {np.max(ibis):.1f} ms\n")
            # HRV metric (RMSSD)
            if len(ibis) > 1:
                rmssd = np.sqrt(np.mean(np.diff(ibis) ** 2))
                f.write(f"  RMSSD (HRV): {rmssd:.1f} ms\n")
    print(f"Saved: {hr_path}")

    return hr_result


def main():
    parser = argparse.ArgumentParser(description='rPPG Inference')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='./rppg_outputs')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--img_size', type=int, default=72)
    parser.add_argument('--frame_depth', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--large_box_coef', type=float, default=1.5)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--num_chunks_per_batch', type=int, default=10,
                        help='Chunks per batch (batch_size = this x frame_depth)')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  rPPG Inference")
    print("=" * 60)
    print(f"Input: {args.input_path}")
    print(f"Model: {args.model_path}")
    print(f"FPS: {args.fps}")

    inference = rPPGInference(
        model_path=args.model_path,
        img_size=args.img_size,
        frame_depth=args.frame_depth,
        device=args.device,
        large_box_coef=args.large_box_coef
    )

    raw_signal, fps = inference.predict(
        args.input_path,
        fps=args.fps,
        max_frames=args.max_frames,
        num_chunks_per_batch=args.num_chunks_per_batch
    )

    filtered_signal = inference.postprocess(raw_signal, fps)

    hr_result = save_results(raw_signal, filtered_signal, fps, args.output_path)

    print(f"\n{'=' * 60}")
    print("COMPLETE!")
    print(f"Output: {args.output_path}")
    print(f"HR: {hr_result['hr_bpm']:.1f} BPM")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
