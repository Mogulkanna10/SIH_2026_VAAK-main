from pathlib import Path
import wave
import numpy as np


# ============================================================
# Paths
# ============================================================

BASE = Path.home() / "split"
OUTPUT = BASE / "features"

# ============================================================
# Audio configuration
# ============================================================

SAMPLE_RATE = 16000

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

FRAME_MS = 30
STEP_MS = 20

FRAME_LENGTH = int(SAMPLE_RATE * FRAME_MS / 1000)
FRAME_STEP = int(SAMPLE_RATE * STEP_MS / 1000)

FFT_SIZE = 512
N_MELS = 40

FMIN = 80.0
FMAX = 7500.0


# ============================================================
# WAV loader
# ============================================================

def load_wav(path: Path) -> np.ndarray:

    with wave.open(str(path), "rb") as wf:

        channels = wf.getnchannels()
        rate = wf.getframerate()
        width = wf.getsampwidth()

        if channels != 1:
            raise ValueError(
                f"{path}: expected mono, got {channels} channels"
            )

        if rate != SAMPLE_RATE:
            raise ValueError(
                f"{path}: expected 16000 Hz, got {rate} Hz"
            )

        if width != 2:
            raise ValueError(
                f"{path}: expected 16-bit PCM, got {width * 8}-bit"
            )

        raw = wf.readframes(wf.getnframes())

    audio = np.frombuffer(
        raw,
        dtype=np.int16
    ).astype(np.float32)

    audio /= 32768.0

    return audio


# ============================================================
# Mel scale
# ============================================================

def hz_to_mel(hz):
    return 2595.0 * np.log10(
        1.0 + hz / 700.0
    )


def mel_to_hz(mel):
    return 700.0 * (
        10.0 ** (mel / 2595.0) - 1.0
    )


def create_mel_filterbank():

    mel_min = hz_to_mel(FMIN)
    mel_max = hz_to_mel(FMAX)

    mel_points = np.linspace(
        mel_min,
        mel_max,
        N_MELS + 2
    )

    hz_points = mel_to_hz(
        mel_points
    )

    bins = np.floor(
        (FFT_SIZE + 1)
        * hz_points
        / SAMPLE_RATE
    ).astype(int)

    filters = np.zeros(
        (N_MELS, FFT_SIZE // 2 + 1),
        dtype=np.float32
    )

    for m in range(1, N_MELS + 1):

        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center <= left:
            center = left + 1

        if right <= center:
            right = center + 1

        for k in range(
            left,
            min(center, filters.shape[1])
        ):
            filters[m - 1, k] = (
                (k - left) /
                (center - left)
            )

        for k in range(
            center,
            min(right, filters.shape[1])
        ):
            filters[m - 1, k] = (
                (right - k) /
                (right - center)
            )

    return filters


MEL_FILTERS = create_mel_filterbank()


# ============================================================
# Fixed 1-second input
# ============================================================

def make_one_second(audio):

    if len(audio) >= WINDOW_SAMPLES:
        return audio[:WINDOW_SAMPLES]

    output = np.zeros(
        WINDOW_SAMPLES,
        dtype=np.float32
    )

    output[:len(audio)] = audio

    return output


# ============================================================
# Log-Mel extraction
# ============================================================

def extract_logmel(audio):

    frames = []

    window = np.hanning(
        FRAME_LENGTH
    )

    for start in range(
        0,
        len(audio) - FRAME_LENGTH + 1,
        FRAME_STEP
    ):

        frame = (
            audio[
                start:start + FRAME_LENGTH
            ] * window
        )

        frames.append(frame)

    frames = np.asarray(
        frames,
        dtype=np.float32
    )

    spectrum = np.fft.rfft(
        frames,
        n=FFT_SIZE,
        axis=1
    )

    power = (
        np.abs(spectrum) ** 2
    ) / FFT_SIZE

    mel_energy = np.dot(
        power,
        MEL_FILTERS.T
    )

    mel_energy = np.maximum(
        mel_energy,
        1e-10
    )

    log_mel = np.log(
        mel_energy
    )

    # Per-file normalization.
    mean = np.mean(log_mel)
    std = np.std(log_mel)

    if std > 1e-8:
        log_mel = (
            log_mel - mean
        ) / std

    return log_mel.astype(
        np.float32
    )


# ============================================================
# Process one split
# ============================================================

def process_split(split_name):

    source = BASE / split_name
    destination = OUTPUT / split_name

    for label in ["positive", "negative"]:

        source_dir = source / label
        destination_dir = destination / label

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        files = sorted(
            source_dir.glob("*.wav")
        )

        print(
            f"{split_name}/{label}: "
            f"{len(files)} files"
        )

        for index, path in enumerate(files, start=1):

            audio = load_wav(path)

            audio = make_one_second(
                audio
            )

            features = extract_logmel(
                audio
            )

            output_path = (
                destination_dir
                / f"{path.stem}.npy"
            )

            np.save(
                output_path,
                features
            )

            if index % 50 == 0:
                print(
                    f"  processed {index}/{len(files)}"
                )

        print()


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("VaaK / Dracarys KWS Feature Preparation")
print("=" * 70)
print()

for split in [
    "train",
    "validation",
    "test"
]:

    process_split(split)


total = len(
    list(OUTPUT.rglob("*.npy"))
)

print("=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    "Total feature files:",
    total
)

print(
    "Output directory:",
    OUTPUT
)
