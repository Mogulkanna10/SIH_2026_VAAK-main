from pathlib import Path
import wave
import numpy as np


BASE = Path.home() / "split"
OUTPUT = BASE / "features"

SAMPLE_RATE = 16000

# FULL 2-second recording
WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(
    SAMPLE_RATE * WINDOW_SECONDS
)

FRAME_MS = 30
STEP_MS = 20

FRAME_LENGTH = int(
    SAMPLE_RATE * FRAME_MS / 1000
)  # 480

FRAME_STEP = int(
    SAMPLE_RATE * STEP_MS / 1000
)  # 320

FFT_SIZE = 512
N_MELS = 40

FMIN = 80.0
FMAX = 7500.0


def load_wav(path):

    with wave.open(str(path), "rb") as wf:

        channels = wf.getnchannels()
        rate = wf.getframerate()
        width = wf.getsampwidth()

        if channels != 1:
            raise ValueError(
                f"{path}: expected mono, got {channels}"
            )

        if rate != SAMPLE_RATE:
            raise ValueError(
                f"{path}: expected {SAMPLE_RATE} Hz, got {rate}"
            )

        if width != 2:
            raise ValueError(
                f"{path}: expected 16-bit PCM"
            )

        raw = wf.readframes(
            wf.getnframes()
        )

    audio = np.frombuffer(
        raw,
        dtype=np.int16
    ).astype(np.float32)

    audio /= 32768.0

    return audio


def make_two_second_window(audio):

    # All of our dataset recordings should be 2 seconds.
    # Still handle shorter/longer files safely.

    if len(audio) == WINDOW_SAMPLES:
        return audio

    if len(audio) < WINDOW_SAMPLES:

        output = np.zeros(
            WINDOW_SAMPLES,
            dtype=np.float32
        )

        output[:len(audio)] = audio

        return output

    # Keep the middle 2 seconds if a file is longer.
    start = (
        len(audio) - WINDOW_SAMPLES
    ) // 2

    return audio[
        start:start + WINDOW_SAMPLES
    ]


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

    points = np.linspace(
        mel_min,
        mel_max,
        N_MELS + 2
    )

    hz_points = mel_to_hz(points)

    bins = np.floor(
        (FFT_SIZE + 1)
        * hz_points
        / SAMPLE_RATE
    ).astype(int)

    filters = np.zeros(
        (
            N_MELS,
            FFT_SIZE // 2 + 1
        ),
        dtype=np.float32
    )

    for m in range(
        1,
        N_MELS + 1
    ):

        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        center = max(
            center,
            left + 1
        )

        right = max(
            right,
            center + 1
        )

        for k in range(
            left,
            min(
                center,
                filters.shape[1]
            )
        ):

            filters[m - 1, k] = (
                k - left
            ) / max(
                center - left,
                1
            )

        for k in range(
            center,
            min(
                right,
                filters.shape[1]
            )
        ):

            filters[m - 1, k] = (
                right - k
            ) / max(
                right - center,
                1
            )

    return filters


MEL_FILTERS = create_mel_filterbank()


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
            ]
            * window
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

    features = np.log(
        mel_energy
    ).astype(np.float32)

    # Normalize each feature matrix consistently.
    low = np.percentile(
        features,
        5
    )

    high = np.percentile(
        features,
        95
    )

    if high > low:

        features = (
            features - low
        ) / (
            high - low
        )

        features = np.clip(
            features,
            0.0,
            1.0
        )

    return features.astype(
        np.float32
    )


def process_split(split):

    source = BASE / split
    destination = OUTPUT / split

    total = 0

    for label in [
        "positive",
        "negative"
    ]:

        src = source / label
        dst = destination / label

        dst.mkdir(
            parents=True,
            exist_ok=True
        )

        files = sorted(
            src.glob("*.wav")
        )

        print(
            f"{split}/{label}: "
            f"{len(files)} files"
        )

        for path in files:

            audio = load_wav(path)

            audio = make_two_second_window(
                audio
            )

            features = extract_logmel(
                audio
            )

            output_path = (
                dst
                / f"{path.stem}.npy"
            )

            np.save(
                output_path,
                features
            )

            total += 1

    return total


print("=" * 70)
print("DRACARYS V3 — FULL 2-SECOND FEATURES")
print("=" * 70)
print()

grand_total = 0

for split in [
    "train",
    "validation",
    "test"
]:

    grand_total += process_split(
        split
    )

    print()


print("=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    "Total feature files:",
    grand_total
)

print(
    "Expected time frames:",
    (
        WINDOW_SAMPLES
        - FRAME_LENGTH
    ) // FRAME_STEP + 1
)

print(
    "Mel bins:",
    N_MELS
)

print(
    "Expected feature shape:",
    (
        (
            WINDOW_SAMPLES
            - FRAME_LENGTH
        ) // FRAME_STEP + 1,
        N_MELS
    )
)
