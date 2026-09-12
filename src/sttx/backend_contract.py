from __future__ import annotations

from typing import Final, TypedDict


class RecognizerSettings(TypedDict):
    num_threads: int
    sample_rate: int
    feature_dim: int
    decoding_method: str
    provider: str
    model_type: str


class SileroSettings(TypedDict):
    threshold: float
    min_silence_duration: float
    min_speech_duration: float
    window_size: int
    max_speech_duration: float


class VadModelSettings(TypedDict):
    sample_rate: int
    num_threads: int
    provider: str
    debug: bool


class VadDetectorSettings(TypedDict):
    buffer_size_in_seconds: float


class VadSettings(TypedDict):
    silero: SileroSettings
    model: VadModelSettings
    detector: VadDetectorSettings


PARAKEET_REPO_ID: Final = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
PARAKEET_REVISION: Final = "2bda32ec70b097a55adaa07d9a7173915b43cc78"
PARAKEET_FILENAMES: Final = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)
PARAKEET_SHA256: Final[dict[str, str]] = {
    "encoder.int8.onnx": "acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247",
    "decoder.int8.onnx": "179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e",
    "joiner.int8.onnx": "3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3",
    "tokens.txt": "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
}
SILERO_FILENAME: Final = "silero_vad.onnx"
SILERO_URL: Final = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
SILERO_SHA256: Final = "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"

FIXTURE_REPO_ID: Final = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
FIXTURE_REVISION: Final = "2bda32ec70b097a55adaa07d9a7173915b43cc78"
FIXTURE_FILENAME: Final = "test_wavs/en.wav"
FIXTURE_SHA256: Final = "148b936b43ce7c546a866e64da059f0458aee2d65e617f16e9d94f06e8d99ed6"

ENCODER_FRAME_US: Final = 80_000
ENCODER_FRAME: Final = 0.08
MAX_TOKEN_OVERHANG_US: Final = 1_000_000
MAX_TOKEN_OVERHANG: Final = 1.0
MAX_CHUNK_SAMPLES: Final = 480_000
VAD_WINDOW_SAMPLES: Final = 512
PROBE_VERSION: Final = 1
QUALIFIED_RUNS_SHA256: Final = "d88b5643b7d52dad0eeeec132574d1a180d74d0221169e0dc198b2f1f6ded382"

RECOGNIZER_SETTINGS: Final[RecognizerSettings] = {
    "num_threads": 1,
    "sample_rate": 16000,
    "feature_dim": 80,
    "decoding_method": "greedy_search",
    "provider": "cpu",
    "model_type": "nemo_transducer",
}
VAD_SETTINGS: Final[VadSettings] = {
    "silero": {
        "threshold": 0.5,
        "min_silence_duration": 0.5,
        "min_speech_duration": 0.25,
        "window_size": 512,
        "max_speech_duration": 60,
    },
    "model": {
        "sample_rate": 16000,
        "num_threads": 1,
        "provider": "cpu",
        "debug": False,
    },
    "detector": {"buffer_size_in_seconds": 240.0},
}


class ContractPayload(TypedDict):
    parakeet_repo_id: str
    parakeet_revision: str
    parakeet_filenames: list[str]
    parakeet_sha256: dict[str, str]
    silero_filename: str
    silero_url: str
    silero_sha256: str
    fixture_repo_id: str
    fixture_revision: str
    fixture_filename: str
    fixture_sha256: str
    encoder_frame_us: int
    encoder_frame: float
    max_token_overhang_us: int
    max_token_overhang: float
    max_chunk_samples: int
    vad_window_samples: int
    probe_version: int
    qualified_runs_sha256: str
    recognizer_settings: RecognizerSettings
    vad_settings: VadSettings


def contract_payload() -> ContractPayload:
    return {
        "parakeet_repo_id": PARAKEET_REPO_ID,
        "parakeet_revision": PARAKEET_REVISION,
        "parakeet_filenames": list(PARAKEET_FILENAMES),
        "parakeet_sha256": PARAKEET_SHA256.copy(),
        "silero_filename": SILERO_FILENAME,
        "silero_url": SILERO_URL,
        "silero_sha256": SILERO_SHA256,
        "fixture_repo_id": FIXTURE_REPO_ID,
        "fixture_revision": FIXTURE_REVISION,
        "fixture_filename": FIXTURE_FILENAME,
        "fixture_sha256": FIXTURE_SHA256,
        "encoder_frame_us": ENCODER_FRAME_US,
        "encoder_frame": ENCODER_FRAME,
        "max_token_overhang_us": MAX_TOKEN_OVERHANG_US,
        "max_token_overhang": MAX_TOKEN_OVERHANG,
        "max_chunk_samples": MAX_CHUNK_SAMPLES,
        "vad_window_samples": VAD_WINDOW_SAMPLES,
        "probe_version": PROBE_VERSION,
        "qualified_runs_sha256": QUALIFIED_RUNS_SHA256,
        "recognizer_settings": RECOGNIZER_SETTINGS.copy(),
        "vad_settings": {
            "silero": VAD_SETTINGS["silero"].copy(),
            "model": VAD_SETTINGS["model"].copy(),
            "detector": VAD_SETTINGS["detector"].copy(),
        },
    }
