from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Final, TypeAlias, get_type_hints

import pytest

import sttx.backend_contract as contract
import sttx.cli as cli
from tests.integration import qualification_probes as probes
from sttx.model import ModelBundle

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
EXPECTED_PAYLOAD: Final[dict[str, JsonValue]] = {
    "parakeet_repo_id": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    "parakeet_revision": "2bda32ec70b097a55adaa07d9a7173915b43cc78",
    "parakeet_filenames": ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"],
    "parakeet_sha256": {
        "encoder.int8.onnx": "acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247",
        "decoder.int8.onnx": "179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e",
        "joiner.int8.onnx": "3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3",
        "tokens.txt": "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
    },
    "silero_filename": "silero_vad.onnx",
    "silero_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
    "silero_sha256": "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
    "fixture_repo_id": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    "fixture_revision": "2bda32ec70b097a55adaa07d9a7173915b43cc78",
    "fixture_filename": "test_wavs/en.wav",
    "fixture_sha256": "148b936b43ce7c546a866e64da059f0458aee2d65e617f16e9d94f06e8d99ed6",
    "encoder_frame_us": 80_000,
    "encoder_frame": 0.08,
    "max_token_overhang_us": 1_000_000,
    "max_token_overhang": 1.0,
    "max_chunk_samples": 480_000,
    "vad_window_samples": 512,
    "probe_version": 1,
    "qualified_runs_sha256": "a" * 64,
    "recognizer_settings": {
        "num_threads": 1, "sample_rate": 16000, "feature_dim": 80,
        "decoding_method": "greedy_search", "provider": "cpu", "model_type": "nemo_transducer",
    },
    "vad_settings": {
        "silero": {"threshold": 0.5, "min_silence_duration": 0.5,
                   "min_speech_duration": 0.25, "window_size": 512, "max_speech_duration": 60},
        "model": {"sample_rate": 16000, "num_threads": 1, "provider": "cpu", "debug": False},
        "detector": {"buffer_size_in_seconds": 240.0},
    },
}


def test_canonical_payload_when_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "QUALIFIED_RUNS_SHA256", "a" * 64)
    expected = json.dumps(EXPECTED_PAYLOAD, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    actual = json.dumps(contract.contract_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    assert actual == expected


def test_public_constants_when_exported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "QUALIFIED_RUNS_SHA256", "a" * 64)
    expected_names = set(EXPECTED_PAYLOAD)

    exported = {
        "parakeet_repo_id": contract.PARAKEET_REPO_ID,
        "parakeet_revision": contract.PARAKEET_REVISION,
        "parakeet_filenames": contract.PARAKEET_FILENAMES,
        "parakeet_sha256": contract.PARAKEET_SHA256,
        "silero_filename": contract.SILERO_FILENAME,
        "silero_url": contract.SILERO_URL,
        "silero_sha256": contract.SILERO_SHA256,
        "fixture_repo_id": contract.FIXTURE_REPO_ID,
        "fixture_revision": contract.FIXTURE_REVISION,
        "fixture_filename": contract.FIXTURE_FILENAME,
        "fixture_sha256": contract.FIXTURE_SHA256,
        "encoder_frame_us": contract.ENCODER_FRAME_US,
        "encoder_frame": contract.ENCODER_FRAME,
        "max_token_overhang_us": contract.MAX_TOKEN_OVERHANG_US,
        "max_token_overhang": contract.MAX_TOKEN_OVERHANG,
        "max_chunk_samples": contract.MAX_CHUNK_SAMPLES,
        "vad_window_samples": contract.VAD_WINDOW_SAMPLES,
        "probe_version": contract.PROBE_VERSION,
        "qualified_runs_sha256": contract.QUALIFIED_RUNS_SHA256,
        "recognizer_settings": contract.RECOGNIZER_SETTINGS,
        "vad_settings": contract.VAD_SETTINGS,
    }

    assert set(exported) == expected_names
    assert json.dumps(exported, sort_keys=True) == json.dumps(EXPECTED_PAYLOAD, sort_keys=True)


def test_settings_when_typed_have_exact_required_nested_keys() -> None:
    expected_recognizer = {
        "num_threads": int, "sample_rate": int, "feature_dim": int,
        "decoding_method": str, "provider": str, "model_type": str,
    }
    expected_silero = {
        "threshold": float, "min_silence_duration": float, "min_speech_duration": float,
        "window_size": int, "max_speech_duration": float,
    }
    expected_model = {"sample_rate": int, "num_threads": int, "provider": str, "debug": bool}
    expected_detector = {"buffer_size_in_seconds": float}

    settings_types = (
        (contract.RecognizerSettings, expected_recognizer, contract.RECOGNIZER_SETTINGS),
        (contract.SileroSettings, expected_silero, contract.VAD_SETTINGS["silero"]),
        (contract.VadModelSettings, expected_model, contract.VAD_SETTINGS["model"]),
        (contract.VadDetectorSettings, expected_detector, contract.VAD_SETTINGS["detector"]),
    )

    for settings_type, expected, settings in settings_types:
        assert get_type_hints(settings_type) == expected
        assert settings_type.__required_keys__ == set(expected) == set(settings)
        assert settings_type.__optional_keys__ == frozenset()
    assert get_type_hints(contract.VadSettings) == {
        "silero": contract.SileroSettings,
        "model": contract.VadModelSettings,
        "detector": contract.VadDetectorSettings,
    }
    assert contract.VadSettings.__required_keys__ == {"silero", "model", "detector"}
    assert set(contract.VAD_SETTINGS) == {"silero", "model", "detector"}


def _assert_digest_format(digest: str) -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", digest) is not None


def test_identities_when_published_have_complete_hex_formats() -> None:
    payload = contract.contract_payload()

    digests = [*payload["parakeet_sha256"].values(), payload["silero_sha256"], payload["fixture_sha256"]]

    for digest in digests:
        _assert_digest_format(digest)
    for revision in (payload["parakeet_revision"], payload["fixture_revision"]):
        assert re.fullmatch(r"[0-9a-f]{40}", revision) is not None


def test_digest_format_when_synthetic_length_is_mutated_rejects() -> None:
    payload = contract.contract_payload()
    original = payload["fixture_sha256"]
    payload["fixture_sha256"] = original[:-1]

    with pytest.raises(AssertionError):
        _assert_digest_format(payload["fixture_sha256"])

    assert contract.contract_payload()["fixture_sha256"] == original


CapturedCalls: TypeAlias = dict[str, dict[str, str | int | float]]
BUNDLE: Final = ModelBundle(
    encoder=Path("/sentinel/encoder.onnx"), decoder=Path("/sentinel/decoder.onnx"),
    joiner=Path("/sentinel/joiner.onnx"), tokens=Path("/sentinel/vocabulary.txt"),
    silero=Path("/sentinel/vad.onnx"),
)


@pytest.fixture
def constructors(monkeypatch: pytest.MonkeyPatch) -> CapturedCalls:
    calls: CapturedCalls = {}

    def recognizer(**kwargs: str | int | float) -> str:
        calls["recognizer"] = kwargs
        return "recognizer"

    def silero(**kwargs: str | int | float) -> str:
        calls["silero"] = kwargs
        return "silero-config"

    def model(**kwargs: str | int | float) -> str:
        calls["model"] = kwargs
        return "model-config"

    def detector(config: str, **kwargs: str | int | float) -> str:
        calls["detector"] = {"config": config, **kwargs}
        return "detector"

    monkeypatch.setattr(cli.sherpa_onnx.OfflineRecognizer, "from_transducer", recognizer)
    monkeypatch.setattr(cli.sherpa_onnx, "SileroVadModelConfig", silero)
    monkeypatch.setattr(cli.sherpa_onnx, "VadModelConfig", model)
    monkeypatch.setattr(cli.sherpa_onnx, "VoiceActivityDetector", detector)
    return calls


def _assert_constructor_calls(
    calls: CapturedCalls, settings: contract.ContractPayload, group: str | None = None,
) -> None:
    expected = {
        "recognizer": {"encoder": str(BUNDLE.encoder), "decoder": str(BUNDLE.decoder),
                       "joiner": str(BUNDLE.joiner), "tokens": str(BUNDLE.tokens),
                       **settings["recognizer_settings"]},
        "silero": {"model": str(BUNDLE.silero), **settings["vad_settings"]["silero"]},
        "model": {"silero_vad": "silero-config", **settings["vad_settings"]["model"]},
        "detector": {"config": "model-config", **settings["vad_settings"]["detector"]},
    }
    if group is None:
        assert calls == expected
    else:
        assert calls[group] == expected[group]


def test_constructor_kwargs_when_defaults_are_used(constructors: CapturedCalls) -> None:
    expected = contract.contract_payload()

    _ = cli._make_recognizer_from_bundle(BUNDLE)
    _ = cli._make_vad_from_bundle(BUNDLE)

    _assert_constructor_calls(constructors, expected)
    assert cli.VAD_BUFFER_SECONDS == 240.0


@pytest.mark.parametrize("scenario", [(live, group, gate) for live, gate in ((False, False), (True, False), (True, True))
                                     for group in ("recognizer", "silero", "model", "detector")])
def test_constructor_kwargs_when_sentinels_are_supplied(
    constructors: CapturedCalls, monkeypatch: pytest.MonkeyPatch, scenario: tuple[bool, str, bool],
) -> None:
    live_defaults, group, gate = scenario
    expected = contract.contract_payload()
    recognizer: contract.RecognizerSettings = {
        "num_threads": 3, "sample_rate": 8000, "feature_dim": 40,
        "decoding_method": "modified_beam_search", "provider": "cuda", "model_type": "zipformer",
    }
    vad: contract.VadSettings = {
        "silero": {"threshold": 0.7, "min_silence_duration": 0.8,
                   "min_speech_duration": 0.4, "window_size": 256, "max_speech_duration": 17.0},
        "model": {"sample_rate": 8000, "num_threads": 2, "provider": "cuda", "debug": True},
        "detector": {"buffer_size_in_seconds": 123.0},
    }
    expected["recognizer_settings"], expected["vad_settings"] = recognizer, vad
    if live_defaults:
        monkeypatch.setattr(contract, "RECOGNIZER_SETTINGS", recognizer)
        monkeypatch.setattr(contract, "VAD_SETTINGS", vad)

    if gate:
        _ = probes.make_backends(BUNDLE)
    elif group == "recognizer":
        if live_defaults:
            _ = cli._make_recognizer_from_bundle(BUNDLE)
        else:
            _ = cli._make_recognizer_from_bundle(BUNDLE, settings=recognizer)
    else:
        if live_defaults:
            _ = cli._make_vad_from_bundle(BUNDLE)
        else:
            _ = cli._make_vad_from_bundle(BUNDLE, settings=vad)

    _assert_constructor_calls(constructors, expected, group)


def test_production_consumption_when_contract_constants_change(tmp_path: Path) -> None:
    driver = '''
import sys
from pathlib import Path
import sttx.backend_contract as contract
contract.VAD_WINDOW_SAMPLES = 256
contract.MAX_CHUNK_SAMPLES = 16000
contract.ENCODER_FRAME = 0.16
contract.MAX_TOKEN_OVERHANG = 0.25
import sttx.asr as asr
from tests.test_asr import FakeResult, _run, _segment
assert (asr.VAD_WINDOW_SAMPLES, asr.MAX_CHUNK_SAMPLES,
        asr.ENCODER_FRAME, asr.MAX_TOKEN_OVERHANG) == (256, 16000, 0.16, 0.25)
vad, recognizer, transcript = _run(
    Path(sys.argv[1]), sample_count=32001, segments=(_segment(0, 32001),),
    results=(FakeResult("", (), ()),) * 3)
assert vad.fed_sizes == [256] * 125 + [1]
assert recognizer.chunk_lengths == [16000, 16000, 1]
assert asr._valid_starts((1.15,), 1.0)
assert not asr._valid_starts((1.17,), 1.0)
assert asr._word_events(FakeResult("word", ("word",), (0.0,), (1.24,)),
                        offset=0.0, chunk_duration=1.0)[0].end == 1.0
try:
    asr._word_events(FakeResult("word", ("word",), (0.0,), (1.26,)),
                     offset=0.0, chunk_duration=1.0)
except asr.TranscriptionError:
    pass
else:
    raise AssertionError("contract overhang bound was not consumed")
'''

    result = subprocess.run([sys.executable, "-c", driver, str(tmp_path)], capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
