from __future__ import annotations

from pathlib import Path

import pytest

from sttx.asr import transcribe
from tests.integration.qualification_recording import Event, Front, Recording
from tests.integration.timing_lattice import ObservationError
from tests.test_asr import FakeRecognizer, FakeResult, FakeVad, _segment, _prepared_wav


def test_recording_when_production_traverses_long_segment_records_chunks(tmp_path: Path) -> None:
    from tests.integration.qualification_recording import Recording
    from tests.test_asr import FakeRecognizer, FakeResult, FakeVad, _segment, _prepared_wav
    from sttx.asr import transcribe
    raw = FakeRecognizer([FakeResult("word", ("word",), (0.0,))] * 2)
    recording = Recording("long", 576000)
    recognizer, vad = recording.wrap(raw, FakeVad([_segment(0, 576000)]))

    with _prepared_wav(tmp_path, 576000) as audio:
        transcript = transcribe(audio, recognizer=recognizer, vad=vad)
        runs = recording.validate()
        assert [run["sample_count"] for run in runs] == [480000, 96000]
        assert [run["chunk_index"] for run in runs] == [0, 1]
        assert [run["start_sample"] for run in runs] == [0, 480000]
        assert transcript.text == "word word"


def test_recording_when_final_drain_missing_rejects() -> None:
    from tests.integration.qualification_recording import Event, Recording
    from tests.integration.timing_lattice import ObservationError

    recording = Recording("original", 1)
    recording.events.extend([Event("feed", 1), Event("empty", 1), Event("flush", 0)])

    with pytest.raises(ObservationError):
        recording.validate()


def test_raw_probes_when_prefixes_requested_use_fresh_streams(tmp_path: Path) -> None:
    from tests.integration.qualification_probes import collect_raw
    from tests.test_asr import FakeRecognizer, FakeResult, _prepared_wav

    recognizer = FakeRecognizer([FakeResult("word", ("word",), (0.08,))] * 4)
    with _prepared_wav(tmp_path, 16000) as audio:
        runs = collect_raw(audio, recognizer)
        assert recognizer.chunk_lengths == [16000, 16160, 16480, 16800]
        assert [run["variant"] for run in runs] == [0, 160, 480, 800]
        assert [run["timestamps_us"] for run in runs] == [[80000]] * 4


@pytest.fixture
def recorded_trace(tmp_path: Path) -> Recording:
    recording = Recording("long", 576000)
    raw = FakeRecognizer([FakeResult("word", ("word",), (0.08,))] * 2)
    recognizer, vad = recording.wrap(raw, FakeVad([_segment(0, 576000)]))
    with _prepared_wav(tmp_path, 576000) as audio:
        _ = transcribe(audio, recognizer=recognizer, vad=vad)
    return recording


@pytest.mark.parametrize("mutation", ["final-drain", "feed-drain", "feed-size", "flush-count", "pop-count",
    "fresh-stream", "chunk-data", "chunk-size", "stream-order", "front-bounds", "decode", "final-pop"])
def test_recording_when_traversal_mutated_rejects(recorded_trace: Recording, mutation: str) -> None:
    recording = recorded_trace
    if mutation == "final-drain":
        _ = recording.events.pop()
    elif mutation == "feed-drain":
        del recording.events[1]
    elif mutation == "feed-size":
        recording.events[0] = Event("feed", 513)
    elif mutation == "flush-count":
        recording.events.append(Event("flush", 0))
        recording.events.append(Event("empty", 1))
    elif mutation == "pop-count":
        index = next(index for index, event in enumerate(recording.events) if event.kind == "pop")
        recording.events.insert(index, recording.events[index])
    elif mutation == "fresh-stream":
        recording.streams[1] = recording.streams[0]
    elif mutation == "chunk-data":
        recording.accepted[0][0] = 0.5
    elif mutation == "chunk-size":
        recording.accepted[0] = recording.accepted[0][:-1]
    elif mutation == "stream-order":
        index = next(index for index, event in enumerate(recording.events) if event.kind == "create")
        recording.events[index] = Event("create", 1)
    elif mutation == "front-bounds":
        recording.fronts[0] = Front(1, recording.fronts[0].samples)
    else:
        kind = "decode" if mutation == "decode" else "pop"
        index = next(index for index, event in enumerate(recording.events) if event.kind == kind)
        del recording.events[index]

    with pytest.raises(ObservationError):
        recording.validate()


def test_long_probe_when_written_tiles_first_in_bounds_pcm_slice(tmp_path: Path) -> None:
    import wave
    import numpy as np
    from sttx.audio import PreparedAudio
    from tests.integration.qualification_probes import write_long_probe

    samples = np.arange(320, dtype=np.int16)
    original_path = tmp_path / "original.wav"
    with wave.open(str(original_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(samples.tobytes())
    recording = Recording("original", 320)
    recording.fronts.append(Front(20, samples[20:100].astype(np.float32) / 32768.0))
    long_path = tmp_path / "long.wav"

    with PreparedAudio(original_path, 320) as original, PreparedAudio(long_path, 576000) as long_audio:
        write_long_probe(original, recording, long_audio)
        with wave.open(str(long_path), "rb") as source:
            assert source.getnframes() == 576000
            assert source.getframerate() == 16000 and source.getsampwidth() == 2 and source.getnchannels() == 1
            assert source.readframes(576000) == samples[20:100].tobytes() * 7200
    assert not original_path.exists() and not long_path.exists()


def test_collection_when_fixture_hash_wrong_keeps_observations_and_owned_audio(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    from sttx import backend_contract as contract
    from sttx.audio import PreparedAudio, normalize_media
    from sttx.model import ModelBundle
    from tests.backend_qualification_helpers import observation_case
    from tests.integration import qualification_probes as probes
    from tests.integration.qualification_validation import validate_candidate

    _, expected = observation_case()
    model_dir = tmp_path / contract.PARAKEET_REVISION
    model_dir.mkdir()
    paths = [model_dir / name for name in (*contract.PARAKEET_FILENAMES, contract.SILERO_FILENAME)]
    for path in paths:
        _ = path.write_bytes(b"offline model asset")
    bundle = ModelBundle(paths[0], paths[1], paths[2], paths[3], paths[4])
    monkeypatch.setattr(probes, "resolve_bundle", lambda: bundle)
    normalized: list[Path] = []
    acquired: list[tuple[str, str, str]] = []
    vad_lengths = iter([16000, 576000])
    result = FakeResult("word", tuple("▁word" for _ in range(8)), tuple(0.08 * index for index in range(1, 9)))

    def recognizer(bundle: ModelBundle) -> FakeRecognizer:
        del bundle
        return FakeRecognizer([result] * 4)

    def vad(bundle: ModelBundle) -> FakeVad:
        del bundle
        return FakeVad([_segment(0, next(vad_lengths))])

    def normalize(path: Path) -> PreparedAudio:
        audio = normalize_media(path)
        normalized.append(audio.path)
        return audio

    with _prepared_wav(tmp_path, 16000) as fixture:
        def download(*, repo_id: str, revision: str, filename: str, cache_dir: Path,
                     local_files_only: bool, force_download: bool) -> str:
            assert cache_dir == tmp_path / "fixture-cache" and force_download and not local_files_only
            acquired.append((repo_id, revision, filename))
            return str(fixture.path)

        monkeypatch.setattr(probes, "hf_hub_download", download)
        monkeypatch.setattr(probes, "normalize_media", normalize)
        monkeypatch.setattr(probes.cli, "_make_recognizer_from_bundle", recognizer)
        monkeypatch.setattr(probes.cli, "_make_vad_from_bundle", vad)

        with probes.collect_observations(tmp_path, expected) as observations:
            assert observations["fixture"]["sha256"] == hashlib.sha256(fixture.path.read_bytes()).hexdigest()
            assert acquired == [(contract.FIXTURE_REPO_ID, contract.FIXTURE_REVISION, contract.FIXTURE_FILENAME)]
            assert normalized[0].is_file() and (tmp_path / "long.wav").is_file()
            assert [run["sample_count"] for run in observations["production_runs"]] == [16000, 480000, 96000]
            assert [item["probe"] for item in observations["production_transcripts"]] == ["original", "long"]
            codes = {item["code"] for item in validate_candidate(observations, expected)}
            assert codes == {"candidate.fixture_invalid", "candidate.asset_invalid"}
    assert not normalized[0].exists() and not (tmp_path / "long.wav").exists()


def test_recording_when_fronts_overlap_rejects_even_with_valid_local_bounds(tmp_path: Path) -> None:
    recording = Recording("original", 16000)
    raw = FakeRecognizer([FakeResult("word", ("word",), (0.08,))] * 2)
    recognizer, vad = recording.wrap(raw, FakeVad([_segment(0, 8000), _segment(8000, 8000)]))
    with _prepared_wav(tmp_path, 16000) as audio:
        _ = transcribe(audio, recognizer=recognizer, vad=vad)
    recording.fronts[1] = Front(0, recording.fronts[1].samples)

    with pytest.raises(ObservationError):
        recording.validate()
