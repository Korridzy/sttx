from __future__ import annotations

import json
from copy import deepcopy

import pytest

import tests.integration.timing_lattice as lattice
from tests.integration.timing_lattice import JsonValue as Json


def complete_payload() -> lattice.SignedPayload:
    raw = {
        "tokens": ["\u2581caf\u00e9"] * 40,
        "timestamps_us": [index * 80_000 for index in range(40)],
        "durations_us": [40_000] * 40,
        "text": "caf\u00e9",
        "lang": "en",
    }
    return lattice.load_payload(json.dumps({
        "runs": [{"variant": prefix, **raw} for prefix in (0, 160, 480, 800)],
        "production_runs": [
            {"probe": probe, "segment_index": 0, "chunk_index": chunk,
             "start_sample": chunk * 480_000, "sample_count": count, **raw}
            for probe, chunk, count in (("original", 0, 80_000),
                                        ("long", 0, 480_000), ("long", 1, 96_000))
        ],
        "production_transcripts": [
            {"probe": probe, "language": "en", "duration_us": duration,
             "text": "caf\u00e9", "segments": [
                 {"id": 0, "start_us": 0, "end_us": 3_200_000, "text": "caf\u00e9"}]}
            for probe, duration in (("original", 5_000_000), ("long", 36_000_000))
        ],
    }))


@pytest.mark.parametrize(("seconds", "expected"), [
    (0, 0), (0.000005, 0), (0.000015, 20), (0.000025, 20),
    (0.0000145, 10), (0.080000001, 80_000),
])
def test_quantization_when_seconds_are_valid(seconds: float, expected: int) -> None:
    assert lattice.quantize_us(seconds) == expected


@pytest.mark.parametrize("seconds", [
    -1.0, float("nan"), float("inf"), True, 1e308,
    10**303, 10**309, -(10**309),
])
def test_quantization_when_seconds_are_invalid(seconds: float) -> None:
    with pytest.raises(lattice.ObservationError):
        _ = lattice.quantize_us(seconds)


@pytest.mark.parametrize("quantum", [40_000, 80_000, 1_000])
def test_lattice_when_sample_is_sufficient(quantum: int) -> None:
    assert lattice.derive_quantum_us([(index % 8 + 1) * quantum for index in range(30)]) == quantum


@pytest.mark.parametrize("times", [[], [0] * 40, [80_000 * value for value in range(1, 8)] * 5,
    [index * 80_000 for index in range(1, 30)],
    [index * 80_000 for index in range(1, 30)] + [2_400_010],
    [index * 990 for index in range(1, 31)], [True] * 30, [-10] * 30,
    *[[zero, *range(80_000, 2_480_000, 80_000)] for zero in (False, 0.0)],
])
def test_lattice_when_sample_is_inconclusive(times: list[Json]) -> None:
    with pytest.raises(lattice.ObservationError):
        _ = lattice.derive_quantum_us(times)


def test_complete_payload_when_identical() -> None:
    payload = complete_payload()
    encoded = lattice.canonical_bytes(payload)
    assert lattice.compare_payloads(payload, lattice.load_payload(encoded)) == ()
    assert encoded == json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False).encode("utf-8")
    assert b"\\u00e9" not in encoded and not encoded.endswith(b"\n")


@pytest.mark.parametrize("collection", ["runs", "production_runs", "production_transcripts"])
def test_signature_when_any_collection_changes(collection: str) -> None:
    payload = complete_payload()
    changed = mutate(payload, (collection, 0, "text"), "changed")
    assert lattice.signature_sha256(lattice.load_payload(changed)) != lattice.signature_sha256(payload)


def node_at(document: Json, path: tuple[str | int, ...]) -> Json:
    target = document
    for component in path:
        if isinstance(component, str):
            assert isinstance(target, dict)
            target = target[component]
        else:
            assert isinstance(target, list)
            target = target[component]
    return target


def mutate(payload: lattice.SignedPayload, path: tuple[str | int, ...], value: Json) -> str:
    document: Json = json.loads(json.dumps(payload))
    target = node_at(document, path[:-1])
    last = path[-1]
    if isinstance(last, str):
        assert isinstance(target, dict)
        target[last] = value
    else:
        assert isinstance(target, list)
        target[last] = value
    return json.dumps(document)


@pytest.mark.parametrize(("path", "value"), [
    (("extra",), 1), (("runs",), []), (("runs", 0, "extra"), 1),
    (("runs", 0, "variant"), True), (("runs", 1, "variant"), 0),
    (("runs", 0, "tokens"), "wrong"), (("runs", 0, "tokens"), []),
    (("runs", 0, "tokens", 0), 1), (("runs", 0, "text"), None),
    (("runs", 0, "lang"), []), (("runs", 0, "timestamps_us", 0), True),
    (("runs", 0, "timestamps_us", 0), -10),
    (("runs", 0, "timestamps_us", 0), float("nan")),
    (("runs", 0, "timestamps_us", 0), float("inf")),
    (("runs", 0, "timestamps_us", 0), 1.0),
    (("runs", 0, "timestamps_us", 0), 1),
    (("runs", 0, "timestamps_us", 1), 3_200_000),
    (("runs", 0, "durations_us"), [10]),
    (("production_runs",), []), (("production_runs", 0, "probe"), ""),
    (("production_runs", 0, "segment_index"), -1),
    (("production_runs", 0, "chunk_index"), 1),
    (("production_runs", 0, "start_sample"), -1),
    (("production_runs", 0, "sample_count"), 0),
    (("production_runs", 0, "sample_count"), 480_001),
    (("production_runs", 0, "sample_count"), 80_001),
    (("production_runs", 2, "start_sample"), 480_001),
    (("production_runs", 2, "chunk_index"), 0),
    (("production_transcripts",), []),
    (("production_transcripts", 1, "probe"), "original"),
    (("production_transcripts", 0, "duration_us"), True),
    (("production_transcripts", 0, "segments", 0, "id"), 1),
    (("production_transcripts", 0, "segments", 0, "end_us"), 5_000_010),
    (("production_transcripts", 0, "segments", 0, "start_us"), 3_200_010),
    (("production_transcripts", 0, "segments", 0, "text"), False),
])
def test_parser_when_shape_or_order_is_malformed(path: tuple[str | int, ...], value: Json) -> None:
    with pytest.raises(lattice.ObservationError):
        _ = lattice.load_payload(mutate(complete_payload(), path, value))


@pytest.mark.parametrize("raw", ['{}', '[]', 'null', '{',
    '{"runs":[],"runs":[]}', '{"nested":{"x":1,"x":2}}',
    '{"runs":NaN}', b'\xff', '{"runs":"\\ud800"}',
])
def test_parser_when_json_is_invalid(raw: str | bytes) -> None:
    with pytest.raises(lattice.ObservationError):
        _ = lattice.load_payload(raw)


@pytest.mark.parametrize(("path", "value"), [
    (("runs", 0, "text"), "drift"), (("runs", 0, "lang"), "fr"),
    (("runs", 0, "tokens", 0), "different"), (("runs", 0, "durations_us"), []),
    (("production_runs", 0, "text"), "drift"),
    (("production_runs", 0, "sample_count"), 79_999),
    (("production_transcripts", 0, "language"), "fr"),
    (("production_transcripts", 0, "duration_us"), 5_000_010),
    (("production_transcripts", 0, "text"), "drift"),
    (("production_transcripts", 0, "segments", 0, "text"), "drift"),
    (("production_transcripts", 0, "segments"), []),
])
def test_comparator_when_identity_drifts(path: tuple[str | int, ...], value: Json) -> None:
    payload = complete_payload()
    changed = lattice.load_payload(mutate(payload, path, value))
    assert lattice.compare_payloads(payload, changed)


@pytest.mark.parametrize(("delta", "count", "accepted"), [
    (39_990, 40, True), (40_000, 1, True), (40_000, 2, False),
    (79_990, 40, False), (80_000, 1, True), (80_000, 2, False),
    (80_010, 1, False),
])
@pytest.mark.parametrize("field", ["timestamps_us", "durations_us"])
def test_comparator_when_timing_crosses_threshold(field: str, delta: int,
                                                  count: int, accepted: bool) -> None:
    payload = complete_payload()
    source = payload["runs"][0]
    values = source["timestamps_us"] if field == "timestamps_us" else source["durations_us"]
    changed = lattice.load_payload(mutate(payload, ("runs", 0, field),
        [value + (delta if index >= len(values) - count else 0)
         for index, value in enumerate(values)]))
    assert (lattice.compare_payloads(payload, changed) == ()) is accepted


def test_empty_duration_mode_when_preserved() -> None:
    payload = complete_payload()
    for run in payload["runs"]:
        run["durations_us"] = []
    assert lattice.load_payload(lattice.canonical_bytes(payload))["runs"][0]["durations_us"] == []
    assert lattice.compare_payloads(payload, deepcopy(payload)) == ()


def test_canonicalizer_when_tuple_would_be_silently_coerced() -> None:
    malformed: dict[str, lattice.JsonValue] = json.loads(json.dumps(complete_payload()))
    runs = malformed["runs"]
    assert isinstance(runs, list)
    malformed["runs"] = tuple(runs)
    with pytest.raises(lattice.ObservationError):
        _ = lattice.canonical_bytes(malformed)


@pytest.mark.parametrize("path", [(), ("runs", 0), ("production_runs", 0),
    ("production_transcripts", 0), ("production_transcripts", 0, "segments", 0)])
def test_parser_when_any_required_field_is_missing(path: tuple[str | int, ...]) -> None:
    document: Json = json.loads(json.dumps(complete_payload()))
    target = node_at(document, path)
    assert isinstance(target, dict)
    for key in tuple(target):
        value = target.pop(key)
        with pytest.raises(lattice.ObservationError):
            _ = lattice.load_payload(json.dumps(document))
        target[key] = value


@pytest.mark.parametrize("collection", ["runs", "production_runs", "production_transcripts"])
def test_parser_when_collection_is_reversed(collection: str) -> None:
    document: dict[str, Json] = json.loads(json.dumps(complete_payload()))
    items = document[collection]
    assert isinstance(items, list)
    items.reverse()
    with pytest.raises(lattice.ObservationError):
        _ = lattice.load_payload(json.dumps(document))


@pytest.mark.parametrize(("size", "count", "accepted"), [(49, 1, True), (99, 2, False),
    (100, 2, True), (100, 3, False), (150, 3, True), (150, 4, False)])
@pytest.mark.parametrize("field", ["timestamps_us", "durations_us", "start_us", "end_us"])
def test_material_budget_when_array_size_changes(size: int, count: int, accepted: bool,
                                                field: str) -> None:
    payload = complete_payload()
    for transcript in payload["production_transcripts"]:
        transcript["duration_us"] = 40_000_000
        transcript["segments"] = [{"id": index, "start_us": index * 100_000,
            "end_us": index * 100_000 + 90_000, "text": "word"} for index in range(size // 2)]
    for run in payload["runs"]:
        run["tokens"] = ["word"] * size
        run["timestamps_us"] = [index * 100_000 for index in range(size)]
        run["durations_us"] = [90_000] * size
    changed = deepcopy(payload)
    if field in ("start_us", "end_us"):
        for segment in changed["production_transcripts"][0]["segments"][-count:]:
            segment[field] += 80_000
    else:
        values = (changed["runs"][0]["timestamps_us"] if field == "timestamps_us"
                  else changed["runs"][0]["durations_us"])
        values[-count:] = [value + 80_000 for value in values[-count:]]
    assert (lattice.compare_payloads(payload, changed) == ()) is accepted


def test_parser_when_json_integer_exceeds_decoder_limit() -> None:
    with pytest.raises(lattice.ObservationError):
        _ = lattice.load_payload('{"runs":' + "1" * 5_000 + '}')


@pytest.mark.parametrize("field", ["start_us", "end_us"])
@pytest.mark.parametrize(("delta", "accepted"), [(39_990, True), (40_000, True),
    (79_990, True), (80_000, True), (80_010, False)])
def test_segment_bound_when_hard_limit_is_crossed(field: str, delta: int, accepted: bool) -> None:
    payload = complete_payload()
    changed = deepcopy(payload)
    changed["production_transcripts"][0]["segments"][0][field] += delta
    assert (lattice.compare_payloads(payload, changed) == ()) is accepted


def test_comparator_when_independent_arrays_each_use_their_budget() -> None:
    payload = complete_payload()
    changed = deepcopy(payload)
    changed["runs"][0]["timestamps_us"][-1] += 80_000
    changed["runs"][0]["durations_us"][0] += 80_000
    assert lattice.compare_payloads(payload, changed) == ()


@pytest.mark.parametrize("collection", ["runs", "production_runs", "production_transcripts"])
def test_comparator_when_valid_lengths_differ(collection: str) -> None:
    payload = complete_payload()
    changed = deepcopy(payload)
    if collection == "runs":
        run = changed["runs"][0]
        _ = run["tokens"].pop()
        _ = run["timestamps_us"].pop()
        _ = run["durations_us"].pop()
    elif collection == "production_runs":
        _ = changed["production_runs"].pop()
    else:
        changed["production_runs"] = changed["production_runs"][:1]
        _ = changed["production_transcripts"].pop()
    assert lattice.compare_payloads(payload, changed)


def test_comparator_when_both_segment_bounds_shift_materially() -> None:
    payload = complete_payload()
    changed = deepcopy(payload)
    changed["production_transcripts"][0]["segments"] = [
        {"id": 0, "start_us": 79_990, "end_us": 3_279_990, "text": "caf\u00e9"}]
    assert lattice.compare_payloads(payload, changed)
