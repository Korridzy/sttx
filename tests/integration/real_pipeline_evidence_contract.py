from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit

from .real_pipeline_artifacts import JsonValue

HF_ASSETS = ("encoder", "decoder", "joiner", "tokens")
RUNTIME_ASSETS = (*HF_ASSETS, "silero")


class EvidenceContractError(AssertionError):
    pass


def assert_todo10_contract(evidence: Mapping[str, JsonValue]) -> None:
    cold_env = _mapping(evidence, "cold_env")
    cold_values = _mapping(cold_env, "values")
    hf_hub = Path(_text(cold_values, "HF_HUB_CACHE")).resolve()
    model = _mapping(evidence, "model")
    _text(model, "repo_id")
    _text(model, "silero_official_url")
    _assert_qa_wav(_mapping(evidence, "qa_wav"), _text(model, "commit"), _text(model, "repo_id"))
    assets = _mapping(model, "assets")
    for name in HF_ASSETS:
        identity = _mapping(assets, name)
        _assert_descendant(Path(_text(identity, "path")), hf_hub, f"{name}.path")
        _assert_descendant(
            Path(_text(identity, "resolved_path")),
            hf_hub,
            f"{name}.resolved_path",
        )
    _assert_warm_identity(evidence, model)
    _assert_signal_barriers(
        _sequence(_mapping(evidence, "signals"), "probes"),
        _text(model, "silero_official_url"),
    )
    _assert_cleanup(_mapping(evidence, "cleanup"))


def _assert_qa_wav(qa_wav: Mapping[str, JsonValue], commit: str, repo_id: str) -> None:
    if _text(qa_wav, "runtime_asset") != "false":
        raise EvidenceContractError("qa_wav is not explicitly marked non-runtime")
    if _text(qa_wav, "repo_id") != repo_id:
        raise EvidenceContractError("qa_wav repo id does not match model repo id")
    if _text(qa_wav, "commit") != commit:
        raise EvidenceContractError("qa_wav commit does not match model commit")
    if _number(qa_wav, "sample_rate") <= 0:
        raise EvidenceContractError("qa_wav source sample rate must be a positive integer")


def _assert_warm_identity(
    evidence: Mapping[str, JsonValue],
    model: Mapping[str, JsonValue],
) -> None:
    warm_identity = _mapping(evidence, "warm_identity")
    if _text(warm_identity, "repo_id") != _text(model, "repo_id"):
        raise EvidenceContractError("warm repo id does not match cold repo id")
    if _text(warm_identity, "commit") != _text(model, "commit"):
        raise EvidenceContractError("warm commit does not match cold commit")
    if _text(evidence, "cold_warm_identity_match") != "true":
        raise EvidenceContractError("cold/warm identity comparison is not recorded true")
    cold_assets = _mapping(model, "assets")
    warm_assets = _mapping(warm_identity, "assets")
    for name in RUNTIME_ASSETS:
        cold = _mapping(cold_assets, name)
        warm = _mapping(warm_assets, name)
        for field in ("path", "resolved_path", "size", "sha256"):
            if cold.get(field) != warm.get(field):
                raise EvidenceContractError(f"warm {name}.{field} does not match cold identity")


def _assert_signal_barriers(
    probes: Sequence[JsonValue],
    silero_url: str,
) -> None:
    expected = {
        ("hf", "SIGINT"),
        ("hf", "SIGTERM"),
        ("silero", "SIGINT"),
        ("silero", "SIGTERM"),
        ("cancellable_acquisition", "SIGINT"),
        ("cancellable_acquisition", "SIGTERM"),
        ("native_decode", "SIGINT"),
        ("native_decode", "SIGTERM"),
    }
    seen: set[tuple[str, str]] = set()
    for item in probes:
        probe = _as_mapping(item, "signal probe")
        phase = _text(probe, "phase")
        signum = _text(probe, "signum")
        seen.add((phase, signum))
        barrier = _mapping(probe, "barrier")
        match phase:
            case "hf":
                _assert_hf_barrier(barrier)
            case "silero":
                _assert_silero_barrier(barrier)
            case "cancellable_acquisition":
                _assert_cancellable_acquisition_barrier(barrier, silero_url)
                if _sequence(probe, "staging"):
                    raise EvidenceContractError(
                        "cancelled acquisition left Silero staging files"
                    )
            case "native_decode":
                _assert_native_decode_barrier(barrier)
            case _:
                raise EvidenceContractError(f"unknown signal phase: {phase}")
    missing = sorted(expected - seen)
    if missing:
        raise EvidenceContractError(f"missing signal probes: {missing}")


def _assert_hf_barrier(barrier: Mapping[str, JsonValue]) -> None:
    if _number(barrier, "runtime_complete_count") >= len(HF_ASSETS):
        raise EvidenceContractError("HF barrier fired after all runtime assets completed")
    if _text(barrier, "process_live_at_barrier") != "true":
        raise EvidenceContractError("HF barrier did not prove a live acquisition child")
    candidates = [
        _text(value, "partial asset path")
        for value in _sequence(barrier, "partial_asset_candidates")
    ]
    if not candidates:
        raise EvidenceContractError("HF barrier has no runtime partial candidates")
    if not _sequence(barrier, "missing_runtime_finals"):
        raise EvidenceContractError("HF barrier did not record an absent runtime final")
    sizes = _mapping(barrier, "partial_asset_sizes")
    kinds = _mapping(barrier, "partial_asset_kinds")
    for candidate in candidates:
        if _number(sizes, candidate) <= 0:
            raise EvidenceContractError(f"HF partial is not recorded as nonzero: {candidate}")
        kind = _text(kinds, candidate)
        if not _is_hf_runtime_partial(candidate, kind):
            raise EvidenceContractError(f"HF partial is bookkeeping, not runtime data: {candidate}")


def _assert_silero_barrier(barrier: Mapping[str, JsonValue]) -> None:
    ready = _text(barrier, "ready")
    if not ready.endswith(".tmp"):
        raise EvidenceContractError(f"Silero barrier is not a staging temp file: {ready}")
    staged = _text(barrier, "staged_path")
    if staged != ready:
        raise EvidenceContractError("Silero barrier staged path does not match ready marker")
    if _number(barrier, "staged_size") <= 0:
        raise EvidenceContractError("Silero barrier did not record nonzero staged bytes")


def _assert_cancellable_acquisition_barrier(
    barrier: Mapping[str, JsonValue],
    silero_url: str,
) -> None:
    parsed_url = urlsplit(silero_url)
    expected_target = f"{parsed_url.hostname}:{parsed_url.port or 443}"
    if _text(barrier, "connect_target") != expected_target:
        raise EvidenceContractError("acquisition proxy did not intercept the Silero host")
    if _text(barrier, "process_live_at_barrier") != "true":
        raise EvidenceContractError("acquisition driver was not live at the CONNECT barrier")
    staging = [
        Path(_text(value, "staging path")).name
        for value in _sequence(barrier, "staging_at_barrier")
    ]
    if len(staging) != 1 or not (
        staging[0].startswith(".silero_vad.onnx.") and staging[0].endswith(".tmp")
    ):
        raise EvidenceContractError("acquisition CONNECT barrier has no owned Silero staging file")


def _assert_native_decode_barrier(barrier: Mapping[str, JsonValue]) -> None:
    if "real_decode_entered" not in barrier:
        raise EvidenceContractError("native decode barrier did not record real decode marker")
    if _text(barrier, "real_decode_entered") != "decode_entered":
        raise EvidenceContractError("native decode barrier did not record real decode marker")
    if _number(barrier, "native_decode_us") <= 0:
        raise EvidenceContractError("native decode barrier did not record time spent inside the native call")
    if _text(barrier, "output_finals_exist") != "false":
        raise EvidenceContractError("native decode barrier fired after final outputs existed")


def _assert_cleanup(cleanup: Mapping[str, JsonValue]) -> None:
    probe = _mapping(cleanup, "live_process_probe")
    output = _text(probe, "output")
    phase = _text(probe, "phase")
    requires_refresh = _text(probe, "requires_post_pytest_refresh")
    if "test_real_pipeline" in output and (phase, requires_refresh) != ("in_pytest_non_final", "true"):
        raise EvidenceContractError("cleanup process probe captured pytest without a non-final marker")


def _assert_descendant(path: Path, root: Path, label: str) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise EvidenceContractError(f"{label} is outside isolated HF cache: {resolved}")


def _is_hf_runtime_partial(path: str, kind: str) -> bool:
    parts = Path(path).parts
    match kind:
        case "hub_incomplete":
            return path.endswith(".incomplete")
        case "xet_data_partial":
            return (
                "xet" in parts
                and not path.endswith("CACHEDIR.TAG")
                and not any(part in parts for part in ("logs", "metadata", "refs", "snapshots"))
                and any(part in parts for part in ("chunk-cache", "chunks", "data", "staging"))
            )
        case _:
            return False


def _mapping(source: Mapping[str, JsonValue], key: str) -> Mapping[str, JsonValue]:
    return _as_mapping(source.get(key), key)


def _as_mapping(value: JsonValue, label: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise EvidenceContractError(f"{label} is not an object")
    return value


def _sequence(source: Mapping[str, JsonValue], key: str) -> Sequence[JsonValue]:
    value = source.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise EvidenceContractError(f"{key} is not a list")
    return value


def _text(source: Mapping[str, JsonValue] | JsonValue, key: str) -> str:
    value = source.get(key) if isinstance(source, Mapping) else source
    if isinstance(value, bool):
        return "true" if value else "false"
    if not isinstance(value, str):
        raise EvidenceContractError(f"{key} is not text")
    return value


def _number(source: Mapping[str, JsonValue], key: str) -> int:
    value = source.get(key)
    if not isinstance(value, int):
        raise EvidenceContractError(f"{key} is not an integer")
    return value
