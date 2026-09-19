"""Typed qualification documents and canonical, stdlib-only serialization."""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, TypedDict

if TYPE_CHECKING or __package__:
    from tests.integration import timing_lattice as lattice


STALE: Final = frozenset("baseline." + name + "_stale" for name in (
    "fingerprint", "probe_version", "model_repo", "model_revision", "model_assets", "dependencies", "quantum"))
BEHAVIOR: Final = frozenset({"behavior.length", "behavior.identity", "behavior.timing"})
INTEGRITY: Final = frozenset({"baseline.missing", "baseline.invalid", "candidate.invalid",
    "candidate.lattice_inconclusive", "candidate.quantum_invalid", "candidate.overhang_invalid",
    "candidate.model_invalid", "candidate.asset_invalid", "candidate.fixture_invalid",
    "candidate.dependencies_invalid", "candidate.coverage_invalid"})
IDENTITY: Final = STALE | {"baseline.anchor_stale", "comparison.fixture_invalid", "comparison.platform_invalid"}
COLLECTIONS: Final = frozenset({"runs", "production_runs", "production_transcripts"})
OBSERVATION_KEYS: Final = COLLECTIONS | {"record_kind", "schema_version", "fingerprint", "probe_version",
    "model", "dependencies", "fixture", "platform", "derived_quantum_us", "inconclusive_reason",
    "observed_max_overhang_us", "signature_sha256"}
RECORD_KEYS: Final = OBSERVATION_KEYS | {"baseline_comparison"}


class ModelIdentity(TypedDict):
    repo: str
    revision: str
    sha256: dict[str, str]


class FixtureIdentity(TypedDict):
    repo: str
    revision: str
    name: str
    sha256: str


class PlatformIdentity(TypedDict):
    system: str
    machine: str


class Observations(lattice.SignedPayload):
    record_kind: str
    schema_version: int
    fingerprint: str
    probe_version: int
    model: ModelIdentity
    dependencies: dict[str, str]
    fixture: FixtureIdentity
    platform: PlatformIdentity
    derived_quantum_us: int | None
    inconclusive_reason: str | None
    observed_max_overhang_us: int
    signature_sha256: str


class BaselineComparison(TypedDict):
    comparable: bool
    integrity: list[lattice.Diagnostic]
    identity: list[lattice.Diagnostic]
    behavior: list[lattice.Diagnostic]


class Candidate(Observations):
    baseline_comparison: BaselineComparison


class Promotion(TypedDict):
    previous_fingerprint: str | None
    previous_anchor: str | None
    comparison_to_previous: BaselineComparison
    decision: str


class Decision(StrEnum):
    BOOTSTRAP = "bootstrap"
    ROUTINE = "routine"
    ACCEPTED_BEHAVIOR = "accepted_behavior"


@dataclass(frozen=True, slots=True)
class ParsedRecord:
    observations: Observations
    comparison: BaselineComparison
    promotion: Promotion | None


@dataclass(frozen=True, slots=True)
class BaselineLoadResult:
    observations: Observations | None
    diagnostics: tuple[lattice.Diagnostic, ...]


@dataclass(frozen=True, slots=True)
class Expectations:
    model: ModelIdentity
    fixture: FixtureIdentity
    dependencies: dict[str, str]
    platform: PlatformIdentity
    probe_version: int
    quantum_us: int
    max_overhang_us: int
    fingerprint: str


def encode(value: lattice.JsonValue | Observations | BaselineComparison | Promotion) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def json_value(value: Observations | BaselineComparison | Promotion) -> lattice.JsonValue:
    decoded: lattice.JsonValue = json.loads(encode(value))
    return decoded


def document(value: Observations | BaselineComparison | Promotion) -> dict[str, lattice.JsonValue]:
    return lattice._shape(json_value(value), frozenset(value))


def signed_payload(observations: Observations) -> lattice.SignedPayload:
    return {"runs": observations["runs"], "production_runs": observations["production_runs"],
            "production_transcripts": observations["production_transcripts"]}


def candidate(observations: Observations, comparison: BaselineComparison) -> Candidate:
    return {**observations, "baseline_comparison": comparison}
