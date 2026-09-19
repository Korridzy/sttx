from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import pytest

JsonValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | Sequence["JsonValue"]
    | Mapping[str, "JsonValue"]
)
IdentityWriter: TypeAlias = Callable[[Mapping[str, JsonValue]], None]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--qualification-output", type=Path, default=None,
                     help="write qualification candidate or failure evidence to this JSON path")
    parser.addoption(
        "--identity-output",
        type=Path,
        default=None,
        help="write integration identity evidence to this JSON path",
    )
    parser.addoption(
        "--abstractor-baseline",
        type=Path,
        default=None,
        help="path to the abstractor baseline JSON artifact",
    )
    parser.addoption(
        "--abstractor-baseline-sha256",
        type=str,
        default=None,
        help="expected SHA-256 digest for the abstractor baseline artifact",
    )


@pytest.fixture
def identity_output(request: pytest.FixtureRequest) -> Path | None:
    value = request.config.getoption("--identity-output")
    return value if isinstance(value, Path) else None


@pytest.fixture
def write_identity_json(identity_output: Path | None) -> IdentityWriter:
    def write(payload: Mapping[str, JsonValue]) -> None:
        if identity_output is None:
            return
        write_json(identity_output, payload)

    return write


def write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    staging = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            _ = stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)
