from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

type JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | list[JsonValue]
    | Mapping[str, JsonValue]
)
type IdentityWriter = Callable[[Mapping[str, JsonValue]], None]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--identity-output",
        type=Path,
        default=None,
        help="write integration identity evidence to this JSON path",
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
        identity_output.parent.mkdir(parents=True, exist_ok=True)
        staging = identity_output.with_name(f".{identity_output.name}.tmp")
        with staging.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, identity_output)

    return write
