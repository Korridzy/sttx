from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .real_pipeline_artifacts import JsonValue

STTX_BIN: Final = Path(sys.executable).with_name("sttx")
ENV_UNSET: Final = (
    "HUGGINGFACE_HUB_CACHE",
    "HUGGINGFACE_ASSETS_CACHE",
    "TRANSFORMERS_CACHE",
)
ENV_KEYS: Final = (
    "HOME",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
    "HF_ASSETS_CACHE",
    "HF_TOKEN_PATH",
)


@dataclass(frozen=True, slots=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "args": list(self.args),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True, slots=True)
class CacheEnv:
    root: Path
    values: dict[str, str]

    def subprocess_env(self, *, offline: bool = False) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if key not in ENV_UNSET}
        env.update(self.values)
        if offline:
            env["HF_HUB_OFFLINE"] = "1"
        return env

    def json(self) -> dict[str, JsonValue]:
        return {"root": str(self.root), "values": self.values}


def sanitized_env(root: Path) -> CacheEnv:
    values = {
        "HOME": str(root / "home"),
        "XDG_CACHE_HOME": str(root / "xdg"),
        "HF_HOME": str(root / "hf"),
        "HF_HUB_CACHE": str(root / "hf" / "hub"),
        "HF_XET_CACHE": str(root / "hf" / "xet"),
        "HF_ASSETS_CACHE": str(root / "hf" / "assets"),
        "HF_TOKEN_PATH": str(root / "hf" / "token"),
    }
    for path in values.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return CacheEnv(root=root, values=values)


def poisoned_env(root: Path) -> CacheEnv:
    values = {
        "HOME": str(root / "home"),
        "XDG_CACHE_HOME": str(root / "xdg"),
        "HF_HOME": str(root / "hf"),
        "HF_HUB_CACHE": str(root / "hf-hub"),
        "HF_XET_CACHE": str(root / "hf-xet"),
        "HF_ASSETS_CACHE": str(root / "hf-assets"),
        "HF_TOKEN_PATH": str(root / "hf-token"),
    }
    root.mkdir(parents=True, exist_ok=True)
    for path in values.values():
        Path(path).write_text("poison", encoding="utf-8")
    return CacheEnv(root=root, values=values)


def run_command(
    args: tuple[str, ...],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout: float = 900.0,
) -> CommandResult:
    completed = subprocess.run(
        args,
        check=False,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return CommandResult(args, completed.returncode, completed.stdout, completed.stderr)


def run_sttx(
    media: Path,
    outdir: Path,
    *,
    env: CacheEnv,
    model_dir: Path | None = None,
    offline: bool = False,
    timeout: float = 900.0,
) -> CommandResult:
    args = (str(STTX_BIN), str(media), "-d", str(outdir))
    if model_dir is not None:
        args = (*args, "--model-dir", str(model_dir))
    return run_command(
        args,
        env=env.subprocess_env(offline=offline),
        cwd=Path.cwd(),
        timeout=timeout,
    )


def trace_command(
    trace_prefix: Path,
    command: tuple[str, ...],
    *,
    network_only: bool,
) -> tuple[str, ...]:
    traces = "network" if network_only else "network,file"
    return (
        "strace",
        "-ff",
        "-qq",
        "-s",
        "0",
        "-yy",
        "-e",
        f"trace={traces}",
        "-o",
        str(trace_prefix),
        "unshare",
        "--map-root-user",
        "--net",
        "--kill-child",
        "--",
        *command,
    )
