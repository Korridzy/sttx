from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

NETWORK_SYSCALLS = ("connect(", "sendto(", "sendmsg(")
PATH_SYSCALL = re.compile(
    r"(?P<call>openat|newfstatat|statx|access|faccessat2)\("
    r"(?P<dirfd>[^,]+), (?P<path>\"(?:[^\"\\]|\\.)*\"|NULL)"
)
CHDIR_SYSCALL = re.compile(r"chdir\(\"(?P<path>(?:[^\"\\]|\\.)*)\"\)")
DIRFD_PATH = re.compile(r"(?P<num>\d+)(?:<(?P<path>[^>]+)>)?")


@dataclass(frozen=True, slots=True)
class TraceVerdict:
    network_attempts: tuple[str, ...]
    poisoned_accesses: tuple[str, ...]
    unresolved_path_syscalls: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return (
            not self.network_attempts
            and not self.poisoned_accesses
            and not self.unresolved_path_syscalls
        )


def parse_trace_family(prefix: Path, poisoned_roots: tuple[Path, ...] = ()) -> TraceVerdict:
    attempts: list[str] = []
    poisoned: list[str] = []
    unresolved: list[str] = []
    for trace in sorted(prefix.parent.glob(f"{prefix.name}*")):
        cwd = Path.cwd()
        dirfds: dict[str, Path] = {}
        for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(syscall in line for syscall in NETWORK_SYSCALLS):
                attempts.append(f"{trace.name}: {line}")
            chdir = CHDIR_SYSCALL.search(line)
            if chdir is not None:
                cwd = _resolve_path(cwd, Path(chdir.group("path")))
            path_match = PATH_SYSCALL.search(line)
            if path_match is None:
                continue
            resolved = _resolve_trace_path(cwd, dirfds, path_match)
            if resolved is None:
                unresolved.append(f"{trace.name}: {line}")
                continue
            _remember_dirfd(dirfds, line, resolved)
            for poisoned_root in poisoned_roots:
                if _is_relative_to(resolved, poisoned_root):
                    poisoned.append(f"{trace.name}: {line} -> {resolved}")
    return TraceVerdict(tuple(attempts), tuple(poisoned), tuple(unresolved))


def assert_clean_trace(prefix: Path, poisoned_roots: tuple[Path, ...] = ()) -> TraceVerdict:
    verdict = parse_trace_family(prefix, poisoned_roots)
    if not verdict.clean:
        raise AssertionError(
            "trace is not clean: "
            f"network={verdict.network_attempts[:3]} "
            f"poisoned={verdict.poisoned_accesses[:3]} "
            f"unresolved={verdict.unresolved_path_syscalls[:3]}"
        )
    return verdict


def _resolve_trace_path(
    cwd: Path,
    dirfds: dict[str, Path],
    match: re.Match[str],
) -> Path | None:
    raw_path = match.group("path")
    if raw_path == "NULL":
        return cwd
    path = Path(bytes(raw_path[1:-1], "utf-8").decode("unicode_escape"))
    if path.is_absolute():
        return path.resolve(strict=False)
    raw_dirfd = match.group("dirfd")
    if raw_dirfd == "AT_FDCWD":
        return _resolve_path(cwd, path)
    dirfd = DIRFD_PATH.fullmatch(raw_dirfd.strip())
    if dirfd is None:
        return None
    dirfd_path = dirfd.group("path")
    if dirfd_path is not None:
        return _resolve_path(Path(dirfd_path), path)
    base = dirfds.get(dirfd.group("num"))
    if base is None:
        return None
    return _resolve_path(base, path)


def _remember_dirfd(dirfds: dict[str, Path], line: str, resolved: Path) -> None:
    if " = " not in line or not resolved.is_dir():
        return
    suffix = line.rsplit(" = ", maxsplit=1)[-1]
    fd = suffix.split("<", maxsplit=1)[0].strip()
    if fd.isdigit():
        dirfds[fd] = resolved


def _resolve_path(base: Path, path: Path) -> Path:
    return (base / path).resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True
