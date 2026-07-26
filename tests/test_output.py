import json
import os
from pathlib import Path

import pytest

from sttx.output import (
    OutputPathError,
    OutputPaths,
    OutputWriteError,
    Segment,
    Transcript,
    output_paths,
    write_outputs,
)


def russian_transcript() -> Transcript:
    return Transcript(
        language="ru",
        duration=12.345,
        segments=(
            Segment(id=0, start=0.004, end=1.234, text="  Привет\t\nмир  "),
            Segment(id=1, start=2.345, end=12.345, text="Финальная   фраза без точки"),
        ),
    )


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_json_and_txt_share_joined_segment_text(tmp_path: Path) -> None:
    # Given: a transcript with whitespace noise and a trailing unpunctuated segment.
    paths = output_paths(
        Path("source.mp4"),
        outdir=tmp_path,
        name="joined",
        cwd=tmp_path / "cwd",
    )
    transcript = Transcript(
        language="ru",
        duration=3.0,
        segments=(
            Segment(id=0, start=0.0, end=1.0, text=" первая\tфраза "),
            Segment(id=1, start=1.0, end=2.0, text="   "),
            Segment(id=2, start=2.0, end=3.0, text="\nфинал без точки"),
        ),
    )

    # When: outputs are written.
    write_outputs(transcript, paths)

    # Then: JSON text and TXT are the same normalized segment join.
    assert read_json(paths.json_path)["text"] == "первая фраза финал без точки"
    assert paths.txt_path.read_text(encoding="utf-8") == "первая фраза финал без точки"


def test_utf8_indented_schema(tmp_path: Path) -> None:
    # Given: a Cyrillic transcript and a hand-authored golden schema fixture.
    paths = output_paths(Path("voice.mp4"), outdir=tmp_path, name="schema", cwd=tmp_path)
    expected = json.loads(
        Path("tests/fixtures/output_transcript.json").read_text(encoding="utf-8")
    )

    # When: the transcript is serialized.
    write_outputs(russian_transcript(), paths)

    # Then: the JSON shape, UTF-8 text, indentation, and newline are deterministic.
    json_bytes = paths.json_path.read_bytes()
    assert json_bytes.endswith(b"\n")
    assert b"\\u041f" not in json_bytes
    assert b'  "task": "transcribe"' in json_bytes
    assert read_json(paths.json_path) == expected
    assert paths.txt_path.read_bytes().decode("utf-8") == expected["text"]


@pytest.mark.parametrize(
    ("input_path", "outdir", "name", "cwd", "expected_json", "expected_txt"),
    [
        (
            Path("clip.mp4"),
            None,
            None,
            Path("/work"),
            Path("/work/transcriptions/clip.json"),
            Path("/work/transcriptions/clip.txt"),
        ),
        (
            Path("clip.mp4"),
            Path("nested/results"),
            None,
            Path("/work"),
            Path("/work/nested/results/clip.json"),
            Path("/work/nested/results/clip.txt"),
        ),
        (
            Path("archive.tar.mp4"),
            Path("/abs/results"),
            None,
            Path("/work"),
            Path("/abs/results/archive.tar.json"),
            Path("/abs/results/archive.tar.txt"),
        ),
        (
            Path("voice.mp4"),
            None,
            "episode.one",
            Path("/work"),
            Path("/work/transcriptions/episode.one.json"),
            Path("/work/transcriptions/episode.one.txt"),
        ),
    ],
)
def test_output_path_precedence_table(
    tmp_path: Path,
    input_path: Path,
    outdir: Path | None,
    name: str | None,
    cwd: Path,
    expected_json: Path,
    expected_txt: Path,
) -> None:
    # Given: absolute and relative path options rooted in the pytest temp tree.
    actual_cwd = tmp_path / cwd.relative_to("/")
    actual_outdir = tmp_path / outdir.relative_to("/") if outdir and outdir.is_absolute() else outdir
    actual_json = tmp_path / expected_json.relative_to("/")
    actual_txt = tmp_path / expected_txt.relative_to("/")

    # When: output paths are resolved.
    paths = output_paths(input_path, outdir=actual_outdir, name=name, cwd=actual_cwd)

    # Then: outdir/name precedence and suffix appending are exact.
    assert paths == OutputPaths(json_path=actual_json, txt_path=actual_txt)
    assert actual_json.parent.is_dir()


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "/absolute", "nested/name", "nested\\name", "ready.json", "ready.txt"],
)
def test_invalid_output_names(tmp_path: Path, name: str) -> None:
    # Given: a forbidden output stem class.
    # When/Then: path resolution rejects it as an output path error.
    with pytest.raises(OutputPathError):
        output_paths(Path("input.mp4"), outdir=tmp_path, name=name, cwd=tmp_path)

    # Given: an existing non-directory output component.
    blocker = tmp_path / "blocked"
    blocker.write_text("file", encoding="utf-8")

    # When/Then: recursive output directory creation rejects the component.
    with pytest.raises(OutputPathError):
        output_paths(Path("input.mp4"), outdir=blocker / "child", name=None, cwd=tmp_path)


def test_output_paths_and_overwrite(tmp_path: Path) -> None:
    # Given: nested output paths and stale final artifacts.
    paths = output_paths(Path("clip.mp4"), outdir=Path("a/b"), name=None, cwd=tmp_path)
    paths.json_path.write_text("stale json", encoding="utf-8")
    paths.txt_path.write_text("stale txt", encoding="utf-8")

    # When: outputs are written again.
    write_outputs(russian_transcript(), paths)

    # Then: both stale finals are overwritten and recursive parents exist.
    assert paths.json_path.parent == tmp_path / "a/b"
    assert paths.json_path.parent.is_dir()
    assert read_json(paths.json_path)["text"] == "Привет мир Финальная фраза без точки"
    assert paths.txt_path.read_text(encoding="utf-8") == "Привет мир Финальная фраза без точки"


def test_staging_files_are_cleaned_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: stale finals and an injected failure on the second atomic replace.
    paths = output_paths(Path("clip.mp4"), outdir=tmp_path, name="atomic", cwd=tmp_path)
    paths.json_path.write_text("stale json", encoding="utf-8")
    paths.txt_path.write_text("stale txt", encoding="utf-8")
    real_replace = os.replace
    calls: list[tuple[str, str]] = []

    def fail_second_replace(source: str, destination: str) -> None:
        calls.append((source, destination))
        if len(calls) == 2:
            raise OSError("injected replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_replace)

    # When: the second final replacement fails.
    with pytest.raises(OutputWriteError):
        write_outputs(russian_transcript(), paths)

    # Then: staging siblings are removed and per-file atomicity is preserved.
    assert paths.json_path.read_text(encoding="utf-8") != "stale json"
    assert paths.txt_path.read_text(encoding="utf-8") == "stale txt"
    assert not list(tmp_path.glob(".atomic.*.tmp"))
