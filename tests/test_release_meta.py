"""Tests for `.github/scripts/release_meta.py`."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "release_meta.py"
_SPEC = importlib.util.spec_from_file_location("release_meta", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
release_meta = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release_meta)


def test_read_pyproject_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "pitwall-f1"\nversion = "2.1.0"\n', encoding="utf-8")
    assert release_meta.read_pyproject_version(pyproject) == "2.1.0"


def test_validate_tag_matches_and_mismatches() -> None:
    release_meta.validate_tag("2.1.0", "v2.1.0")
    release_meta.validate_tag("2.1.0", None)
    with pytest.raises(ValueError, match="does not match"):
        release_meta.validate_tag("2.1.0", "v2.1.1")


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.1.0", False),
        ("2.1.0.dev14", True),
        ("2.0.0rc2", True),
        ("2.0.0a1", True),
        ("2.0.0b3", True),
    ],
)
def test_is_prerelease(version: str, expected: bool) -> None:
    assert release_meta.is_prerelease(version) is expected


def test_extract_changelog_section(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n"
        "## 2.1.0 — Stable release\n"
        "- Added auto release\n"
        "- Fixed bugs\n\n"
        "## 2.0.0rc2 — Phase 13\n"
        "- Old notes\n",
        encoding="utf-8",
    )
    notes = release_meta.extract_changelog_section(changelog, "2.1.0")
    assert notes == "- Added auto release\n- Fixed bugs"
    assert release_meta.extract_changelog_section(changelog, "9.9.9") == ""


def test_main_writes_github_output_and_notes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "2.1.0.dev14"\n', encoding="utf-8")
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("## 2.1.0.dev14 — Phase 14\n- Quant analysis\n", encoding="utf-8")
    gh_out = tmp_path / "gh_output.txt"
    notes_out = tmp_path / "notes.md"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))

    rc = release_meta.main([
        "--pyproject", str(pyproject),
        "--changelog", str(changelog),
        "--tag", "v2.1.0.dev14",
        "--notes-out", str(notes_out),
    ])
    assert rc == 0
    assert gh_out.read_text(encoding="utf-8").splitlines() == [
        "version=2.1.0.dev14",
        "prerelease=true",
        "has_notes=true",
    ]
    assert notes_out.read_text(encoding="utf-8").strip() == "- Quant analysis"
