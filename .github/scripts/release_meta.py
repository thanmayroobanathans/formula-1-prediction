"""Release metadata helper for CI workflows.

Reads `project.version` from `pyproject.toml`, validates an optional git tag
(`v<version>`), detects PEP 440 pre-release status, and extracts matching
release notes from `CHANGELOG.md`.
"""
from __future__ import annotations

import argparse
import os
import re
import tomllib
from pathlib import Path

_PRERELEASE_RE = re.compile(r"(a|b|rc|\.dev)\d*", re.IGNORECASE)


def read_pyproject_version(pyproject_path: Path) -> str:
    """Return `[project].version` from `pyproject.toml`."""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"Missing [project].version in {pyproject_path}")
    return version.strip()


def validate_tag(version: str, tag: str | None) -> None:
    """Ensure a `v*` git tag matches `v<version>` when provided."""
    if not tag:
        return
    normalized = tag.removeprefix("v")
    if normalized != version:
        raise ValueError(
            f"Git tag {tag!r} does not match pyproject.toml version {version!r} "
            f"(expected 'v{version}')"
        )


def is_prerelease(version: str) -> bool:
    """Return True if `version` is a PEP 440 pre-release or dev release."""
    return bool(_PRERELEASE_RE.search(version))


def extract_changelog_section(changelog_path: Path, version: str) -> str:
    """Extract the `## <version>...` section from `CHANGELOG.md`, or empty string."""
    if not changelog_path.is_file():
        return ""
    lines = changelog_path.read_text(encoding="utf-8").splitlines()
    header_prefix = f"## {version}"
    start_idx: int | None = None
    for idx, line in enumerate(lines):
        if line.startswith(header_prefix):
            rest = line[len(header_prefix):]
            if not rest or rest[0] in (" ", "\t", "—", "-", ":"):
                start_idx = idx + 1
                break
    if start_idx is None:
        return ""

    collected: list[str] = []
    for line in lines[start_idx:]:
        if line.startswith("## "):
            break
        collected.append(line)
    return "\n".join(collected).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and emit release metadata.")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--tag", default=None, help="Git tag name (e.g. v2.1.0)")
    parser.add_argument(
        "--notes-out",
        type=Path,
        default=None,
        help="Optional path to write extracted changelog notes",
    )
    args = parser.parse_args(argv)

    version = read_pyproject_version(args.pyproject)
    validate_tag(version, args.tag)
    prerelease = is_prerelease(version)
    notes = extract_changelog_section(args.changelog, version)

    if args.notes_out is not None and notes:
        args.notes_out.parent.mkdir(parents=True, exist_ok=True)
        args.notes_out.write_text(notes + "\n", encoding="utf-8")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as fh:
            fh.write(f"version={version}\n")
            fh.write(f"prerelease={'true' if prerelease else 'false'}\n")
            fh.write(f"has_notes={'true' if notes else 'false'}\n")

    print(f"version={version}")
    print(f"prerelease={'true' if prerelease else 'false'}")
    print(f"has_notes={'true' if notes else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
