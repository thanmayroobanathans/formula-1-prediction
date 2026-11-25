# Releasing `pitwall-f1`

The [`.github/workflows/release.yml`](../.github/workflows/release.yml) workflow automates verification, building, PyPI/TestPyPI publishing, and GitHub Release creation.

## How It Works

1. **`verify`**: Runs `ruff check .` and `pytest --cov=pitwall --cov-report=term-missing`.
2. **`build`**:
   - Validates `project.version` in `pyproject.toml` (and enforces `tag == v<version>` on `v*` tag pushes).
   - Builds the source distribution (`.tar.gz`) and wheel (`.whl`) with `python -m build`.
   - Checks package metadata with `twine check dist/*`.
   - Installs the wheel in a clean virtual environment and runs `pitwall-f1 --version`, `pitwall-f1 --help`, and `import pitwall`.
3. **`publish-pypi` / `publish-testpypi`**:
   - Uses PyPI Trusted Publishing (OIDC, no long-lived API tokens required).
   - **Fallback behavior**: If PyPI Trusted Publishing is not configured yet, the publish step logs a warning and the workflow still proceeds to attach the built `.whl` and `.tar.gz` artifacts to the GitHub Release.
4. **`github-release`**:
   - Creates a GitHub Release with `dist/*` attached.
   - Marks `a`, `b`, `rc`, and `.dev` versions as pre-releases.
   - Uses the matching `## <version>` section from `CHANGELOG.md` when present, or falls back to GitHub auto-generated release notes.

## One-Time Setup (PyPI Trusted Publishing)

1. In your GitHub repository, go to **Settings → Environments** and create two environments:
   - `pypi` (optionally add required reviewers)
   - `testpypi`
2. On [PyPI](https://pypi.org/manage/account/publishing/), add a **Pending Publisher** (or project publisher):
   - **PyPI Project Name**: `pitwall-f1`
   - **Owner**: `notaayushsrivastava`
   - **Repository name**: `formula-1-prediction`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`
3. On [TestPyPI](https://test.pypi.org/manage/account/publishing/), add the same publisher with environment `testpypi`.

## Cutting a Release

1. Update `version` in `pyproject.toml` (e.g. `2.1.0`).
2. Add a matching `## 2.1.0 — ...` section at the top of `CHANGELOG.md`.
3. Commit and push to `main`:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: release 2.1.0"
   git push origin main
   ```
4. Tag and push the tag to trigger the release workflow:
   ```bash
   git tag v2.1.0
   git push origin v2.1.0
   ```

## Manual Dry-Runs & TestPyPI

You can also run **Actions → release → Run workflow** manually:
- `target: testpypi`: Builds and publishes to TestPyPI (default).
- `target: pypi`: Builds and publishes to PyPI.
- `target: build-only`: Runs checks, builds, and smoke-tests without publishing.
- `create_github_release`: Optionally create a GitHub Release from a manual run.
