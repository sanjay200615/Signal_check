from pathlib import Path

import app.core.preflight as preflight


def test_live_preflight_does_not_require_local_index(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EVIDENCE_PROVIDER", "live")

    checks = preflight.run_preflight(tmp_path)

    names = {check.name for check in checks}
    assert "data/index/evidence.faiss" not in names
    assert "data/index/metadata.json" not in names
    assert "python:ddgs" in names
    assert "python:rapidocr_onnxruntime" in names
    assert "python:faster_whisper" in names
    assert all(check.ok for check in checks)


def test_live_preflight_reports_missing_search_dependency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVIDENCE_PROVIDER", "live")
    find_spec = preflight.importlib.util.find_spec
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda name: None if name == "ddgs" else find_spec(name),
    )

    checks = preflight.run_preflight(tmp_path)

    ddgs_check = next(check for check in checks if check.name == "python:ddgs")
    assert not ddgs_check.ok
    assert ddgs_check.detail == "package is not installed"


def test_local_preflight_reports_missing_index(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EVIDENCE_PROVIDER", "local")

    checks = preflight.run_preflight(tmp_path)

    missing = {
        check.name for check in checks if not check.ok
    }
    assert "data/index/evidence.faiss" in missing
    assert "data/index/metadata.json" in missing
