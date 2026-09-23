import json

from typer.testing import CliRunner

from mapd.cli import app


def test_status_reports_partial_artifact_progress(tmp_path):
    artifact_dir = tmp_path / "run"
    artifact_dir.mkdir()
    (artifact_dir / "artifacts.jsonl").write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")

    result = CliRunner().invoke(app, ["status", "--artifact-dir", str(artifact_dir)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["artifact_count"] == 2
    assert payload["complete"] is False
