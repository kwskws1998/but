"""Tests for the trajectory-matched result ZIP packager."""

import json
import zipfile

from cognitive_model_comparsion.scripts.package_provo_skew4_trajectory_results import (
    write_archive,
)


def test_result_archive_contains_manifest_and_passes_crc(tmp_path):
    """The package records source hashes and verifies every ZIP member."""
    first = tmp_path / "first.csv"
    second = tmp_path / "second.json"
    first.write_text("value\n1\n", encoding="utf-8")
    second.write_text('{"status": "completed"}\n', encoding="utf-8")
    archive_path = tmp_path / "verification.zip"

    result = write_archive(
        {
            "analysis/first.csv": first,
            "run/second.json": second,
        },
        {
            "status": "complete",
            "comparison_design": "trajectory_matched_skew3_and_skew4",
        },
        archive_path,
    )

    assert result["zip_integrity"] == "passed"
    assert result["zip_member_count"] == 3
    assert archive_path.is_file()
    assert archive_path.with_suffix(".zip.sha256").is_file()
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        assert manifest["validation"]["status"] == "complete"
        assert {
            record["archive_path"] for record in manifest["files"]
        } == {"analysis/first.csv", "run/second.json"}
