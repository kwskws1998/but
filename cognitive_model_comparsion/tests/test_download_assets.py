"""Unit tests for asset integrity and archive safety."""

import hashlib
import io
import tarfile
import zipfile

import pytest

import cognitive_model_comparsion.download_assets as download_module
from cognitive_model_comparsion.download_assets import (
    download_assets,
    extract_subtlex,
    load_manifest,
    validate_tar_members,
    verify_file,
    verify_tree,
    verify_tree_against_archive,
)


def test_verify_file_accepts_exact_size_and_hash(tmp_path):
    """An exact local artifact passes both integrity checks."""
    path = tmp_path / "asset.bin"
    content = b"exact artifact"
    path.write_bytes(content)

    verify_file(
        path,
        expected_bytes=len(content),
        expected_sha256=hashlib.sha256(content).hexdigest(),
    )


def test_verify_file_rejects_wrong_hash(tmp_path):
    """A same-size but modified artifact fails checksum validation."""
    path = tmp_path / "asset.bin"
    content = b"modified"
    path.write_bytes(content)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_file(
            path,
            expected_bytes=len(content),
            expected_sha256="0" * 64,
        )


def test_validate_tar_members_rejects_parent_traversal():
    """An archive member cannot escape the extraction directory."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("../outside.txt")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    buffer.seek(0)

    with tarfile.open(fileobj=buffer, mode="r:gz") as archive:
        with pytest.raises(ValueError, match="Unsafe archive member"):
            validate_tar_members(archive)


def test_verify_tree_requires_every_sentinel(tmp_path):
    """An extracted source tree fails if a required file is absent."""
    (tmp_path / "present.txt").write_text("present", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing.txt"):
        verify_tree(
            tmp_path,
            ("present.txt", "missing.txt"),
            "reference",
        )


def test_extract_subtlex_rejects_unexpected_members(tmp_path):
    """The SUBTLEX extractor rejects archives with extra members."""
    archive_path = tmp_path / "subtlex.zip"
    destination = tmp_path / "SUBTLEX_UK.txt"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("SUBTLEX-UK.txt", "expected")
        archive.writestr("extra.txt", "unexpected")
    asset = {
        "archive_member": "SUBTLEX-UK.txt",
        "extract_destination": str(destination),
        "extracted_bytes": len(b"expected"),
        "extracted_sha256": hashlib.sha256(b"expected").hexdigest(),
    }

    with pytest.raises(ValueError, match="Unexpected SUBTLEX-UK"):
        extract_subtlex(archive_path, asset)


def test_tree_verification_rejects_modified_extracted_source(tmp_path):
    """Pinned archives detect edits outside the small sentinel set."""
    archive_path = tmp_path / "source.tar.gz"
    content = b"pinned source"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("source-root/src/runtime.py")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    destination = tmp_path / "source"
    (destination / "src").mkdir(parents=True)
    extracted_path = destination / "src/runtime.py"
    extracted_path.write_bytes(content)

    verify_tree_against_archive(archive_path, destination, "source")
    extracted_path.write_bytes(b"locally modified")

    with pytest.raises(ValueError, match="differs from pinned archive"):
        verify_tree_against_archive(archive_path, destination, "source")


def test_onestop_manifest_pins_official_ordinary_archive():
    """OneStop metadata pins the official Ordinary interest-area ZIP exactly."""
    asset = load_manifest()["onestop_ordinary_interest_areas"]

    assert asset == {
        "source": "https://osf.io/download/xkgfz/",
        "landing_page": "https://osf.io/zn9sq/",
        "file_api": "https://api.osf.io/v2/files/683738581943ba131a53944c/",
        "destination": "data/raw/onestop/ia_Paragraph_ordinary.csv.zip",
        "archive_member": "ia_Paragraph_ordinary.csv",
        "bytes": 177291322,
        "sha256": (
            "8883478946ee52381e7057683c9e84dc"
            "69fcea9054acc34f0c900463a6b546e9"
        ),
        "license_eye_tracking": "CC BY 4.0",
        "license_text_and_annotations": "CC BY-SA 4.0",
    }


def test_onestop_download_route_selects_only_pinned_archive(monkeypatch):
    """The OneStop asset route delegates once without touching other assets."""
    calls = []

    def record_download(asset, destination_key):
        calls.append((asset, destination_key))

    monkeypatch.setattr(download_module, "download_file", record_download)
    download_assets("onestop")

    asset = load_manifest()["onestop_ordinary_interest_areas"]
    assert calls == [(asset, "destination")]
