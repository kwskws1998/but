"""Unit tests for asset integrity and archive safety."""

import hashlib
import io
import tarfile
import zipfile

import pytest

from cognitive_model_comparsion.download_assets import (
    extract_subtlex,
    validate_tar_members,
    verify_file,
    verify_tree,
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
