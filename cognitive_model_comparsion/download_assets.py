"""Download and verify the exact Provo, OB1, and ET2 reference assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "asset_manifest.json"
TREE_SENTINELS = {
    "ob1_provo_2024": (
        "README.md",
        "src/main.py",
        "src/evaluation.py",
        "src/parameters.py",
    ),
    "et2_torontocl_cmcl_2021": (
        "Readme.md",
        "data/provo.csv",
        "notebooks/ProvoProcess.py",
        "scripts/run_roberta.py",
    ),
}


def load_manifest() -> dict:
    """Load the checked-in asset manifest."""
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)["assets"]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    """Raise if a downloaded file does not match the manifest."""
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Size mismatch for {path}: expected {expected_bytes}, got {actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )


def download_file(asset: dict, destination_key: str) -> Path:
    """Download one file atomically and verify it before installation."""
    destination = ROOT / asset[destination_key]
    if destination.is_file():
        verify_file(destination, asset["bytes"], asset["sha256"])
        print(f"verified {destination.relative_to(ROOT)}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        asset["source"],
        headers={"User-Agent": "GazeReward-cognitive-comparison/1.0"},
    )
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".part",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(request) as response:
                shutil.copyfileobj(response, temporary)
            verify_file(temporary_path, asset["bytes"], asset["sha256"])
            temporary_path.replace(destination)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    print(f"downloaded {destination.relative_to(ROOT)}")
    return destination


def validate_tar_members(archive: tarfile.TarFile) -> str:
    """Validate archive paths and return the single top-level directory."""
    top_levels = set()
    for member in archive.getmembers():
        member_path = PurePosixPath(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe archive member: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"Links are not allowed in the archive: {member.name}")
        if member_path.parts:
            top_levels.add(member_path.parts[0])
    if len(top_levels) != 1:
        raise ValueError(f"Expected one archive root, found {sorted(top_levels)}")
    return top_levels.pop()


def verify_tree(destination: Path, sentinels: tuple[str, ...], label: str) -> None:
    """Check that an extracted pinned source tree contains its required files."""
    missing = [name for name in sentinels if not (destination / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{label} extraction is incomplete at {destination}; missing {missing}"
        )


def extract_tree(
    archive_path: Path,
    destination: Path,
    sentinels: tuple[str, ...],
    label: str,
) -> None:
    """Extract a pinned source archive without overwriting an existing tree."""
    if destination.exists():
        verify_tree(destination, sentinels, label)
        print(f"verified {destination.relative_to(ROOT)}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{label.lower()}_extract_",
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive_root = validate_tar_members(archive)
            archive.extractall(temporary_root, filter="data")
        extracted_root = temporary_root / archive_root
        verify_tree(extracted_root, sentinels, label)
        extracted_root.replace(destination)
    print(f"extracted {destination.relative_to(ROOT)}")


def verify_subtlex(asset: dict) -> None:
    """Verify the official SUBTLEX-UK archive and extracted text table."""
    verify_file(
        ROOT / asset["archive_destination"],
        asset["bytes"],
        asset["sha256"],
    )
    verify_file(
        ROOT / asset["extract_destination"],
        asset["extracted_bytes"],
        asset["extracted_sha256"],
    )
    print(f"verified {asset['archive_destination']}")
    print(f"verified {asset['extract_destination']}")


def extract_subtlex(archive_path: Path, asset: dict) -> None:
    """Extract the single pinned SUBTLEX-UK text member atomically."""
    destination = ROOT / asset["extract_destination"]
    if destination.is_file():
        verify_file(
            destination,
            asset["extracted_bytes"],
            asset["extracted_sha256"],
        )
        print(f"verified {destination.relative_to(ROOT)}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        expected_member = asset["archive_member"]
        if names != [expected_member]:
            raise ValueError(
                f"Unexpected SUBTLEX-UK archive members: {names}"
            )
        member_path = PurePosixPath(expected_member)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe ZIP member: {expected_member}")
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                with archive.open(expected_member) as source:
                    shutil.copyfileobj(source, temporary)
                verify_file(
                    temporary_path,
                    asset["extracted_bytes"],
                    asset["extracted_sha256"],
                )
                temporary_path.replace(destination)
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
    print(f"extracted {destination.relative_to(ROOT)}")


def verify_source_archive(asset: dict, asset_name: str, label: str) -> None:
    """Verify one pinned archive and its extracted source tree."""
    verify_file(
        ROOT / asset["archive_destination"],
        asset["bytes"],
        asset["sha256"],
    )
    verify_tree(
        ROOT / asset["extract_destination"],
        TREE_SENTINELS[asset_name],
        label,
    )
    print(f"verified {asset['archive_destination']}")
    print(f"verified {asset['extract_destination']}")


def download_source_archive(asset: dict, asset_name: str, label: str) -> None:
    """Download and extract one pinned source archive."""
    archive_path = download_file(asset, "archive_destination")
    extract_tree(
        archive_path,
        ROOT / asset["extract_destination"],
        TREE_SENTINELS[asset_name],
        label,
    )


def verify_assets(selected: str) -> None:
    """Verify already-downloaded assets without network access."""
    assets = load_manifest()
    if selected in {"all", "provo"}:
        for name in ("provo_eye_tracking", "provo_predictability"):
            asset = assets[name]
            verify_file(
                ROOT / asset["destination"],
                asset["bytes"],
                asset["sha256"],
            )
            print(f"verified {asset['destination']}")
    if selected in {"all", "ob1"}:
        verify_source_archive(assets["ob1_provo_2024"], "ob1_provo_2024", "OB1")
    if selected in {"all", "et2-reference"}:
        verify_source_archive(
            assets["et2_torontocl_cmcl_2021"],
            "et2_torontocl_cmcl_2021",
            "ET2",
        )
    if selected in {"all", "subtlex"}:
        verify_subtlex(assets["subtlex_uk"])


def download_assets(selected: str) -> None:
    """Download selected data and extract selected source archives."""
    assets = load_manifest()
    if selected in {"all", "provo"}:
        for name in ("provo_eye_tracking", "provo_predictability"):
            download_file(assets[name], "destination")
    if selected in {"all", "ob1"}:
        download_source_archive(assets["ob1_provo_2024"], "ob1_provo_2024", "OB1")
    if selected in {"all", "et2-reference"}:
        download_source_archive(
            assets["et2_torontocl_cmcl_2021"],
            "et2_torontocl_cmcl_2021",
            "ET2",
        )
    if selected in {"all", "subtlex"}:
        asset = assets["subtlex_uk"]
        archive_path = download_file(asset, "archive_destination")
        extract_subtlex(archive_path, asset)


def parse_args() -> argparse.Namespace:
    """Parse the asset selection and offline verification mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset",
        choices=("all", "provo", "ob1", "et2-reference", "subtlex"),
        default="all",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the downloader or the offline integrity check."""
    args = parse_args()
    if args.verify_only:
        verify_assets(args.asset)
    else:
        download_assets(args.asset)


if __name__ == "__main__":
    main()
