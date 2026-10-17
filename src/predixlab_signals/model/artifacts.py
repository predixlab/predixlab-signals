"""Integrity and atomic-write helpers for model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA_VERSION = 1
BUNDLE_MAGIC = b"PREDIXLAB_PIVOT_BUNDLE_V1\n"


class ArtifactIntegrityError(RuntimeError):
    """Raised before deserialization when an artifact cannot be trusted."""


@dataclass(frozen=True)
class VerifiedArtifact:
    """One immutable read of a verified model bundle."""

    bundle: bytes
    payload: bytes
    metadata: dict[str, Any]

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "sha256": sha256_bytes(self.bundle),
            "size_bytes": len(self.bundle),
        }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_path(path: Path) -> Path:
    return Path(path).with_suffix(".meta.json")


def artifact_identity(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_bytes(
        path,
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
    )


def build_bundle(
    payload: bytes, metadata: dict[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    """Embed the authoritative integrity manifest beside the pickle payload."""
    manifest = dict(metadata)
    manifest["artifact"] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "format": "python-pickle",
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }
    header = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    bundle = BUNDLE_MAGIC + struct.pack(">Q", len(header)) + header + payload
    return bundle, manifest


def _read_legacy_artifact(path: Path, bundle: bytes) -> VerifiedArtifact:
    """Verify the pre-bundle two-file format for migration compatibility."""
    sidecar = metadata_path(path)
    if not sidecar.exists():
        raise ArtifactIntegrityError(f"Model metadata is missing: {sidecar}")
    try:
        metadata = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Model metadata is unreadable: {sidecar}") from exc
    artifact = metadata.get("artifact") or {}
    expected = artifact.get("sha256")
    expected_size = artifact.get("size_bytes")
    actual = sha256_bytes(bundle)
    if not expected or expected_size is None:
        raise ArtifactIntegrityError(
            f"Legacy model has no trusted artifact identity: {path}. "
            "Refit it with this build, or restore a copy whose SHA-256 you "
            "trust, before deploying this version."
        )
    if len(bundle) != int(expected_size) or actual != expected:
        raise ArtifactIntegrityError(
            f"Model integrity check failed for {path}: expected {expected}, got {actual}"
        )
    return VerifiedArtifact(bundle=bundle, payload=bundle, metadata=metadata)


def read_verified_artifact(path: Path) -> VerifiedArtifact:
    """Read one immutable bundle and verify it before pickle deserialization."""
    path = Path(path)
    bundle = path.read_bytes()
    if not bundle.startswith(BUNDLE_MAGIC):
        return _read_legacy_artifact(path, bundle)
    offset = len(BUNDLE_MAGIC)
    if len(bundle) < offset + 8:
        raise ArtifactIntegrityError(f"Model bundle header is truncated: {path}")
    header_size = struct.unpack(">Q", bundle[offset : offset + 8])[0]
    header_start = offset + 8
    payload_start = header_start + header_size
    if payload_start > len(bundle):
        raise ArtifactIntegrityError(f"Model bundle manifest is truncated: {path}")
    try:
        metadata = json.loads(bundle[header_start:payload_start])
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError(f"Model bundle manifest is unreadable: {path}") from exc
    payload = bundle[payload_start:]
    artifact = metadata.get("artifact") or {}
    expected = artifact.get("sha256")
    expected_size = artifact.get("size_bytes")
    actual = sha256_bytes(payload)
    if not expected or expected_size is None:
        raise ArtifactIntegrityError(f"Model bundle has no artifact identity: {path}")
    if len(payload) != int(expected_size) or actual != expected:
        raise ArtifactIntegrityError(
            f"Model integrity check failed for {path}: expected {expected}, got {actual}"
        )
    return VerifiedArtifact(bundle=bundle, payload=payload, metadata=metadata)


def read_verified_bytes(path: Path) -> tuple[bytes, dict]:
    artifact = read_verified_artifact(path)
    return artifact.payload, artifact.metadata
