"""Content-addressed blob registry.

A :class:`ContentStore` writes immutable byte blobs into the configured
:class:`Storage` keyed by ``library/<aa>/<sha256><ext>`` and records a small
``.meta.json`` sidecar at the same prefix. Identical bytes register exactly
once (dedup by sha256).
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
from datetime import UTC, datetime
from pathlib import Path

from video_generation.store.models import ContentMeta, ContentRef
from video_generation.store.storage import Storage

logger = logging.getLogger(__name__)

LIBRARY_PREFIX = "library"

# Map kinds to the suffix we want in storage when an original_name is unknown.
_DEFAULT_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "text/markdown": ".md",
    "text/plain": ".txt",
    "application/json": ".json",
}

# Map suffixes -> kind label used in metadata.
_KIND_BY_SUFFIX = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".mp4": "video",
    ".mov": "video",
    ".md": "text",
    ".txt": "text",
    ".json": "json",
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _infer_mime(original_name: str | None, mime: str | None) -> str | None:
    if mime:
        return mime
    if original_name:
        guess, _ = mimetypes.guess_type(original_name)
        return guess
    return None


def _infer_kind(original_name: str | None, mime: str | None) -> str:
    if mime:
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("video/"):
            return "video"
        if mime == "application/json":
            return "json"
        if mime.startswith("text/"):
            return "text"
    if original_name:
        suffix = Path(original_name).suffix.lower()
        if suffix in _KIND_BY_SUFFIX:
            return _KIND_BY_SUFFIX[suffix]
    return "blob"


def _suffix_for(original_name: str | None, mime: str | None) -> str:
    if original_name:
        suffix = Path(original_name).suffix.lower()
        if suffix:
            return suffix
    if mime and mime in _DEFAULT_EXTENSIONS:
        return _DEFAULT_EXTENSIONS[mime]
    return ""


def _blob_key(content_id: str, suffix: str) -> str:
    return f"{LIBRARY_PREFIX}/{content_id[:2]}/{content_id}{suffix}"


def _meta_key(content_id: str) -> str:
    return f"{LIBRARY_PREFIX}/{content_id[:2]}/{content_id}.meta.json"


def _caption_key(content_id: str) -> str:
    return f"{LIBRARY_PREFIX}/{content_id[:2]}/{content_id}.caption.json"


class ContentStore:
    """Content-addressed blob registry on top of a :class:`Storage` backend."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def register_bytes(
        self,
        data: bytes,
        kind: str | None = None,
        original_name: str | None = None,
        mime: str | None = None,
        produced_by: dict[str, str] | None = None,
    ) -> ContentRef:
        """Register a byte blob and return its content reference.

        If a blob with the same sha256 is already present, the existing
        metadata is returned unchanged (the bytes are not rewritten).
        """
        content_id = _sha256_hex(data)
        meta_key = _meta_key(content_id)

        if self.storage.exists(meta_key):
            existing = ContentMeta.from_dict(json.loads(self.storage.get(meta_key)))
            return ContentRef(content_id=content_id, meta=existing)

        resolved_mime = _infer_mime(original_name, mime)
        resolved_kind = kind or _infer_kind(original_name, resolved_mime)
        suffix = _suffix_for(original_name, resolved_mime)

        meta = ContentMeta(
            sha256=content_id,
            size=len(data),
            mime=resolved_mime,
            kind=resolved_kind,
            original_name=original_name,
            registered_at=_utc_now_iso(),
            produced_by=produced_by,
        )

        self.storage.put(_blob_key(content_id, suffix), data, mime=resolved_mime)
        self.storage.put(
            meta_key,
            json.dumps(meta.to_dict(), indent=2).encode("utf-8"),
            mime="application/json",
        )
        logger.debug(
            "Registered content %s (%d bytes, kind=%s)", content_id[:16], len(data), resolved_kind
        )
        return ContentRef(content_id=content_id, meta=meta)

    def register_path(
        self,
        path: Path,
        kind: str | None = None,
        produced_by: dict[str, str] | None = None,
    ) -> ContentRef:
        """Register a file from disk; original_name is taken from the path."""
        if not path.is_file():
            raise FileNotFoundError(f"Cannot register, not a file: {path}")
        return self.register_bytes(
            data=path.read_bytes(),
            kind=kind,
            original_name=path.name,
            produced_by=produced_by,
        )

    def get_bytes(self, content_id: str) -> bytes:
        meta = self.get_meta(content_id)
        suffix = _suffix_for(meta.original_name, meta.mime)
        return self.storage.get(_blob_key(content_id, suffix))

    def get_meta(self, content_id: str) -> ContentMeta:
        meta_key = _meta_key(content_id)
        if not self.storage.exists(meta_key):
            raise KeyError(f"Unknown content_id: {content_id}")
        return ContentMeta.from_dict(json.loads(self.storage.get(meta_key)))

    def get_ref(self, content_id: str) -> ContentRef:
        return ContentRef(content_id=content_id, meta=self.get_meta(content_id))

    def materialize(self, content_id: str, dest: Path) -> Path:
        """Write the blob's bytes to ``dest`` and return the path.

        ``dest`` may be a directory (the blob's preferred filename is used)
        or a full file path. Useful for handing bytes to libraries that need
        an actual filesystem path (e.g. fal_client uploads).
        """
        meta = self.get_meta(content_id)
        suffix = _suffix_for(meta.original_name, meta.mime)
        if dest.is_dir() or (not dest.exists() and dest.suffix == ""):
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / f"{content_id[:16]}{suffix}"
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            target = dest
        target.write_bytes(self.get_bytes(content_id))
        return target

    def blob_key(self, content_id: str) -> str:
        """Storage key where the bytes for ``content_id`` live."""
        meta = self.get_meta(content_id)
        return _blob_key(content_id, _suffix_for(meta.original_name, meta.mime))

    # ------------------------------------------------------------------ #
    # Captions (persistent per-asset annotation)                         #
    # ------------------------------------------------------------------ #
    #
    # A caption is a short, role-explicit description of a content blob
    # (currently only used for product images). It lives next to the blob
    # in a sidecar at ``library/<aa>/<sha>.caption.json`` so it persists
    # with the asset across runs and across pipelines. Once a blob has
    # been captioned, no later run needs to re-invoke an LLM for it.
    #
    # The on-disk schema is intentionally tiny::
    #
    #     {
    #         "text": "ON A REAL MODEL'S EAR \u2014 ...",
    #         "model": "gemini-3.1-pro-preview",
    #         "generated_at": "2026-04-18T12:34:56Z"
    #     }
    #
    # Captions are immutable in spirit but mutable on disk: callers may
    # overwrite a caption to manually correct it.

    def get_caption_record(self, content_id: str) -> dict[str, str] | None:
        """Return the full caption sidecar for ``content_id``, or None if absent."""
        key = _caption_key(content_id)
        if not self.storage.exists(key):
            return None
        return dict(json.loads(self.storage.get(key)))

    def get_caption(self, content_id: str) -> str | None:
        """Return just the caption text, or None if no caption exists yet."""
        record = self.get_caption_record(content_id)
        if record is None:
            return None
        return record.get("text") or None

    def set_caption(self, content_id: str, text: str, model: str) -> None:
        """Persist a caption for ``content_id``.

        Validates that the content exists. Writes the caption as JSON; any
        existing caption is overwritten.
        """
        if not text.strip():
            raise ValueError(f"Refusing to set empty caption for {content_id[:16]}")
        self.get_meta(content_id)  # raises KeyError if the blob is unknown
        record = {
            "text": text.strip(),
            "model": model,
            "generated_at": _utc_now_iso(),
        }
        self.storage.put(
            _caption_key(content_id),
            json.dumps(record, indent=2).encode("utf-8"),
            mime="application/json",
        )

    def delete_caption(self, content_id: str) -> bool:
        """Remove the caption sidecar for ``content_id``. Returns True if removed."""
        key = _caption_key(content_id)
        if not self.storage.exists(key):
            return False
        self.storage.delete(key)
        return True
