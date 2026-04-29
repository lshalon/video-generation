"""Build a :class:`Storage` from a spec string.

The spec format is ``<scheme>:<location>`` where:

- ``local:<path>``        -> :class:`LocalStorage` rooted at ``<path>``
- ``gdrive:<folder-id>``  -> :class:`GoogleDriveStorage` rooted at the given Drive folder

The spec keeps storage selection out of the rest of the codebase: pipelines
just take a ``storage_spec`` string from config and call :func:`make_storage`.
"""

from __future__ import annotations

from video_generation.store.gdrive import GoogleDriveStorage
from video_generation.store.storage import LocalStorage, Storage


def make_storage(spec: str) -> Storage:
    """Parse a spec string and return the matching backend.

    Args:
        spec: A spec like ``"local:./data"`` or ``"gdrive:0AbCdEf..."``.

    Returns:
        A :class:`Storage` instance.
    """
    if ":" not in spec:
        raise ValueError(
            f"Storage spec must be '<scheme>:<location>', got: {spec!r}. "
            "Examples: 'local:./data', 'gdrive:0AbCdEf...'"
        )
    scheme, location = spec.split(":", 1)
    scheme = scheme.strip().lower()
    location = location.strip()
    if not location:
        raise ValueError(f"Storage spec is missing a location: {spec!r}")

    if scheme == "local":
        return LocalStorage(location)
    if scheme == "gdrive":
        return GoogleDriveStorage(folder_id=location)

    raise ValueError(f"Unknown storage scheme {scheme!r} in spec {spec!r}")
