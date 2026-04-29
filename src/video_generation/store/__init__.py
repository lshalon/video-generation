"""Content-addressed storage and run-tracking primitives.

This package provides three layers:

1. :class:`Storage` — a thin byte-level interface (put/get/exists/list/delete/url).
   Concrete backends: :class:`LocalStorage`, :class:`GoogleDriveStorage` (stub).
2. :class:`ContentStore` — content-addressed blob registry that hashes each
   blob with sha256 and persists a small ``.meta.json`` sidecar describing it.
3. :class:`RunStore` — record of a pipeline execution: which content was
   provided in which role, what params/code-version were used, and what each
   step produced (also tracked as content).
"""

from video_generation.store.code_version import current_code_version
from video_generation.store.content import ContentStore
from video_generation.store.context import StepContext
from video_generation.store.factory import make_storage
from video_generation.store.models import (
    CodeVersion,
    ContentMeta,
    ContentRef,
    RunInputs,
    RunParams,
    RunRecord,
    StepRecord,
)
from video_generation.store.runs import RunStore
from video_generation.store.storage import LocalStorage, Storage

__all__ = [
    "CodeVersion",
    "ContentMeta",
    "ContentRef",
    "ContentStore",
    "LocalStorage",
    "RunInputs",
    "RunParams",
    "RunRecord",
    "RunStore",
    "StepContext",
    "StepRecord",
    "Storage",
    "current_code_version",
    "make_storage",
]
