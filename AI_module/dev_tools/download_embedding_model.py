"""
Download the embedding model into AI_module/data/models/... for offline runs and tests.

Requires network once. After this, config resolves EMBEDDING_MODEL_NAME to the local folder
(see AI_module.config._resolve_embedding_model_name).

Run from repo root:
    python AI_module/dev_tools/download_embedding_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if _root not in [Path(p).resolve() for p in sys.path]:
    sys.path.insert(0, str(_root))

from AI_module.config import (  # noqa: E402
    EMBEDDING_MODEL_HUB_ID,
    EMBEDDING_MODEL_LOCAL_DIR,
)


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        return 1

    dest = EMBEDDING_MODEL_LOCAL_DIR.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {EMBEDDING_MODEL_HUB_ID!r} -> {dest} ...", flush=True)
    snapshot_download(repo_id=EMBEDDING_MODEL_HUB_ID, local_dir=str(dest))
    if not (dest / "config.json").exists():
        print("Warning: config.json not found after download; load may still fail.", file=sys.stderr)
        return 1
    print("Done. You can run tests and the app offline for this model.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
