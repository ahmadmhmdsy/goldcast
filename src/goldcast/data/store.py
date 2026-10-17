"""Raw-data store: one parquet per series plus a manifest recording when it was fetched."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from goldcast.config import Config
from goldcast.data.sources import FETCHERS


def _path(cfg: Config, name: str) -> Path:
    return cfg.raw_dir / f"{name}.parquet"


def fetch_all(cfg: Config, only: list[str] | None = None) -> dict[str, dict]:
    """Fetch every configured series; a failed source is reported, not fatal."""
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for spec in cfg.series:
        if only and spec.name not in only:
            continue
        try:
            s = FETCHERS[spec.source](spec.id, cfg.start)
        except Exception as exc:  # noqa: BLE001 - report every source failure
            manifest[spec.name] = {"source": spec.source, "id": spec.id, "error": repr(exc)}
            print(f"FAIL  {spec.name:<8} {spec.source}:{spec.id}  {exc!r}")
            continue
        path = _path(cfg, spec.name)
        s.to_frame("value").to_parquet(path)
        manifest[spec.name] = {
            "source": spec.source,
            "id": spec.id,
            "rows": int(s.size),
            "first": str(s.index.min().date()),
            "last": str(s.index.max().date()),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        }
        print(f"OK    {spec.name:<8} {spec.source}:{spec.id}  {s.size} rows  "
              f"{manifest[spec.name]['first']}..{manifest[spec.name]['last']}")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_raw(cfg: Config) -> dict[str, pd.Series]:
    """Load every stored series. Missing series are skipped with a notice."""
    out = {}
    for spec in cfg.series:
        path = _path(cfg, spec.name)
        if not path.exists():
            if spec.block:  # R-004 rows are optional for the R-002 commands
                continue
            print(f"missing raw series: {spec.name} (run `goldcast fetch`)")
            continue
        out[spec.name] = pd.read_parquet(path)["value"]
    if cfg.target not in out:
        raise FileNotFoundError(f"target series '{cfg.target}' has no raw data")
    return out
