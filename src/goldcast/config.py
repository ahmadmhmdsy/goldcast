"""Load the TOML experiment config into small typed objects."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.toml"


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    source: str
    id: str
    lag_bdays: int
    kinds: tuple[str, ...]
    rvol_days: tuple[int, ...] = ()
    smooth_days: int = 0
    block: str = ""  # "" = R-002 base feature set; "B1".."B4" = R-004 blocks


@dataclass(frozen=True)
class FreqSpec:
    name: str
    rule: str
    windows: tuple[int, ...]
    refit_every: int
    baseline_quantile_window: int
    periods_per_year: int


@dataclass(frozen=True)
class Config:
    start: str
    raw_dir: Path
    target: str
    test_start: str
    regime_start: str
    quantiles: tuple[float, ...]
    cost_bp: float
    alpha: float
    freqs: dict[str, FreqSpec]
    series: tuple[SeriesSpec, ...] = field(default_factory=tuple)

    def spec(self, name: str) -> SeriesSpec:
        return next(s for s in self.series if s.name == name)

    @property
    def base_series(self) -> tuple[SeriesSpec, ...]:
        return tuple(s for s in self.series if not s.block)

    @property
    def blocks(self) -> list[str]:
        return sorted({s.block for s in self.series if s.block})


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    exp = raw["experiment"]
    freqs = {
        name: FreqSpec(
            name=name,
            rule=f["rule"],
            windows=tuple(f["windows"]),
            refit_every=f["refit_every"],
            baseline_quantile_window=f["baseline_quantile_window"],
            periods_per_year=f["periods_per_year"],
        )
        for name, f in raw["freq"].items()
    }
    series = tuple(
        SeriesSpec(
            name=s["name"],
            source=s["source"],
            id=s["id"],
            lag_bdays=s.get("lag_bdays", 0),
            kinds=tuple(s["kinds"]),
            rvol_days=tuple(s.get("rvol_days", ())),
            smooth_days=s.get("smooth_days", 0),
            block=s.get("block", ""),
        )
        for s in raw["series"]
    )
    raw_dir = Path(raw["data"]["raw_dir"])
    return Config(
        start=raw["data"]["start"],
        raw_dir=raw_dir if raw_dir.is_absolute() else ROOT / raw_dir,
        target=exp["target"],
        test_start=exp["test_start"],
        regime_start=exp["regime_start"],
        quantiles=tuple(exp["quantiles"]),
        cost_bp=exp["cost_bp"],
        alpha=exp["alpha"],
        freqs=freqs,
        series=series,
    )
