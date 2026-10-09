"""On-disk layout of a run folder."""

from __future__ import annotations

from pathlib import Path


class RunPaths:
    """Every path within a run directory.

    ::

        <run_dir>/
          recipe.json            # editable intent — precious
          fit_config.json        # per-lane fit intent — precious, editable
          manifest.json          # cache ledger — disposable
          cache/
            image/<hash>.npy
            background/<hash>.npz
            profiles/<lane>__<hash>.json
            fits/<lane>__<hash>.json
          render/ ...
          results.csv / results.json / report.md
          log/history.jsonl
    """

    def __init__(self, run_dir: str | Path):
        self.root = Path(run_dir)

    # --- top-level files ---
    @property
    def recipe(self) -> Path:
        return self.root / "recipe.json"

    @property
    def fit_config(self) -> Path:
        return self.root / "fit_config.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def results_csv(self) -> Path:
        return self.root / "results.csv"

    @property
    def results_json(self) -> Path:
        return self.root / "results.json"

    @property
    def report(self) -> Path:
        return self.root / "report.md"

    # --- directories ---
    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def render(self) -> Path:
        return self.root / "render"

    @property
    def render_fits(self) -> Path:
        return self.render / "fits"

    @property
    def log(self) -> Path:
        return self.root / "log"

    @property
    def history(self) -> Path:
        return self.log / "history.jsonl"

    # --- cache artifacts ---
    def image_cache(self, image_hash: str) -> Path:
        return self.cache / "image" / f"{image_hash}.npy"

    def background_cache(self, background_hash: str) -> Path:
        return self.cache / "background" / f"{background_hash}.npz"

    def profile_cache(self, lane_index: int, profile_hash: str) -> Path:
        return self.cache / "profiles" / f"lane_{lane_index:03d}__{profile_hash}.json"

    def fit_cache(self, lane_index: int, fit_hash: str) -> Path:
        return self.cache / "fits" / f"lane_{lane_index:03d}__{fit_hash}.json"

    def fit_render(self, lane_index: int) -> Path:
        return self.render_fits / f"lane_{lane_index:03d}.png"

    def ensure_dirs(self) -> None:
        """Create the full directory skeleton (idempotent)."""
        for d in (
            self.cache / "image",
            self.cache / "background",
            self.cache / "profiles",
            self.cache / "fits",
            self.render_fits,
            self.log,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.recipe.exists()
