"""Manages output directory structure for simulation, test, and analysis results."""

from pathlib import Path
from datetime import datetime


class OutputManager:
    """Organizes and provides paths for result storage by category and format.

    Directory structure:
        output/
        ├── simulations/
        │   ├── images/       # PNG plots
        │   ├── animations/   # GIF/MP4
        │   └── data/         # CSV/HDF5 raw data
        ├── tests/
        │   ├── images/       # Test result plots
        │   ├── animations/   # Test animations
        │   └── reports/      # Test reports (MD/HTML)
        └── analysis/
            ├── images/       # Analysis plots
            ├── animations/   # Analysis animations
            └── reports/      # Analysis reports
    """

    CATEGORIES = ("simulations", "tests", "analysis")
    FORMATS = ("images", "animations", "data", "reports")

    def __init__(self, base_dir: str | Path = "output"):
        self._base = Path(base_dir)
        self._ensure_structure()

    def _ensure_structure(self):
        for cat in self.CATEGORIES:
            for fmt in self.FORMATS:
                (self._base / cat / fmt).mkdir(parents=True, exist_ok=True)

    def get_path(
        self,
        category: str,
        fmt: str,
        filename: str,
        timestamp: bool = True,
    ) -> Path:
        """Get full path for an output file.

        Args:
            category: One of 'simulations', 'tests', 'analysis'.
            fmt: One of 'images', 'animations', 'data', 'reports'.
            filename: Base filename (with extension).
            timestamp: If True, prepend ISO timestamp to filename.
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"Unknown category '{category}'. Use: {self.CATEGORIES}")
        if fmt not in self.FORMATS:
            raise ValueError(f"Unknown format '{fmt}'. Use: {self.FORMATS}")

        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            filename = f"{ts}_{stem}{suffix}"

        return self._base / category / fmt / filename

    def simulation_image(self, name: str, **kw) -> Path:
        return self.get_path("simulations", "images", f"{name}.png", **kw)

    def simulation_animation(self, name: str, ext: str = "gif", **kw) -> Path:
        return self.get_path("simulations", "animations", f"{name}.{ext}", **kw)

    def simulation_data(self, name: str, ext: str = "h5", **kw) -> Path:
        return self.get_path("simulations", "data", f"{name}.{ext}", **kw)

    def test_image(self, name: str, **kw) -> Path:
        return self.get_path("tests", "images", f"{name}.png", **kw)

    def test_animation(self, name: str, ext: str = "gif", **kw) -> Path:
        return self.get_path("tests", "animations", f"{name}.{ext}", **kw)

    def test_report(self, name: str, ext: str = "md", **kw) -> Path:
        return self.get_path("tests", "reports", f"{name}.{ext}", **kw)

    def analysis_image(self, name: str, **kw) -> Path:
        return self.get_path("analysis", "images", f"{name}.png", **kw)

    def analysis_animation(self, name: str, ext: str = "gif", **kw) -> Path:
        return self.get_path("analysis", "animations", f"{name}.{ext}", **kw)

    def analysis_report(self, name: str, ext: str = "md", **kw) -> Path:
        return self.get_path("analysis", "reports", f"{name}.{ext}", **kw)
