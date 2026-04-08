"""Unit tests for OutputManager."""

import pytest
from pathlib import Path
from models.output_manager import OutputManager


class TestOutputManager:
    def test_directory_creation(self, tmp_path):
        om = OutputManager(tmp_path / "output")
        for cat in OutputManager.CATEGORIES:
            for fmt in OutputManager.FORMATS:
                assert (tmp_path / "output" / cat / fmt).is_dir()

    def test_simulation_image_path(self, tmp_path):
        om = OutputManager(tmp_path / "output")
        p = om.simulation_image("test_plot", timestamp=False)
        assert p == tmp_path / "output" / "simulations" / "images" / "test_plot.png"

    def test_test_report_path(self, tmp_path):
        om = OutputManager(tmp_path / "output")
        p = om.test_report("summary", ext="html", timestamp=False)
        assert p == tmp_path / "output" / "tests" / "reports" / "summary.html"

    def test_timestamp_prefix(self, tmp_path):
        om = OutputManager(tmp_path / "output")
        p = om.simulation_image("plot", timestamp=True)
        # Should have timestamp prefix
        assert len(p.stem) > len("plot")

    def test_invalid_category_raises(self, tmp_path):
        om = OutputManager(tmp_path / "output")
        with pytest.raises(ValueError, match="Unknown category"):
            om.get_path("invalid", "images", "test.png")

    def test_invalid_format_raises(self, tmp_path):
        om = OutputManager(tmp_path / "output")
        with pytest.raises(ValueError, match="Unknown format"):
            om.get_path("simulations", "invalid", "test.png")
