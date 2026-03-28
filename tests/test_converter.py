"""Tests for cad_physics_tuner.converter."""

from __future__ import annotations

import pytest

from cad_physics_tuner.converter import ConversionResult, UrdfToUsdConverter


class TestUrdfToUsdConverter:
    def test_converts_urdf_to_stub_usd(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("<robot name='test'></robot>", encoding="utf-8")

        converter = UrdfToUsdConverter(output_dir=tmp_path)
        result = converter.convert(urdf)

        assert isinstance(result, ConversionResult)
        assert result.usd_path.exists()
        assert result.usd_path.suffix == ".usd"
        assert result.source_path == urdf

    def test_default_output_dir_is_input_parent(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("<robot/>", encoding="utf-8")
        converter = UrdfToUsdConverter()  # no explicit output_dir
        result = converter.convert(urdf)
        assert result.usd_path.parent == tmp_path

    def test_missing_input_raises(self, tmp_path):
        converter = UrdfToUsdConverter(output_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            converter.convert(tmp_path / "nonexistent.urdf")

    def test_output_dir_created_if_missing(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("<robot/>", encoding="utf-8")
        out = tmp_path / "subdir" / "out"
        converter = UrdfToUsdConverter(output_dir=out)
        result = converter.convert(urdf)
        assert out.exists()
        assert result.usd_path.parent == out

    def test_stub_content(self, tmp_path):
        urdf = tmp_path / "test.urdf"
        urdf.write_text("<robot/>")
        converter = UrdfToUsdConverter(output_dir=tmp_path)
        result = converter.convert(urdf)
        content = result.usd_path.read_text()
        assert "#usda" in content
