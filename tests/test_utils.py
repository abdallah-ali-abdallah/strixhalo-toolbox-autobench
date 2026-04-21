"""Tests for toolbox_autobench.utils."""

from pathlib import Path

from toolbox_autobench.utils import (
    color,
    format_duration,
    normalize_path,
    normalize_power_profile,
    parse_csv_values,
    power_profile_label,
    safe_report_id,
    validate_power_profiles,
)


class TestFormatDuration:
    def test_zero(self):
        assert format_duration(0) == '00:00:00'

    def test_seconds_only(self):
        assert format_duration(45) == '00:00:45'

    def test_minutes_only(self):
        assert format_duration(90) == '00:01:30'

    def test_hours_only(self):
        assert format_duration(3661) == '01:01:01'

    def test_negative(self):
        assert format_duration(-10) == '00:00:00'


class TestParseCSVValues:
    def test_none(self):
        assert parse_csv_values(None) == []

    def test_single(self):
        assert parse_csv_values('hello') == ['hello']

    def test_multiple(self):
        assert parse_csv_values('a,b,c') == ['a', 'b', 'c']

    def test_with_spaces(self):
        assert parse_csv_values(' a , b , c ') == ['a', 'b', 'c']

    def test_empty_items(self):
        assert parse_csv_values('a,,b,,,c') == ['a', 'b', 'c']


class TestNormalizePath:
    def test_absolute(self, tmp_path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        p = sub / 'file.txt'
        p.touch()
        assert normalize_path(str(p)) == str(p.resolve())

    def test_tilde(self, tmp_path, monkeypatch):
        monkeypatch.setenv('HOME', str(tmp_path))
        result = normalize_path('~/test.txt')
        assert 'test.txt' in result


class TestSafeReportId:
    def test_simple_name(self):
        # Keeps directory components joined with __
        result = safe_report_id('/models/model.gguf')
        assert 'model' in result and not result.endswith('.gguf')

    def test_with_spaces(self):
        result = safe_report_id('/path/to/my model.gguf')
        assert 'my_model' in result and not result.endswith('.gguf')

    def test_nested_path(self):
        result = safe_report_id('/a/b/c/model.gguf')
        assert 'c' in result and 'model' in result

    def test_no_extension(self):
        assert safe_report_id('mymodel') == 'mymodel'


class TestNormalizePowerProfile:
    def test_exact_match(self):
        assert normalize_power_profile('power-saver') == 'power-saver'
        assert normalize_power_profile('balanced') == 'balanced'
        assert normalize_power_profile('performance') == 'performance'

    def test_alias_saving(self):
        for alias in ['saving', 'saver', 'powersave', 'power save']:
            assert normalize_power_profile(alias) == 'power-saver'

    def test_alias_performance(self):
        for alias in ['perf', 'fast', 'high', 'turbo']:
            assert normalize_power_profile(alias) == 'performance'

    def test_mixed_case(self):
        assert normalize_power_profile('Power-Saver') == 'power-saver'


class TestValidatePowerProfiles:
    def test_none_returns_all(self):
        result = validate_power_profiles(None)
        assert set(result) == {'power-saver', 'balanced', 'performance'}

    def test_single_valid(self):
        assert validate_power_profiles('balanced') == ['balanced']

    def test_invalid_raises(self):
        import pytest
        with pytest.raises(ValueError, match='invalid power profile'):
            validate_power_profiles('invalid-profile')

    def test_duplicates_removed(self):
        result = validate_power_profiles('balanced,balanced')
        assert result == ['balanced']


class TestPowerProfileLabel:
    def test_all_labels(self):
        assert power_profile_label('power-saver') == 'Power Saver'
        assert power_profile_label('balanced') == 'Balanced'
        assert power_profile_label('performance') == 'Performance'

    def test_unknown(self):
        # Returns the input string for unknown profiles
        assert power_profile_label('unknown') == 'unknown'


class TestColor:
    def test_no_color_when_no_tty(self, monkeypatch):
        monkeypatch.setattr('os.isatty', lambda fd: False)
        result = color('hello', '\x1b[31m')
        assert result == 'hello'
