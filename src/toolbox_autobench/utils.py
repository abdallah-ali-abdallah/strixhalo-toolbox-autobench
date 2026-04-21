"""Shared utilities for toolbox-autobench."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


# ── ANSI colours ──────────────────────────────────────────────────────────────

ANSI_BLUE = '\x1b[34m'
ANSI_BOLD = '\x1b[1m'
ANSI_CYAN = '\x1b[36m'
ANSI_GREEN = '\x1b[32m'
ANSI_RESET = '\x1b[0m'
ANSI_YELLOW = '\x1b[33m'


def supports_color(stream_fd: int = 1) -> bool:
    """Return True when *stream_fd* appears to be a TTY."""
    return os.isatty(stream_fd)


def color(text: str, code: str) -> str:
    """Wrap *text* in an ANSI colour *code* (no-op when no colour is supported)."""
    if supports_color():
        return f'{code}{text}{ANSI_RESET}'
    return text


def format_duration(seconds: float) -> str:
    """Return *seconds* as ``HH:MM:SS``."""
    seconds = max(0.0, float(seconds))
    sec_int = int(seconds)
    hours = sec_int // 3600
    minutes = (sec_int % 3600) // 60
    seconds = sec_int % 60
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def parse_csv_values(raw: str | None) -> list[str]:
    """Split a comma-separated string into a list of stripped tokens."""
    if raw is None:
        return []
    return [item.strip() for item in raw.split(',') if item.strip()]


def normalize_path(path: str | Path) -> str:
    """Return the canonical absolute path as a string."""
    return str(Path(path).expanduser().resolve())


def safe_report_id(path: str | Path) -> str:
    """Derive a filesystem-safe report ID from *path*."""
    value = str(path).strip().replace('\\', '/').replace(' ', '_')
    value = '/'.join(filter(None, value.split('/')))
    parts = [p for p in value.split('/') if p]
    value = '__'.join(parts)
    value = re.sub(r'[^A-Za-z0-9._-]', '_', value).strip('_')
    if value.endswith('.gguf'):
        value = value[:-5]
    return value or 'model'


def is_hex(value: str) -> bool:
    """Return True when *value* looks like a hexadecimal string (12-64 chars)."""
    return bool(re.fullmatch(r'[0-9a-f]{12,64}', value))


def power_profile_label(profile: str) -> str | None:
    """Human-readable label for a power profile name."""
    mapping = {
        'power-saver': 'Power Saver',
        'balanced': 'Balanced',
        'performance': 'Performance',
    }
    return mapping.get(profile, profile)


def normalize_toolbox_name(name: str) -> str:
    """Normalize a toolbox name (strip slashes, remove ``toolbox/`` prefix)."""
    if not name:
        return name
    name = name.strip()
    name = name.rstrip('/')
    if name.startswith('toolbox/'):
        name = name[len('toolbox/'):]
    return name


def parse_toolbox_list(lines: Iterable[str]) -> Tuple[list[str], dict[str, str]]:
    """Parse ``toolbox list --containers`` output.

    Returns ``(discovered_names, id_to_name_map)`` for containers whose names
    start with ``llama-``.
    """
    discovered: list[str] = []
    id_map: dict[str, str] = {}
    seen: set[str] = set()

    for line in lines:
        if not line:
            continue
        normalized = ' '.join(line.split())
        if not normalized:
            continue
        upper = normalized.upper()
        # Skip header / separator / footer rows
        if (upper.startswith('CONTAINER ID') or upper.startswith('IMAGE')
                or upper.startswith('ID') or upper.startswith('NAME')
                or upper.startswith('----') or upper.startswith('NO')):
            continue

        parts = normalized.split()
        if len(parts) < 2:
            continue

        candidate_id = parts[0]
        if not is_hex(candidate_id):
            continue

        resolved_name = normalize_toolbox_name(parts[1])
        if not resolved_name.startswith('llama-'):
            continue
        if resolved_name in seen:
            continue

        discovered.append(resolved_name)
        seen.add(resolved_name)
        id_map[resolved_name] = candidate_id

    return discovered, id_map


def normalize_power_profile(value: str) -> str:
    """Canonicalise a power-profile string to one of ``power-saver | balanced | performance``."""
    if not value:
        value = ''
    lowered = (value.strip().lower()
               .replace('_', ' ')
               .replace('-', ' '))
    compact = re.sub(r'[^a-z0-9 ]+', ' ', lowered)
    compact = re.sub(r'\s+', ' ', compact).strip()

    direct = {
        'saving': 'power-saver',
        'saver': 'power-saver',
        'savers': 'power-saver',
        'powersave': 'power-saver',
        'powersaving': 'power-saver',
        'power saving': 'power-saver',
        'power saver': 'power-saver',
        'power save': 'power-saver',
        'power saver mode': 'power-saver',
        'power save mode': 'power-saver',
        'power saver profile': 'power-saver',
        'power save profile': 'power-saver',
        'balance': 'balanced',
        'balanced': 'balanced',
        'balanced mode': 'balanced',
        'balance mode': 'balanced',
        'balanced profile': 'balanced',
        'default': 'balanced',
        'normal': 'balanced',
        'performance': 'performance',
        'perf': 'performance',
        'fast': 'performance',
        'high': 'performance',
        'turbo': 'performance',
        'performance mode': 'performance',
        'performance profile': 'performance',
    }

    if compact in direct:
        return direct[compact]

    if re.search(r'\bpower[-\s]*sav(?:e|ing|er)\b', compact):
        return 'power-saver'
    if re.search(r'\bpower\b.*\bsav(?:e|ing|er)\b', compact):
        return 'power-saver'
    if re.search(r'\b(balanced|balance|default|normal)\b', compact):
        return 'balanced'
    if re.search(r'\b(performance|perf|fast|high|turbo)\b', compact):
        return 'performance'

    return compact.replace(' ', '-')


def validate_power_profiles(raw: str | None) -> list[str]:
    """Parse and validate a comma-separated power-profile string.

    Raises ``ValueError`` on invalid input.
    """
    if not raw:
        return ['power-saver', 'balanced', 'performance']

    selected: list[str] = []
    seen: set[str] = set()

    for item in parse_csv_values(raw):
        normalized = normalize_power_profile(item)
        if normalized not in ('power-saver', 'balanced', 'performance'):
            raise ValueError(
                f"invalid power profile '{item}'. "
                "valid values: power-saver, balanced, performance"
            )
        if normalized in seen:
            continue
        selected.append(normalized)
        seen.add(normalized)

    if not selected:
        raise ValueError('no valid power profiles selected')

    return selected


# ── Internal helpers used by discover_toolboxes ────────────────────────────────

def _run(cmd: Sequence[str]) -> str | None:
    """Run *cmd* and return stdout stripped, or ``None`` on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except (subprocess.TimeoutExpired, OSError):
        return None


def discover_toolboxes() -> Tuple[list[str], dict[str, str]]:
    """Discover ``llama-*`` toolboxes via ``toolbox list --containers``."""
    outputs_text = _run(['toolbox', 'list', '--containers'])
    if not outputs_text:
        raise RuntimeError("unable to list toolboxes via 'toolbox list'")

    outputs = outputs_text.splitlines()
    return parse_toolbox_list(outputs)
