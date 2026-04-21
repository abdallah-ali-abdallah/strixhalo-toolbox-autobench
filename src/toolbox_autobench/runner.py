"""Core benchmark execution logic for toolbox-autobench."""

from __future__ import annotations

import glob
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .reporting import build_reports_from_result_folder as _build_reports_from_result_folder
from .utils import (
    discover_toolboxes,
    format_duration,
    normalize_path,
    normalize_power_profile,
    parse_csv_values,
    power_profile_label,
    safe_report_id,
    validate_power_profiles,
)


@dataclass
class ProfileResult:
    """Result of a single power-profile benchmark run."""
    profile: str
    label: str
    status: str = 'unknown'
    runtime_seconds: float = 0.0
    start_time: str = ''
    end_time: str = ''
    model: str = ''
    backend: str = ''
    device: str = ''
    prompt_tokens: int | str = 'n/a'
    gen_tokens: int | str = 'n/a'
    prompt_tps: float | None = None
    gen_tps: float | None = None
    prompt_std: float | None = None
    gen_std: float | None = None
    cpu_ram_used_gib: float | None = None
    cpu_ram_total_gib: float | None = None
    cpu_temp_c: float | None = None
    gpu_temp_c: float | None = None
    gpu_power_w: float | None = None
    model_type: str = 'unknown'
    json: str = ''
    log: str = ''
    results: list[dict] = field(default_factory=list)


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""
    models: list[str]
    toolbox: Optional[str] = None
    run_all_backends: bool = False
    mode: str = 'quick'
    prompt_tokens: list[int] = field(default_factory=lambda: [512])
    gen_tokens: list[int] = field(default_factory=lambda: [512])
    repetitions: int = 1
    power_profiles: list[str] = field(default_factory=lambda: ['power-saver', 'balanced', 'performance'])
    run_root: Optional[str] = None
    dry_run: bool = False
    progress_sink: Optional[Callable] = None


@dataclass
class RunEvent:
    """Event emitted during benchmark execution."""
    type: str  # 'start', 'progress', 'end', 'error'
    model: str = ''
    toolbox: str = ''
    profile: str = ''
    message: str = ''
    data: dict = field(default_factory=dict)


@dataclass
class RunProgress:
    """Accumulated progress state."""
    total: int = 0
    completed: int = 0
    current_model: str = ''
    current_toolbox: str = ''
    current_profile: str = ''
    events: list[RunEvent] = field(default_factory=list)


def discover_toolboxes(run_all_backends: bool = False) -> Tuple[list[str], dict[str, str]]:
    """Discover llama-* toolboxes, optionally filtering for all backends."""
    if run_all_backends:
        return _discover_toolboxes()
    # Return default toolbox
    return ['llama-rocm7-nightlies'], {'llama-rocm7-nightlies': 'default'}


def discover_toolboxes_from_env() -> Tuple[list[str], dict[str, str]]:
    """Discover toolboxes from environment variables or fallback."""
    env_toolbox = os.environ.get('TOOLBOX_AUTOBENCH_TOOLBOX')
    if env_toolbox:
        return [env_toolbox], {env_toolbox: 'env'}
    return _discover_toolboxes()


def collect_system_metadata() -> dict:
    """Collect system metadata for reports."""
    try:
        uname = os.uname()
    except AttributeError:
        uname = type('uname', (), {'sysname': '', 'release': '', 'version': '', 'machine': '', 'nodename': ''})()

    # Try to read OS release info
    os_info = {}
    try:
        with open('/etc/os-release') as f:
            for line in f:
                if '=' in line:
                    key, _, val = line.strip().partition('=')
                    os_info[key] = val.strip('"')
    except OSError:
        pass

    # Try to get CPU model
    cpu_model = 'unknown'
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    cpu_model = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass

    # Try to get RAM total
    ram_total_gib = 0.0
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    kb = int(line.split()[1])
                    ram_total_gib = kb / (1024 * 1024)
                    break
    except OSError:
        pass

    return {
        'machine_name': socket.gethostname(),
        'machine_model': os_info.get('PRETTY_NAME', ''),
        'cpu_model': cpu_model,
        'architecture': uname.machine or platform.machine(),
        'kernel_version': uname.release or '',
        'kernel_name': uname.sysname or '',
        'kernel_build': uname.version or '',
        'os_name': os_info.get('NAME', uname.sysname or ''),
        'os_id': os_info.get('ID', ''),
        'os_version_id': os_info.get('VERSION_ID', ''),
        'os_version': os_info.get('VERSION', ''),
        'ram_total_gib': ram_total_gib,
    }


def list_models(source: str | Path) -> list[str]:
    """List GGUF models from a file path or folder."""
    source = normalize_path(source)
    path = Path(source)

    if path.is_file():
        if path.suffix.lower() == '.gguf':
            return [str(path)]
        return []

    if path.is_dir():
        # Non-recursive glob for .gguf files
        models = sorted(glob.glob(str(path / '*.gguf')))
        # Filter to keep first shard of split GGUFs
        seen_prefixes: set[str] = set()
        result = []
        for m in models:
            basename = os.path.basename(m)
            # Check if it's a split shard (e.g., -00001-of-000NN.gguf)
            shard_match = re.search(r'-\d{5}-of-\d{5}\.gguf$', basename)
            if shard_match:
                prefix = basename[:shard_match.start()]
                if prefix not in seen_prefixes:
                    seen_prefixes.add(prefix)
                    result.append(m)
            else:
                result.append(m)
        return result

    return []


def resolve_toolbox(choice: str | None, run_all_backends: bool) -> list[str]:
    """Resolve toolbox name(s) from user choice."""
    if run_all_backends:
        toolboxes, _ = _discover_toolboxes()
        return toolboxes
    if choice:
        return [choice]
    return ['llama-rocm7-nightlies']


def validate_dependencies(require_toolbox: bool = True, require_power_profiles: bool = True) -> None:
    """Validate that required system tools are available."""
    if require_toolbox:
        try:
            subprocess.run(['toolbox', '--version'], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise RuntimeError("'toolbox' command not found. Install it first.")

    if require_power_profiles:
        try:
            subprocess.run(['powerprofilesctl', 'get'], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # Not fatal, just means we can't sweep power profiles


def make_run_config(
    model_paths: list[str],
    toolbox: str | None = None,
    run_all_backends: bool = False,
    mode: str = 'quick',
    prompt_tokens: str | None = None,
    gen_tokens: str | None = None,
    repetitions: int = 1,
    power_profiles: str | None = None,
    run_root: str | None = None,
    dry_run: bool = False,
    progress_sink: Callable | None = None,
) -> RunConfig:
    """Build a RunConfig from CLI arguments."""
    # Resolve models
    models: list[str] = []
    for source in model_paths:
        found = list_models(source)
        if found:
            models.extend(found)
        elif Path(source).is_file():
            models.append(source)

    if not models:
        raise ValueError(f'no GGUF models found from: {model_paths}')

    # Resolve toolbox
    toolboxes = resolve_toolbox(toolbox, run_all_backends)
    primary_toolbox = toolboxes[0] if toolboxes else 'llama-rocm7-nightlies'

    # Token presets
    prompt_tok: list[int] = [512]
    gen_tok: list[int] = [512]

    if mode == 'medium':
        prompt_tok = [1024, 4096, 8192, 16384]
        gen_tok = [512, 2048]
    elif mode == 'custom' and prompt_tokens:
        prompt_tok = [int(x) for x in parse_csv_values(prompt_tokens)]
        if not gen_tokens:
            raise ValueError('must provide --gen-tokens with custom --prompt-tokens')
        gen_tok = [int(x) for x in parse_csv_values(gen_tokens)]

    # Power profiles
    profiles = validate_power_profiles(power_profiles)

    # Run root
    if run_root is None:
        ts = time.strftime('%d-%m-%y-%H-%M-%S')
        run_root = str(Path.home() / 'llama-bench-reports' / ts)

    return RunConfig(
        models=models,
        toolbox=primary_toolbox if not run_all_backends else None,
        run_all_backends=run_all_backends,
        mode=mode,
        prompt_tokens=prompt_tok,
        gen_tokens=gen_tok,
        repetitions=repetitions,
        power_profiles=profiles,
        run_root=run_root,
        dry_run=dry_run,
        progress_sink=progress_sink,
    )


def run_profile(
    profile: str,
    label: str,
    toolbox_name: str,
    model_path: str,
    report_dir: Path,
    cmd_cfg: dict,
    progress: Optional[Callable] = None,
) -> ProfileResult:
    """Run a single power-profile benchmark inside a toolbox."""
    result = ProfileResult(
        profile=profile,
        label=label,
    )

    # Switch power profile
    try:
        subprocess.run(
            ['powerprofilesctl', 'set', profile],
            capture_output=True, text=True, timeout=10
        )
        # Verify active profile
        verify = subprocess.run(
            ['powerprofilesctl', 'get'],
            capture_output=True, text=True, timeout=10
        )
        if verify.returncode == 0:
            active = normalize_power_profile(verify.stdout.strip())
            if active != normalize_power_profile(profile):
                result.status = 'warning'
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Build llama-bench command
    cmd = [
        'toolbox', 'run', '-c', toolbox_name, '--',
        'llama-bench',
        '-m', model_path,
        '-mmp', '0',
        '-fa', '1',
    ]

    # Add prompt/gen tokens based on mode
    if cmd_cfg.get('mode') == 'custom':
        for i, (pt, gt) in enumerate(zip(cmd_cfg['prompt_tokens'], cmd_cfg['gen_tokens'])):
            cmd.extend(['-p', str(pt), '-n', str(gt)])
    else:
        # Quick/medium mode: single run per profile with preset tokens
        pt = cmd_cfg['prompt_tokens'][0] if cmd_cfg['prompt_tokens'] else 512
        gt = cmd_cfg['gen_tokens'][0] if cmd_cfg['gen_tokens'] else 512
        cmd.extend(['-p', str(pt), '-n', str(gt)])

    cmd.extend(['-r', str(cmd_cfg.get('repetitions', 1)), '-o', 'json'])

    # Execute
    start_time = time.time()
    result.start_time = time.strftime('%Y-%m-%dT%H:%M:%S')

    log_path = report_dir / f'{safe_report_id(model_path)}-{toolbox_name}-{profile}.log'
    json_path = report_dir / f'{safe_report_id(model_path)}-{toolbox_name}-{profile}.json'

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )
        end_time = time.time()
        result.runtime_seconds = end_time - start_time
        result.end_time = time.strftime('%Y-%m-%dT%H:%M:%S')

        # Save log
        log_content = proc.stdout + proc.stderr
        log_path.write_text(log_content, encoding='utf-8')
        result.log = str(log_path)

        if proc.returncode != 0:
            result.status = 'fail'
            return result

        # Parse JSON output
        try:
            json_data = json.loads(proc.stdout.strip())
            result.json = str(json_path)
            json_path.write_text(
                json.dumps(json_data, indent=2), encoding='utf-8'
            )
            result.status = 'success'

            # Extract metrics
            if isinstance(json_data, dict):
                result.prompt_tps = json_data.get('prompt_tps')
                result.gen_tps = json_data.get('gen_tps')
                result.prompt_std = json_data.get('prompt_std')
                result.gen_std = json_data.get('gen_std')
                result.prompt_tokens = json_data.get('prompt_tokens', 'n/a')
                result.gen_tokens = json_data.get('gen_tokens', 'n/a')
                result.model_type = json_data.get('model_type', 'unknown')
                result.backend = json_data.get('backend', 'unknown')
                result.device = json_data.get('device', 'unknown')
                result.cpu_ram_used_gib = json_data.get('cpu_ram_used_gib')
                result.cpu_ram_total_gib = json_data.get('cpu_ram_total_gib')
                result.cpu_temp_c = json_data.get('cpu_temp_c')
                result.gpu_temp_c = json_data.get('gpu_temp_c')
                result.gpu_power_w = json_data.get('gpu_power_w')

                # Handle results array
                results_list = json_data.get('results', [])
                if isinstance(results_list, list):
                    result.results = results_list
        except (json.JSONDecodeError, KeyError):
            result.status = 'partial'

    except subprocess.TimeoutExpired:
        result.status = 'fail'
        result.runtime_seconds = 3600.0
        log_path.write_text('Benchmark timed out after 1 hour', encoding='utf-8')
    except Exception as e:
        result.status = 'fail'
        log_path.write_text(f'Error: {e}', encoding='utf-8')

    return result


def run_model_toolbox(
    model_path: str,
    toolbox_name: str,
    cfg: RunConfig,
    progress: Optional[Callable] = None,
) -> Tuple[dict, list[ProfileResult]]:
    """Run benchmarks for a single model across all power profiles in one toolbox."""
    report_dir = Path(cfg.run_root) / safe_report_id(model_path) / toolbox_name
    report_dir.mkdir(parents=True, exist_ok=True)

    results: list[ProfileResult] = []

    cmd_cfg = {
        'mode': cfg.mode,
        'prompt_tokens': cfg.prompt_tokens,
        'gen_tokens': cfg.gen_tokens,
        'repetitions': cfg.repetitions,
    }

    for profile in cfg.power_profiles:
        label = power_profile_label(profile) or profile

        if progress:
            progress(RunEvent(
                type='start',
                model=model_path,
                toolbox=toolbox_name,
                profile=profile,
                message=f'Starting {label}...',
            ))

        result = run_profile(
            profile=profile,
            label=label,
            toolbox_name=toolbox_name,
            model_path=model_path,
            report_dir=report_dir,
            cmd_cfg=cmd_cfg,
            progress=progress,
        )
        results.append(result)

        if progress:
            progress(RunEvent(
                type='progress',
                model=model_path,
                toolbox=toolbox_name,
                profile=profile,
                message=f'{label}: {result.status} ({result.runtime_seconds:.1f}s)',
                data={'result': result},
            ))

    # Build summary
    system_meta = collect_system_metadata()
    summary = {
        'model': model_path,
        'toolbox': toolbox_name,
        'mode': cfg.mode,
        'status': 'success' if all(r.status == 'success' for r in results) else 'partial',
        'profiles': [
            {
                'label': r.label,
                'profile': r.profile,
                'status': r.status,
                'runtime_seconds': r.runtime_seconds,
                'start_time': r.start_time,
                'end_time': r.end_time,
                'model_type': r.model_type,
                'backend': r.backend,
                'device': r.device,
                'prompt_tokens': r.prompt_tokens,
                'gen_tokens': r.gen_tokens,
                'prompt_tps': r.prompt_tps,
                'gen_tps': r.gen_tps,
                'prompt_std': r.prompt_std,
                'gen_std': r.gen_std,
                'cpu_ram_used_gib': r.cpu_ram_used_gib,
                'cpu_ram_total_gib': r.cpu_ram_total_gib,
                'cpu_temp_c': r.cpu_temp_c,
                'gpu_temp_c': r.gpu_temp_c,
                'gpu_power_w': r.gpu_power_w,
                'json': r.json,
                'log': r.log,
                'results': r.results,
            }
            for r in results
        ],
        **system_meta,
        'prompt_tokens': cfg.prompt_tokens,
        'gen_tokens': cfg.gen_tokens,
        'repetitions': cfg.repetitions,
        'benchmark_command': f'toolbox run -c {toolbox_name} -- llama-bench ...',
    }

    summary_path = report_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    # Generate per-model reports
    reporting.write_model_reports(
        str(summary_path),
        str(report_dir / 'index.html'),
        str(report_dir / 'report.md'),
    )

    return summary, results


def run_benchmarks(cfg: RunConfig) -> dict:
    """Execute the full benchmark sweep."""
    # Validate dependencies
    try:
        _validate_dependencies(require_toolbox=True, require_power_profiles=not cfg.dry_run)
    except RuntimeError:
        if not cfg.dry_run:
            raise

    if cfg.dry_run:
        print(Colors.YELLOW + '[dry-run] Skipping dependency checks' + Colors.RESET)
        # Print plan
        toolboxes = resolve_toolbox(cfg.toolbox, cfg.run_all_backends)
        for model in cfg.models:
            for toolbox in toolboxes:
                for profile in cfg.power_profiles:
                    label = power_profile_label(profile) or profile
                    print(f'  {model} | {toolbox} | {label}')
        return {'dry_run': True, 'models': len(cfg.models), 'toolboxes': len(toolboxes)}

    # Create run root
    os.makedirs(cfg.run_root, exist_ok=True)

    toolboxes = resolve_toolbox(cfg.toolbox, cfg.run_all_backends)
    all_summaries: list[dict] = []
    total_runs = len(cfg.models) * len(toolboxes) * len(cfg.power_profiles)

    completed = 0
    for model in cfg.models:
        for toolbox in toolboxes:
            summary, results = run_model_toolbox(model, toolbox, cfg)
            all_summaries.append(summary)
            completed += len(cfg.power_profiles)

            if cfg.progress_sink:
                cfg.progress_sink(RunEvent(
                    type='progress',
                    model=model,
                    toolbox=toolbox,
                    message=f'Completed {completed}/{total_runs} runs',
                ))

    # Generate sweep-level report
    ts = time.strftime('%y-%b-%d_%H_%M_%S').lower()
    out_html = Path(cfg.run_root) / f'index-{ts}.html'
    out_md = Path(cfg.run_root) / f'index-{ts}.md'

    summary_paths = [str(s.get('model', '')) for s in all_summaries if s]
    toolboxes_str = ', '.join(toolboxes)
    profiles_str = ', '.join(cfg.power_profiles)

    _build_reports_from_result_folder(cfg.run_root)

    return {
        'run_root': cfg.run_root,
        'models_run': len(cfg.models),
        'toolboxes_used': len(toolboxes),
        'total_runs': total_runs,
        'completed': completed,
        'out_html': str(out_html),
        'out_md': str(out_md),
    }


# ── Rebuild entry point ───────────────────────────────────────────────────────

def rebuild_reports(result_dir: str) -> dict:
    """Rebuild reports from existing benchmark results."""
    return _build_reports_from_result_folder(result_dir)
