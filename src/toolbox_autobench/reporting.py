"""Report generation for toolbox-autobench."""

from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(value: Any, default: str = '') -> str:
    """Return *value* as a safely escaped string, or *default*."""
    if value is None or value == '':
        return default
    return html.escape(str(value))


def _safe_read_json(path: Path) -> dict | list | None:
    """Read and parse JSON from *path*, returning ``None`` on any error."""
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _looks_like_profile_payload(payload: Any, path: Path) -> bool:
    """Heuristic: does this JSON look like a llama-bench profile output?"""
    if not isinstance(payload, dict):
        return False
    # Either has 'results' key or has known metric keys directly
    if 'results' in payload:
        return True
    for key in ('prompt_tps', 'gen_tps', 'runtime_seconds', 'prompt_tokens'):
        if key in payload:
            return True
    # Fallback: check file name patterns
    name = path.name.lower()
    return any(kw in name for kw in ('power-saver', 'balanced', 'performance'))


def _build_synthesized_summary(run_dir: Path, profile_payloads: list) -> dict:
    """Build a summary.json from raw llama-bench profile JSON payloads."""
    profiles = []
    for json_path, payload in profile_payloads:
        if not isinstance(payload, dict):
            continue
        label = json_path.parent.name  # e.g. power-saver
        status = 'success'
        entry = {
            'label': label,
            'status': status,
            **payload,
        }
        profiles.append(entry)

    return {
        'model': str(run_dir.name),
        'toolbox': profile_payloads[0][0].parent.parent.name if profile_payloads else 'unknown',
        'profiles': profiles,
        'mode': 'rebuild',
    }


def _build_svg_chart(profiles: list, metric_key: str, title: str, color: str) -> str:
    """Build an inline SVG horizontal bar chart for *metric_key*."""
    data = []
    for p in profiles:
        val = p.get(metric_key, 0)
        try:
            val = float(val) if val else 0.0
        except (ValueError, TypeError):
            val = 0.0
        label = p.get('label', 'unknown')
        data.append((label, val))

    if not data:
        return '<svg width="100" height="20"></svg>'

    max_val = max(v for _, v in data) or 1
    bar_width = 500
    bar_height = 24
    padding = 120
    svg_h = len(data) * (bar_height + 4) + 30

    lines = [f'<svg width="{bar_width + padding}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg">']
    for i, (label, val) in enumerate(data):
        bar_w = max(2, int(val / max_val * bar_width))
        y = i * (bar_height + 4) + 10
        lines.append(
            f'<rect x="{padding}" y="{y}" width="{bar_w}" height="{bar_height - 4}" '
            f'fill="{color}" rx="4"/>'
        )
        lines.append(
            f'<text x="{padding - 8}" y="{y + bar_height / 2 + 4}" '
            f'text-anchor="end" font-size="12" fill="#333">{label}</text>'
        )
        lines.append(
            f'<text x="{padding + bar_w + 6}" y="{y + bar_height / 2 + 4}" '
            f'font-size="12" fill="#555">{val:.2f}</text>'
        )
    lines.append('</svg>')
    return '\n'.join(lines)


def _markdown_profile_table(profiles: list) -> str:
    """Render a Markdown table from profile entries."""
    lines = [
        '| Profile | Status | Runtime (s) | Prompt TPS | Gen TPS |',
        '|---|---|---:|---:|---:',
    ]
    for p in profiles:
        label = _safe(p.get('label', 'unknown'))
        status = _safe(p.get('status', 'unknown'))
        runtime = f"{float(p.get('runtime_seconds', 0)):.3f}"
        prompt_tps = f"{float(p.get('prompt_tps', 0)):.3f}"
        gen_tps = f"{float(p.get('gen_tps', 0)):.3f}"
        lines.append(f'| {label} | {status} | {runtime} | {prompt_tps} | {gen_tps} |')
    return '\n'.join(lines)


def _fmt_ram_pair(used: Any, total: Any) -> str:
    """Format RAM used/total as a human-readable string."""
    u = float(used) if used else 0.0
    t = float(total) if total else 0.0
    return f'{u:.1f}/{t:.1f} GiB'


def _fmt_metric(value: Any, decimals: int = 1) -> str:
    """Format a metric value to *decimals* places."""
    if value is None or value == '':
        return 'n/a'
    try:
        return f'{float(value):.{decimals}f}'
    except (ValueError, TypeError):
        return 'n/a'


def _coerce_float(value: Any) -> float:
    """Coerce *value* to float, returning 0.0 on failure."""
    if value is None or value == '':
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _grid_context_label(ctx: dict) -> str:
    """Build a context label from prompt/gen token values."""
    pts = ctx.get('prompt_tokens', 'n/a')
    gts = ctx.get('gen_tokens', 'n/a')
    return f'P{pts}/G{gts}'


def _grid_quant_from_model_name(model_name: str) -> str:
    """Extract quantisation tag from a model name."""
    if not model_name:
        return 'unknown'
    # Common quant patterns
    for q in ['Q8_0', 'Q8_K', 'Q6_K', 'Q5_0', 'Q5_K', 'Q4_0', 'Q4_K', 'Q3_K', 'Q2_K', 'IQ4_XS']:
        if q in model_name:
            return q
    # Last segment after last dot or underscore that looks like a quant
    parts = re.split(r'[-_.]', model_name)
    for part in reversed(parts):
        if re.match(r'^Q[0-9]', part):
            return part
    return 'unknown'


def _grid_params_b_from_model_name(model_name: str) -> str:
    """Extract parameter count (in billions) from a model name."""
    if not model_name:
        return 'unknown'
    m = re.search(r'(\d+\.?\d*)B', model_name, re.IGNORECASE)
    if m:
        return f'{m.group(1)}B'
    return 'unknown'


def _as_relative(path: Path, base: Path) -> str:
    """Return a relative path from *base* to *path*."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


# ── Leaderboard rendering ─────────────────────────────────────────────────────

def render_leaderboard(profiles: Iterable[dict], metric_key: str) -> list[str]:
    """Render a Markdown leaderboard table for *metric_key*."""
    ranking = sorted(
        profiles,
        key=lambda p: float(p.get(metric_key, 0) or 0),
        reverse=True,
    )
    lines = [
        '| Rank | Profile | Value | Runtime (s) | Status |',
        '|---|---|---:|---:|---|',
    ]
    for idx, entry in enumerate(ranking, start=1):
        label = _safe(entry.get('label'), 'unknown')
        val = f"{float(entry.get(metric_key, 0) or 0):.3f}"
        runtime = f"{float(entry.get('runtime_seconds', 0) or 0):.3f}"
        status = _safe(entry.get('status'), 'unknown')
        lines.append(f'| {idx} | {label} | {val} | {runtime} | {status} |')
    return lines


# ── Per-model reports ─────────────────────────────────────────────────────────

def write_model_reports(summary_path: str, html_path: str, markdown_path: str) -> None:
    """Generate HTML and Markdown reports for a single model run."""
    data = json.loads(Path(summary_path).read_text(encoding='utf-8'))
    profiles = data.get('profiles', [])
    model_path_value = data.get('model', '') or ''

    if model_path_value:
        model_name = Path(model_path_value).name
    else:
        model_name = 'unknown'

    # Build markdown
    leaderboard_prompt = '\n'.join(render_leaderboard(profiles, 'prompt_tps'))
    leaderboard_gen = '\n'.join(render_leaderboard(profiles, 'gen_tps'))

    chart_prompt = _build_svg_chart(profiles, 'prompt_tps', 'Prompt TPS Leaderboard', '#4c7ce9')
    chart_gen = _build_svg_chart(profiles, 'gen_tps', 'Generation TPS Leaderboard', '#2ea8a0')
    chart_runtime = _build_svg_chart(profiles, 'runtime_seconds', 'Runtime (seconds)', '#f0a202')

    command = data.get('benchmark_command', '') or ''
    if model_path_value and command:
        command = str(command).replace(model_path_value, model_name)

    md_lines = [
        '# llama-bench Power Profile Sweep Report',
        '',
        f'- Model: `{model_name}`',
        f'- Toolbox: `{data.get("toolbox", "unknown")}`',
        f'- Machine: `{data.get("machine_name", "n/a")}`',
        f'- Machine model: `{data.get("machine_model", "n/a")}`',
        f'- CPU model: `{data.get("cpu_model", "n/a")}`',
        f'- Mode: `{data.get("mode", "unknown")}`',
        f'- Kernel: `{data.get("kernel_version", "")}`',
        f'- Kernel name/build: `{data.get("kernel_name", "n/a")} / {data.get("kernel_build", "n/a")}`',
        f'- OS: `{data.get("os_name", "n/a")}` · ID: `{data.get("os_id", "n/a")}` · Version: `{data.get("os_version_id", "n/a")}`',
        f'- OS version text: `{data.get("os_version", "n/a")}`',
        f'- Architecture: `{data.get("architecture", "n/a")}`',
        f'- Llama-bench version: `{data.get("llama_bench_version", "unknown")}`',
        f'- Llama-bench build: commit `{data.get("llama_bench_build_commit", "unknown")}`, number `{data.get("llama_bench_build_number", "unknown")}`',
        f'- Prompt tokens: `{data.get("prompt_tokens", "")}`',
        f'- Generation tokens: `{data.get("gen_tokens", "")}`',
        f'- Repetitions: `{data.get("repetitions", "")}`',
        f'- Command: `{command}`',
        '',
        '## Benchmark summary table',
        '',
        _markdown_profile_table(profiles),
        '',
        '## Leaderboard by prompt TPS',
        '',
        leaderboard_prompt,
        '',
        '## Leaderboard by generation TPS',
        '',
        leaderboard_gen,
        '',
        '## Artifacts',
        '',
        '- [index.html](index.html)',
        f'- [summary.json]({Path(summary_path).name})',
        '',
    ]

    Path(markdown_path).write_text('\n'.join(md_lines) + '\n', encoding='utf-8')

    # Build HTML
    model_safe = _safe(model_name)
    toolbox_safe = _safe(data.get('toolbox', 'unknown'))
    mode_safe = _safe(data.get('mode', 'unknown'))

    rows_html = []
    for p in profiles:
        status_class = 'ok' if p.get('status') == 'success' else 'fail'
        rows_html.append(
            f'<tr><td><strong>{_safe(p.get("label"))}</strong></td>'
            f'<td class="status {status_class}">{_safe(p.get("status"))}</td>'
            f'<td>{float(p.get("runtime_seconds", 0) or 0):.3f}</td>'
            f'<td><code>{_safe(p.get("start_time"))}</code></td>'
            f'<td><code>{_safe(p.get("end_time"))}</code></td>'
            f'<td>{float(p.get("prompt_tps", 0) or 0):.3f}</td>'
            f'<td>{float(p.get("gen_tps", 0) or 0):.3f}</td>'
            f'<td>{_fmt_ram_pair(p.get("cpu_ram_used_gib"), p.get("cpu_ram_total_gib"))}</td>'
            f'<td>{_fmt_metric(p.get("cpu_temp_c"))}</td>'
            f'<td>{_fmt_metric(p.get("gpu_temp_c"))}</td>'
            f'<td>{_fmt_metric(p.get("gpu_power_w"))}</td>'
            f'<td>{_safe(p.get("model_type", "unknown"))}</td>'
            f'<td>{_safe(p.get("backend", "unknown"))}</td>'
            f'<td>{_safe(p.get("device", "unknown"))}</td>'
            f'<td><a href="{_safe(p.get("json"), "#")}">json</a> · <a href="{_safe(p.get("log"), "#")}">log</a></td>'
            f'</tr>'
        )

    prompt_val = data.get('prompt_tokens', '')
    if isinstance(prompt_val, list):
        prompt_str = ', '.join(str(x) for x in prompt_val)
    else:
        prompt_str = str(prompt_val or '')

    gen_val = data.get('gen_tokens', '')
    if isinstance(gen_val, list):
        gen_str = ', '.join(str(x) for x in gen_val)
    else:
        gen_str = str(gen_val or '')

    html_parts = [
        '<!doctype html>\n<html>\n<head>\n'
        '  <meta charset="utf-8" />\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        '  <title>llama-bench Power Profile Sweep</title>\n'
        '  <style>\n'
        '    :root {\n'
        '      --bg: #f4f7fb;\n'
        '      --card: #ffffff;\n'
        '      --line: #dbe5f2;\n'
        '      --line-strong: #c5d5ea;\n'
        '      --text: #1f2a44;\n'
        '      --muted: #60708d;\n'
        '      --ok: #0f9d58;\n'
        '      --fail: #d93025;\n'
        '      --link: #1d4ed8;\n'
        '    }\n'
        '    * { box-sizing: border-box; }\n'
        '    body {\n'
        '      margin: 0;\n'
        '      padding: 28px;\n'
        '      font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;\n'
        '      color: var(--text);\n'
        '      background: var(--bg);\n'
        '      min-height: 100vh;\n'
        '    }\n'
        '    h1 { margin-top: 0; margin-bottom: 10px; }\n'
        '    .card {\n'
        '      background: var(--card);\n'
        '      border: 1px solid var(--line);\n'
        '      border-radius: 12px;\n'
        '      padding: 18px;\n'
        '      margin-bottom: 16px;\n'
        '      box-shadow: 0 8px 24px rgba(22, 40, 75, 0.08);\n'
        '      animation: lift 0.22s ease;\n'
        '    }\n'
        '    table { width: 100%; border-collapse: collapse; font-size: 14px; }\n'
        '    th, td { padding: 9px; border-bottom: 1px solid var(--line); text-align: left; }\n'
        '    th { color: var(--muted); border-bottom: 1px solid var(--line-strong); }\n'
        '    tr:hover td { background: #f8fbff; }\n'
        '    code { background: #f3f7fc; border: 1px solid var(--line); padding: 2px 6px; border-radius: 6px; }\n'
        '    .status.ok { color: var(--ok); font-weight: 700; }\n'
        '    .status.fail { color: var(--fail); font-weight: 700; }\n'
        '    .chart-wrap { overflow-x: auto; }\n'
        '    .leader {\n'
        '      display: grid;\n'
        '      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));\n'
        '      gap: 12px;\n'
        '    }\n'
        '    a { color: var(--link); text-decoration: none; }\n'
        '    a:hover { text-decoration: underline; }\n'
        '    @keyframes lift {\n'
        '      from { transform: translateY(6px); opacity: 0; }\n'
        '      to { transform: translateY(0); opacity: 1; }\n'
        '    }\n'
        '    @media (max-width: 768px) {\n'
        '      table, th, td { font-size: 12px; }\n'
        '    }\n'
        '  </style>\n'
        '</head>\n<body>\n'
        '  <h1>llama-bench Power Profile Sweep</h1>\n'
        f'  <p>Model: <code>{model_safe}</code><br/>\n'
        f'  Toolbox: <code>{toolbox_safe}</code><br/>\n'
        f'  Machine: <code>{_safe(data.get("machine_name"))}</code> · <strong>Model</strong>: <code>{_safe(data.get("machine_model"))}</code><br/>\n'
        f'  CPU: <code>{_safe(data.get("cpu_model"))}</code> · Architecture: <code>{_safe(data.get("architecture"))}</code><br/>\n'
        f'  Mode: <code>{mode_safe}</code> · Prompt tokens: <code>{prompt_str}</code>, Gen tokens: <code>{gen_str}</code>, Repetitions: <code>{data.get("repetitions", "n/a")}</code><br/>\n'
        f'  Kernel: <code>{data.get("kernel_version", "")}</code> ({_safe(data.get("kernel_name"))} / {_safe(data.get("kernel_build"))}) · OS: <code>{data.get("os_name", "")}</code> ({data.get("os_id", "")}:{data.get("os_version_id", "")})<br/>\n'
        f'  OS version: <code>{data.get("os_version", "")}</code> · Llama-bench: <code>{data.get("llama_bench_version", "unknown")}</code> (commit {_safe(data.get("llama_bench_build_commit"))}, build {_safe(data.get("llama_bench_build_number"))})<br/>\n'
        f'  Command: <code>{_safe(command)}</code>\n'
        '  </p>\n'
        '  <div class="card">\n'
        '    <h2>Summary</h2>\n'
        '    <table>\n'
        '      <thead>\n'
        '        <tr><th>Profile</th><th>Status</th><th>Runtime (s)</th><th>Start</th><th>End</th>'
        '<th>Prompt TPS</th><th>Gen TPS</th><th>RAM Avg Used/Total (GiB)</th>'
        '<th>CPU Temp Avg (C)</th><th>GPU Temp Avg (C)</th><th>GPU Power Avg (W)</th>'
        '<th>Model</th><th>Backend</th><th>Device</th><th>Details</th></tr>\n'
        '      </thead>\n'
        '      <tbody>\n'
        f'        {"\n".join(rows_html)}\n'
        '      </tbody>\n'
        '    </table>\n'
        '  </div>\n'
        '  <div class="leader">\n'
        '    <div class="card"><h2>Prompt TPS Leaderboard</h2><div class="chart-wrap">' + chart_prompt + '</div></div>\n'
        '    <div class="card"><h2>Generation TPS Leaderboard</h2><div class="chart-wrap">' + chart_gen + '</div></div>\n'
        '    <div class="card"><h2>Runtime Leaderboard</h2><div class="chart-wrap">' + chart_runtime + '</div></div>\n'
        '  </div>\n'
        '  <div class="card">\n'
        '    <h2>Artifacts</h2>\n'
        '    <p><a href="summary.json">summary.json</a> · <a href="index.html">index.html</a></p>\n'
        '  </div>\n'
        '</body>\n</html>\n',
    ]

    Path(html_path).write_text('\n'.join(html_parts), encoding='utf-8')


# ── Sweep-level report ────────────────────────────────────────────────────────

def write_run_summary(
    summary_paths: list[str],
    out_html: str,
    out_md: str,
    toolboxes: list[str],
    mode: str,
    profiles: list[str],
    prompt_tokens: str,
    gen_tokens: str,
    repetitions: int,
    command_template: str,
    run_root: str,
) -> None:
    """Build sweep-level HTML and Markdown reports from multiple summary files."""
    summary_root = Path(out_html).resolve().parent

    model_rows: list[dict] = []
    context_entries: list[dict] = []
    host_summary: dict = {}

    for path in summary_paths:
        payload_raw = _safe_read_json(Path(path))
        if not isinstance(payload_raw, dict):
            continue
        payload = payload_raw
        profiles_payload = payload.get('profiles', [])
        summary_path = Path(path).resolve()
        index_path = summary_path.parent / 'index.html'

        model_val = payload.get('model', '') or ''
        if model_val:
            model_name = Path(model_val).name
        else:
            model_name = 'unknown'

        toolbox_name = str(payload.get('toolbox', 'unknown') or 'unknown')

        if not host_summary:
            host_summary = payload

        for profile_row in profiles_payload:
            profile_label = str(profile_row.get('label', 'n/a'))
            status = str(
                profile_row.get('status', payload.get('status', 'unknown'))
                or 'unknown'
            )
            runtime_seconds = float(profile_row.get('runtime_seconds', 0) or 0)

            profile_index = _as_relative(index_path, summary_root)

            context_rows = _context_rows_for_profile(
                summary_path,
                summary_path.parent,
                payload,
                profile_row,
                model_name,
                toolbox_name,
            )

            if not context_rows:
                continue

            for ctx in context_rows:
                prompt_tps_value = _coerce_float(ctx.get('prompt_tps'))
                gen_tps_value = _coerce_float(ctx.get('gen_tps'))
                prompt_std_value = _coerce_float(ctx.get('prompt_std'))
                gen_std_value = _coerce_float(ctx.get('gen_std'))
                context_label = _grid_context_label(ctx)

                model_row = {
                    'model_name': model_name,
                    'quant': _grid_quant_from_model_name(model_name),
                    'params_b': _grid_params_b_from_model_name(model_name),
                    'toolbox': toolbox_name,
                    'profile': profile_label,
                    'status': status,
                    'context_label': context_label,
                    'prompt_tokens': ctx.get('prompt_tokens', 'n/a'),
                    'gen_tokens': ctx.get('gen_tokens', 'n/a'),
                    'prompt_tps': prompt_tps_value,
                    'gen_tps': gen_tps_value,
                    'prompt_std': prompt_std_value,
                    'gen_std': gen_std_value,
                    'runtime_seconds': runtime_seconds,
                    'cpu_ram_used_gib': profile_row.get('cpu_ram_used_gib'),
                    'cpu_ram_total_gib': profile_row.get('cpu_ram_total_gib'),
                    'cpu_temp_c': profile_row.get('cpu_temp_c'),
                    'gpu_temp_c': profile_row.get('gpu_temp_c'),
                    'gpu_power_w': profile_row.get('gpu_power_w'),
                    'index': profile_index,
                }
                model_rows.append(model_row)

                context_entries.append({
                    'model_name': model_name,
                    'toolbox': toolbox_name,
                    'profile': profile_label,
                    'prompt_tokens': ctx.get('prompt_tokens', 'n/a'),
                    'gen_tokens': ctx.get('gen_tokens', 'n/a'),
                    'prompt_tps': prompt_tps_value,
                    'prompt_std': prompt_std_value,
                    'gen_tps': gen_tps_value,
                    'gen_std': gen_std_value,
                })

    if not model_rows:
        raise ValueError('no profile rows available to build sweep report')

    # Group by model name and find bests
    by_model_name: dict[str, dict] = {}
    for row in model_rows:
        key = str(row['model_name'])
        current = by_model_name.get(key)
        if current is None:
            by_model_name[key] = {
                'best_gen_row': row,
                'best_prompt_row': row,
            }
        else:
            cur_best_gen = _coerce_float(current['best_gen_row'].get('gen_tps'))
            new_gen = _coerce_float(row.get('gen_tps'))
            if new_gen > cur_best_gen:
                current['best_gen_row'] = row

            cur_best_prompt = _coerce_float(current['best_prompt_row'].get('prompt_tps'))
            new_prompt = _coerce_float(row.get('prompt_tps'))
            if new_prompt > cur_best_prompt:
                current['best_prompt_row'] = row

    # Build sweep-level HTML
    toolboxes_str = ', '.join(sorted(toolboxes)) if toolboxes else 'unknown'
    profiles_str = ', '.join(sorted(profiles)) if profiles else 'unknown'

    html_output = _build_sweep_html(
        model_rows=model_rows,
        by_model_name=by_model_name,
        context_entries=context_entries,
        host_summary=host_summary,
        toolboxes=toolboxes_str,
        mode=mode,
        profiles=profiles_str,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        repetitions=repetitions,
        command_template=command_template,
        run_root=run_root,
    )

    Path(out_html).write_text(html_output, encoding='utf-8')


def _context_rows_for_profile(
    summary_path: Path,
    model_dir: Path,
    payload: dict,
    profile_row: dict,
    model_name: str,
    toolbox_name: str,
) -> list[dict]:
    """Extract context-size rows from a single profile's results."""
    results = profile_row.get('results', [])
    if not isinstance(results, list):
        return []

    rows = []
    for r in results:
        if not isinstance(r, dict):
            continue
        pt = r.get('prompt_tokens', 'n/a')
        gt = r.get('gen_tokens', 'n/a')
        if pt == 'n/a' and gt == 'n/a':
            # If no explicit results, use the top-level profile tokens
            pt = payload.get('prompt_tokens', pt)
            gt = payload.get('gen_tokens', gt)
        rows.append({
            'prompt_tokens': pt,
            'gen_tokens': gt,
            'prompt_tps': r.get('prompt_tps'),
            'gen_tps': r.get('gen_tps'),
            'prompt_std': r.get('prompt_std'),
            'gen_std': r.get('gen_std'),
        })
    return rows


def _build_sweep_html(
    model_rows: list[dict],
    by_model_name: dict,
    context_entries: list[dict],
    host_summary: dict,
    toolboxes: str,
    mode: str,
    profiles: str,
    prompt_tokens: str,
    gen_tokens: str,
    repetitions: int,
    command_template: str,
    run_root: str,
) -> str:
    """Build the sweep-level HTML report."""
    # Build context-size table rows
    ctx_rows = []
    for entry in context_entries:
        ctx_rows.append(
            f'<tr>'
            f'<td>{_safe(entry["model_name"])}</td>'
            f'<td>{_safe(entry["toolbox"])}</td>'
            f'<td>{_safe(entry["profile"])}</td>'
            f'<td>{entry["prompt_tokens"]}</td>'
            f'<td>{entry["gen_tokens"]}</td>'
            f'<td>{float(entry.get("prompt_tps", 0) or 0):.3f}</td>'
            f'<td>{float(entry.get("prompt_std", 0) or 0):.3f}</td>'
            f'<td>{float(entry.get("gen_tps", 0) or 0):.3f}</td>'
            f'<td>{float(entry.get("gen_std", 0) or 0):.3f}</td>'
            f'</tr>'
        )

    # Build model rows grouped by model name
    model_sections = []
    for model_name, bests in sorted(by_model_name.items()):
        best_gen = bests['best_gen_row']
        best_prompt = bests['best_prompt_row']
        model_sections.append(
            f'<div class="card">'
            f'<h3>{_safe(model_name)} <span style="color:var(--muted)">'
            f'({_safe(bests["best_gen_row"]["quant"])}, '
            f'{_safe(bests["best_gen_row"]["params_b"])})</span></h3>'
            f'<p><strong>Best gen TPS:</strong> {float(best_gen.get("gen_tps", 0) or 0):.3f} '
            f'(<code>{_safe(best_gen["profile"])}</code>) · '
            f'<strong>Best prompt TPS:</strong> {float(best_prompt.get("prompt_tps", 0) or 0):.3f} '
            f'(<code>{_safe(best_prompt["profile"])}</code>)</p>'
            f'</div>'
        )

    # Build context comparison table
    ctx_table_rows = []
    for entry in sorted(context_entries, key=lambda e: (e['model_name'], e['toolbox'])):
        ctx_table_rows.append(
            f'<tr>'
            f'<td>{_safe(entry["model_name"])}</td>'
            f'<td>{_safe(entry["toolbox"])}</td>'
            f'<td>{_safe(entry["profile"])}</td>'
            f'<td>{entry["prompt_tokens"]}</td>'
            f'<td>{entry["gen_tokens"]}</td>'
            f'<td>{float(entry.get("prompt_tps", 0) or 0):.3f} ± {float(entry.get("prompt_std", 0) or 0):.3f}</td>'
            f'<td>{float(entry.get("gen_tps", 0) or 0):.3f} ± {float(entry.get("gen_std", 0) or 0):.3f}</td>'
            f'</tr>'
        )

    # Machine info
    machine_name = _safe(host_summary.get('machine_name', 'n/a'))
    machine_model = _safe(host_summary.get('machine_model', 'n/a'))
    cpu_model = _safe(host_summary.get('cpu_model', 'n/a'))
    architecture = _safe(host_summary.get('architecture', 'n/a'))
    kernel_version = host_summary.get('kernel_version', '')
    kernel_name = _safe(host_summary.get('kernel_name', 'n/a'))
    kernel_build = _safe(host_summary.get('kernel_build', 'n/a'))
    os_name = _safe(host_summary.get('os_name', 'n/a'))
    os_id = _safe(host_summary.get('os_id', 'n/a'))
    os_version_id = _safe(host_summary.get('os_version_id', 'n/a'))
    os_version = _safe(host_summary.get('os_version', 'n/a'))
    llama_bench_version = _safe(host_summary.get('llama_bench_version', 'unknown'))
    llama_bench_commit = _safe(host_summary.get('llama_bench_build_commit', 'unknown'))
    llama_bench_build_num = _safe(host_summary.get('llama_bench_build_number', 'unknown'))

    html_parts = [
        '<!doctype html>\n<html>\n<head>\n'
        '  <meta charset="utf-8" />\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        '  <title>llama-bench Sweep Report</title>\n'
        '  <style>\n'
        '    :root {\n'
        '      --bg: #f4f7fb;\n'
        '      --card: #ffffff;\n'
        '      --line: #dbe5f2;\n'
        '      --text: #1f2a44;\n'
        '      --muted: #60708d;\n'
        '      --ok: #0f9d58;\n'
        '      --fail: #d93025;\n'
        '      --link: #1d4ed8;\n'
        '    }\n'
        '    * { box-sizing: border-box; }\n'
        '    body {\n'
        '      margin: 0;\n'
        '      padding: 28px;\n'
        '      font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;\n'
        '      color: var(--text);\n'
        '      background: var(--bg);\n'
        '    }\n'
        '    h1 { margin-top: 0; }\n'
        '    .card {\n'
        '      background: var(--card);\n'
        '      border: 1px solid var(--line);\n'
        '      border-radius: 12px;\n'
        '      padding: 18px;\n'
        '      margin-bottom: 16px;\n'
        '    }\n'
        '    table { width: 100%; border-collapse: collapse; font-size: 14px; }\n'
        '    th, td { padding: 8px; border-bottom: 1px solid var(--line); text-align: left; }\n'
        '    th { color: var(--muted); }\n'
        '    code { background: #f3f7fc; padding: 2px 6px; border-radius: 4px; }\n'
        '    a { color: var(--link); }\n'
        '    .meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 8px; }\n'
        '    .meta dt { color: var(--muted); font-weight: 600; }\n'
        '    .meta dd { margin-left: 0; }\n'
        '  </style>\n'
        '</head>\n<body>\n'
        '  <h1>llama-bench Sweep Report</h1>\n'
        f'  <dl class="meta">\n'
        f'    <dt>Toolboxes</dt><dd>{_safe(toolboxes)}</dd>\n'
        f'    <dt>Mode</dt><dd>{_safe(mode)}</dd>\n'
        f'    <dt>Profiles</dt><dd>{_safe(profiles)}</dd>\n'
        f'    <dt>Prompt tokens</dt><dd>{_safe(prompt_tokens)}</dd>\n'
        f'    <dt>Gen tokens</dt><dd>{_safe(gen_tokens)}</dd>\n'
        f'    <dt>Repetitions</dt><dd>{repetitions}</dd>\n'
        f'    <dt>Command</dt><dd><code>{_safe(command_template)}</code></dd>\n'
        f'    <dt>Run root</dt><dd><code>{_safe(run_root)}</code></dd>\n'
        '  </dl>\n'
        '  <div class="card">\n'
        '    <h2>Machine Info</h2>\n'
        '    <table>\n'
        f'      <tr><td><strong>Machine</strong></td><td>{machine_name}</td></tr>\n'
        f'      <tr><td><strong>Model</strong></td><td>{machine_model}</td></tr>\n'
        f'      <tr><td><strong>CPU</strong></td><td>{cpu_model}</td></tr>\n'
        f'      <tr><td><strong>Architecture</strong></td><td>{architecture}</td></tr>\n'
        f'      <tr><td><strong>Kernel</strong></td><td>{kernel_version} ({kernel_name} / {kernel_build})</td></tr>\n'
        f'      <tr><td><strong>OS</strong></td><td>{os_name} ({os_id}:{os_version_id}) — {os_version}</td></tr>\n'
        f'      <tr><td><strong>Llama-bench</strong></td><td>{llama_bench_version} (commit {llama_bench_commit}, build {llama_bench_build_num})</td></tr>\n'
        '    </table>\n'
        '  </div>\n'
        '  <div class="card">\n'
        '    <h2>Model Summaries</h2>\n'
        f'    {"\n".join(model_sections)}\n'
        '  </div>\n'
        '  <div class="card">\n'
        '    <h2>Context-Size Comparison</h2>\n'
        '    <table>\n'
        '      <thead>\n'
        '        <tr><th>Model</th><th>Toolbox</th><th>Profile</th>'
        '<th>Prompt tok</th><th>Gen tok</th>'
        '<th>Prompt TPS</th><th>Prompt σ</th>'
        '<th>Gen TPS</th><th>Gen σ</th></tr>\n'
        '      </thead>\n'
        '      <tbody>\n'
        f'        {"\n".join(ctx_table_rows)}\n'
        '      </tbody>\n'
        '    </table>\n'
        '  </div>\n'
        '</body>\n</html>\n',
    ]

    return '\n'.join(html_parts)


# ── Report rebuild ────────────────────────────────────────────────────────────

def build_reports_from_result_folder(result_root: str) -> dict[str, int]:
    """Rebuild reports from existing JSON artifacts under *result_root*.

    Returns a dict with summary/synthesis counts.
    """
    root = Path(result_root).expanduser().resolve()

    if not root.exists():
        raise ValueError(f'result folder not found: {root}')
    if not root.is_dir():
        raise ValueError(f'result folder must be a directory: {root}')

    # Find all summary.json files
    summary_paths = sorted(root.rglob('summary.json'))
    summary_dirs: set[Path] = set()
    for sp in summary_paths:
        summary_dirs.add(sp.parent.resolve())

    # Group profile JSONs by their parent model directory
    grouped_profiles: defaultdict[Path, list] = defaultdict(list)
    skipped_json = 0

    for json_path in sorted(root.rglob('*.json')):
        if json_path.name == 'summary.json':
            continue
        if json_path.parent.resolve() not in summary_dirs:
            continue

        payload = _safe_read_json(json_path)
        if payload is None or not _looks_like_profile_payload(payload, json_path):
            skipped_json += 1
            continue

        if not isinstance(payload, dict):
            continue

        # Wrap raw results in a 'results' key if needed
        if 'results' not in payload:
            payload = {'results': payload}

        grouped_profiles[json_path.parent.resolve()].append((json_path, payload))

    # Synthesize summary.json for directories that only have profile JSONs
    synthesized = 0
    for run_dir, profile_payloads in grouped_profiles.items():
        if not profile_payloads:
            continue

        summary_payload = _build_synthesized_summary(run_dir, profile_payloads)
        summary_path = run_dir / 'summary.json'
        summary_path.write_text(
            json.dumps(summary_payload, indent=2), encoding='utf-8'
        )
        summary_paths.append(summary_path)
        synthesized += 1

    # Deduplicate and process summaries
    unique_summaries = sorted(set(str(p.resolve()) for p in summary_paths))

    if not unique_summaries:
        raise ValueError(f'no summary.json or valid llama-bench profile json files found under: {root}')

    # Write per-model reports
    for summary_path in unique_summaries:
        write_model_reports(summary_path, str(Path(summary_path).parent / 'index.html'),
                            str(Path(summary_path).parent / 'report.md'))

    # Collect payloads for sweep-level report
    payloads: list[dict] = []
    for summary_path in unique_summaries:
        payload = _safe_read_json(Path(summary_path))
        if isinstance(payload, dict):
            payloads.append(payload)

    if not payloads:
        raise ValueError(f'no readable summary payloads found under: {root}')

    # Gather metadata from first payload
    first = payloads[0]
    toolboxes_set: set[str] = set()
    for item in payloads:
        tb = item.get('toolbox')
        if tb:
            toolboxes_set.add(str(tb))
    toolboxes = sorted(toolboxes_set) or ['unknown']

    mode = str(first.get('mode', '')) or 'rebuild'

    profiles_set: set[str] = set()
    for item in payloads:
        for profile in item.get('profiles', []):
            if isinstance(profile, dict):
                pid = str(profile.get('id') or profile.get('label', '') or 'unknown')
                profiles_set.add(pid)
    profiles = sorted(profiles_set) or ['unknown']

    prompt_tokens_raw = first.get('prompt_tokens', 'n/a')
    gen_tokens_raw = first.get('gen_tokens', 'n/a')

    if isinstance(prompt_tokens_raw, list):
        prompt_tokens = ','.join(str(x) for x in prompt_tokens_raw)
    else:
        prompt_tokens = str(prompt_tokens_raw or '')

    if isinstance(gen_tokens_raw, list):
        gen_tokens = ','.join(str(x) for x in gen_tokens_raw)
    else:
        gen_tokens = str(gen_tokens_raw or '')

    repetitions = max(1, int(first.get('repetitions', 1)))

    timestamp_suffix = datetime.now().strftime('%y-%b-%d_%H_%M_%S').lower()

    out_html_path = root / f'index-{timestamp_suffix}.html'
    out_md_path = root / f'index-{timestamp_suffix}.md'

    collision = 1
    while out_html_path.exists() or out_md_path.exists():
        out_html_path = root / f'index-{timestamp_suffix}-{collision}.html'
        out_md_path = root / f'index-{timestamp_suffix}-{collision}.md'
        collision += 1

    write_run_summary(
        summary_paths=unique_summaries,
        out_html=str(out_html_path),
        out_md=str(out_md_path),
        toolboxes=toolboxes,
        mode=mode,
        profiles=profiles,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        repetitions=repetitions,
        command_template='reconstructed from existing llama-bench json artifacts',
        run_root=str(root),
    )

    return {
        'summary_count': len(unique_summaries),
        'synthesized_count': synthesized,
        'skipped_json_count': skipped_json,
        'out_html': str(out_html_path),
        'out_md': str(out_md_path),
    }
