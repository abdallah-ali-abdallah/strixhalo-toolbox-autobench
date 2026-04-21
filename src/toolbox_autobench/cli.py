"""Command-line interface for toolbox-autobench."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

try:
    import typer
except ImportError:
    typer = None  # type: ignore[assignment]

from .runner import make_run_config, run_benchmarks, rebuild_reports
from .utils import (
    ANSI_BLUE as _BLUE,
    ANSI_CYAN as _CYAN,
    ANSI_GREEN as _GREEN,
    ANSI_RESET as _RESET,
    ANSI_YELLOW as _YELLOW,
    color,
    discover_toolboxes,
    normalize_power_profile,
    parse_csv_values,
    validate_power_profiles,
)


class Colors:
    """ANSI colour aliases for CLI output."""
    BLUE = _BLUE
    CYAN = _CYAN
    GREEN = _GREEN
    RESET = _RESET
    YELLOW = _YELLOW


def _version_callback(value: bool):
    if value:
        try:
            from . import __version__
            print(f'toolbox-autobench {__version__}')
        except ImportError:
            print('toolbox-autobench unknown')
        raise typer.Exit()


def _get_app():
    """Create the Typer app."""
    if typer is None:
        raise RuntimeError("'typer' is required. Install with: uv pip install typer")

    app = typer.Typer(
        name='toolbox-autobench',
        help='Automated llama-bench sweeps across toolboxes and power profiles.',
        add_completion=False,
    )

    @app.callback()
    def main(version: bool = typer.Option(False, '--version', callback=_version_callback, is_eager=True)):
        pass

    @app.command()
    def bench(
        model_paths: Optional[List[str]] = typer.Argument(
            None,
            help='GGUF file paths or folders. Comma-separated for multiple.',
        ),
        toolbox: Optional[str] = typer.Option(
            None, '--toolbox', '-t',
            help='Target toolbox name.',
        ),
        run_all_backends: bool = typer.Option(
            False, '--run-all-backends',
            help='Benchmark all discovered llama-* toolboxes.',
        ),
        mode: str = typer.Option(
            'quick', '--mode', '-m',
            help='Token preset: quick, medium, or custom.',
        ),
        prompt_tokens: Optional[str] = typer.Option(
            None, '--prompt-tokens', '-p',
            help='Custom prompt token sizes (comma-separated).',
        ),
        gen_tokens: Optional[str] = typer.Option(
            None, '--gen-tokens', '-n',
            help='Custom generation token sizes (comma-separated).',
        ),
        repetitions: int = typer.Option(
            1, '--repetitions', '-r',
            help='Number of repetitions per profile.',
        ),
        power_profiles: Optional[str] = typer.Option(
            None, '--power-profile',
            help='Power profiles to test (comma-separated). Default: all.',
        ),
        run_root: Optional[str] = typer.Option(
            None, '--run-root',
            help='Output directory for reports.',
        ),
        dry_run: bool = typer.Option(
            False, '--dry-run',
            help='Print the planned sweep without executing.',
        ),
        build_report_from_result: Optional[str] = typer.Option(
            None, '--build-report-from-result',
            help='Rebuild reports from an existing results directory.',
        ),
    ):
        """Run llama-bench sweeps across toolboxes and power profiles."""

        # Handle --build-report-from-result mode
        if build_report_from_result:
            result_dir = str(Path(build_report_from_result).expanduser().resolve())
            print(color(f'Rebuilding reports from: {result_dir}', Colors.CYAN))
            try:
                report = rebuild_reports(result_dir)
                print(color(f'Summary files found: {report.get("summary_count", 0)}', Colors.GREEN))
                print(color(f'Synthesized summaries: {report.get("synthesized_count", 0)}', Colors.YELLOW))
                print(color(f'Skipped JSON files: {report.get("skipped_json_count", 0)}', Colors.YELLOW))
                if report.get('out_html'):
                    print(color(f'HTML report: {report["out_html"]}', Colors.CYAN))
                if report.get('out_md'):
                    print(color(f'Markdown report: {report["out_md"]}', Colors.CYAN))
            except ValueError as e:
                typer.echo(color(str(e), Colors.YELLOW))
                raise typer.Exit(1)
            return

        # Validate inputs
        if not model_paths:
            typer.echo(color('Error: at least one model path or folder is required.', Colors.YELLOW))
            raise typer.Exit(1)

        # Parse comma-separated paths
        parsed_paths = []
        for p in model_paths:
            parts = [x.strip() for x in p.split(',') if x.strip()]
            parsed_paths.extend(parts)

        if not parsed_paths:
            typer.echo(color('Error: no valid model paths provided.', Colors.YELLOW))
            raise typer.Exit(1)

        # Determine mode
        actual_mode = mode
        if mode == 'quick':
            actual_mode = 'quick'
        elif mode == 'medium':
            actual_mode = 'medium'
        else:
            actual_mode = 'custom'

        # Validate power profiles early
        if power_profiles:
            try:
                validate_power_profiles(power_profiles)
            except ValueError as e:
                typer.echo(color(str(e), Colors.YELLOW))
                raise typer.Exit(1)

        # Build config
        cfg = make_run_config(
            model_paths=parsed_paths,
            toolbox=toolbox,
            run_all_backends=run_all_backends,
            mode=actual_mode,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            repetitions=repetitions,
            power_profiles=power_profiles,
            run_root=run_root,
            dry_run=dry_run,
        )

        # Print plan for dry-run
        if cfg.dry_run:
            print(color('\n=== Benchmark Plan (dry-run) ===', Colors.CYAN))
            toolboxes = ['llama-rocm7-nightlies']
            if run_all_backends:
                tbs, _ = discover_toolboxes()
                toolboxes = tbs or [cfg.toolbox or 'llama-rocm7-nightlies']

            for model in cfg.models:
                for tb in toolboxes:
                    for profile in cfg.power_profiles:
                        label = power_profile_label(profile) or profile
                        print(f'  {Path(model).name} | {tb} | {label}')
            print()
            return

        # Run benchmarks
        print(color('\n=== Starting Benchmark Sweep ===', Colors.CYAN))
        print(f'  Models:    {len(cfg.models)}')
        toolboxes = ['llama-rocm7-nightlies']
        if run_all_backends:
            tbs, _ = discover_toolboxes()
            toolboxes = tbs or [cfg.toolbox or 'llama-rocm7-nightlies']
        print(f'  Toolboxes: {len(toolboxes)}')
        print(f'  Profiles:  {", ".join(cfg.power_profiles)}')
        print(f'  Run root:  {cfg.run_root}')
        print()

        result = run_benchmarks(cfg)

        print(color('\n=== Benchmark Complete ===', Colors.GREEN))
        print(f'  Models run:      {result.get("models_run", 0)}')
        print(f'  Toolboxes used:  {result.get("toolboxes_used", 0)}')
        print(f'  Total runs:      {result.get("total_runs", 0)}')
        print(f'  Completed:       {result.get("completed", 0)}')
        if result.get('out_html'):
            print(color(f'  HTML report:     {result["out_html"]}', Colors.CYAN))
        if result.get('out_md'):
            print(color(f'  Markdown report: {result["out_md"]}', Colors.CYAN))
        print()

    return app


# CLI entry point
app = _get_app()


def main():
    """Main entry point when invoked as a script."""
    app()


if __name__ == '__main__':
    main()
