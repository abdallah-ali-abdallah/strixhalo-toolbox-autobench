"""Textual-based TUI for live benchmark progress monitoring."""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import signal
import subprocess
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .runner import (
    ProfileResult,
    RunConfig,
    RunEvent,
    RunProgress,
    collect_system_metadata,
    format_duration,
    power_profile_label,
    run_benchmarks as _run_benchmarks,
)


def run_with_tui(cfg: RunConfig) -> dict:
    """Run benchmarks with an interactive Textual TUI overlay."""
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Container, Horizontal, Vertical
        from textual.screen import Screen
        from textual.widgets import Footer, Header, Label, Static, DataTable
    except ImportError:
        print('Textual is required for TUI mode. Install with: uv pip install textual')
        raise

    # Shared state
    progress = RunProgress()
    event_queue: queue.Queue[RunEvent] = queue.Queue()
    system_info: dict = {}
    results_lock = threading.Lock()
    all_results: list[dict] = []
    run_completed = threading.Event()

    def _progress_callback(event: RunEvent) -> None:
        event_queue.put(event)

    # Override cfg with progress callback
    cfg = RunConfig(
        models=cfg.models,
        toolbox=cfg.toolbox,
        run_all_backends=cfg.run_all_backends,
        mode=cfg.mode,
        prompt_tokens=cfg.prompt_tokens,
        gen_tokens=cfg.gen_tokens,
        repetitions=cfg.repetitions,
        power_profiles=cfg.power_profiles,
        run_root=cfg.run_root,
        dry_run=cfg.dry_run,
        progress_sink=_progress_callback,
    )

    def _collect_system_info():
        nonlocal system_info
        try:
            system_info = collect_system_metadata()
        except Exception:
            system_info = {'machine_name': 'unknown'}

    info_thread = threading.Thread(target=_collect_system_info, daemon=True)
    info_thread.start()

    def _run_bench():
        try:
            result = _run_benchmarks(cfg)
            with results_lock:
                all_results.append(result)
        except Exception as e:
            event_queue.put(RunEvent(type='error', message=str(e)))
        finally:
            run_completed.set()

    bench_thread = threading.Thread(target=_run_bench, daemon=True)
    bench_thread.start()

    class BenchScreen(Screen):
        """Main TUI screen."""

        def compose(self) -> ComposeResult:
            yield Header()
            yield Vertical(
                Label('Collecting system info...', id='sys-info'),
                Label('', id='status'),
                DataTable(id='results'),
                Label('', id='progress-bar'),
                id='main',
            )
            yield Footer()

        def on_mount(self) -> None:
            self.query_one('#results', DataTable).add_columns(
                'Model', 'Toolbox', 'Profile', 'Status', 'Runtime', 'Gen TPS'
            )
            self.update_interval = 0.25  # Refresh every 250ms

        def _process_events(self) -> None:
            while not event_queue.empty():
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    break

                if event.type == 'start':
                    self.query_one('#status', Label).update(
                        f'{event.message}'
                    )
                    progress.current_model = event.model
                    progress.current_toolbox = event.toolbox
                    progress.current_profile = event.profile

                elif event.type == 'progress':
                    progress.completed += 1
                    self.query_one('#status', Label).update(
                        f'[{progress.completed}/{progress.total}] {event.message}'
                    )
                    if 'result' in event.data:
                        r: ProfileResult = event.data['result']
                        runtime_str = format_duration(r.runtime_seconds) if r.runtime_seconds else 'n/a'
                        gen_tps_str = f'{r.gen_tps:.2f}' if r.gen_tps else 'n/a'
                        self.query_one('#results', DataTable).add_row(
                            Path(event.model).name,
                            event.toolbox,
                            event.profile,
                            r.status,
                            runtime_str,
                            gen_tps_str,
                        )

                elif event.type == 'error':
                    self.query_one('#status', Label).update(f'ERROR: {event.message}')
                    run_completed.set()

        def on_idle(self) -> None:
            self._process_events()

    class BenchApp(App):
        """Main TUI application."""

        BINDINGS = [
            ('q', 'quit', 'Quit'),
            ('h', 'help', 'Help'),
        ]

        CSS = """
        #main {
            padding: 1;
            layout: grid;
            grid-size: 1;
            grid-rows: auto auto 1fr auto;
        }
        #sys-info {
            text-align: center;
            color: $accent;
        }
        #status {
            text-align: center;
            padding: 1;
            background: $surface;
        }
        #results {
            height: 1fr;
        }
        #progress-bar {
            text-align: center;
            color: $primary;
        }
        DataTable {
            width: 100%;
        }
        """

        def on_mount(self) -> None:
            self.push_screen(BenchScreen())

    # Run the TUI
    app = BenchApp()
    result = app.run()

    # Wait for benchmark to finish
    run_completed.wait(timeout=3600)
    bench_thread.join(timeout=5)

    if all_results:
        return all_results[0]
    return {}


# ── Lightweight console progress (no Textual dependency) ──────────────────────

def run_with_console_progress(cfg: RunConfig) -> dict:
    """Run benchmarks with simple console-based progress display."""
    total_runs = len(cfg.models) * len(cfg.power_profiles)
    if cfg.run_all_backends:
        try:
            _, tb_list = __import__('.utils', fromlist=['discover_toolboxes']).discover_toolboxes()
            total_runs *= len(tb_list)
        except Exception:
            pass

    completed = 0
    current_info = {'model': '', 'toolbox': '', 'profile': ''}
    results_table: list[list] = []

    def _progress(event: RunEvent) -> None:
        nonlocal completed
        if event.type == 'start':
            current_info['model'] = Path(event.model).name
            current_info['toolbox'] = event.toolbox
            current_info['profile'] = event.profile
            print(f'\r  [{event.toolbox}] {current_info["model"]} → {event.profile}...', end='', flush=True)

        elif event.type == 'progress' and 'result' in event.data:
            r: ProfileResult = event.data['result']
            completed += 1
            runtime_str = format_duration(r.runtime_seconds) if r.runtime_seconds else 'n/a'
            gen_tps = f'{r.gen_tps:.2f} t/s' if r.gen_tps else 'n/a'
            print(f'\r  [{completed}/{total_runs}] {current_info["model"]} | {event.profile}: '
                  f'{r.status} ({runtime_str}, {gen_tps})', flush=True)

            results_table.append([
                Path(event.model).name,
                event.toolbox,
                event.profile,
                r.status,
                runtime_str,
                gen_tps,
            ])

    # Override config
    cfg = RunConfig(
        models=cfg.models,
        toolbox=cfg.toolbox,
        run_all_backends=cfg.run_all_backends,
        mode=cfg.mode,
        prompt_tokens=cfg.prompt_tokens,
        gen_tokens=cfg.gen_tokens,
        repetitions=cfg.repetitions,
        power_profiles=cfg.power_profiles,
        run_root=cfg.run_root,
        dry_run=cfg.dry_run,
        progress_sink=_progress,
    )

    result = _run_benchmarks(cfg)

    # Print summary table
    if results_table:
        print(f'\n{"="*80}')
        print(f'{"Model":<30} {"Toolbox":<25} {"Profile":<15} {"Status":<8} {"Runtime":<10} {"Gen TPS"}')
        print(f'{"-"*80}')
        for row in results_table:
            print(f'{row[0]:<30} {row[1]:<25} {row[2]:<15} {row[3]:<8} {row[4]:<10} {row[5]}')
        print()

    return result



