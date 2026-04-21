# toolbox-autobench

`toolbox-autobench` automates `llama-bench` sweeps across one or more GGUF models, one or more toolbox environments, and multiple Linux power profiles. It generates per-model reports plus a sweep-level HTML and Markdown report with leaderboards and context-size comparisons.

The project is a Python rewrite of an earlier shell workflow and is intended for repeatable local benchmarking on systems that use `toolbox`.

## Available Toolboxes

Prebuilt `llama-*` toolboxes for **AMD Strix Halo** with a `llama.cpp` backend are available at
[**kyuz0/amd-strix-halo-toolboxes**](https://github.com/kyuz0/amd-strix-halo-toolboxes).
These images provide a ready-to-use environment for local inference benchmarking on Strix Halo laptops.

This project automates the full testing workflow — including sweeps across **power profiles** (`power-saver`,
`balanced`, `performance`) — making it straightforward to compare performance characteristics on real hardware.

## What It Does

- Benchmarks one model, many models, or every `.gguf` file in one or more folders
- Runs against a single toolbox or every discovered `llama-*` toolbox
- Sweeps power profiles: `power-saver`, `balanced`, `performance`
- Supports quick, medium, and custom token plans
- Produces:
  - per-model `summary.json`
  - per-model `index.html`
  - per-model `report.md`
  - sweep-level `index.html`
  - sweep-level `index.md`
- Can rebuild reports from an existing results folder without rerunning benchmarks
- Offers an optional Textual TUI for live progress and system charts

## Requirements

### Python

- Python `>=3.10`
- `uv` for environment management

### System Tools

- `toolbox`
- `llama-bench` available inside the selected toolbox/container
- `powerprofilesctl` for normal benchmark runs

Notes:

- `powerprofilesctl` is not required for `--dry-run`
- `--run-all-backends` only uses toolboxes whose names begin with `llama-`

## Installation

```bash
cd ~/toolbox-autobench
uv sync
```

For development and tests:

```bash
cd ~/toolbox-autobench
uv sync --extra dev
```

## CLI Usage

```bash
uv run toolbox-autobench --help
```

Basic form:

```bash
uv run toolbox-autobench [options] <model_path_or_folder[,more]>
```

Accepted inputs:

- A single `.gguf` file
- Multiple `.gguf` files as a comma-separated list
- A folder path as a positional argument
- Multiple folders via positional comma-separated input
- One or more folders via `--folder`
- A mix of `--folder` plus positional `.gguf` files

Model folders are scanned non-recursively.

## Common Examples

Single model:

```bash
uv run toolbox-autobench --quick /home/abdallah/models/LFM2-24B-A2B-Q8_0.gguf
```

Single model with TUI:

```bash
uv run toolbox-autobench --tui --quick /home/abdallah/models/LFM2-24B-A2B-Q8_0.gguf
```

All models in one folder using all discovered llama toolboxes:

```bash
uv run toolbox-autobench --folder /home/abdallah/models --run-all-backends --quick
```

Dry run with explicit power profiles:

```bash
uv run toolbox-autobench \
  --dry-run \
  --run-all-backends \
  --power-profile power-saver,balanced \
  /home/abdallah/models/LFM2-24B-A2B-Q8_0.gguf
```

Multiple folders:

```bash
uv run toolbox-autobench \
  --folder /models/set_a,/models/set_b,/models/set_c \
  --run-all-backends \
  --quick
```

Custom tokens:

```bash
uv run toolbox-autobench \
  --toolbox llama-rocm7-nightlies \
  -t 1024,4096,8192 \
  -n 512,2048 \
  /home/abdallah/models/model.gguf
```

Rebuild reports from an existing results directory:

```bash
uv run toolbox-autobench \
  --build-report-from-result \
  /home/abdallah/toolbox-autobench/reports/12-04-26-03-44-23
```

## Main Options

### Execution

- `--toolbox <name>`: run against one toolbox. Default: `llama-rocm7-nightlies`
- `--run-all-backends`: benchmark all discovered `llama-*` toolboxes
- `--tui`: run inside the interactive terminal UI
- `--dry-run`: print the planned sweep without executing commands
- `--build-report-from-result <dir>`: rebuild reports from existing JSON artifacts
- `-r, --repetitions <int>`: repetitions per profile

### Token Modes

- `--quick`: preset `-p 512` and `-n 512`
- `--medium`: preset `-p 1024,4096,8192,16384` and `-n 512,2048`
- `-t, --tokens <csv>`: custom prompt token list
- `-n, --gen-tokens <csv>`: custom generation token list

If you use custom tokens, you must provide both `-t` and `-n`.

### Power Profiles

- `--power-profile <csv>`
- `--power-profil <csv>`: backward-compatible alias

Supported values:

- `power-saver`
- `balanced`
- `performance`

## Output Layout

Benchmark runs are written under the repo-local `reports/` directory:

```text
reports/
  12-04-26-03-44-23/
    index.html
    index.md
    <model-id>/
      <toolbox-name>/
        summary.json
        index.html
        report.md
        power-saver.json
        balanced.json
        performance.json
        *.log
```

Notes:

- The sweep-level report lives at `reports/<timestamp>/index.html` and `index.md`
- Each model/toolbox directory also gets its own HTML and Markdown report
- Rebuilding reports from existing results writes fresh sweep reports into the selected run directory

## Reports

The generated reports include:

- Machine and OS metadata
- Kernel version and build
- CPU model and system architecture
- `llama-bench` version and build metadata
- Prompt and generation token settings
- Profile-by-profile results
- Sweep leaderboards
- Context-size comparisons
- Interactive HTML charts

The Markdown sweep report includes separate leaderboards for:

- Token Generation per second
- Prompt Processing per second

## How Runs Are Executed

For each model:

1. Resolve the selected toolbox or toolboxes
2. Switch to the requested power profile
3. Verify the active power profile
4. Run `llama-bench` inside the toolbox
5. Capture JSON and logs
6. Generate per-model reports
7. Generate a sweep-level report after all runs finish

The command template used for benchmarking is:

```bash
toolbox run -c <toolbox> -- llama-bench -m <model> -mmp 0 -fa 1 -p '<prompt>' -n '<gen>' -r <reps> -o json
```

## TUI Mode

Run with:

```bash
uv run toolbox-autobench --tui --quick /path/to/model.gguf
```

The TUI shows:

- live benchmark progress
- current run status
- leaderboard snapshots
- CPU monitoring
- GPU/power monitoring

## Testing

Run the test suite:

```bash
uv run pytest -q
```

Coverage is enabled by default through `pyproject.toml`.

If `pytest` is missing, sync the dev dependencies first:

```bash
uv sync --extra dev
```

## Troubleshooting

### `toolbox` not found

Install `toolbox` and ensure it is available in `PATH`.

### `powerprofilesctl` not found

Install it if you want real benchmark runs. It is not needed for `--dry-run`.

### No models found

Check that:

- paths exist
- model files end with `.gguf`
- folders contain `.gguf` files directly

### Report rebuild found no valid inputs

`--build-report-from-result` expects a directory containing either:

- existing `summary.json` files, or
- valid `llama-bench` JSON profile outputs that can be synthesized into summaries

## Development Notes

- Source package: [`src/toolbox_autobench`](src/toolbox_autobench)
- Tests: [`tests`](tests)
- Reports directory is ignored by Git via [`.gitignore`](.gitignore)
