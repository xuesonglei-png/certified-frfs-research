# Certified FRFS Experiments

Code and compact data files for experiments on robust fuzzy-rough margins under interval uncertainty.

## Repository layout

- `src/certified_frfs/`: reusable implementations of the fuzzy-rough scores, certificates, and selectors.
- `scripts/`: experiment, analysis, verification, and result-generation entry points.
- `tests/`: automated tests.
- `data/`: compact benchmark files and dataset metadata.
- `requirements-lock.txt`: pinned Python dependencies.

Generated result files are not tracked. The commands below create them locally under `results/`.

## Requirements

- Python 3.11
- PowerShell on Windows, or an equivalent shell on other platforms

Create an environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Reproduction

Run the primary experiments:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py --workers 3 --output results\primary
```

Run the sensitivity experiments and build summary files and figures:

```powershell
.\.venv\Scripts\python.exe scripts\run_sensitivity.py --workers 2
.\.venv\Scripts\python.exe scripts\summarize_results.py
.\.venv\Scripts\python.exe scripts\build_paper_results.py
```

The verification scripts cover the theoretical examples and the stored experiment records:

```powershell
.\.venv\Scripts\python.exe scripts\run_prototypes.py
.\.venv\Scripts\python.exe scripts\validate_theory.py
.\.venv\Scripts\python.exe scripts\verify_reproduction.py
```

The scripts write generated outputs to `results/`, `output/`, or the paper figure directory. Existing result folders should be preserved before starting a run with a different protocol or configuration.

## Data

`data/manifest.json` records the source, preprocessing information, and license information for the included datasets. `data/download_datasets.py` can retrieve the public sources when a fresh download is needed.

## Code and paper

<https://github.com/xuesonglei-png/certified-frfs-research>
