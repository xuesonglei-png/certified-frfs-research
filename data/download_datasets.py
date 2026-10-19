"""Download the 15 named UCI datasets and produce auditable local NPZ files.

Run from the repository root: python data/download_datasets.py
Only NumPy and the Python standard library are required. Raw archives and UCI
page snapshots are retained so provenance does not depend on a future website.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent
SPECS = [
    ("iris", 53, "iris", ["iris.data"], ",", 0, -1, (), "All four predictors are continuous measurements."),
    ("wine", 109, "wine", ["wine.data"], ",", 0, 0, (), "All thirteen predictors are numerical chemical measurements."),
    ("wdbc", 17, "breast+cancer+wisconsin+diagnostic", ["wdbc.data"], ",", 0, 1, (0,), "Thirty continuous predictors; patient/sample ID removed."),
    ("ionosphere", 52, "ionosphere", ["ionosphere.data"], ",", 0, -1, (), "Numeric inputs include two binary/constant indicator columns; not exclusively continuous."),
    ("sonar", 151, "connectionist+bench+sonar+mines+vs+rocks", ["sonar.all-data"], ",", 0, -1, (), "Sixty numerical sonar energy measurements."),
    ("glass", 42, "glass+identification", ["glass.data"], ",", 0, -1, (0,), "Nine continuous numerical predictors; identifying row number removed."),
    ("seeds", 236, "seeds", ["seeds_dataset.txt"], None, 0, -1, (), "Seven numerical geometric measurements."),
    ("ecoli", 39, "ecoli", ["ecoli.data"], None, 0, -1, (0,), "Seven numerical predictors include binary protein indicators; sequence identifier removed."),
    ("yeast", 110, "yeast", ["yeast.data"], None, 0, -1, (0,), "Eight numerical localization predictors include discrete indicator-like scores; sequence identifier removed."),
    ("vehicle", 149, "statlog+vehicle+silhouettes", [f"xa{letter}.dat" for letter in "abcdefghi"], None, 0, -1, (), "Eighteen numerical silhouette descriptors, recorded as integer values."),
    ("transfusion", 176, "blood+transfusion+service+center", ["transfusion.data"], ",", 1, -1, (), "Four numerical count/time/volume predictors; includes algebraically redundant frequency and volume."),
    ("banknote", 267, "banknote+authentication", ["data_banknote_authentication.txt"], ",", 0, -1, (), "Four continuous wavelet summary measurements."),
    ("haberman", 43, "haberman+s+survival", ["haberman.data"], ",", 0, -1, (), "Three numerical age/year/count predictors, recorded as integers."),
    ("heart_statlog", 145, "statlog+heart", ["heart.dat"], None, 0, -1, (), "Thirteen numerical columns include categorical codes and binary indicators; ordinal-distance treatment is an explicit baseline limitation."),
    ("vertebral", 212, "vertebral+column", ["column_3C.dat"], None, 0, -1, (), "Six continuous biomechanical measurements; original three-class task."),
]

# These flags describe source semantics; they do not silently recode predictors.
CATEGORICAL_FEATURES = {
    "ionosphere": [0, 1],
    "ecoli": [2, 3],
    "heart_statlog": [1, 2, 5, 6, 8, 10, 12],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, path: Path, refresh: bool) -> None:
    if path.exists() and not refresh:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "academic-reproducibility/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        payload = response.read()
    path.write_bytes(payload)


def member_bytes(archive: zipfile.ZipFile, basename: str) -> bytes:
    found = [n for n in archive.namelist() if Path(n).name == basename]
    if len(found) == 1:
        return archive.read(found[0])
    # UCI vertebral-column archives have also been distributed as nested ZIPs.
    for name in archive.namelist():
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(name))) as child:
                try:
                    return member_bytes(child, basename)
                except FileNotFoundError:
                    pass
    raise FileNotFoundError(f"{basename}: available members {archive.namelist()}")


def process(spec, refresh: bool = False) -> dict:
    name, uci_id, slug, members, delimiter, skip, target_col, ignored, notes = spec
    raw_path = ROOT / "raw" / f"{name}.zip"
    page_path = ROOT / "sources" / f"{name}.html"
    archive_url = f"https://archive.ics.uci.edu/static/public/{uci_id}/{slug}.zip"
    source_url = f"https://archive.ics.uci.edu/dataset/{uci_id}/{slug}"
    fetch(archive_url, raw_path, refresh)
    page_error = None
    try:
        fetch(source_url, page_path, refresh)
    except Exception as error:
        page_error = str(error)
    rows = []
    with zipfile.ZipFile(raw_path) as archive:
        archive_members = archive.namelist()
        for member in members:
            lines = member_bytes(archive, member).decode("utf-8-sig").splitlines()[skip:]
            for line in lines:
                if not line.strip():
                    continue
                row = next(csv.reader([line], delimiter=delimiter)) if delimiter else line.split()
                rows.append([entry.strip() for entry in row])
    if len({len(row) for row in rows}) != 1:
        raise ValueError(f"Inconsistent column counts in {name}")
    table = np.asarray(rows, dtype=str)
    target_index = target_col % table.shape[1]
    feature_indices = [i for i in range(table.shape[1]) if i != target_index and i not in ignored]
    labels, y = np.unique(table[:, target_index], return_inverse=True)
    X = np.asarray([[float(value) if value not in {"?", "", "NA", "NaN", "nan"} else np.nan
                     for value in row] for row in table[:, feature_indices]], dtype=np.float64)
    y = y.astype(np.int64)
    processed_path = ROOT / "processed" / f"{name}.npz"
    np.savez_compressed(processed_path, X=X, y=y)
    page = page_path.read_text(encoding="utf-8", errors="replace") if page_path.exists() else ""
    license_marker = "creativecommons.org/licenses/by/4.0"
    license_verified = license_marker in page
    license_paragraph = re.search(r"<p[^>]*>This dataset is licensed under.*?</p>", page, re.DOTALL)
    license_quote = (re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", license_paragraph.group(0)))).strip()
                     if license_paragraph else None)
    citation_match = re.search(r"10\.24432/[A-Za-z0-9]+", page)
    return {
        "name": name, "uci_id": uci_id, "source_url": source_url,
        "archive_url": archive_url, "archive_members": archive_members,
        "parsed_members": members, "archive_sha256": sha256(raw_path),
        "processed_sha256": sha256(processed_path),
        "array_sha256": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "n_classes": int(len(labels)), "class_labels": labels.tolist(),
        "class_counts": np.bincount(y).tolist(), "missing_values": int(np.isnan(X).sum()),
        "missing_handling": "Missing source values remain NaN. Fit imputation on each training fold only.",
        "dropped_source_columns_zero_based": list(ignored),
        "target_source_column_zero_based": target_index,
        "feature_source_columns_zero_based": feature_indices,
        "notes": notes,
        "categorical_or_indicator_features_zero_based": CATEGORICAL_FEATURES.get(name, []),
        "all_predictors_numerically_encoded": True,
        "license": "CC BY 4.0" if license_verified else "Not verified from downloaded page",
        "license_url": "https://creativecommons.org/licenses/by/4.0/" if license_verified else None,
        "license_evidence": license_marker if license_verified else None,
        "license_quote": license_quote,
        "doi": citation_match.group(0) if citation_match else None,
        "source_page_sha256": sha256(page_path) if page_path.exists() else None,
        "source_page_error": page_error,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Fetch archives and source pages again.")
    parser.add_argument("--only", nargs="+", choices=[s[0] for s in SPECS])
    args = parser.parse_args()
    for folder in ("raw", "processed", "sources"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    specs = [s for s in SPECS if not args.only or s[0] in args.only]
    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(process, spec, args.refresh): spec[0] for spec in specs}
        for future in concurrent.futures.as_completed(jobs):
            name = jobs[future]
            try:
                result = future.result()
                results.append(result)
                print(f"{name}: {result['n_samples']} x {result['n_features']}, {result['n_classes']} classes", flush=True)
            except Exception as error:
                errors.append((name, repr(error)))
                print(f"ERROR {name}: {error!r}", flush=True)
    manifest_path = ROOT / "manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if args.only and manifest_path.exists() else {}
    prior.update({item["name"]: item for item in results})
    manifest_path.write_text(json.dumps(dict(sorted(prior.items())), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(f"Failed datasets: {errors}")


if __name__ == "__main__":
    main()
