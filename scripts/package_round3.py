"""Build two local round-three review archives, with verified byte manifests.

Run only after the manuscript/PDF and submission drafts have been finalized.
Existing archives are never overwritten. This script performs no upload.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
REVIEW = OUTPUT / "certified-frfs-round3-review.zip"
SUBMISSION = OUTPUT / "certified-frfs-round3-submission-source.zip"
REPORT = OUTPUT / "round3-bundle-manifest.json"

PAPER_REQUIRED = (
    "paper-round3/main.tex", "paper-round3/results.tex",
    "paper-round3/references.bib", "paper-round3/main.bbl",
    "paper-round3/main.pdf", "paper-round3/figures/gap-validation.png",
)
EVIDENCE_REQUIRED = (
    "results/round3/margin-pilot.json",
    "results/round3/margin-prototypes.json",
    "results/round3/margin-mixture-benchmark.json",
    "results/round3/margin-gap-benchmark.json",
    "results/round3/sat-reduction-audit.json",
    "results/round3/test-run-final.json",
    "results/round3/test-run-final.log",
    "results/round3/source-snapshots/margin-pilot-margin.py",
    "results/round3/source-snapshots/README.md",
    "results/real-subtable-verification.json",
    "research/round3/protocol.md",
    "research/round3/mixture-benchmark-protocol.md",
    "research/round3/margin-gap-benchmark-protocol.md",
    "submission/round3/reproducibility.md",
)
BIB_JSON = (
    "crossref-classics-and-recent.json", "crossref-initial.json",
    "verified-recent-records.json", "Sion1958.crossref.json",
    "Cook1971.crossref.json", "margin2004_metadata.json",
)
EXCLUSIONS = [
    "Third-party literature PDFs, reference-article PDFs, and text extracts",
    "Whole search/web/journal page caches, including the PMLR HTML page",
    "submission/round3/author-metadata.json and private author-reference files",
    "Raw dataset ZIPs and source-page caches; processed data and provenance retained",
    "Virtual environments, bytecode, pytest cache, and temporary files",
    "Earlier-round ZIP files and unrelated historical experiment outputs",
    "LaTeX auxiliary, log, and synctex files except the requested main.bbl",
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


class Contents:
    def __init__(self) -> None:
        self.payload: dict[str, bytes] = {}
        self.origin: dict[str, dict] = {}

    def add(self, name: str, data: bytes, origin: dict) -> None:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Unsafe archive path: {name}")
        if name in self.payload:
            if self.payload[name] != data:
                raise ValueError(f"Conflicting archive entry: {name}")
            return
        self.payload[name] = data
        self.origin[name] = origin

    def file(self, path: Path, name: str | None = None) -> None:
        path = path.resolve()
        relative = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        self.add(name or relative, data, {"source_path": relative, "source_sha256": sha(data)})

    def generated(self, name: str, content: str, description: str) -> None:
        self.add(name, content.encode("utf-8"), {"generated": description})


def bibliographic_item(item: dict) -> dict:
    """Keep the fields actually consumed by the existing bibliography builder."""
    keys = ("DOI", "title", "container-title", "published", "issued", "page", "volume", "type", "URL")
    result = {key: item[key] for key in keys if key in item}
    if item.get("author") is not None:
        result["author"] = [
            {key: author[key] for key in ("family", "given", "name") if key in author}
            for author in item["author"]
        ]
    return result


def add_curated_bibliography(contents: Contents) -> None:
    directory = ROOT / "research/round3/search"
    for filename in BIB_JSON:
        path = directory / filename
        raw = path.read_bytes()
        data = json.loads(raw)
        if filename.startswith("crossref-"):
            reduced = [
                {"status": row.get("status"),
                 "items": [bibliographic_item(item) for item in row.get("items", [])]}
                for row in data
            ]
        elif filename == "verified-recent-records.json":
            reduced = []
            for row in data:
                record = {key: row[key] for key in ("doi", "status", "url") if key in row}
                message = row.get("result", {}).get("message")
                record["result"] = {"message": bibliographic_item(message)} if isinstance(message, dict) else {}
                reduced.append(record)
        else:
            reduced = {"status": data.get("status"), "message": bibliographic_item(data["message"])}
        contents.add(path.relative_to(ROOT).as_posix(), json_bytes(reduced), {
            "source_path": path.relative_to(ROOT).as_posix(), "source_sha256": sha(raw),
            "transformation": "Curated bibliographic fields only; response bodies, abstracts, links and affiliations removed",
        })
    # The builder prefers this verified record to its local HTML fallback.
    verified = directory / "Voracek2022.verified.bib"
    if not verified.is_file():
        raise FileNotFoundError(f"Standalone PMLR bibliography required before packaging: {verified}")
    for path in sorted(directory.glob("*.bib")):
        contents.file(path)
    contents.file(ROOT / "paper/references.bib")


def attribution(manifest: dict) -> str:
    lines = [
        "# Dataset attribution and transformations", "",
        "These 15 datasets originate from the UCI Machine Learning Repository. "
        "Credit the original dataset creators through each official source/DOI record below. "
        "The recorded license is CC BY 4.0; the linked official dataset citation remains authoritative.",
        "", "The packaged NPZ files contain parsed, unscaled numerical arrays and integer-encoded labels. "
        "No global imputation, scaling or feature selection is applied to them. "
        "Training-fold preprocessing is recorded separately in the saved experiment inputs.", "",
    ]
    for name, row in sorted(manifest.items()):
        doi = row.get("doi", "")
        lines.extend([
            f"## {name}", "",
            f"- Official source: {row['source_url']}",
            f"- Dataset DOI: https://doi.org/{doi}" if doi else "- Dataset DOI: not recorded",
            f"- License: {row.get('license', 'not recorded')} — {row.get('license_url', '')}",
            f"- Retrieved: {row.get('retrieved_utc', 'not recorded')}",
            f"- Shape: {row['n_samples']} objects, {row['n_features']} attributes; {row['n_classes']} classes.",
            f"- Parsed source files: {', '.join(row.get('parsed_members', []))}",
            f"- Removed source columns (zero-based): {row.get('dropped_source_columns_zero_based', [])}.",
            f"- Target column (zero-based): {row.get('target_source_column_zero_based', 'not recorded')}.",
            f"- Encoded label mapping: {json.dumps(row.get('class_labels', []), ensure_ascii=False)}.",
            f"- Transformations/interpretation: {row.get('notes', '')}",
            f"- Missing-value handling: {row.get('missing_handling', '')}",
            f"- Processed archive SHA-256: `{row['processed_sha256']}`.",
            f"- Array-content SHA-256: `{row.get('array_sha256', 'not recorded')}`.", "",
        ])
    lines.extend([
        "All source/licensing evidence and parsing fields are retained in `data/manifest.json`. "
        "The original web-page caches and downloaded raw ZIPs are excluded from this review archive. "
        "Dataset attribution does not license the authors' code or third-party articles.", "",
    ])
    return "\n".join(lines)


def add_paper(contents: Contents, *, flatten: bool) -> None:
    for relative in PAPER_REQUIRED:
        path = ROOT / relative
        contents.file(path, path.relative_to(ROOT / "paper-round3").as_posix() if flatten else None)
    for path in sorted((ROOT / "paper-round3").glob("*.tex")):
        contents.file(path, path.name if flatten else None)
    for path in sorted((ROOT / "paper-round3/figures").glob("*.png")):
        contents.file(path, "figures/" + path.name if flatten else None)


def add_submission_docs(contents: Contents) -> None:
    directory = ROOT / "submission/round3"
    for suffix in ("*.md", "*.txt", "*.docx"):
        for path in sorted(directory.glob(suffix)):
            contents.file(path)
    access = directory / "journal-guidance/access.json"
    if access.is_file():
        contents.file(access)


def collect_review() -> Contents:
    contents = Contents()
    add_paper(contents, flatten=False)
    for name in ("pyproject.toml", "requirements-lock.txt", "data/manifest.json", "data/download_datasets.py"):
        contents.file(ROOT / name)
    for directory in ("src", "tests", "scripts"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            if "__pycache__" not in path.parts:
                contents.file(path)
    # Only third-round top-level research documents; no search or author-reference tree.
    for path in sorted((ROOT / "research/round3").glob("*.md")):
        contents.file(path)
    for path in sorted((ROOT / "results/round3").glob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".log", ".xml"}:
            contents.file(path)
    for path in sorted((ROOT / "results/round3/source-snapshots").glob("*")):
        if path.is_file() and path.suffix.lower() in {".py", ".md"}:
            contents.file(path)
    contents.file(ROOT / "results/real-subtable-verification.json")
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != 15:
        raise ValueError("Round-three bundle expects the retained fifteen-dataset manifest")
    for name, record in sorted(manifest.items()):
        path = ROOT / "data/processed" / f"{name}.npz"
        if sha(path.read_bytes()) != record["processed_sha256"]:
            raise ValueError(f"Processed dataset hash mismatch: {name}")
        contents.file(path)
        contents.file(ROOT / "results/primary" / f"{name}-s17-f0.json")
    contents.generated("data/ATTRIBUTION.md", attribution(manifest), "Dataset DOI/license/transformations from data/manifest.json")
    add_curated_bibliography(contents)
    add_submission_docs(contents)
    contents.generated("README.md", REVIEW_README, "Round-three archive entry point")
    return contents


REVIEW_README = """# Third-round research review and reproducibility package

This is a local review archive, not a submitted or publicly deposited paper.
Start with `research/round3/研究进展与投稿清单.md`, then read
`research/round3/final-evidence-audit.md` and `submission/round3/reproducibility.md`.
The editable manuscript and PDF are in `paper-round3/`.

The target is the robust leave-one-out nearest-friend/enemy margin. Earlier
dependency-preservation results are different targets and are not relabelled here.
The retained 15-dataset first-fold pilot has limited compression and no established
predictive advantage. The full 384-object synthetic benchmark adds no positive
certifications despite 56 improved mixture bounds. Keep these negative results.

Create a Python 3.11 environment and install `requirements-lock.txt`. Relevant
entry points are `run_margin_prototypes.py`, `run_margin_pilot.py`,
`verify_margin_sat_reduction.py`, `benchmark_margin_mixtures.py`,
`benchmark_margin_gap.py` and `build_margin_paper.py` under `scripts/`.
The gap pass consumes the mixture benchmark. Use a separate copy/new outputs for
reruns: some scripts refuse existing outputs while others overwrite fixed paths.
The other retained scripts are historical source context and may need inputs
outside this third-round archive; do not run all scripts indiscriminately.

The final whole-project test snapshot is 303 passed; the earlier 286-test snapshot
is retained separately. See `results/round3/test-run-final.json` for the source
hashes. `source-snapshots/margin-pilot-margin.py` under the same directory is the
exact historical pilot module. A later defensive mixture-helper change did not
affect the pilot's MarginWorkspace computation. It must not silently replace the
original run fingerprint. Ordinary floating arithmetic is not formal certification.

Dataset credit/license/transformation records are in `data/ATTRIBUTION.md` and
`data/manifest.json`. The bundle's bibliographic JSONs are reduced metadata with
the original source and packaged hashes recorded in `ROUND3-MANIFEST.json`.
The standalone Voracek2022 BibTeX replaces the optional local PMLR page fallback.
No third-party article PDFs, full text extracts, web-page caches, private author
metadata, virtual environments or prior-round ZIPs are included.
Author drafts under `submission/round3/` remain review materials.

The manifest lists every payload file with SHA-256, byte length and provenance;
its own bytes are checked in the external build report rather than self-hashed.
No publication or external release is performed by the packaging script.
"""

SUBMISSION_README = """# Editable manuscript source package — local author review

This archive is not a submission confirmation. `main.tex`, `results.tex`,
`references.bib`, `main.bbl`, `main.pdf` and `figures/` are the editable paper.
Draft cover letter, highlights and author-review notes, where present, are under
`submission/round3/`. The official-access status record is included without the
fetched publisher page caches. Private author-metadata.json is excluded.

Compile from this directory with XeLaTeX, BibTeX, XeLaTeX, XeLaTeX, for example:
    xelatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    xelatex -interaction=nonstopmode -halt-on-error main.tex
    xelatex -interaction=nonstopmode -halt-on-error main.tex
Install elsarticle, fontspec, Latin Modern Roman and the packages declared in
main.tex. The saved main.bbl is also provided. No numerical experiments need to
be rerun to compile the saved manuscript.

For proofs, raw numerical records, code, tests, datasets and split inputs use the
separate certified-frfs-round3-review.zip. Author statements, target-journal
requirements and the outstanding theorem-level prior-art comparison still need
author review. The manifest describes the archive bytes, not editorial approval.
"""


def build_archive(target: Path, contents: Contents, created: str, kind: str) -> dict:
    records = {name: {"sha256": sha(data), "bytes": len(data), **contents.origin[name]}
               for name, data in sorted(contents.payload.items())}
    manifest = {
        "created_utc": created, "kind": kind,
        "purpose": "Private local review; no upload or submission",
        "files": records, "payload_files": len(records),
        "payload_bytes": sum(len(data) for data in contents.payload.values()),
        "exclusions": EXCLUSIONS,
        "manifest_scope": "All payloads except this manifest; external report hashes the manifest and ZIP",
    }
    payload = {**contents.payload, "ROUND3-MANIFEST.json": json_bytes(manifest)}
    with target.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in sorted(payload.items()):
                archive.writestr(name, data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(payload):
            raise RuntimeError(f"ZIP integrity or entry inventory failure: {target}")
        if len(archive.namelist()) != len(payload):
            raise RuntimeError(f"Duplicate ZIP entries: {target}")
        for name, data in payload.items():
            if sha(archive.read(name)) != sha(data):
                raise RuntimeError(f"ZIP roundtrip hash failure: {name}")
    return {
        "path": target.relative_to(ROOT).as_posix(), "sha256": sha(target.read_bytes()),
        "bytes": target.stat().st_size, "entries": len(payload),
        "uncompressed_bytes": sum(len(data) for data in payload.values()),
        "roundtrip_verified": True,
        "files": {name: {"sha256": sha(data), "bytes": len(data)} for name, data in sorted(payload.items())},
        "exclusions": EXCLUSIONS,
    }


def main() -> None:
    for path in (REVIEW, SUBMISSION, REPORT):
        if path.exists():
            raise FileExistsError(f"Preserve the existing build before creating a new one: {path}")
    for relative in (*PAPER_REQUIRED, *EVIDENCE_REQUIRED):
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(f"Finalize required manuscript/evidence before packaging: {relative}")
    historical = [OUTPUT / "certified-frfs-research-v1.zip", OUTPUT / "certified-frfs-round2-exploration.zip"]
    old_hashes = {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in historical if p.is_file()}
    review = collect_review()
    submission = Contents()
    add_paper(submission, flatten=True)
    add_submission_docs(submission)
    submission.generated("README.txt", SUBMISSION_README, "Editable-source compilation and scope notes")
    # Snapshot all bytes before creating outputs; active manuscript edits must finish first.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    report = {"created_utc": created, "purpose": "Local review only", "bundles": [
        build_archive(REVIEW, review, created, "research-review"),
        build_archive(SUBMISSION, submission, created, "editable-submission-source"),
    ]}
    for relative, expected in old_hashes.items():
        if sha((ROOT / relative).read_bytes()) != expected:
            raise RuntimeError(f"Historical bundle changed during packaging: {relative}")
    report["historical_bundles_unchanged"] = old_hashes
    with REPORT.open("xb") as stream:
        stream.write(json_bytes(report))
    print(json.dumps({"report": REPORT.relative_to(ROOT).as_posix(), "bundles": [
        {key: item[key] for key in ("path", "sha256", "bytes", "entries", "roundtrip_verified")}
        for item in report["bundles"]]}, indent=2))


if __name__ == "__main__":
    main()
