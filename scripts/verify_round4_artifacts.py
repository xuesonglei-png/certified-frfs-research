"""Check round-four manuscript integration without rerunning experiments."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def main():
    paper = ROOT / "paper-round4"
    paths = sorted(paper.glob("*.tex"))
    tex = "\n".join(p.read_text(encoding="utf-8") for p in paths)
    keys = re.findall(r"@\w+\{([^,]+),", read("paper-round4/references.bib"))
    cites = {key.strip() for group in re.findall(
        r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", tex) for key in group.split(",")}
    assert len(keys) == len(set(keys)) == 20
    assert set(keys) == cites, (set(keys) - cites, cites - set(keys))
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    assert len(labels) == len(set(labels))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", tex))
    assert refs <= set(labels), refs - set(labels)
    for name in re.findall(r"\\input\{([^}]+)\}", tex):
        assert (paper / name).is_file(), name
    log = read("paper-round4/main.log")
    for marker in ["Overfull", "undefined references", "Citation `", "LaTeX Error"]:
        assert marker not in log, marker
    pdf = PdfReader(paper / "main.pdf")
    contents = "\n".join(p.extract_text() for p in pdf.pages)
    for value in ["Songlei Xue", "Zhongying Suo", "zaiyujian0101@163.com",
                  "suozhongying@sohu.com", "This research received no external funding."]:
        assert value in contents, value
    audit = json.loads(read("results/round4/projected-mixture-audit.json"))
    s = audit["summary"]
    assert s["objects"] == 384 and s["enemy_pairs"] == 1536
    assert s["pair_check_failures"] == s["object_check_failures"] == 0
    assert s["objects_all_enemies_old_zero"] == 170
    assert s["objects_all_enemies_new_zero"] == 218
    assert audit["new_MILP_calls"] == 0 and audit["inputs_sources_tests_unchanged"]
    coupling = json.loads(read("results/round4/fr-separate-joint-audit.json"))
    cs = coupling["summary"]
    assert cs["objects"] == 384
    assert cs["separate_positive"] == 59 and cs["joint_positive"] == 73
    assert cs["joint_strictly_tighter"] == 346 and cs["separate_strictly_tighter"] == 0
    assert coupling["new_MILP_calls"] == 0
    repeated = json.loads(read("results/summary.json"))
    recent = json.loads(read("results/recent-sfrip/summary.json"))
    assert recent["completed_splits"] == 132 and recent["n_datasets"] == 15
    assert recent["budget_pass_splits"] == 24
    assert len(list((ROOT / "results/recent-sfrip").glob("*-s*-f*.json"))) == 132
    assert repeated["completed_runs"] == 132 and repeated["datasets"] == 15
    assert abs(repeated["mean_removed_fraction"] - 0.21859413280981907) < 1e-12
    assert abs(repeated["knn_clean_comparisons"][0]["mean_delta"] + 0.01704729394376449) < 1e-12
    runtime = json.loads(read("results/round4/inference-runtime-summary.json"))
    assert runtime["completed_splits"] == 132 and runtime["threads"] == 1
    assert runtime["knn"]["median_speedup"] > 1
    assert runtime["svm"]["median_speedup"] > 1
    assert runtime["fuzzy_rough"]["median_speedup"] > 1
    for name, expected in audit["source_hashes"].items():
        assert sha(name) == expected, name
    previous = json.loads(read("results/round3/test-run-final.json"))
    for name, expected in previous["source_and_test_hashes"].items():
        assert sha(name) == expected, name
    examples = json.loads(read("results/round4/fr-examples.json"))
    assert examples["status"] == "passed"
    assert len(examples["examples"]) == 2
    bib_check = json.loads(read("research/round4/bibliography-check.log"))
    assert bib_check["valid_entries"] == bib_check["total_entries"] == 19
    assert not bib_check["missing_in_bib"] and not bib_check["unused_in_tex"]
    assert not [i for i in bib_check["issues"] if i["severity"] == "error"]
    (ROOT / "results/round4/bibliography-skill-check.json").write_text(
        json.dumps(bib_check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifacts = [str(p.relative_to(ROOT)).replace("\\", "/") for p in paths]
    artifacts += ["paper-round4/main.pdf", "paper-round4/references.bib",
                  "paper-round4/figures/gap-validation.png",
                  "research/round4/核心增量与模糊粗糙价值.md",
                  "research/round4/theorem-delta-audit.md",
                  "research/round4/manuscript-math-review.md",
                  "research/round4/fr-bridge-theory.md",
                  "scripts/check_round4_fr_examples.py",
                  "results/round4/fr-examples.json",
                  "results/round4/projected-mixture-audit.json",
                  "results/round4/fr-separate-joint-audit.json",
                  "results/summary.json", "results/overall-means.csv",
                  "results/statistics-knn-clean.csv",
                  "scripts/benchmark_inference_runtime.py",
                  "results/round4/inference-runtime.csv",
                  "results/round4/inference-runtime-summary.json"]
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "pdf_pages": len(pdf.pages), "references": len(keys),
        "undefined_labels": [], "missing_citations": [], "unused_references": [],
        "source_and_tests_match_retained_303_pass_run": True,
        "frozen_projected_audit_inputs_match": True,
        "projected_audit_summary": s,
        "fuzzy_rough_coupling_summary": cs,
        "repeated_prediction_summary": {
            "completed_runs": repeated["completed_runs"],
            "datasets": repeated["datasets"],
            "mean_removed_fraction": repeated["mean_removed_fraction"],
            "certified_vs_all_mean_delta": repeated["knn_clean_comparisons"][0]["mean_delta"],
            "certified_vs_all_holm_p": repeated["knn_clean_comparisons"][0]["p_holm"]},
        "inference_runtime_summary": runtime,
        "rational_examples": examples["status"],
        "artifact_sha256": {name: sha(name) for name in artifacts},
        "scope": "Integration check only; no rerun of experiments or test suite.",
        "limits": ["Novelty not established field-wide; see theorem-delta-audit.md.",
                   "Floating-point numerical audits are not formal certificates.",
                   "Bib checker has DOI recommendations for existing URL-based references.",
                   "BibTeX empty-pages warning remains for the CCCG2008 entry; no invented range."],
        "visual_review": "research/round4/artifact-qa.md"
    }
    (ROOT / "results/round4/artifact-integration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "pdf_pages", "references",
                                           "rational_examples"]}))


if __name__ == "__main__":
    main()
