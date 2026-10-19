"""Preserve the frozen runner and generate a resumable equivalent replay.

The original runner accumulated parent hashes during execution, so its
completed-cache check incorrectly compared a partial fingerprint dictionary.
V2 fingerprints all parents before opening a checkpoint. Models unchanged.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    text = (ROOT/'scripts/diagnose_forest_sample_depth.py').read_text(encoding='utf-8')
    text = text.replace("FOLDER = ROOT/'results/forest-sample-depth-diagnosis'",
                        "FOLDER = ROOT/'results/forest-sample-depth-reproduction'")
    old = "    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}"
    new = "    paths += [ROOT/'results/predictive-iteration41'/(name+'-s101-f'+str(fold)+'.json') for fold in range(3)]\n"+old
    assert text.count(old) == 1
    text = text.replace(old, new)
    destination = ROOT/'scripts/diagnose_forest_sample_depth_reproduction.py'
    assert not destination.exists(), 'preserve any prior generated source'
    destination.write_text(text, encoding='utf-8')
