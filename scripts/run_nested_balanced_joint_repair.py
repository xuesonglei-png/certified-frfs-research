"""Reuse identical reference fits; freshly repair, refit, score and select."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import run_nested_triangular as original
from certified_frfs.datasets import load_dataset
from certified_frfs.balanced_joint_repair_stage import refit_aligned_stage

ROOT = original.runner.ROOT
SOURCE = ROOT/'results/nested-balanced-joint'
original.runner.FOLDER = ROOT/'results/nested-balanced-joint-aligned-repair'
CACHE = {}


def identity(raw, y, C):
    return (hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
            hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(), float(C))


def fit_stage(raw, y, *, C):
    return refit_aligned_stage(raw, y, CACHE[identity(raw, y, C)])


original.runner.fit_stage = fit_stage


def run(name):
    CACHE.clear()
    source = SOURCE/(name+'.json')
    saved = json.loads(source.read_text())
    X, y = load_dataset(name)
    for split in saved['splits']:
        ri = np.array(split['reference_indices'])
        raw, labels = X[ri], y[ri]
        for inner in split['inner']:
            a = np.array(inner['reference_indices'])
            for C, record in inner['by_C'].items():
                CACHE[identity(raw[a], labels[a], float(C))] = record['stage']
        for C, stage in split['outer_stages'].items():
            CACHE[identity(raw, labels, float(C))] = stage
    original.run(name)
    path = original.runner.FOLDER/(name+'.json')
    result = json.loads(path.read_text())
    files = [Path(__file__), source, ROOT/'src/certified_frfs/balanced_joint_repair_stage.py',
             ROOT/'src/certified_frfs/certificate_repair.py',
             ROOT/'src/certified_frfs/constrained_fuzzy_logistic.py',
             ROOT/'research/nested-balanced-joint-aligned-repair-protocol.md']
    result['fingerprints'].update({str(f.relative_to(ROOT)):
                                  hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    CACHE.clear()


if __name__ == '__main__':
    original.runner.FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    original.runner.summarize(names)
