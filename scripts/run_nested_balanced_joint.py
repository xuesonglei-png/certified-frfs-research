"""Fresh nested balanced grouped fitting, with ordinary teacher anchors."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import run_nested_triangular as original
from certified_frfs.nested_joint_group_stage import fit_stage as grouped_stage


def fit_stage(raw, y, *, C):
    return grouped_stage(raw, y, C=C, class_balanced=True)


original.runner.FOLDER = original.runner.ROOT/'results/nested-balanced-joint'
original.runner.fit_stage = fit_stage


def run(name):
    original.run(name)
    runner = original.runner
    path = runner.FOLDER/(name+'.json')
    saved = json.loads(path.read_text())
    files = [Path(__file__), runner.ROOT/'src/certified_frfs/nested_joint_group_stage.py',
             runner.ROOT/'src/certified_frfs/joint_group_fuzzy.py',
             runner.ROOT/'src/certified_frfs/constrained_fuzzy_logistic.py',
             runner.ROOT/'src/certified_frfs/certificate_repair.py',
             runner.ROOT/'research/nested-balanced-joint-protocol.md']
    saved['fingerprints'].update({str(p.relative_to(runner.ROOT)):
                                 hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    path.write_text(json.dumps(saved, indent=2), encoding='utf-8')


if __name__ == '__main__':
    runner = original.runner
    runner.FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((runner.ROOT/
             'results/joint-group-fuzzy-pilot/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    runner.summarize(names)
