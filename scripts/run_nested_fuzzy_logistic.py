"""Fresh inner support selection under a shared logistic classifier."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import run_nested_triangular as original
from certified_frfs.nested_fuzzy_logistic_stage import fit_stage

original.runner.FOLDER = original.runner.ROOT / 'results/nested-fuzzy-logistic'
original.runner.fit_stage = fit_stage


def run(name):
    original.run(name)
    path = original.runner.FOLDER / (name + '.json')
    saved = json.loads(path.read_text())
    files = [Path(__file__), original.runner.ROOT / 'src/certified_frfs/nested_fuzzy_logistic_stage.py',
             original.runner.ROOT / 'src/certified_frfs/certificate_repair.py',
             original.runner.ROOT / 'research/nested-fuzzy-logistic-protocol.md']
    saved['fingerprints'].update({str(p.relative_to(original.runner.ROOT)):
                                 hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    path.write_text(json.dumps(saved, indent=2), encoding='utf-8')


if __name__ == '__main__':
    original.runner.FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((original.runner.ROOT /
             'results/nested-triangular-repair/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    original.runner.summarize(names)
