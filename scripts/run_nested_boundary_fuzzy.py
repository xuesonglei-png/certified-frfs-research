"""Fresh nested boundary-score supports with unchanged dual screening."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import run_nested_fuzzy_logistic as original
import certified_frfs.nested_fuzzy_logistic_stage as stage_module
from certified_frfs.boundary_fuzzy_support import boundary_score_supports

stage_module.fuzzy_score_supports = boundary_score_supports
original.original.runner.FOLDER = original.original.runner.ROOT / 'results/nested-boundary-fuzzy'


def run(name):
    original.run(name)
    runner = original.original.runner
    path = runner.FOLDER / (name+'.json')
    saved = json.loads(path.read_text())
    files = [Path(__file__), runner.ROOT/'src/certified_frfs/boundary_fuzzy_support.py',
             runner.ROOT/'research/nested-boundary-fuzzy-protocol.md']
    saved['fingerprints'].update({str(p.relative_to(runner.ROOT)):
                                 hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    path.write_text(json.dumps(saved, indent=2), encoding='utf-8')


if __name__ == '__main__':
    runner = original.original.runner
    runner.FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((runner.ROOT/
             'results/nested-fuzzy-logistic/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    runner.summarize(names)
