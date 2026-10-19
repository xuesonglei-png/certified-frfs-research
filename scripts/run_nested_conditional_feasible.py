"""Fresh inner selection after teacher-only certified-path feasibility repair."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import run_nested_balanced_joint_repair as cached
from certified_frfs.conditional_feasible_stage import refit_feasible_stage

cached.SOURCE = cached.ROOT/'results/nested-conditional-backward'
cached.original.runner.FOLDER = cached.ROOT/'results/nested-conditional-feasible'


def fit_stage(raw, y, *, C):
    return refit_feasible_stage(raw, y, cached.CACHE[cached.identity(raw, y, C)])


cached.original.runner.fit_stage = fit_stage


def run(name):
    cached.run(name)
    path = cached.original.runner.FOLDER/(name+'.json')
    saved = json.loads(path.read_text())
    for f in (Path(__file__), cached.ROOT/'src/certified_frfs/conditional_feasible_stage.py',
              cached.ROOT/'src/certified_frfs/conditional_certified_backward.py',
              cached.ROOT/'research/conditional-feasible-protocol.md'):
        saved['fingerprints'][str(f.relative_to(cached.ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    path.write_text(json.dumps(saved, indent=2), encoding='utf-8')


if __name__ == '__main__':
    cached.original.runner.FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((cached.SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    cached.original.runner.summarize(names)
