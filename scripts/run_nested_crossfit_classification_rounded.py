"""Fresh reference-internal classification paths in every nested stage."""
import copy
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import run_nested_balanced_joint_repair as cached
from certified_frfs.conditional_feasible_stage import refit_feasible_stage
from certified_frfs.crossfit_classification_rounded import crossfit_classification_path

cached.SOURCE = cached.ROOT/'results/nested-conditional-feasible'
cached.original.runner.FOLDER = cached.ROOT/'results/nested-crossfit-classification-rounded'


def fit_stage(raw, y, *, C):
    source = copy.deepcopy(cached.CACHE[cached.identity(raw, y, C)])
    source['conditional_path'] = crossfit_classification_path(raw, y)
    stage = refit_feasible_stage(raw, y, source)
    stage['support_rule'] = 'crossfit_classification_rounded_feasibility_repair'
    return stage


cached.original.runner.fit_stage = fit_stage


def run(name):
    cached.run(name)
    path = cached.original.runner.FOLDER/(name+'.json')
    saved = json.loads(path.read_text())
    for f in (Path(__file__), cached.ROOT/'src/certified_frfs/conditional_feasible_stage.py',
              cached.ROOT/'src/certified_frfs/crossfit_classification_rounded.py',
              cached.ROOT/'src/certified_frfs/crossfit_label_backward.py',
              cached.ROOT/'research/crossfit-classification-rounded-protocol.md'):
        saved['fingerprints'][str(f.relative_to(cached.ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    path.write_text(json.dumps(saved, indent=2), encoding='utf-8')


if __name__ == '__main__':
    cached.original.runner.FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((cached.SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    cached.original.runner.summarize(names)
