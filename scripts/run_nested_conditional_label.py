"""Fresh nested label reconstruction paths, original teacher certification."""
import copy
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import run_nested_balanced_joint_repair as cached
from certified_frfs.conditional_feasible_stage import refit_feasible_stage
from certified_frfs.conditional_label_backward import conditional_label_path
from certified_frfs.validated_pool import preprocess_fit

cached.SOURCE = cached.ROOT/'results/nested-conditional-feasible'
cached.original.runner.FOLDER = cached.ROOT/'results/nested-conditional-label'


def fit_stage(raw, y, *, C):
    source = copy.deepcopy(cached.CACHE[cached.identity(raw, y, C)])
    A, _ = preprocess_fit(raw, raw)
    source['conditional_path'] = conditional_label_path(A, y)
    stage = refit_feasible_stage(raw, y, source)
    stage['support_rule'] = 'conditional_label_feasibility_repair'
    return stage


cached.original.runner.fit_stage = fit_stage


def run(name):
    cached.run(name)
    path = cached.original.runner.FOLDER/(name+'.json')
    saved = json.loads(path.read_text())
    for f in (Path(__file__), cached.ROOT/'src/certified_frfs/conditional_feasible_stage.py',
              cached.ROOT/'src/certified_frfs/conditional_label_backward.py',
              cached.ROOT/'research/conditional-label-protocol.md'):
        saved['fingerprints'][str(f.relative_to(cached.ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    path.write_text(json.dumps(saved, indent=2), encoding='utf-8')


if __name__ == '__main__':
    cached.original.runner.FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((cached.SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    cached.original.runner.summarize(names)
