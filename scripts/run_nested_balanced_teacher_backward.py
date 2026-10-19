"""Complete fresh balanced teacher/group/path/refit nested experiment."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import run_nested_triangular as original
from certified_frfs.balanced_teacher_backward_stage import fit_stage

original.runner.FOLDER = original.runner.ROOT/'results/nested-balanced-teacher-backward'
original.runner.fit_stage = fit_stage


def run(name):
    original.run(name)
    runner = original.runner
    path = runner.FOLDER/(name+'.json')
    saved = json.loads(path.read_text())
    files = [Path(__file__), runner.ROOT/'src/certified_frfs/balanced_teacher_backward_stage.py',
             runner.ROOT/'src/certified_frfs/joint_group_fuzzy.py',
             runner.ROOT/'src/certified_frfs/joint_certified_backward.py',
             runner.ROOT/'src/certified_frfs/constrained_fuzzy_logistic.py',
             runner.ROOT/'src/certified_frfs/certificate_repair.py',
             runner.ROOT/'research/nested-balanced-teacher-backward-protocol.md']
    saved['fingerprints'].update({str(f.relative_to(runner.ROOT)):
                                 hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    path.write_text(json.dumps(saved, indent=2), encoding='utf-8')


if __name__ == '__main__':
    runner = original.runner
    runner.FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((runner.ROOT/
             'results/nested-joint-backward/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    runner.summarize(names)
