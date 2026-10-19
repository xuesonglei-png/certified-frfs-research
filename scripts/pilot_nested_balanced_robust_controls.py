"""Shared original margins/classifier controls after fresh balanced selection."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import pilot_balanced_robust_controls as original

original.SOURCE = original.ROOT/'results/nested-balanced-joint'
original.SUPPORTS = original.ROOT/'results/nested-balanced-joint-common-budget'
original.FOLDER = original.ROOT/'results/nested-balanced-joint-robust-controls'


if __name__ == '__main__':
    original.FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((original.SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(original.run, name) for name in names]):
            future.result()
    original.summarize(names)
    path = original.FOLDER/'complete.json'
    summary = json.loads(path.read_text())
    summary['limitations'] = ('Fresh nested balanced proposed selection; baseline budgets and classifier C selected by balanced unconstrained support ablation. '
        'This fixed-C constrained refit isolates support quality, not a complete native baseline selector or independent confirmation. '
        'Initial counts match; solver failure deploys full; no certificate claimed for baseline FRS metric. '
        'FGS and SFRIP are budget adaptations with an explicit shared-margin wrapper.')
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
