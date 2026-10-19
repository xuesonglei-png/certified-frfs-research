"""Shared balanced teacher anchors and C after fresh balanced support choice."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import pilot_balanced_robust_controls as original

original.SOURCE = original.ROOT/'results/nested-balanced-teacher-backward'
original.SUPPORTS = original.ROOT/'results/nested-balanced-teacher-backward-common-budget'
original.FOLDER = original.ROOT/'results/balanced-teacher-robust-controls'


if __name__ == '__main__':
    original.FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((original.SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(original.run, name) for name in names]):
            future.result()
    original.summarize(names)
    path = original.FOLDER/'complete.json'
    summary = json.loads(path.read_text())
    summary['scope'] = 'all15_all44_shared_BALANCEDteacher_C_halfmargin_fixed_support_mechanism_controls'
    summary['limitations'] = ('Supports freshly selected under balanced loss; shared teacher now balanced, not ordinary. '
        'Baseline settings selected via balanced unconstrained budget fits; fixed-C constrained refits are mechanism controls, not native pipelines. '
        'Initial counts match; failed fits deploy full with actual counts reported. No independent confirmation.')
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
