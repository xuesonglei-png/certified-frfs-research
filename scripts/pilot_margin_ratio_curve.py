"""Three fixed positive-margin strengths on identical audited supports and C."""
import hashlib
import json
from functools import partial
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pilot_constrained_logistic as original
from certified_frfs.constrained_fuzzy_logistic import fit_constrained_logistic

RATIOS = (.5, .25, .1)


def run(name, ratio):
    original.FOLDER = original.ROOT/'results'/('margin-ratio-'+str(ratio))
    original.fit_constrained_logistic = partial(fit_constrained_logistic, ratio=ratio)
    original.run(name)
    path = original.FOLDER/(name+'.json')
    saved = json.loads(path.read_text())
    saved['margin_ratio'] = ratio
    files = [Path(__file__), original.ROOT/'research/margin-ratio-curve-protocol.md']
    saved['fingerprints'].update({str(p.relative_to(original.ROOT)):
                                 hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    path.write_text(json.dumps(saved, indent=2), encoding='utf-8')


if __name__ == '__main__':
    names = [r['dataset'] for r in json.loads((original.SOURCE/'complete.json').read_text())['rows']]
    reports = {}
    for ratio in RATIOS:
        original.FOLDER = original.ROOT/'results'/('margin-ratio-'+str(ratio))
        original.FOLDER.mkdir(exist_ok=True)
        with ProcessPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(run, name, ratio) for name in names]):
                future.result()
        original.summarize(names)
        reports[str(ratio)] = json.loads((original.FOLDER/'complete.json').read_text())
    (original.ROOT/'results/margin-ratio-curve.json').write_text(json.dumps({
        'ratios': reports,
        'scope': 'fixed_audited_supports_C_all15_all44_predeclared_margin_ratio_curve',
        'limitations': 'Not independently nested ratio selection. No outer-selected best ratio. '
                      'All strengths retain ALL positive reference anchors at their stated fraction; '
                      'neither population/test-risk nor untouched confirmation.'}, indent=2), encoding='utf-8')
