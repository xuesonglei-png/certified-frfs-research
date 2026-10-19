"""Fixed teacher-distilled LP iteration; historical artifacts are immutable."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import diagnose_robust_linear_refit as search
from certified_frfs.distilled_box_linear_refit import fit_distilled_box_linear

search.FOLDER = search.ROOT / 'results/distilled-box-linear-refit'
search.fit_robust_linear = fit_distilled_box_linear

def run(name):
    search.run(name)
    path = search.FOLDER / f'{name}.json'
    p = json.loads(path.read_text())
    additions = [Path(__file__), search.ROOT / 'src/certified_frfs/distilled_box_linear_refit.py', search.ROOT / 'research/distilled-box-protocol.md']
    p['fingerprints'].update({str(f.relative_to(search.ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in additions})
    path.write_text(json.dumps(p, indent=2), encoding='utf-8')

if __name__ == '__main__':
    names = [r['dataset'] for r in json.loads((search.ROOT / 'results/linear-retention/complete.json').read_text())['rows']]
    search.FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run, n) for n in names]):
            f.result()
    search.summarize(names)
