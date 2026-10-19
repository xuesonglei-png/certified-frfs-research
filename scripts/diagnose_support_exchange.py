from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import numpy as np
import diagnose_certified_query_refinement as runner
from certified_frfs.certified_support_exchange import refine_support_exchange

runner.FOLDER = runner.ROOT/'results/support-exchange'
runner.refine_certified_query = refine_support_exchange


def fingerprints(name):
    paths = [Path(__file__), Path(runner.__file__), runner.ROOT/'research/support-exchange-protocol.md',
             runner.ROOT/'results/long-capped-sparse'/f'{name}.json', runner.ROOT/'data/processed'/f'{name}.npz']
    paths.extend(sorted((runner.ROOT/'src/certified_frfs').glob('*.py')))
    return {str(p.relative_to(runner.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


runner.fingerprints = fingerprints


def main():
    prior = json.loads((runner.ROOT/'results/long-capped-sparse/complete.json').read_text())
    names = [r['dataset'] for r in prior['rows']]; runner.FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(runner.run, n) for n in names]):
            future.result()
    runner.summarize(names)
    path = runner.FOLDER/'complete.json'; result = json.loads(path.read_text())
    result['gate_passed'] = bool(result['gate_passed'] and result['mean_removed'] >= np.mean([r['removed'] for r in prior['rows']])-1e-12)
    result['scope'] = 'all15_all44_certified_support_exchange_exposed_development_not_confirmation'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
