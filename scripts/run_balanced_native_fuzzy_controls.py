"""Fresh balanced classifier selection on native and fixed-budget recent paths."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import run_common_fuzzy_classifier as original
from balanced_fuzzy_controls import train, final_metrics

original.FOLDER = original.ROOT/'results/balanced-native-fuzzy-controls'
original.train = train
original.final_metrics = final_metrics


def run(name):
    original.run(name)
    path = original.FOLDER/(name+'.json')
    saved = json.loads(path.read_text())
    for f in (Path(__file__), original.ROOT/'scripts/balanced_fuzzy_controls.py',
              original.ROOT/'research/balanced-native-fuzzy-controls-protocol.md'):
        saved['fingerprints'][str(f.relative_to(original.ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    saved['class_balanced'] = True
    path.write_text(json.dumps(saved, indent=2), encoding='utf-8')


if __name__ == '__main__':
    original.FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((original.ROOT/
        'results/nested-joint-backward/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    original.summarize(names)
    path = original.FOLDER/'complete.json'
    saved = json.loads(path.read_text())
    saved['scope'] = 'fresh_inner_balanced_classifier_native_and_budget_recent_paths_exposed_development'
    saved['limitations'] = ('Historical proposed and matched_full rows inherit old recipe and must not be compared to new methods. '
        'Use full, fgs2023_native and sfrip2025_native as new balanced controls. '
        'Path definitions/conventions inherit documented baseline fidelity limits.')
    path.write_text(json.dumps(saved, indent=2), encoding='utf-8')
