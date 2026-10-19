"""Separate failed ranking, metric repair and classifier deployment mechanisms."""
import argparse
import json
from collections import Counter
import numpy as np
from review_box_iterations import ROOT


def main(source):
    folder = ROOT/'results'/source
    summary = json.loads((folder/'complete.json').read_text())
    counts = Counter()
    rows = []
    for row in summary['rows']:
        saved = json.loads((folder/(row['dataset']+'.json')).read_text())
        additions = []
        for split in saved['splits']:
            for scope, stage in [('outer', split['stage'])]+[
                ('inner', r['stage']) for inner in split['inner'] for r in inner['by_C'].values()]:
                if not stage['joint_fit']['feasible']:
                    counts[scope+':joint_failed'] += 1
                    continue
                seen = set()
                for key, record in stage['policies'].items():
                    if key == '1.0':
                        continue
                    candidate = record['failure']['requested_candidate'] if (
                        record['fallback'] and 'requested_candidate' in record['failure']) else record['candidate']
                    identity = tuple(candidate['features'])
                    if identity in seen:
                        continue
                    seen.add(identity)
                    trace = candidate['repair_trace']
                    restored = len(trace[-1]['features'])-len(trace[0]['features'])
                    additions.append(restored/stage['d'])
                    counts[scope+':supports'] += 1
                    counts[scope+':restored_features'] += restored
                    counts[scope+':classifier_failed'] += record['fallback']
                    for step in trace[:-1]:
                        cert = step['certificate']
                        counts[scope+':majority_failed_steps'] += not cert['majority_feasible']
                        counts[scope+':dependency_failed_steps'] += cert['dependency_certificate']['upper_bound'] > .1
        rows.append({'dataset': row['dataset'], 'mean_restored_fraction_across_unique_candidates':
                     float(np.mean(additions)) if additions else None})
    report = {'counts': dict(counts), 'rows': rows,
              'limitations': 'Unique actual supports deduplicated per stage; fractions repaired to the same cached full support can hide requested trajectories. Counts are diagnostic, not infeasibility proofs.'}
    (folder/'repair-diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report['counts'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='nested-balanced-joint')
    main(parser.parse_args().source)
