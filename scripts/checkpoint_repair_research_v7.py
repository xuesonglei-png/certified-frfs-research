"""Append-only research checkpoints; partial runs never masquerade as success."""
import json
import time
from pathlib import Path
from diagnose_repair_bottlenecks_v7 import ROOT, NAMES, digest


def main():
    studies = {}
    files = {}
    for name in ('graph-consensus-inner-v6', 'certificate-barrier-support-v1',
                 'anchor-subdivision-dependency-v1', 'anchor-subdivision-dependency-v2'):
        folder = ROOT/'results'/name
        status = {}
        for dataset in NAMES:
            p = folder/(dataset+'.json')
            if not p.exists():
                status[dataset] = {'recorded': False}
                continue
            data = json.loads(p.read_text())
            files[p.relative_to(ROOT).as_posix()] = digest(p)
            status[dataset] = {'recorded': True, 'complete': data.get('complete', False),
                               'panels': len(data.get('panels', [])),
                               'new_outer_scoring': data.get('new_outer_scoring')}
        summaries = []
        for pattern in ('*summary*.json', 'complete*.json', '*failure*.json'):
            for p in folder.glob(pattern):
                summaries.append(p.relative_to(ROOT).as_posix())
                files[p.relative_to(ROOT).as_posix()] = digest(p)
        studies[name] = {'datasets': status, 'reports': sorted(set(summaries))}
    reports = {}
    for rel in ('research/repair-iterations-v4-v5-review.json',
                'research/repair-bottlenecks-inner-v7.json',
                'results/nominal-dependency-headroom-v7/summary.json',
                'results/expanded-majority-admission-v1/summary-v1.json',
                'research/graph-consensus-inner-v6-reviewed-summary.json'):
        p = ROOT/rel
        if p.exists():
            data = json.loads(p.read_text()); files[rel] = digest(p)
            reports[rel] = {k: data[k] for k in (
                'complete', 'datasets', 'counts', 'all_method_audits_passed',
                'all_baseline_audits_passed', 'goal_achieved',
                'classification_superiority_established', 'gate_passed',
                'development_gate_passed') if k in data}
    for rel, sha in files.items():
        assert digest(ROOT/rel) == sha, 'Artifact changed during checkpoint: '+rel
    record = {'observed_unix': time.time(), 'goal_achieved': False,
              'classification_superiority_established': False,
              'latex_modified_or_compiled': False, 'studies': studies,
              'reports': reports, 'artifact_hashes': files,
              'next_action': 'Finish predeclared training screens and independent audits; '
                             'advance only a passing mechanism to a new nested development study.',
              'limits': ['Exposed datasets cannot establish confirmation after adaptive method iteration.',
                         'No nominal violation is not robust feasibility.',
                         'Partial results are not favorable-stopping permission.',
                         'This observer does not fit methods or audit mathematical primitives.']}
    out = ROOT/'research/repair-checkpoints-v7'; out.mkdir(exist_ok=True)
    path = out/(str(time.time_ns())+'.json')
    path.write_text(json.dumps(record, indent=2), encoding='utf8')
    print(path)
    print(json.dumps({'reports': reports, 'partial_studies': {
        k: sum(v.get('complete', False) for v in s['datasets'].values())
        for k,s in studies.items()}}, indent=2))


if __name__ == '__main__':
    main()
