"""Check all 15 reference populations/folds/anchors before prospective design."""
import json
import time
from run_simplex_soft5_knn_pilot import ROOT, digest
from continue_soft_topk_knn_pilot import verify
from pair_transition_inner_design import alignment, STRONG


def main():
    atomic = ROOT/'results/atomic-pruning-classifier-v2'
    canonical = ROOT/'results/majority-hinge-canonical-benchmark'
    audit = verify(atomic); rows = []
    for name in sorted(audit['artifact_hashes']):
        a, c = atomic/(name+'.json'), canonical/(name+'.json')
        source = json.loads(a.read_text()); control = json.loads(c.read_text())
        folds = alignment(source, control)
        rows.append({'dataset':name, 'inner_folds':folds,
            'atomic_sha256':digest(a), 'canonical_sha256':digest(c),
            'native_candidate_counts':{m:len(control['inner'][0]['options'][m]) for m in STRONG}})
        print(name, 'all reference/split/count/anchor identities agree', flush=True)
    assert len(rows) == 15 and sum(r['inner_folds'] for r in rows) == 44
    report = {'passed':True, 'created_unix':time.time(), 'rows':rows,
        'datasets':15, 'inner_folds':44, 'goal_achieved':False,
        'canonical_full_independent_audit_still_required':True,
        'scope':'index_and_anchor_alignment_only_no_scores_or_predictions_examined'}
    out = ROOT/'research/pair-transition-native-control-alignment.json'
    assert not out.exists(), 'Preserve previous alignment evidence'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
