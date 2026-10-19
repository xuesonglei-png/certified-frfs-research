"""Audited negative capacity experiment plus its refinement-penalty caveat."""
import json
import numpy as np
from scipy.special import logsumexp
from run_nested_joint_path import ROOT,digest,source_verified
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from spline_forest_audit_oracle import score


def main():
    folder = ROOT/'results/spline-capacity-pilot'; summary = json.loads((folder/'complete.json').read_text())
    audit = json.loads((folder/'complete-audit.json').read_text()); assert audit['passed']
    exact = json.loads((ROOT/'results/functional-refinement-exact.json').read_text()); assert exact['passed']
    rows = []; cases = []
    for row in summary['rows']:
        name = row['dataset']; saved = json.loads((folder/(name+'.json')).read_text())
        checkpoint = json.loads((folder/(name+'-audit.json')).read_text())
        assert checkpoint['fingerprints']['artifact_sha256'] == digest(folder/(name+'.json'))
        assert checkpoint['fingerprints']['auditor_sha256'] == digest(ROOT/'scripts/audit_spline_capacity_pilot.py')
        assert checkpoint['fingerprints']['oracle_sha256'] == digest(ROOT/'scripts/spline_forest_audit_oracle.py')
        source_verified(name); X,y = load_dataset(name); training = []
        for record in saved['records']:
            ri,a = np.array(record['outer_reference_indices']),np.array(record['reference_indices'])
            A,_ = preprocess_fit(X[ri[a]],X[ri[a]]); labels = y[ri[a]]; metrics = {}
            for method in ('full3','full5_fixed3'):
                fit = record['fits'][method]; classes = np.array(fit['classes']); q = len(classes)
                values = score(A,fit); own = np.argmax(labels[:,None] == classes[None],axis=1)
                balance = np.array([1/(q*sum(labels == c)) for c in labels])
                ce = float(balance@(logsumexp(values,axis=1)-values[np.arange(len(A)),own]))
                pred = classes[values.argmax(axis=1)]; accuracy = float(np.mean([np.mean(pred[labels == c] == c) for c in classes]))
                k,d,e = len(fit['knots'][0]),A.shape[1],len(fit['edges'])
                W,V = np.array(fit['coef']),np.array(fit['pair_coef']).reshape(q,e,k,k)
                if q == 2: W,V = W[1:2],V[1:2]
                else: W,V = W-W.mean(axis=0,keepdims=True),V-V.mean(axis=0,keepdims=True)
                nodal_penalty = float(np.sum(W**2)+fit['pair_penalty']*np.sum(V**2))
                metrics[method] = {'train_logloss':ce,'train_BA':accuracy,'penalized_objective':ce+nodal_penalty/(2*fit['C']*len(A)),
                    'coefficient_dimension':(1 if q == 2 else q)*(d*(k-1)+e*(k-1)**2),
                    'reference_samples':len(A),'query_BA':record['metrics'][method]['clean']}
            training.append(metrics)
            train_gain = metrics['full5_fixed3']['train_BA']-metrics['full3']['train_BA']
            query_gain = metrics['full5_fixed3']['query_BA']-metrics['full3']['query_BA']
            if abs(query_gain)+abs(train_gain) > 1e-12:
                cases.append({'dataset':name,'outer_index':record['outer_index'],'inner_index':record['inner_index'],
                    'training':metrics,'train_gain_pp':100*train_gain,'query_gain_pp':100*query_gain,
                    'train_improves_query_declines':train_gain > 1e-12 and query_gain < -1e-12})
        rows.append({'dataset':name,'training':{m:{key:float(np.mean([r[m][key] for r in training])) for key in training[0][m]}
            for m in ('full3','full5_fixed3')}})
    result = {'summary':summary,'audit':audit,'exact_refinement_penalty_counterexample':exact,'training_diagnostics':rows,
        'all_nonzero_capacity_cases':cases,'train_improves_query_declines':sum(r['train_improves_query_declines'] for r in cases),
        'scope':'exposed_development_negative_capacity_and_nodal_penalty_confound_not_causal_overfit_or_SOTA_proof','goal_achieved':False}
    (ROOT/'results/spline-capacity-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines = ['# Five-knot capacity pilot: audited negative result and regularization confound','',
        '2026-10-10. All15/all128 fixed-config experiments and independent audit complete. No TeX edits.',
        'Goal active/unachieved. No universal classification or recent-SOTA gain.','',
        '|Clean contrast|Gain pp|Descriptive p|', '|---|---:|---:|']
    for key,value in summary['ablations']['clean'].items(): lines.append(f"|{key}|{value['mean_pp']:.4f}|{value['sign_flip_two_sided_p']:.6f}|")
    for key,value in summary['method_contrasts']['clean'].items(): lines.append(f"|proposed5 versus {key}|{value['mean_pp']:.4f}|{value['sign_flip_two_sided_p']:.6f}|")
    lines += ['',f"Old/new proposed removal: {100*summary['means']['old3']['removed']:.4f}%/{100*summary['means']['proposed5']['removed']:.4f}%; all128 actual count matches,69 unchanged supports. Representation and method gates both=False.",
        'Native SFRIP gain is accompanied by different support counts; count-matched controls establish no advantage. Do not select another knot count, penalty or significant contrast from these query results.','',
        '## Independent checks','',
        f"{audit['counts']['independent_classifier_refits']} independent model refits, {audit['counts']['independent_lp_solves']} integral marginal LP solves, {audit['counts']['independent_leaf_replays']} all-query leaf-elimination replays, {audit['counts']['raw_witness_replays']} raw witness score checks, {audit['counts']['raw_projection_probes']} direct projection probes.",
        f"Independent interpolation/basis arithmetic causes small LBFGS stopping differences. Observed maxima: {audit['precision_maxima']}. Numerical refit tolerance1e-5 is supplemented by gradient<=2e-6 and objective gap<=1e-9; actual maximum objective gap is6.73e-14. Experiment outputs were not changed. The numerical optimizer is not a certified global-optimum oracle.",'',
        '## Why this did not isolate pure capacity','',
        'The frozen field name basis_only denotes fixed old edges with a3-to5 nodal-basis change. It also changes functional regularization: the SAME linear function gets25% more nodal unary penalty and the SAME pure pair function gets56.25% more penalty. These exact ratios do not depend on query outcomes. Hence neither pure representation benefit nor a causal overfitting explanation is established by this fixed-C comparison.',
        f"Training BA improves while query BA declines in{result['train_improves_query_declines']} references; all nonzero cases, losses and parameter/sample diagnostics are archived. This is observational evidence, not proof of the cause.",'',
        '## Implemented next correction','',
        'functional_spline_forest.py uses a raw-domain hat Gram, zero raw means and orthonormal functional contrasts. Coarse-to-fine transfer preserves scores, unary/pair penalties and exact box margins. Exact SymPy proof and quadrature/score-transfer tests pass; these are established functional-ANOVA/spline ideas pending a novelty comparison.',
        'functional_pipeline_cv.py and frozen functional-normalization-pilot-protocol.md rerun selectors, scaling and trees independently in each reference-internal training fold, with common3/5-knot and regularization grids for every method. The full15/all128 run has been started separately. Its outcome and independent audit are not implied by these mathematical tests.',
        'All15/compact8 remain exposed development; faithful FD+/FCE+ and an untouched confirmation cohort remain required.']
    (ROOT/'research/spline-capacity-review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'representation_gate':summary['representation_gate_passed'],'method_gate':summary['method_gate_passed'],
        'train_improves_query_declines':result['train_improves_query_declines'],'goal_achieved':False},flush=True)


if __name__ == '__main__': main()
