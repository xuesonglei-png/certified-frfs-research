"""Keep26--29 failed predictive gates and28 numerical success separate."""
import json
from run_nested_joint_path import ROOT,digest

FOLDERS = ('stable-support-pool-screen','bagged-forest-mechanism-screen','forest-robust-logloss-pilot','robust-logloss-classifier-screen')


if __name__ == '__main__':
    summaries,audits = {},{}
    for folder in FOLDERS:
        path = ROOT/'results'/folder
        saved = json.loads((path/'complete.json').read_text()); audit = json.loads((path/'complete-audit.json').read_text()); assert audit['passed'] and not saved['goal_achieved']
        for row in saved['rows']:
            artifact = path/(row['dataset']+'.json'); data = json.loads(artifact.read_text())
            for rel,sha in data['fingerprints'].items(): assert digest(ROOT/rel) == sha
            if 'artifact_hashes' in audit: assert audit['artifact_hashes'][row['dataset']] == digest(artifact)
            else:
                independent = json.loads((path/(row['dataset']+'-audit.json')).read_text())
                assert independent['fingerprints']['artifact_sha256'] == digest(artifact)
        summaries[folder],audits[folder] = saved,audit
    diagnosis = json.loads((ROOT/'results/robust-logloss-scaling-diagnosis.json').read_text()); assert diagnosis['complete'] and not diagnosis['validation_scores_used']
    for rel,sha in diagnosis['fingerprints'].items(): assert digest(ROOT/rel) == sha
    result = {'goal_achieved':False,'summaries':summaries,'audits':audits,'numerical_scaling_diagnosis':diagnosis,
        'scope':'allnegative_cases_preserved_numerical_training_success_not_prediction_advantage'}
    (ROOT/'results/stability-and-robust-risk-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    stable,bag,numerical,classification = (summaries[f] for f in FOLDERS)
    text = f'''# Stability and direct robust-risk development: audited review

2026-10-10. Goal ACTIVE / UNACHIEVED. No TeX work. Original15 remain exposed development data.

## Conclusion

No significant predictive advantage over recent baselines is established. Diverse two-partition support selection fails its third-partition mechanism gate; fixed-forest bagging worsens classification; direct label-targeted robust loss reduces training envelope objectives and validation envelope loss, but fails the primary clean classification gate. The numerical28 success is not a substitute for the user's prediction-superiority goal.

## Code and experiments

26 fixes a pool of old one-exchanges plus all9FGS2023/all6SFRIP2025 same-count ranking-prefix proposals. Two partitions select by worst improvement, and a third frozen partition never changes selection. All128 reference certificates preserve original FRS conditions. Minimum selection gain {stable['mean_minimum_selection_gain_pp']:.6f}pp becomes third-check gain {stable['mean_third_check_gain_pp']:.6f}pp;37 supports change. Resource gateFAIL. The partitions use the SAME observations; this is not independent confirmation. Exact raw-RBF audit validates all losses/provenance/ranking/admission. No new query scoring follows.

27 holds a single learned forest fixed within each training fold, trains12 stratified with-replacement logistic fits, and averages coefficients. Mean logits remain in exactly the original forest score family; averaging probabilities, votes or different graphs would be a different model. All preprocessing/graph learning/fitting exclude internal validation labels. All15/all128/381 internal folds: balanced accuracy gain {bag['BA_gain_pp']:.6f}pp; balanced logloss gain {bag['logloss_gain']:.6f}, both adverse. GateFAIL. Independent audit refits381 controls and4572 members and checks15 LP probes. No bootstrap-count/configuration search is added.

28 implements a different convex training objective: class-balanced logsumexp of negative true/rival exact forest-box minimum contrasts, plus the same ordinary L2 penalty. Binary loss is exact box-adversarial logistic loss. Multiclass is an upper envelope and can be strictly loose; equality requires a common raw minimizer for every true/rival contrast. These elementary convexity/exactness arguments are not verified novelty. Neither original teacher half-margin preservation nor a reduct's FRS certificate follows automatically. Solver labels and normalized objective differences are numerical diagnostics, not rigorous global-optimum certificates.

Three prespecified reference-only numerical cases reduce robust objectives by0.105570%,0.382735%,2.003017%. All pass exact separation; independent legacy DP checks253 pair margins,560 cuts,15 LPs(maxerror1.621e-8). This justifies testing classification, not claiming accuracy improvement.

29 freezes unchanged settings before isolated internal validation on ALL15 first references. Control is27's exact audited ordinary classifier; all methods use full attributes, so no support-quality/SOTA claim is possible. All44 fresh robust fits are independently replayed. Clean accuracy PRIMARY gain {classification['clean_BA_gain_pp']:.6f}pp; only3 positive datasets. Validation envelope-loss improvement {classification['validation_envelope_loss_gain']:.9f} does not replace the failed primary. Seven solver failures remain explicit ordinary-control fallbacks and independently reproduce. GateFAIL; no all128 or feature-selector expansion. Certified balanced correctness also slightly decreases. All88 validation score replays and30 LP probes pass.

## Numerical diagnosis, kept separate

Positive scaling by training sample count n preserves the convex objective's mathematical minimizers. A separate immutable-source diagnostic re-solves ONLY7 rejected training folds, without validation scoring, solver/tolerance changes or original-result replacement. It recovers usable numerical separation in {diagnosis['recovered']}/7 cases; {diagnosis['remaining_failed']} remain unusable. Recovered coefficients still require separate independent audit before scientific use. Their status recovery is NOT a new prediction result and does not reopen29's failed gate. Original source/results/fallbacks remain intact.

## Remaining requirement

No predictive hypothesis above earns expansion. Do not select favorable datasets, seeds, bootstrap counts or robust secondary metrics. A next mechanism must be justified before query evaluation. Investigate mathematically equivalent numerical epigraph representations using training data only; redundant own-class inequalities are a possible numerical issue, not an established cause. Fully faithfulFD+/FCE+, comparable recent tuning, feature-count controls and untouched confirmation still remain. A corrected optimizer alone cannot establish predictive superiority.
'''
    (ROOT/'research/stability-and-robust-risk-review.md').write_text(text,encoding='utf-8')
    print({'goal_achieved':False,'audited_experiments':list(FOLDERS),'scaling_recovered':diagnosis['recovered'],'scaling_remaining':diagnosis['remaining_failed']},flush=True)
