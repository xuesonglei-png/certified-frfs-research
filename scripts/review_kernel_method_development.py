"""Consolidate audited experiments18--25, including every adverse dataset."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_nested_joint_path import ROOT,digest


def verified(folder):
    path = ROOT/'results'/folder
    summary = json.loads((path/'complete.json').read_text()); audit = json.loads((path/'complete-audit.json').read_text())
    assert audit['passed'] and not summary['goal_achieved']
    for row in summary['rows']:
        artifact = path/(row['dataset']+'.json'); saved = json.loads(artifact.read_text())
        for rel,sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
        checkpoint = path/(row['dataset']+'-audit.json')
        if checkpoint.exists():
            independent = json.loads(checkpoint.read_text())
            fp = independent['fingerprints']
            assert fp.get('artifact_sha256',fp.get(artifact.name)) == digest(artifact)
    return summary,audit


if __name__ == '__main__':
    folders = ('flexible-frs-screen','calibrated-forest-pilot','crossfit-kernel-screen','crossfit-kernel-prediction',
        'crossfit-rbf-refit-screen','crossfit-rbf-refit-prediction','exhaustive-refit-screen','refit-repartition-diagnosis')
    data = {f:verified(f) for f in folders}; forest = data[folders[1]][0]
    imputation,refit = data[folders[3]][0],data[folders[5]][0]
    paths = ROOT/'results/figures/method-development'; paths.mkdir(exist_ok=True)
    screens = [data[folders[j]][0] for j in (2,4)]; predictions = [imputation,refit]
    local,stability = data[folders[6]][0],data[folders[7]][0]
    per_dataset = []
    for name in sorted(row['dataset'] for row in imputation['rows']):
        row = {'dataset':name}
        for label,screen,prediction in zip(('imputation','refit'),screens,predictions):
            train = next(r for r in screen['rows'] if r['dataset'] == name)
            query = next(r for r in prediction['rows'] if r['dataset'] == name)
            row[label] = {'internal_selected_gain_pp':train['mean_internal_error_gain_pp'],
                'query_old_gain_pp':100*(query['methods']['new']['svm']['clean']-query['methods']['old']['svm']['clean']),
                'query_baseline_gains_pp':{m:100*(query['methods']['new']['svm']['clean']-query['methods'][m]['svm']['clean']) for m in ('full','fgs_matched','sfrip_matched')},
                'query_SVM_metrics':query['methods']}
        diagnosis = next(r for r in stability['rows'] if r['dataset'] == name)
        row['refit']['same_data_repartition_gain_pp'] = diagnosis['mean_gains_pp']['repartition']['refit']
        row['local'] = diagnosis['mean_gains_pp']
        per_dataset.append(row)
    result = {'goal_achieved':False,'scope':'all15_exposed_development_audited_failures_no_universal_prediction_claim',
        'source_hashes':{str((ROOT/'results'/f/'complete.json').relative_to(ROOT)):digest(ROOT/'results'/f/'complete.json') for f in folders},
        'summaries':{f:v[0] for f,v in data.items()},'audits':{f:v[1] for f,v in data.items()},'per_dataset':per_dataset,
        'interpretation':'Internal selected gains are optimistic; paired internal/query difference is descriptive, not causal evidence. Fixed native/matched methods and classifiers have different scopes.'}
    (ROOT/'results/kernel-method-development-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fig,axes = plt.subplots(1,2,figsize=(12,6),sharey=True,layout='constrained')
    position = np.arange(15)
    for ax,label in zip(axes,('imputation','refit')):
        ax.barh(position-.16,[r[label]['internal_selected_gain_pp'] for r in per_dataset],height=.30,color='#9cb9d1',label='Adaptive internal selected gain')
        ax.barh(position+.16,[r[label]['query_old_gain_pp'] for r in per_dataset],height=.30,color='#b44c3b',label='Isolated development query gain')
        if label == 'refit':
            ax.plot([r[label]['same_data_repartition_gain_pp'] for r in per_dataset],position,'o',color='#796619',markersize=4,label='Same-data repartition gain')
            ax.legend(loc='lower right',fontsize=8)
        ax.axvline(0,color='black',linewidth=.7); ax.set_yticks(position,[r['dataset'] for r in per_dataset])
        ax.set_title('Frozen teacher imputation' if label == 'imputation' else 'Actual refit + foldwise descent')
        ax.set_xlabel('Balanced accuracy gain over old support (pp)'); ax.grid(axis='x',alpha=.15)
    axes[0].legend(loc='lower right',fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('Training selection gains did not establish query superiority\nAll 15 exposed development datasets; same support count and common SVM',fontsize=11)
    for extension in ('png','svg'): fig.savefig(paths/('kernel-selection-transfer.'+extension),dpi=180)
    plt.close(fig)
    lines = ['# Kernel and forest method development: audited review','',
        '2026-10-10. Goal ACTIVE / UNACHIEVED. No TeX work. All15 are repeatedly exposed development data.','',
        '## Outcome','',
        'None of sparse calibrated forests, frozen-RBF imputation, or actual RBF refitting plus foldwise descent establishes prediction superiority over both recent equal-count controls. Tight certification alone does not satisfy the stated goal.','',
        '|Method|Common classifier|Clean BA %|Removal %|FGS matched gain pp|SFRIP matched gain pp|Primary adjusted p|Gate|',
        '|---|---|---:|---:|---:|---:|---|---|']
    entries = [('Calibrated forest','forest',forest['means'],forest['contrasts']['clean'],forest['matched_holm'],forest['development_numeric_gate'])]
    for name,p in [('Frozen RBF imputation',imputation),('Actual RBF refits, foldwise descent',refit)]:
        entries.append((name,'SVM',p['means']['svm'],p['contrasts']['svm']['clean'],p['primary_holm'],p['development_numeric_gate']))
    for name,family,means,contrasts,adjusted,gate in entries:
        lines.append(f"|{name}|{family}|{100*means['new']['clean']:.4f}|{100*means['new']['removed']:.4f}|{contrasts['fgs_matched']['mean_pp']:.4f}|{contrasts['sfrip_matched']['mean_pp']:.4f}|{adjusted}|{'PASS' if gate else 'FAIL'}|")
    lines += ['', '## What changed and what the experiments show','',
        'Enemy-specific top8 friend witnesses rescued only3 distinct candidates and3 references from177 previously rejected attempts; the resource gate failed. No witness relaxation entered prediction experiments.', '',
        'Permutation-calibrated synergy edges replace forced spanning trees with sparse subforests.35/128 supports change and92 references have zero edges. The maxT interpretation is global-null exchangeability only, not strong partial-null FWER or causal interaction. Predictive gate fails.', '',
        f"Frozen full RBF teachers optimize center-imputed-query error. The adaptive training screen gains{screens[0]['mean_internal_error_gain_pp']:.6f}pp, but common-SVM query gain over old is{imputation['contrasts']['svm']['clean']['old']['mean_pp']:.6f}pp. No error-transfer theorem was established.", '',
        f"Actual support-specific balanced RBF refits recompute gamma and require every internal fold to avoid error worsening. The screen uses{screens[1]['counts']['refits']}fits and changes44 supports. Adaptive training gain{screens[1]['mean_internal_error_gain_pp']:.6f}pp becomes{refit['contrasts']['svm']['clean']['old']['mean_pp']:.6f}pp on isolated development queries (raw p={refit['contrasts']['svm']['clean']['old']['sign_flip_two_sided_p']}). Common 5NN is secondary and cannot replace the failed SVM primary.", '',
        'The objective mismatch was a testable hypothesis; correcting it did not establish a large transferable gain. These data do not prove that this mismatch was the unique cause. Original FRS constraints and actual support counts remain identical. The search folds adaptively reuse labels and feasibility uses the whole reference; internal numbers are selection evidence, not honest complete-selector risk estimates.', '',
        '## Comparability and limits','',
        'FGS2023 and SFRIP2025 use fixed published algorithm variants/configurations in this pilot. Their original native stopping and same-count prefixes are separate arms. Native SFRIP removes about64% on average, whereas proposed removes10.63%; larger native accuracy contrasts cannot establish superior support quality. Fully comparable tuning remains required. Faithful FD+/FCE+ numerical operators/defaults remain unresolved; no proxy is relabeled.', '',
        'Public code discovery found no author repository in the checked queries and no valid supplement from the attempted IEEE routes. arXiv2505.15250 HTML provides mathematical content but not unique RF/S/N numerical defaults. These access/search results do not establish nonexistence.', '',
        'Experiments20--23 introduce no exact SVM adversarial-box or population-accuracy guarantee. Original FRS reference certificates do not imply refitted SVM/5NN margin preservation. Query noise is fixed uniform perturbation of query rows only.', '',
        'All selected-support and prediction artifacts passed separate audits. Experiment22 independently refits all47160 raw-RBF models; hinge discrepancy <=8.882e-16. Experiment23 refits1024 SVM and1024 5NN models and replays8192 predictions. Full meaningful suite662passed3warnings. Failed early wine audit invocation was missing an unfinished artifact, not a model failure; its log remains.', '',
        '## Removing search truncation and measuring partition sensitivity','',
        f"Exhaustive certified one-exchange descent removes top8 and two-round cutoffs. Its finite descending discrete objective proves termination and local stationarity only, a standard argument without global or population-risk claim. Every terminal neighborhood on all128 references is independently replayed. It adds{local['counts']['new_refits']}refits and changes12 supports relative to22; additional internal gain{local['mean_additional_internal_gain_pp']:.6f}pp fails the .5pp/20reference gate. No archived query predictions were computed for this local-search support.", '',
        f"A frozen repartition seed20261025 refits the old,22 and24 supports on the SAME original reference observations. Selection gain for22 {stability['dataset_equal_means_pp']['selection']['refit']:.6f}pp becomes{stability['dataset_equal_means_pp']['repartition']['refit']:.6f}pp; for24 {stability['dataset_equal_means_pp']['selection']['local']:.6f}pp becomes{stability['dataset_equal_means_pp']['repartition']['local']:.6f}pp. Of44 internally improved22 references,18 remain positive,15 turn negative and11 tie. This is partition sensitivity, not new independent confirmation, and no inferential p-values are calculated. Independent audit refits all1143 SVM models and manually replays RBF scores (maxerror2.976e-14).", '',
        '## Next research requirement','',
        'Candidate-search truncation has now been tested and its additional-opportunity gate failed. Larger local search increases selected internal gain without a comparable repartition gain. Do not query-score24, retune its gate, or expand query-driven C/radius/count grids. Any next predictive experiment needs a substantially different, justified method and complete training-only selection. Fully faithful recent baseline tuning and untouched confirmation remain required after any development win. All adverse dataset effects are retained in the JSON and scientific figure.']
    (ROOT/'research/kernel-method-development-review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'goal_achieved':False,'experiments_audited':list(folders),'primary_gates':[entry[-1] for entry in entries]},flush=True)
