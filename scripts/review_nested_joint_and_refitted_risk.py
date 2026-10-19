"""Review path selection and actual refit criterion without hiding failures."""
import json
from collections import Counter
from review_box_iterations import ROOT


def main():
    joint_dir = ROOT/'results/nested-joint-path'; risk_dir = ROOT/'results/refitted-risk-geometry-pilot'
    joint = json.loads((joint_dir/'complete.json').read_text()); risk = json.loads((risk_dir/'complete.json').read_text())
    cost_dir = ROOT/'results/dependency-only-guard-cost'; cost = json.loads((cost_dir/'complete.json').read_text())
    for folder in (joint_dir,risk_dir,cost_dir): assert json.loads((folder/'complete-audit.json').read_text())['passed']
    original_dir = ROOT/'results/nested-geometry-forest'; counts = Counter(); joint_selections = []; cases = []
    for row in joint['rows']:
        name = row['dataset']; saved = json.loads((joint_dir/(name+'.json')).read_text()); old = json.loads((original_dir/(name+'.json')).read_text())
        for i,(s,previous) in enumerate(zip(saved['splits'],old['splits'])):
            r = s['results']['proposed']; original = s['results']['original']; chosen = s['trace'][r['chosen_index']]
            counts['same_outer_features_as_original'] += r['features'] == original['features']
            counts['same_outer_fit_as_original'] += s['models'][r['model_id']] == s['models'][original['model_id']]
            original_clean = max(v['mean_clean'] for v in s['trace'][:48]); joint_clean = max(v['mean_clean'] for v in s['trace'][48:])
            counts['joint_strictly_better_inner'] += joint_clean > original_clean+1e-12
            counts['joint_inner_tied'] += abs(joint_clean-original_clean) <= 1e-12
            if chosen['family'] == 'joint':
                joint_selections.append({'dataset':name,'outer_index':i,'model_index':chosen['model_index'],'fraction':chosen['fraction'],
                    'features':r['features'],'original_features':original['features'],
                    'inner_improvement_pp':100*(chosen['mean_clean']-original_clean),
                    'selected_outer_pair_jumps':s['stages']['proposed']['path']['accepted_pair_jumps']})
        actual = json.loads((risk_dir/(name+'.json')).read_text())
        for record in actual['records']:
            stage = record['stage']; scores = stage['risk']['scores']; trace = stage['path']['trace']
            initial = scores[trace[0]['risk_score_id']]['mean_balanced_logloss']; final = scores[trace[-1]['risk_score_id']]['mean_balanced_logloss']
            queries = record['metrics']; gain = queries['refitted']['clean']-queries['old']['clean']
            counts['refit_cv_improves_vs_full'] += final < initial-1e-12
            counts['refit_cv_improves_but_query_loses_to_old'] += final < initial-1e-12 and gain < -1e-12
            counts['refit_cv_improves_but_query_loses_to_full'] += final < initial-1e-12 and queries['refitted']['clean'] < queries['full']['clean']-1e-12
            counts['refit_query_wins_vs_old'] += gain > 1e-12; counts['refit_query_losses_vs_old'] += gain < -1e-12
            if abs(gain) > 1e-12:
                cases.append({'dataset':name,'outer_index':record['outer_index'],'inner_index':record['inner_index'],
                    'old_features':record['fits']['old']['features'],'new_features':record['fits']['refitted']['features'],
                    'old_count':len(record['fits']['old']['features']),'new_count':len(record['fits']['refitted']['features']),
                    'gain_pp':100*gain,'cv_logloss_full':initial,'cv_logloss_selected':final,
                    'scored_supports':len(scores),'fold_models':len(stage['risk']['models'])})
    result = {'joint':{k:v for k,v in joint.items() if k != 'rows'},'refitted_risk':{k:v for k,v in risk.items() if k != 'rows'},
        'dependency_only_guard_cost':{k:v for k,v in cost.items() if k != 'rows'},
        'diagnostics':dict(counts),'selected_joint_cases':joint_selections,'refitted_nonzero_query_cases':cases,
        'goal_achieved':False,'scope':'all_datasets_audited_exposed_development_failed_or_positive_gates_do_not_equal_final_superiority'}
    (ROOT/'results/nested-joint-refitted-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    def comparisons(values):
        lines = ['|Comparator|Gain pp|Descriptive95% interval pp|Sign-flip p|W/L/T|','|---|---:|---|---:|---|']
        for name,v in values.items():
            lines.append(f"|{name}|{v['mean_pp']:.4f}|[{v['bootstrap95_pp'][0]:.4f}, {v['bootstrap95_pp'][1]:.4f}]|{v['sign_flip_two_sided_p']:.6f}|{v['wins']}/{v['losses']}/{v['ties']}|")
        return lines
    lines = ['# Nested joint search and actual refitted-risk ranking', '',
        '2026-10-10. All three new experiments and independent audits are complete. No TeX changes or compilation. Goal remains active and unachieved; these are exposed development experiments.', '',
        '## Training-selected original versus joint path', '',
        f"All15/44outer/128inner partitions,96 choices; {joint['counts']['selected_joint']} outer selected joint, {joint['counts']['inner_pair_jumps']} accepted internal pair jumps. Primary gate={joint['gate_passed']}.", '',
        f"Mean proposed BA={100*joint['means']['proposed']['clean']:.4f}%, actual removal={100*joint['means']['proposed']['removed']:.4f}%.", '']
    lines += comparisons(joint['contrasts']['clean'])
    lines += ['', 'The exact structural counterexample remains valid, but more accessible certified supports did not establish a prediction improvement. All old choices remain available; train-internal selection is not assumed to improve held-out accuracy. Source full/modern controls share the original independently audited search and exact partition identities.', '',
        'Joint selections: '+str(joint_selections), '', '## Actual per-candidate classifier-refit pilot', '',
        f"Fixed C1/penalty1/fraction.8, all15/128reference-query partitions. Risk models={risk['counts']['risk_models']}; optimizer caps={risk['counts']['risk_model_caps']}; count mismatches={risk['counts']['matched_count_mismatches']}; gate={risk['gate_passed']}.", '',
        f"Old/new actual removal={100*risk['means']['old']['removed']:.4f}%/{100*risk['means']['refitted']['removed']:.4f}%.", '']
    lines += comparisons(risk['contrasts']['clean'])
    lines += ['', 'This evaluates actual refitted candidate classifiers rather than projected fixed-teacher scores. Every fold learns scaling and its support-specific tree locally. However, FRS screening uses the whole search reference and candidate scores are adaptively reused during greedy search; these are training selection utilities, not independent or unbiased risk estimates. The archived reference/query labels do not enter selection.', '',
        '## Dependency-only auxiliary-majority cost', '',
        f"All128 exact count matches, {cost['counts']['deployed_majority_violations']} deployed subsets violate the original majority condition, {cost['counts']['unchanged_supports']} supports remain identical. Original dependency upper certificate<=.1 remains verified on all128. Gate={cost['gate_passed']}.", '']
    lines += comparisons(cost['contrasts']['clean'])
    lines += ['', 'Original certificate dictionaries keep their combined/majority meanings. A separate dependency-only admissibility flag governs this diagnostic; original majority-retention and classifier guarantees are not claimed. Relaxing that condition did not establish improved classification, so there is no evidence to discard the majority guarantee for a predictive gain.', '',
        '## Diagnostics', '', str(dict(counts)), '',
        'All nonzero query cases are archived in nested-joint-refitted-review.json, including both wins and losses and actual support sizes. Internal validation improvement can coexist with query deterioration; no query case may be relabeled as a confirmatory result.', '',
        '## Next decision', '',
        'Use the frozen gate and independent audit to decide whether actual-refit ranking merits a complete nested recent-baseline benchmark. A fixed-config development gain, even if positive, cannot demonstrate superiority over native or same-budget modern controls. If the gate fails, retain this run as an unsuccessful criterion test rather than silently choosing another fraction or penalty from its query results. Faithful FD+/FCE+ reproduction and an untouched confirmation cohort remain necessary.']
    (ROOT/'research/nested-joint-refitted-review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    from plot_certified_method_development import plot
    plot(joint,risk,cost)
    print({k:v for k,v in result.items() if k not in ('joint','refitted_nonzero_query_cases')},flush=True)


if __name__ == '__main__': main()
