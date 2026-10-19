"""Preserved-control generator with independent contingency-table auditor."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

INDEPENDENT_FOREST = '''
def independent_penalized_forest(X, y):
    bins = np.where(X <= .25, 0, np.where(X <= .75, 1, 2))
    n, d = X.shape
    q = len(np.unique(y))
    def information(values):
        _, a = np.unique(values, return_inverse=True)
        _, b = np.unique(y, return_inverse=True)
        counts = np.zeros((a.max()+1, b.max()+1), dtype=int)
        np.add.at(counts, (a, b), 1)
        expected = counts.sum(axis=1)[:, None]*counts.sum(axis=0)[None]
        mask = counts > 0
        return float(np.sum(counts[mask]/n*np.log(n*counts[mask]/expected[mask])))
    single = [information(bins[:, a]) for a in range(d)]
    categories = [len(np.unique(bins[:, a])) for a in range(d)]
    candidates = []
    for u in range(d):
        for v in range(u+1, d):
            joint_bins = 3*bins[:, u]+bins[:, v]
            joint = information(joint_bins)
            gain = max(0., joint-max(single[u], single[v]))
            df = (len(np.unique(joint_bins))-max(categories[u], categories[v]))*(q-1)
            penalty = float(df*np.log(n)/(2*n))
            candidates.append({'edge': [u, v], 'joint_MI': joint, 'complementarity': gain,
                               'degrees_increment': int(df), 'bic_penalty': penalty,
                               'adjusted_gain': gain-penalty})
    candidates.sort(key=lambda r: (-r['adjusted_gain'], r['edge']))
    parent, chosen = list(range(d)), []
    def root(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for edge in candidates:
        if edge['adjusted_gain'] <= 0:
            break
        u, v = edge['edge']
        a, b = root(u), root(v)
        if a != b:
            parent[a] = b
            chosen.append(edge)
    return {'edges': [r['edge'] for r in chosen], 'selected': chosen, 'single_MI': single,
            'scope': 'reference_only_BIC_adjusted_maximum_weight_forest_not_FDR_causal_or_risk_guarantee'}

'''


def generated():
    source = (ROOT/'scripts/run_crossfit_label_loss_recent_pilot.py').read_text(encoding='utf-8')
    source = source.replace('crossfit-label-loss-recent-pilot', 'penalized-forest-recent-pilot')
    source = source.replace('from certified_frfs.fuzzy_interaction_tree import interaction_tree',
                            'from certified_frfs.penalized_fuzzy_forest import penalized_forest as interaction_tree')
    source = source.replace('from certified_frfs.crossfit_label_loss_stage import fit_crossfit_label_loss_stage',
                            'from certified_frfs.penalized_crossfit_stage import fit_penalized_crossfit_stage')
    source = source.replace('fit_crossfit_label_loss_stage(', 'fit_penalized_crossfit_stage(')
    source = source.replace("ROOT/'research/crossfit-label-loss-pilot-protocol.md',",
                            "ROOT/'research/crossfit-label-loss-pilot-protocol.md', ROOT/'research/penalized-forest-recent-pilot-protocol.md', ROOT/'scripts/prepare_penalized_forest_recent_pilot.py',")
    source = source.replace('original_FRS_crossfit_label_loss_and_tuned_recent', 'original_FRS_penalized_forest_and_tuned_recent')
    audit = (ROOT/'scripts/audit_crossfit_label_loss_recent_pilot.py').read_text(encoding='utf-8')
    audit = audit.replace('from run_crossfit_label_loss_recent_pilot import', 'from run_penalized_forest_recent_pilot import')
    audit = audit.replace('from certified_frfs.fuzzy_interaction_tree import interaction_tree\n', '')
    audit = audit.replace('TREE_CACHE = {}', INDEPENDENT_FOREST+'\nTREE_CACHE = {}')
    audit = audit.replace('interaction_tree(X, y)', 'independent_penalized_forest(X, y)')
    return {'run_penalized_forest_recent_pilot.py': source,
            'audit_penalized_forest_recent_pilot.py': audit}


if __name__ == '__main__':
    for name, source in generated().items():
        target = ROOT/'scripts'/name
        assert not target.exists(), 'preserve existing source'
        compile(source, str(target), 'exec')
        target.write_text(source, encoding='utf-8')
