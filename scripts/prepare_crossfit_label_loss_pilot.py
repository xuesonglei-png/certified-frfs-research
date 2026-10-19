"""Generate a separate OOF-ranking pilot without changing preserved sources."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def generated():
    source = (ROOT/'scripts/run_projected_label_loss_recent_pilot.py').read_text(encoding='utf-8')
    source = source.replace('projected-label-loss-recent-pilot', 'crossfit-label-loss-recent-pilot')
    source = source.replace('certified_frfs.label_loss_certified_stage import fit_label_loss_certified_stage',
                            'certified_frfs.crossfit_label_loss_stage import fit_crossfit_label_loss_stage')
    source = source.replace('fit_label_loss_certified_stage(', 'fit_crossfit_label_loss_stage(')
    source = source.replace("ROOT/'research/projected-label-loss-next-protocol.md',",
                            "ROOT/'research/projected-label-loss-next-protocol.md', ROOT/'research/crossfit-label-loss-pilot-protocol.md', ROOT/'scripts/prepare_crossfit_label_loss_pilot.py',")
    source = source.replace('original_FRS_label_loss_and_tuned_recent', 'original_FRS_crossfit_label_loss_and_tuned_recent')
    audit = (ROOT/'scripts/audit_projected_label_loss_recent_pilot.py').read_text(encoding='utf-8')
    audit = audit.replace('from run_projected_label_loss_recent_pilot import', 'from run_crossfit_label_loss_recent_pilot import')
    start = audit.index('    alpha = np.array(', audit.index('def check_stage('))
    end = audit.index('    path, active,', start)
    replacement = '''    fold_values, centers = [], []
    for (a, b), block in zip(splits, selection['folds']):
        F, Q = preprocess_fit(raw[a], raw[b])
        alpha = np.array([1/(len(np.unique(y[a]))*sum(y[a] == c)) for c in y[a]])
        center = alpha@F
        centers.append(center.tolist())
        fold_values.append((Q, y[b], block['models'][chosen], center))
    np.testing.assert_allclose(centers, stage['path']['fold_centers'], atol=1e-12)
    loss_cache = {}
    def crossfit_loss(fs):
        key = tuple(fs)
        if key not in loss_cache:
            loss_cache[key] = float(np.mean([direct_projection(Q, truth, model, center, fs)
                                             for Q, truth, model, center in fold_values]))
        return loss_cache[key]
    np.testing.assert_allclose(crossfit_loss(list(range(d))), options[chosen]['mean_balanced_logloss'], atol=1e-12)
'''
    audit = audit[:start]+replacement+audit[end:]
    audit = audit.replace('direct_projection(A, y, teacher, center, fs)', 'crossfit_loss(fs)')
    audit = audit.replace('direct_projection(A, y, teacher, center, [a for a in active if a != r])', 'crossfit_loss([a for a in active if a != r])')
    audit = audit.replace('projected_balanced_logloss', 'crossfit_balanced_logloss')
    return {'run_crossfit_label_loss_recent_pilot.py': source,
            'audit_crossfit_label_loss_recent_pilot.py': audit}


if __name__ == '__main__':
    for name, source in generated().items():
        target = ROOT/'scripts'/name
        assert not target.exists(), 'preserve existing source'
        compile(source, str(target), 'exec')
        target.write_text(source, encoding='utf-8')
