"""Generate nested repaired paths plus an independently replayed auditor."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def generated():
    source = (ROOT/'scripts/run_crossfit_label_loss_recent_pilot.py').read_text(encoding='utf-8')
    source = source.replace('crossfit-label-loss-recent-pilot', 'repaired-crossfit-recent-pilot')
    source = source.replace('from certified_frfs.crossfit_label_loss_stage import fit_crossfit_label_loss_stage',
                            'from certified_frfs.repaired_crossfit_stage import fit_repaired_crossfit_stage')
    source = source.replace('fit_crossfit_label_loss_stage(', 'fit_repaired_crossfit_stage(')
    source = source.replace("ROOT/'research/crossfit-label-loss-pilot-protocol.md',",
                            "ROOT/'research/crossfit-label-loss-pilot-protocol.md', ROOT/'research/repaired-crossfit-recent-pilot-protocol.md', ROOT/'scripts/prepare_repaired_crossfit_recent_pilot.py',")
    source = source.replace('original_FRS_crossfit_label_loss_and_tuned_recent', 'original_FRS_repaired_crossfit_and_tuned_recent')
    marker = "            counts['teacher_optimizer_caps'] += block['stage']['teacher_selection']['optimizer_caps']"
    repair_counts = '''            for repair in block['stage']['path']['repair_blocks']:
                counts['metric_repair_attempts'] += len(repair['attempts'])
                counts['metric_solver_limits'] += sum(r['fit']['solver_status'] == 1 for r in repair['attempts'])
                counts['accepted_metric_repairs'] += sum(r['fit']['status'] == 'original_certificate_passed' for r in repair['attempts'])
                counts['rejected_metric_repairs'] += sum(r['fit']['status'] != 'original_certificate_passed' for r in repair['attempts'])
                counts['failed_metric_repair_steps'] += repair['attempts'][-1]['fit']['status'] != 'original_certificate_passed'
'''
    assert source.count(marker) == 1
    source = source.replace(marker, repair_counts+marker)
    audit = (ROOT/'scripts/audit_crossfit_label_loss_recent_pilot.py').read_text(encoding='utf-8')
    audit = audit.replace('from run_crossfit_label_loss_recent_pilot import', 'from run_repaired_crossfit_recent_pilot import')
    audit = audit.replace('from review_box_iterations import paired', 'from review_box_iterations import paired\nfrom audit_flexible_majority_repair_admission import check_repair')
    audit = audit.replace('    cache = {}\n    def candidate(fs):\n        key = tuple(fs)',
                          '    cache = {}\n    repair_offset = 0\n    def candidate(fs, override=None):\n        key = (tuple(fs), None if override is None else tuple(override))')
    old = '            embedded = np.zeros(d)\n            embedded[fs] = weights'
    new = '''            if override is not None:
                weights = np.array(override)
                assert weights.shape == (len(fs),) and np.all(weights >= 0)
                np.testing.assert_allclose(weights.sum(), mass, atol=1e-10)
                assert np.max(weights/mass) <= 2/d+1e-10
            embedded = np.zeros(d)
            embedded[fs] = weights'''
    assert audit.count(old) == 1
    audit = audit.replace(old, new)
    audit = audit.replace("    for entry in path['rankings']:", "    for ranking_index, entry in enumerate(path['rankings']):")
    old = "        if accepted is None:\n            assert entry is path['rankings'][-1]\n            break\n        active = accepted['features']"
    new = '''        if accepted is None:
            repair = path['repair_blocks'][repair_offset]
            repair_offset += 1
            assert repair['ranking_index'] == ranking_index
            assert 0 < len(repair['attempts']) <= len(ranking)
            for (loss, removed), record in zip(ranking, repair['attempts']):
                assert record['removed'] == removed and record['loss'] == loss
                info = record['fit']
                counts['metric_repair_attempts'] += 1
                counts['metric_solver_limits'] += info['solver_status'] == 1
                fs = [a for a in active if a != removed]
                assert info['features'] == fs and info['time_limit'] == 10.
                assert not candidate(fs)['certificate']['feasible']
                if 'weights' in info:
                    repaired = candidate(fs, info['weights'])
                    equivalent(repaired['certificate'], info['certificate'])
                    if info['status'] == 'original_certificate_passed':
                        assert repaired['certificate']['feasible'] and record is repair['attempts'][-1]
                        check_repair(A, y, info, counts)
                        accepted = repaired
                        accepted_records.append(accepted)
                        counts['accepted_metric_repairs'] += 1
                        break
                    assert not repaired['certificate']['feasible']
                else:
                    assert info['status'] in ('no_incumbent', 'incumbent_constraints_failed', 'normalized_cap_failed')
                counts['rejected_metric_repairs'] += 1
            if accepted is None:
                counts['failed_metric_repair_steps'] += 1
                assert len(repair['attempts']) == len(ranking) and entry is path['rankings'][-1]
                break
        active = accepted['features']
    assert repair_offset == len(path['repair_blocks'])'''
    assert audit.count(old) == 1
    audit = audit.replace(old, new)
    return {'run_repaired_crossfit_recent_pilot.py': source,
            'audit_repaired_crossfit_recent_pilot.py': audit}


if __name__ == '__main__':
    for name, source in generated().items():
        target = ROOT/'scripts'/name
        assert not target.exists(), 'preserve existing source'
        compile(source, str(target), 'exec')
        target.write_text(source, encoding='utf-8')
