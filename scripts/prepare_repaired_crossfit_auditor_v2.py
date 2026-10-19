"""Preserve failed v1; fix only independent floating cost comparison."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def generated():
    source = (ROOT/'scripts/audit_repaired_crossfit_recent_pilot.py').read_text(encoding='utf-8')
    old = "                assert record['removed'] == removed and record['loss'] == loss"
    new = "                assert record['removed'] == removed\n                np.testing.assert_allclose(record['loss'], loss, rtol=0., atol=1e-12)"
    assert source.count(old) == 1
    source = source.replace(old, new)
    source = source.replace("(name+'-audit.json')", "(name+'-audit-v2.json')")
    source = source.replace("FOLDER/'complete-audit.json'", "FOLDER/'complete-audit-v2.json'")
    return source


if __name__ == '__main__':
    target = ROOT/'scripts/audit_repaired_crossfit_recent_pilot_v2.py'
    assert not target.exists()
    source = generated()
    compile(source, str(target), 'exec')
    target.write_text(source, encoding='utf-8')
