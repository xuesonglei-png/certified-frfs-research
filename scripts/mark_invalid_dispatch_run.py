"""Retain outputs but mark the failed iteration11 dispatch as scientifically invalid."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
directory=ROOT/'results/predictive-iteration11'
reason='constrained_method_names_routed_to_legacy_prefix_selector_no_constraint_search_executed'
for path in sorted(directory.glob('*.json')):
    obj=json.loads(path.read_text())
    obj['status']='invalid_implementation_protocol_not_executed'
    obj['invalid_reason']=reason
    obj['superseded_by']='predictive-iteration12'
    path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(path.name)
