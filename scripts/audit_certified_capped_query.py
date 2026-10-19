"""Use full query certificate audit, with explicit pernode cap assertions."""
import json
import numpy as np
import audit_feasible_query as audit
import diagnose_feasible_query as base
from diagnose_certified_capped_query import configure


if __name__=='__main__':
    configure();audit.FOLDER=base.FOLDER;audit.ARCHIVE_NAME='certified-capped-query-evidence.zip'
    for name in base.NAMES:
        p=json.loads((base.FOLDER/f'{name}.json').read_text());cap=2/np.sqrt(p['d'])
        for s in p['splits']:
            assert s['fit']['weight_cap_ratio']==2.
            for node in s['fit']['trace']+s['fit']['oracles']:
                assert np.all(np.array(node['weights'])<=cap+1e-10)
    audit.main()
