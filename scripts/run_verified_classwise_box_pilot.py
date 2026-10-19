"""Immutable separate pilot accepting only verified MILP incumbents."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
import run_classwise_box_pilot as pilot
from certified_frfs.verified_classwise_box_linear_refit import fit_verified_classwise_box_linear

pilot.FOLDER=pilot.ROOT/'results/verified-classwise-box-pilot'
pilot.fit_classwise_box_linear=fit_verified_classwise_box_linear

def run(name):
    pilot.run(name)
    path=pilot.FOLDER/f'{name}.json';p=json.loads(path.read_text())
    files=[Path(__file__),pilot.ROOT/'research/verified-classwise-box-protocol.md',pilot.ROOT/'src/certified_frfs/verified_classwise_box_linear_refit.py']
    p['fingerprints'].update({str(f.relative_to(pilot.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    path.write_text(json.dumps(p,indent=2),encoding='utf-8')

if __name__=='__main__':
    pilot.FOLDER.mkdir(exist_ok=True);names=[r['dataset'] for r in json.loads((pilot.ROOT/'results/classwise-box-pilot/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    pilot.summarize(names)
