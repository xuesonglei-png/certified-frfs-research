"""Isolated grouped-outer wrapper; original all-eight run stays immutable."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import run_newdata_compact as experiment


class PatientOuterFolds:
    def __init__(self, n_splits, *, shuffle, random_state):
        self.folds=StratifiedGroupKFold(n_splits,shuffle=shuffle,random_state=random_state)

    def split(self,X,y):
        path=experiment.DATA/'parkinsons-parkinsons.data'
        records=list(csv.DictReader(path.open(encoding='utf-8-sig')))
        groups=np.array([r['name'].rsplit('_',1)[0] for r in records])
        assert len(groups)==len(y)
        for group in np.unique(groups):assert len(np.unique(y[groups==group]))==1
        for tr,te in self.folds.split(X,y,groups):
            assert set(groups[tr]).isdisjoint(groups[te])
            yield tr,te


original_sources=experiment.sources


def grouped_sources(name):
    hashes=original_sources(name)
    for path in (Path(__file__),experiment.ROOT/'research/parkinsons-grouped-protocol.md'):
        hashes[str(path.relative_to(experiment.ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def configure():
    experiment.FOLDER=experiment.ROOT/'results/parkinsons-grouped'
    experiment.StratifiedKFold=PatientOuterFolds
    experiment.sources=grouped_sources
    experiment.FOLDER.mkdir(exist_ok=True)


if __name__=='__main__':
    configure()
    experiment.run('parkinsons')
    print('Patient-grouped follow-up saved; aggregate review waits for all-eight completion.',flush=True)
