"""Conservative same-identifier grouping sensitivity on saved sampled rows."""
import csv
import hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import run_newdata_compact as experiment


class IdentifierFolds:
    def __init__(self,n_splits,*,shuffle,random_state):
        self.folds=StratifiedGroupKFold(n_splits,shuffle=shuffle,random_state=random_state)

    def split(self,X,y):
        path=experiment.DATA/'breast_original-breast-cancer-wisconsin.data'
        records=list(csv.reader(path.open(encoding='utf-8-sig')))
        with np.load(experiment.DATA/'breast_original.npz') as data:ids=data['source_indices']
        groups=np.array([r[0] for r in records])[ids];assert len(groups)==len(y)
        for tr,te in self.folds.split(X,y,groups):
            assert set(groups[tr]).isdisjoint(groups[te])
            yield tr,te


original_sources=experiment.sources


def grouped_sources(name):
    hashes=original_sources(name)
    for path in (Path(__file__),experiment.ROOT/'research/breast-original-grouped-protocol.md'):
        hashes[str(path.relative_to(experiment.ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def configure():
    experiment.FOLDER=experiment.ROOT/'results/breast-original-grouped'
    experiment.StratifiedKFold=IdentifierFolds;experiment.sources=grouped_sources
    experiment.FOLDER.mkdir(exist_ok=True)


if __name__=='__main__':
    configure();experiment.run('breast_original')
