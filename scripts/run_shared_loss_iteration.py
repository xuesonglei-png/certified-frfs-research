"""Iteration35 frozen dual-constraint pilot with shared-anchor loss bound."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from run_utility_search_iteration import run


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['iris','wine','sonar','wdbc','ionosphere'])
    p.add_argument('--folds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,(name,fold),iteration=35,dependency_eta=.1,shared_loss=True) for name in a.datasets for fold in a.folds]
        for f in as_completed(futures):f.result()
