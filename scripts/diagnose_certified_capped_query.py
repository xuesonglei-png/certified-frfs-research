"""Fixed certified ratio2 cap, all15/all44 inner splits."""
from pathlib import Path
import diagnose_feasible_query as base


def configure():
    base.FOLDER=base.ROOT/'results/certified-capped-query-diagnostic'
    base.ORACLE_SECONDS=10.;base.WEIGHT_CAP_RATIO=2.
    base.EXTRA_SOURCE_PATHS=[Path(__file__),base.ROOT/'research/certified-capped-query-protocol.md']


def run(name):
    configure();return base.run(name)


if __name__=='__main__':
    configure();base.FOLDER.mkdir(exist_ok=True)
    with base.ProcessPoolExecutor(max_workers=2) as pool:
        for future in base.as_completed([pool.submit(run,name) for name in base.NAMES]):future.result()
    base.analyze()
