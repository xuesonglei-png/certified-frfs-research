"""Ten-second oracle sensitivity rerun; output is separate from the 2-second run."""
from pathlib import Path
import diagnose_feasible_query as base


def configure():
    base.FOLDER = base.ROOT/'results/feasible-query-long-diagnostic'
    base.ORACLE_SECONDS = 10.
    base.EXTRA_SOURCE_PATHS = [Path(__file__), base.ROOT/'research/feasible-query-long-protocol.md']


def run(name):
    configure()
    return base.run(name)


def main():
    configure()
    base.FOLDER.mkdir(exist_ok=True)
    with base.ProcessPoolExecutor(max_workers=2) as pool:
        for future in base.as_completed([pool.submit(run, name) for name in base.NAMES]): future.result()
    base.analyze()


if __name__ == '__main__': main()
