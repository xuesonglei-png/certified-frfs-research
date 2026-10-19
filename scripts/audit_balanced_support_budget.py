"""Replay all balanced inner and outer common-budget classifier fits."""
import audit_nested_support_budget as original
from balanced_fuzzy_controls import train

original.train = train

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='nested-balanced-joint')
    original.main(parser.parse_args().source)
