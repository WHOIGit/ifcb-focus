#!/usr/bin/env python
"""
CLI script for scoring IFCB bins using trained models.

Bins are discovered through an ifcbkit data directory constructed from a root
directory of IFCB raw filesets.
"""

import sys
import argparse
import os
import joblib
from ifcbkit import SyncIfcbDataDirectory

from ifcb_focus.scoring import score_bin


def main():
    parser = argparse.ArgumentParser(
        description='Score IFCB bins using a trained model.'
    )
    parser.add_argument(
        '-p', '--pid', type=str,
        help='Bin ID (e.g., D20230101T120000_IFCB123)'
    )
    parser.add_argument(
        '-d', '--directory', type=str,
        help='Root directory containing local IFCB raw filesets'
    )
    parser.add_argument(
        '-n', '--top_n', type=int, default=200,
        help='Number of ROIs to score per bin (default: 200)'
    )
    parser.add_argument(
        '-m', '--model', type=str, default='slim_student_model.pkl',
        help='Path to the trained model file (default: slim_student_model.pkl)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print verbose scoring information'
    )

    args = parser.parse_args()

    # A directory is required (local scoring only; remote scoring was dropped).
    if not args.directory:
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.directory):
        print(f"Error: Directory {args.directory} does not exist.")
        sys.exit(1)

    # Load model
    try:
        model = joblib.load(args.model)
    except FileNotFoundError:
        print(f"Error: Model file {args.model} not found.")
        sys.exit(1)

    dd = SyncIfcbDataDirectory(args.directory)

    # Score based on provided arguments
    if args.pid:
        # Score a specific bin from the local directory
        score = score_bin(
            dd, args.pid, model,
            top_n=args.top_n, verbose=args.verbose)
        print(f'{args.pid},{score:.4f}')
    else:
        # Score all bins in the directory
        print('bin_id,score')
        for fs in dd.list():
            score = score_bin(
                dd, fs['pid'], model,
                top_n=args.top_n, verbose=args.verbose)
            print(f"{fs['pid']},{score:.4f}")


if __name__ == "__main__":
    main()
