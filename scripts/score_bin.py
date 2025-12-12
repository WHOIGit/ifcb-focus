#!/usr/bin/env python
"""
CLI script for scoring IFCB bins using trained models.

This script provides a command-line interface for scoring individual bins,
remote bins, or entire directories of bins.
"""

import sys
import argparse
import os
import joblib
from ifcb import DataDirectory

from ifcb_focus.scoring import score_bin, score_remote_bin


def main():
    parser = argparse.ArgumentParser(
        description='Score IFCB bins using a trained model.'
    )
    parser.add_argument(
        '--host', type=str,
        help='Host for remote bins (e.g., ifcb-data.whoi.edu)'
    )
    parser.add_argument(
        '-p', '--pid', type=str,
        help='Bin ID (e.g., D20230101T120000_IFCB123)'
    )
    parser.add_argument(
        '-d', '--directory', type=str,
        help='Directory containing local bins'
    )
    parser.add_argument(
        '-n', '--top_n', type=int, default=200,
        help='Number of largest ROIs to score per bin (default: 200)'
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

    # Validate arguments
    if args.directory and not os.path.exists(args.directory):
        print(f"Error: Directory {args.directory} does not exist.")
        sys.exit(1)

    # Load model
    try:
        model = joblib.load(args.model)
    except FileNotFoundError:
        print(f"Error: Model file {args.model} not found.")
        sys.exit(1)

    # Score based on provided arguments
    if args.host and args.pid:
        # Score remote bin
        score = score_remote_bin(args.host, args.pid, model)
        print(f'{args.pid},{score:.4f}')

    elif args.directory and args.pid:
        # Score specific bin from local directory
        dd = DataDirectory(args.directory)
        b = dd[args.pid]
        score = score_bin(b, model, top_n=args.top_n, verbose=args.verbose)
        print(f'{b.lid},{score:.4f}')

    elif args.directory:
        # Score all bins in directory
        dd = DataDirectory(args.directory)
        print('bin_id,score')
        for b in dd:
            score = score_bin(b, model, top_n=args.top_n, verbose=args.verbose)
            print(f'{b.lid},{score:.4f}')

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
