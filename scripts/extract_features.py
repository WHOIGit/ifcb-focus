#!/usr/bin/env python
"""
CLI script for extracting features from IFCB bins.

This script extracts features from all bins listed in ground_truth.csv
and saves them to individual CSV files.
"""

import argparse
import os

from ifcb_focus.workflows import feature_extraction


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from IFCB bins listed in ground_truth.csv'
    )
    parser.add_argument(
        '-d', '--data-dir', type=str, default=None,
        help='Data directory containing ground_truth.csv and raw_data/. '
             'Defaults to IFCB_DATA_DIR environment variable or ./data'
    )

    args = parser.parse_args()

    # Set data directory from environment if not provided
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.environ.get('IFCB_DATA_DIR', './data')

    print(f"Extracting features from bins in: {data_dir}")
    print("=" * 60)

    feature_extraction(data_dir=data_dir)

    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print(f"Features saved to: {os.path.join(data_dir, 'features')}")


if __name__ == "__main__":
    main()
