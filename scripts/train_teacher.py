#!/usr/bin/env python
"""
CLI script for training a teacher model for IFCB focus detection.

This script trains a Random Forest classifier using the full feature set.
"""

import argparse
import os

from ifcb_focus.training import train_random_forest


def main():
    parser = argparse.ArgumentParser(
        description='Train a teacher model for IFCB focus detection.'
    )
    parser.add_argument(
        '-d', '--data-dir', type=str, default=None,
        help='Data directory containing train.csv and validation.csv. '
             'Defaults to IFCB_DATA_DIR environment variable or ./data'
    )
    parser.add_argument(
        '-m', '--model-path', type=str, default=None,
        help='Path to save the trained model. '
             'Defaults to {data_dir}/teacher_model.pkl'
    )

    args = parser.parse_args()

    # Set data directory from environment if not provided
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.environ.get('IFCB_DATA_DIR', './data')

    print(f"Training teacher model with data from: {data_dir}")
    print("=" * 60)

    model = train_random_forest(data_dir=data_dir, model_path=args.model_path)

    print("\n" + "=" * 60)
    print("Teacher model training complete!")


if __name__ == "__main__":
    main()
