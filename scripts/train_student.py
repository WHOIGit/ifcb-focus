#!/usr/bin/env python
"""
CLI script for training a student model for IFCB focus detection.

This script trains a lightweight Random Forest regressor using feature
distillation from a teacher model.
"""

import argparse
import os

from ifcb_focus.training import train_student_model


def main():
    parser = argparse.ArgumentParser(
        description='Train a student model for IFCB focus detection using '
                    'knowledge distillation from a teacher model.'
    )
    parser.add_argument(
        '-d', '--data-dir', type=str, default=None,
        help='Data directory containing train.csv, validation.csv, and '
             'teacher_model.pkl. Defaults to IFCB_DATA_DIR environment '
             'variable or ./data'
    )
    parser.add_argument(
        '-m', '--model-path', type=str, default=None,
        help='Path to save the trained model. '
             'Defaults to {data_dir}/slim_student_model.pkl'
    )

    args = parser.parse_args()

    # Set data directory from environment if not provided
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.environ.get('IFCB_DATA_DIR', './data')

    print(f"Training student model with data from: {data_dir}")
    print("=" * 60)

    model = train_student_model(data_dir=data_dir, model_path=args.model_path)

    print("\n" + "=" * 60)
    print("Student model training complete!")


if __name__ == "__main__":
    main()
