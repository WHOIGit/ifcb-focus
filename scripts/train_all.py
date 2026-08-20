#!/usr/bin/env python3
"""
End-to-end IFCB focus-model training from a single data directory.

Runs the whole pipeline in order:
    1. feature_extraction  -> features/{pid}_features.csv   (reads raw bins via ifcbkit)
    2. split_true_labels   -> train.csv, validation.csv
    3. train_random_forest -> teacher_model.pkl             (full-feature classifier)
    4. train_student_model -> slim_student_model.pkl        (distilled 4-feature regressor)

Expected input layout:
    data_dir/
        ground_truth.csv    # columns: pid, true_label  (1 = in-focus/good, 0 = bad)
        raw_data/           # raw IFCB filesets: <pid>.{hdr,adc,roi}

Everything (features/, train.csv, validation.csv, the two .pkl models) is written
back into data_dir. Re-running is idempotent (outputs are overwritten).

Usage:
    python scripts/train_all.py --data-dir /path/to/data
    # or:  export IFCB_DATA_DIR=/path/to/data && python scripts/train_all.py
    # retrain just the models from existing features/:
    python scripts/train_all.py --data-dir /path/to/data --skip-extract
"""

import argparse
import os

import pandas as pd

from ifcb_focus.workflows import feature_extraction, split_true_labels
from ifcb_focus.training import (
    train_random_forest,
    train_student_model,
    search_for_features,
)


def _validate(data_dir, raw_dir, skip_extract):
    gt_path = os.path.join(data_dir, 'ground_truth.csv')
    if not os.path.isfile(gt_path):
        raise SystemExit(
            f"error: no ground_truth.csv in {data_dir} "
            f"(expected columns: pid, true_label)"
        )
    gt = pd.read_csv(gt_path)
    missing = [c for c in ('pid', 'true_label') if c not in gt.columns]
    if missing:
        raise SystemExit(
            f"error: ground_truth.csv is missing column(s) {missing}; "
            f"expected: pid, true_label"
        )
    if not skip_extract and not os.path.isdir(raw_dir):
        raise SystemExit(f"error: raw data directory not found: {raw_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Train an IFCB focus model end-to-end '
                    '(extract -> split -> teacher -> student).'
    )
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Base data directory, or set IFCB_DATA_DIR. '
                             'Must contain ground_truth.csv and raw_data/.')
    parser.add_argument('--raw-dir', type=str, default=None,
                        help='Override the raw-filesets directory '
                             '(default: {data_dir}/raw_data).')
    parser.add_argument('--teacher-path', type=str, default=None,
                        help='Where to save the teacher model '
                             '(default: {data_dir}/teacher_model.pkl).')
    parser.add_argument('--student-path', type=str, default=None,
                        help='Where to save the student model '
                             '(default: {data_dir}/slim_student_model.pkl).')
    parser.add_argument('--train-fraction', type=float, default=0.66,
                        help='Fraction of bins for training (default 0.66).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the train/validation split (default 42).')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip feature extraction and reuse an existing features/ dir.')
    parser.add_argument('--search', action='store_true',
                        help='Auto-tune the number of student features instead of '
                             'using the default set.')
    args = parser.parse_args()

    data_dir = args.data_dir or os.environ.get('IFCB_DATA_DIR', './data')
    raw_dir = args.raw_dir or os.path.join(data_dir, 'raw_data')
    teacher_path = args.teacher_path or os.path.join(data_dir, 'teacher_model.pkl')
    student_path = args.student_path or os.path.join(data_dir, 'slim_student_model.pkl')

    _validate(data_dir, raw_dir, args.skip_extract)

    print(f"\n[1/4] Feature extraction  -> {os.path.join(data_dir, 'features')}")
    if args.skip_extract:
        print("      skipped (--skip-extract); using existing features/")
    else:
        print(f"      raw filesets from: {raw_dir}")
        feature_extraction(data_dir, raw_dir=raw_dir)

    print(f"\n[2/4] Splitting labels  (train_fraction={args.train_fraction}, "
          f"seed={args.seed})")
    split_true_labels(data_dir, train_fraction=args.train_fraction,
                      random_state=args.seed)

    print(f"\n[3/4] Training teacher model  -> {teacher_path}")
    train_random_forest(data_dir, model_path=teacher_path)

    print(f"\n[4/4] Distilling student model  -> {student_path}")
    if args.search:
        search_for_features(data_dir, model_path=student_path)
    else:
        train_student_model(data_dir, model_path=student_path)

    print(f"\nDone.")
    print(f"  Teacher: {teacher_path}")
    print(f"  Student: {student_path}")
    print(f"Score with:")
    print(f"  python scripts/score_bin.py --model-path {student_path} --directory <bins>")


if __name__ == '__main__':
    main()
