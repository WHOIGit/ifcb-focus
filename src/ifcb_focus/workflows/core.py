"""
Workflow functions for IFCB focus detection.

This module provides high-level workflow functions for feature extraction
and data splitting.
"""

import os
import pandas as pd
from tqdm import tqdm
from ifcb import DataDirectory

from ifcb_focus.features import preprocess_image, extract_features


def feature_extraction(data_dir=None):
    """
    Extract features from all bins in the validation set.

    This function reads ground_truth.csv and extracts features for all ROIs
    in each bin, saving them to CSV files.

    Args:
        data_dir (str, optional): Base data directory containing ground_truth.csv
            and raw_data/. If None, uses environment variable IFCB_DATA_DIR
            or defaults to './data'.

    Example:
        >>> from ifcb_focus.workflows import feature_extraction
        >>> feature_extraction('/path/to/data')
    """
    if data_dir is None:
        data_dir = os.environ.get('IFCB_DATA_DIR', './data')

    validation_set = pd.read_csv(os.path.join(data_dir, 'ground_truth.csv'))
    dd = DataDirectory(data_dir, 'raw_data')
    pids = validation_set['pid'].tolist()

    # Create features directory if it doesn't exist
    features_dir = os.path.join(data_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)

    for pid in pids:
        b = dd[pid]
        roi_numbers = list(b.images.keys())
        features = []
        for roi in tqdm(roi_numbers, desc=pid):
            roi_pid = f"{pid}_{roi:05d}"
            image = b.images[roi]
            image = preprocess_image(image)
            image_features = {'pid': roi_pid}
            image_features.update(extract_features(image))
            features.append(image_features)
        features_df = pd.DataFrame.from_records(features)
        features_df.to_csv(
            os.path.join(features_dir, f"{pid}_features.csv"),
            float_format='%.5f',
            index=False
        )


def split_true_labels(data_dir=None, train_fraction=0.66, random_state=42):
    """
    Split ground truth data into training and validation sets.

    Args:
        data_dir (str, optional): Base data directory containing ground_truth.csv.
            If None, uses environment variable IFCB_DATA_DIR or defaults to './data'.
        train_fraction (float): Fraction of data to use for training.
            Default is 0.66 (66% training, 34% validation).
        random_state (int): Random seed for reproducibility. Default is 42.

    Example:
        >>> from ifcb_focus.workflows import split_true_labels
        >>> split_true_labels('/path/to/data', train_fraction=0.7)
    """
    if data_dir is None:
        data_dir = os.environ.get('IFCB_DATA_DIR', './data')

    truth = pd.read_csv(os.path.join(data_dir, 'ground_truth.csv'))
    # split into training and validation
    train_set = truth.sample(frac=train_fraction, random_state=random_state)
    validation_set = truth.drop(train_set.index)
    train_set.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    validation_set.to_csv(os.path.join(data_dir, 'validation.csv'), index=False)

    print(f"Split data: {len(train_set)} training samples, {len(validation_set)} validation samples")
