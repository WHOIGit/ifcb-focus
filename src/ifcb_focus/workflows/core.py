"""
Workflow functions for IFCB focus detection.

This module provides high-level workflow functions that combine feature
extraction, model training, and scoring into reproducible pipelines. Bins are
discovered through ``ifcbkit``; the caller passes a root directory of IFCB raw
filesets and ``ifcbkit`` finds the bins using its default include/exclude rules.
"""

import os
import numpy as np
import pandas as pd
from skimage.io import imsave
from ifcbkit import SyncIfcbDataDirectory

from ifcb_focus.features import (
    preprocess_image,
    extract_features,
    STUDENT_MODEL_FEATURES,
)


def feature_extraction(
    data_dir,
    output_dir,
    features_to_use=None,
    target_size=(512, 512),
):
    """
    Extract features from all bins in a dataset.

    Args:
        data_dir (str): Root directory containing IFCB raw filesets. Passed
            directly to ``ifcbkit.SyncIfcbDataDirectory``, which discovers bins
            using its default include/exclude rules. A ``ground_truth.csv`` is
            expected at the same path root (``data_dir/ground_truth.csv``).
        output_dir (str): Directory to save features CSV and preprocessed images.
        features_to_use (list, optional): Feature names to compute.
            Defaults to the student model feature set.
        target_size (tuple): Target (height, width) for preprocessing.

    Returns:
        pandas.DataFrame: Feature matrix with one row per ROI image.
    """
    if features_to_use is None:
        features_to_use = list(STUDENT_MODEL_FEATURES)

    dd = SyncIfcbDataDirectory(data_dir)

    # Load ground truth
    gt_path = os.path.join(data_dir, 'ground_truth.csv')
    df = pd.read_csv(gt_path)

    # Check which bins are available
    df['features_available'] = df['pid'].apply(lambda pid: dd.exists(pid))

    print(f"Found {df['features_available'].sum()} of {len(df)} bins with features")

    # Prepare output directories
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    all_features = []

    for idx, row in df.iterrows():
        if not dd.exists(row['pid']):
            continue

        images = dd.read_images(row['pid'])

        for roi in images:
            image = preprocess_image(images[roi], target_size=target_size)
            image_path = os.path.join(image_dir, f'{row["pid"]}_{roi}.png')
            # PNG is integer-only; save an 8-bit preview (features use the float array)
            imsave(image_path, (np.clip(image, 0.0, 1.0) * 255).round().astype('uint8'))

            feats = extract_features(image, features_to_use)
            all_features.append({
                'image_path': image_path,
                'roi_pid': f"{row['pid']}_{roi:05d}",
                'bin_pid': row['pid'],
                'target_number': roi,
                'ground_truth': row['ground_truth'],
                **feats
            })

            if idx % 100 == 0:
                print(f"Processed bin {idx}: {row['pid']}")

    features_df = pd.DataFrame(all_features)
    features_path = os.path.join(output_dir, 'features.csv')
    features_df.to_csv(features_path, index=False)

    print(f"Saved features to {features_path}")
    return features_df


def split_true_labels(df, train_ratio=0.8, random_seed=42):
    """
    Split "true" labels into train and test sets.

    Args:
        df (pandas.DataFrame): DataFrame with a 'ground_truth' column.
        train_ratio (float): Fraction of data for training (default 0.8).
        random_seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_df, test_df) DataFrames with a 'split' column added.
    """
    from sklearn.model_selection import train_test_split

    true_mask = df['ground_truth'] == 1
    true_df = df[true_mask].copy()

    train_true, test_true = train_test_split(
        true_df,
        train_size=train_ratio,
        random_state=random_seed,
        stratify=true_df['ground_truth']
    )

    train_true['split'] = 'train'
    test_true['split'] = 'test'

    return train_true, test_true
