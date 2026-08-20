"""
Workflow functions for IFCB focus detection.

These functions drive the data-prep stage of the training pipeline: extracting
per-bin feature CSVs from raw IFCB data (via ``ifcbkit``) and splitting the
ground-truth set into train/validation CSVs that the training modules consume.

The on-disk layout these functions assume (and that the training modules read):

    data_dir/
        ground_truth.csv        # columns: pid, true_label
        train.csv               # (written by split_true_labels) pid, true_label
        validation.csv          # (written by split_true_labels) pid, true_label
        raw_data/               # raw IFCB filesets: <pid>.{hdr,adc,roi}
        features/               # (written by feature_extraction) <pid>_features.csv
"""

import os
import pandas as pd
from tqdm import tqdm
from ifcbkit import SyncIfcbDataDirectory

from ifcb_focus.features import preprocess_image, extract_features


def _default_data_dir(data_dir):
    return data_dir if data_dir is not None else os.environ.get('IFCB_DATA_DIR', './data')


def feature_extraction(data_dir=None, features_to_use=None, raw_dir=None):
    """
    Extract features for every bin listed in ``ground_truth.csv``.

    For each bin, one CSV is written to ``{data_dir}/features/{pid}_features.csv``
    with a ``pid`` column (the per-ROI id ``{pid}_{roi:05d}``) followed by the
    computed feature columns. These per-bin files are exactly what
    :func:`ifcb_focus.training.assemble_features` reads, so this function is the
    bridge between raw IFCB data and the (CSV-driven) training pipeline.

    Args:
        data_dir (str, optional): Base data directory containing
            ``ground_truth.csv``. If None, uses ``IFCB_DATA_DIR`` or ``./data``.
        features_to_use (list, optional): Feature names to extract. If None,
            the full feature set is extracted (required for the teacher model,
            which trains on all features).
        raw_dir (str, optional): Root of the raw IFCB filesets that
            ``ifcbkit`` will search. Defaults to ``{data_dir}/raw_data``
            (matching :func:`ifcb_focus.utils.download_bin`). Pass a different
            path if your raw files live elsewhere.

    Example:
        >>> from ifcb_focus.workflows import feature_extraction
        >>> feature_extraction('/path/to/data')
    """
    data_dir = _default_data_dir(data_dir)
    if raw_dir is None:
        raw_dir = os.path.join(data_dir, 'raw_data')

    dd = SyncIfcbDataDirectory(raw_dir)
    gt = pd.read_csv(os.path.join(data_dir, 'ground_truth.csv'))
    pids = gt['pid'].tolist()

    features_dir = os.path.join(data_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)

    n_found = 0
    for pid in pids:
        if not dd.exists(pid):
            continue

        images = dd.read_images(pid)  # Mapping[int target_number, PIL.Image]
        roi_numbers = list(images)

        records = []
        for roi in tqdm(roi_numbers, desc=pid):
            roi_pid = f"{pid}_{roi:05d}"
            image = preprocess_image(images[roi])
            rec = {'pid': roi_pid}
            rec.update(extract_features(image, features_to_use))
            records.append(rec)

        df = pd.DataFrame.from_records(records)
        df.to_csv(
            os.path.join(features_dir, f"{pid}_features.csv"),
            float_format='%.5f',
            index=False,
        )
        n_found += 1

    print(f"Extracted features for {n_found} of {len(pids)} bins -> {features_dir}")


def split_true_labels(data_dir=None, train_fraction=0.66, random_state=42):
    """
    Split ``ground_truth.csv`` into ``train.csv`` and ``validation.csv``.

    A random ``train_fraction`` of the rows is written to ``train.csv``; the rest
    go to ``validation.csv``. Both files keep the original columns (notably
    ``pid`` and ``true_label``), which is what the training modules expect.

    Args:
        data_dir (str, optional): Base data directory containing
            ``ground_truth.csv``. If None, uses ``IFCB_DATA_DIR`` or ``./data``.
        train_fraction (float): Fraction of rows for training (default 0.66).
        random_state (int): Random seed (default 42).

    Example:
        >>> from ifcb_focus.workflows import split_true_labels
        >>> split_true_labels('/path/to/data', train_fraction=0.7)
    """
    data_dir = _default_data_dir(data_dir)

    truth = pd.read_csv(os.path.join(data_dir, 'ground_truth.csv'))
    train_set = truth.sample(frac=train_fraction, random_state=random_state)
    validation_set = truth.drop(train_set.index)

    train_set.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    validation_set.to_csv(os.path.join(data_dir, 'validation.csv'), index=False)

    print(f"Split data: {len(train_set)} training samples, "
          f"{len(validation_set)} validation samples")
