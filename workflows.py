import os

import pandas as pd
from tqdm import tqdm
from ifcb import DataDirectory

from feature_extraction import preprocess_image, extract_features


DATA_DIR = os.environ.get('IFCB_DATA_DIR', './data')


def feature_extraction():
    validation_set = pd.read_csv(os.path.join(DATA_DIR, 'ground_truth.csv'))
    dd = DataDirectory(DATA_DIR, 'raw_data')
    pids = validation_set['pid'].tolist()
    for pid in pids:
        b = dd[pid]
        roi_numbers = list(b.images.keys())
        features = []
        for roi in tqdm(roi_numbers, desc=pid):
            roi_pid = f"{pid}_{roi:05d}"
            image = b.images[roi]
            image = preprocess_image(image)
            image_features = { 'pid': roi_pid }
            image_features.update(extract_features(image))
            features.append(image_features)
        features_df = pd.DataFrame.from_records(features)
        features_df.to_csv(os.path.join(DATA_DIR, 'features', f"{pid}_features.csv"), float_format='%.5f', index=False)

def split_true_labels():
    truth = pd.read_csv(os.path.join(DATA_DIR, 'ground_truth.csv'))
    # split 50/50 into training and validation
    train_set = truth.sample(frac=0.66, random_state=42)
    validation_set = truth.drop(train_set.index)
    train_set.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
    validation_set.to_csv(os.path.join(DATA_DIR, 'validation.csv'), index=False)

if __name__ == "__main__":
    split_true_labels()
