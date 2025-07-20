import os
import sys
import joblib

import numpy as np
from tqdm import tqdm
import pandas as pd

from ifcb import DataDirectory
from feature_extraction import extract_features, preprocess_image

DATA_DIR = '/Users/jfutrelle/Data/ifcb-data/focus'
STUDENT_MODEL_FEATURES = ['laplacian_var', 'dog_var', 'kurtosis', 'mad']

def score_bin(b, model):
    X = extract_slim_features(b).drop(columns=['pid']).values
    y = model.predict(X)
    # compute confidence as the mean of the predicted probabilities
    bad = len(y[y < 0.1])
    good = len(y[y > 0.9])
    score = good / (good + bad) if (good + bad) > 0 else 0.0
    return score

def extract_slim_features(b):
    # first, see if we've already computed the full features
    features_path = os.path.join(DATA_DIR, 'features', f"{b.lid}_features.csv")
    if os.path.exists(features_path):
        features_df = pd.read_csv(features_path)
        return features_df[STUDENT_MODEL_FEATURES + ['pid']]
    roi_numbers = list(b.images.keys())
    features = []
    for roi in roi_numbers:
        roi_pid = f"{b.lid}_{roi:05d}"
        image = b.images[roi]
        image = preprocess_image(image)
        image_features = { 'pid': roi_pid }
        image_features.update(extract_features(image, STUDENT_MODEL_FEATURES))
        features.append(image_features)
    return pd.DataFrame.from_records(features)

def score_remote_bin(host, pid, model):
    from ifcb import open_url
    with open_url(f'https://{host}/data/{pid}.adc') as b:
        return score_bin(b, model)

if __name__ == "__main__":
    model = joblib.load(os.path.join(DATA_DIR, 'student_model.pkl'))
    ground_truth = pd.read_csv(os.path.join(DATA_DIR, 'ground_truth.csv'))
    def get_true_label(pid):
        row = ground_truth[ground_truth['pid'] == pid]
        if not row.empty:
            return row['true_label'].values[0]
        return 0
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
    dd = DataDirectory(RAW_DATA_DIR)
    print('pid,score,label')
    for b in dd:
        score = score_bin(b, model)
        print(f'{b.lid},{score:.4f},{int(score > 0.5)},{get_true_label(b.lid)}')