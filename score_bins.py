import os
import joblib

import pandas as pd

from ifcb import DataDirectory

from feature_extraction import extract_features, preprocess_image

STUDENT_MODEL_FEATURES = ['laplacian_var', 'dog_var', 'kurtosis', 'mad']

def score_bin(b, model, confidence_threshold=0.1, top_n=200, verbose=False, cache_dir=None):
    X = extract_slim_features(b, top_n=top_n, cache_dir=cache_dir).drop(columns=['pid']).values
    if X.shape[0] == 0:
        return 0.0
    y = model.predict(X)
    # compute confidence as the mean of the predicted probabilities
    bad = len(y[y < confidence_threshold])
    good = len(y[y > (1 - confidence_threshold)])
    score = good / (good + bad) if (good + bad) > 0 else 0.0
    if verbose:
        print(f"Scored bin {b.lid}: {score:.4f} (good: {good}, bad: {bad})")
    return score

def extract_slim_features(b, top_n=200, cache_dir=None, features_to_use=STUDENT_MODEL_FEATURES):
    if cache_dir is not None:
        features_path = os.path.join(cache_dir, f"{b.lid}_features.csv")
        if os.path.exists(features_path):
            return pd.read_csv(features_path)
    roi_numbers = list(b.images.keys())
    features = []
    for roi in roi_numbers[:top_n]:
        roi_pid = f"{b.lid}_{roi:05d}"
        image = b.images[roi]
        image = preprocess_image(image)
        image_features = { 'pid': roi_pid }
        image_features.update(extract_features(image, features_to_use))
        features.append(image_features)
    df = pd.DataFrame.from_records(features)
    if df.empty:
        df = pd.DataFrame(columns=['pid'] + features_to_use)
    if cache_dir is not None:
        df.to_csv(features_path, index=False, float_format='%.5f')
    return df

def score_remote_bin(host, pid, model, features=STUDENT_MODEL_FEATURES):
    from ifcb import open_url
    with open_url(f'https://{host}/data/{pid}.adc') as b:
        return score_bin(b, model)

def score_directory(data_dir, model, features=STUDENT_MODEL_FEATURES, verbose=True, cache_dir=None):
    dd = DataDirectory(data_dir)
    results = []

    def process_bin(b):
        score = score_bin(b, model, verbose=verbose, cache_dir=cache_dir)
        return {
            'pid': b.lid,
            'score': score,
            'label': int(score > 0.5)
        }

    for b in dd:
        results.append(process_bin(b))

    return pd.DataFrame.from_records(results)
