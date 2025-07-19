import sys
import joblib
import argparse
import os

import pandas as pd
from tqdm import tqdm

from ifcb import open_url, DataDirectory
import numpy as np

from train import preprocess_image, compute_features, skip_image

def score_bin(b, model, top_n=100, progress_bar=False):
    """
    Score a bin using the trained model.
    If no model is provided, it will use the default model.
    """
    feature_names = None
    features_list = []
    s = b.schema
    width_col = s.ROI_WIDTH
    height_col = s.ROI_HEIGHT
    roi_metadata = b.adc
    # order by area (width * height) descending
    # retain row number in metadata as a new column
    roi_metadata['roi_number'] = roi_metadata.index
    roi_metadata = roi_metadata[[width_col, height_col, 'roi_number']].copy()
    roi_metadata.loc[:, 'area'] = roi_metadata[width_col] * roi_metadata[height_col]
    roi_metadata = roi_metadata.sort_values(by='area', ascending=False).head(top_n)
    for roi_number in tqdm(roi_metadata['roi_number'], disable=not progress_bar):
        if skip_image(b.images[roi_number]):
            continue
        img = preprocess_image(b.images[roi_number])
        features = compute_features(img)
        if feature_names is None:
            feature_names = list(features.keys())
        features = list(features.values())
        features = np.array(features)
        features_list.append(features)

    if model is None:
        model = joblib.load('ifcb_focus_model.pkl')

    X = pd.DataFrame(np.array(features_list), columns=feature_names)
    y_probabilities = model.predict_proba(X)
    y = y_probabilities[:, 1]  # Probability of being a 'good' ROI
    confidence = np.max(y_probabilities, axis=1)  # Confidence of the prediction
    confidence_threshold = 0.66  # example
    highconf = np.where(confidence > confidence_threshold)[0]
    
    score = np.mean(y[highconf]) if len(highconf) > 0 else 0.0
    sample_confidence = np.mean(confidence[highconf]) if len(highconf) > 0 else 0.0

    return score, sample_confidence

def score_remote_bin(host, pid, model, top_n=100, progress_bar=False):
    with open_url(f'https://{host}/data/{pid}.adc') as b:
        return score_bin(b, model, top_n=top_n, progress_bar=progress_bar)

if __name__ == "__main__":
    # support the following command line arguments:
    # host -- for remote bins
    # pid -- for the bin id
    # directory -- for local bins
    args = argparse.ArgumentParser(description='Score IFCB data using a trained model.')
    args.add_argument('--host', type=str, help='Host for remote bins')
    args.add_argument('-p', '--pid', type=str, help='Bin ID for remote or local bins')
    args.add_argument('-d', '--directory', type=str, help='Directory for local bins')
    args.add_argument('-n', '--top_n', type=int, default=100, help='Number of ROIs to score per bin')
    args.add_argument('-m', '--model', type=str, default='ifcb_focus_model.pkl', help='Path to the trained model file')
    args = args.parse_args()

    if args.host and args.pid:
        host = args.host
        pid = args.pid

    if args.directory:
        directory = args.directory
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist.")
            sys.exit(1)

    model = joblib.load(args.model)

    def score_and_print(b):
        score, confidence = score_bin(b, model, top_n=args.top_n)
        print(f'{b.lid},{score},{confidence}')
    if args.directory and args.pid is None:
        dd = DataDirectory(directory)
        for b in dd:
            score_and_print(b)
    elif args.host and args.pid:
        score, confidence = score_remote_bin(host, pid, model, top_n=args.top_n)
        print(f'{pid},{score},{confidence}')
    elif args.directory and args.pid:
        dd = DataDirectory(directory)
        b = dd[args.pid]
        score_and_print(b)
    elif args.host and args.pid is None:
        print("Please provide a PID when using a host.")
    else:
        print("Please provide either a host and PID or a directory with bins.")
