import sys
import joblib
import argparse
import os

import pandas as pd
from tqdm import tqdm

from ifcb import open_url, DataDirectory
import numpy as np

from train import preprocess_image, compute_features, skip_image

def extract_roi_features(b, top_n=100, progress_bar=False):
    """Extract features from the largest ROIs in an IFCB bin.
    
    Processes ROIs sorted by area (largest first), extracting image features
    from each valid ROI that meets minimum size requirements.
    
    Args:
        b: IFCB bin object with images and metadata.
        top_n (int): Maximum number of ROIs to process (largest by area).
        progress_bar (bool): Whether to display a progress bar during processing.
        
    Returns:
        tuple: A tuple containing:
            - features_list (list): List of feature arrays for each ROI.
            - feature_names (list): List of feature names.
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
    
    return features_list, feature_names

def score_bin(b, model, top_n=100, progress_bar=False, confidence_threshold=0.66):
    """Score an IFCB bin using a trained classifier model.
    
    Extracts features from the largest ROIs in the bin, uses the model to
    predict focus quality probabilities, and computes a score based on
    high-confidence predictions only.
    
    Args:
        b: IFCB bin object with images and metadata.
        model: Trained classifier model with predict_proba() method.
        top_n (int): Maximum number of ROIs to process (largest by area).
        progress_bar (bool): Whether to display a progress bar during processing.
        confidence_threshold (float): Minimum confidence threshold for including predictions.
        
    Returns:
        tuple: A tuple containing:
            - score (float): Average probability of high-confidence good predictions.
            - sample_confidence (float): Average confidence of high-confidence predictions.
    """
    features_list, feature_names = extract_roi_features(b, top_n=top_n, progress_bar=progress_bar)
    
    if len(features_list) == 0:
        return 0.0, 0.0

    X = pd.DataFrame(np.array(features_list), columns=feature_names)
    y_probabilities = model.predict_proba(X)
    y = y_probabilities[:, 1]  # Probability of being a 'good' ROI
    confidence = np.max(y_probabilities, axis=1)  # Confidence of the prediction
    highconf = np.where(confidence > confidence_threshold)[0]
    
    score = np.mean(y[highconf]) if len(highconf) > 0 else 0.0
    sample_confidence = np.mean(confidence[highconf]) if len(highconf) > 0 else 0.0

    return score, sample_confidence

def score_remote_bin(host, pid, model, top_n=100, progress_bar=False, confidence_threshold=0.66):
    """Score a remote IFCB bin by downloading and processing it.
    
    Downloads an IFCB bin from a remote host and scores it using the
    provided classifier model.
    
    Args:
        host (str): Hostname of the remote IFCB data server.
        pid (str): Bin PID identifier to download and score.
        model: Trained classifier model with predict_proba() method.
        top_n (int): Maximum number of ROIs to process (largest by area).
        progress_bar (bool): Whether to display a progress bar during processing.
        confidence_threshold (float): Minimum confidence threshold for including predictions.
        
    Returns:
        tuple: A tuple containing:
            - score (float): Average probability of high-confidence good predictions.
            - sample_confidence (float): Average confidence of high-confidence predictions.
    """
    with open_url(f'https://{host}/data/{pid}.adc') as b:
        return score_bin(b, model, top_n=top_n, progress_bar=progress_bar, confidence_threshold=confidence_threshold)

def score_and_print(b, model, top_n=100):
    """Score a bin and print results in CSV format.
    
    Helper function that scores a bin and prints the results to stdout
    in comma-separated format: pid,score,confidence
    
    Args:
        b: IFCB bin object with images and metadata.
        model: Trained classifier model with predict_proba() method.
        top_n (int): Maximum number of ROIs to process (largest by area).
    """
    score, confidence = score_bin(b, model, top_n=top_n)
    print(f'{b.lid},{score},{confidence}')

def process_directory(directory, model, top_n=100):
    """Process all bins in a directory and print scores.
    
    Iterates through all IFCB bins in the specified directory, scoring
    each one and printing results to stdout.
    
    Args:
        directory (str): Directory containing IFCB bin files.
        model: Trained classifier model with predict_proba() method.
        top_n (int): Maximum number of ROIs to process per bin.
    """
    dd = DataDirectory(directory)
    for b in dd:
        score_and_print(b, model, top_n=top_n)

def process_single_bin(directory, pid, model, top_n=100):
    """Process a single bin from a directory and print score.
    
    Loads a specific bin from the directory by PID, scores it, and
    prints results to stdout.
    
    Args:
        directory (str): Directory containing IFCB bin files.
        pid (str): Bin PID identifier to process.
        model: Trained classifier model with predict_proba() method.
        top_n (int): Maximum number of ROIs to process.
    """
    dd = DataDirectory(directory)
    b = dd[pid]
    score_and_print(b, model, top_n=top_n)

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

    if args.directory and args.pid is None:
        process_directory(directory, model, top_n=args.top_n)
    elif args.host and args.pid:
        score, confidence = score_remote_bin(host, pid, model, top_n=args.top_n)
        print(f'{pid},{score},{confidence}')
    elif args.directory and args.pid:
        process_single_bin(directory, args.pid, model, top_n=args.top_n)
    elif args.host and args.pid is None:
        print("Please provide a PID when using a host.")
    else:
        print("Please provide either a host and PID or a directory with bins.")
