import os
import joblib

import pandas as pd

from ifcb import DataDirectory

from feature_extraction import extract_features, preprocess_image

def score_bin(b, model, features_to_use, confidence_threshold=0.1, top_n=200, verbose=False, cache_dir=None):
    """Score an IFCB bin using a trained model.
    
    Extracts features from ROIs in the bin and uses the model to predict
    focus quality, returning a score based on the ratio of good to total predictions.
    
    Args:
        b: IFCB bin object with images and metadata.
        model: Trained model with predict() method.
        features_to_use (list): List of feature names to extract and use.
        confidence_threshold (float): Threshold for determining good/bad predictions.
        top_n (int): Maximum number of ROIs to process from the bin.
        verbose (bool): Whether to print scoring details.
        cache_dir (str): Optional directory to cache extracted features.
        
    Returns:
        float: Score between 0 and 1, representing focus quality ratio.
    """
    X = extract_slim_features(b, features_to_use, top_n=top_n, cache_dir=cache_dir).drop(columns=['pid']).values
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

def extract_slim_features(b, features_to_use, top_n=200, cache_dir=None):
    """Extract a subset of features from ROIs in an IFCB bin.
    
    Processes up to top_n ROIs from the bin, extracting only the specified
    features for efficient computation. Supports caching to avoid recomputation.
    
    Args:
        b: IFCB bin object with images and metadata.
        features_to_use (list): List of feature names to extract.
        top_n (int): Maximum number of ROIs to process.
        cache_dir (str): Optional directory to cache extracted features.
        
    Returns:
        pd.DataFrame: DataFrame with 'pid' column and feature columns.
    """
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

def score_remote_bin(host, pid, model, features_to_use):
    """Score a remote IFCB bin by downloading and processing it.
    
    Downloads an IFCB bin from a remote host and scores it using the
    provided model and feature set.
    
    Args:
        host (str): Hostname of the remote IFCB data server.
        pid (str): Bin PID identifier to download and score.
        model: Trained model with predict() method.
        features_to_use (list): List of feature names to extract and use.
        
    Returns:
        float: Score between 0 and 1, representing focus quality ratio.
    """
    from ifcb import open_url
    with open_url(f'https://{host}/data/{pid}.adc') as b:
        return score_bin(b, model, features_to_use)

def process_single_bin(b, model, features_to_use, verbose=False, cache_dir=None):
    """Process a single bin and return results dictionary.
    
    Helper function that scores a bin and formats the results into a
    dictionary suitable for DataFrame creation.
    
    Args:
        b: IFCB bin object with images and metadata.
        model: Trained model with predict() method.
        features_to_use (list): List of feature names to extract and use.
        verbose (bool): Whether to print scoring details.
        cache_dir (str): Optional directory to cache extracted features.
        
    Returns:
        dict: Dictionary with 'pid', 'score', and 'label' keys.
    """
    score = score_bin(b, model, features_to_use, verbose=verbose, cache_dir=cache_dir)
    return {
        'pid': b.lid,
        'score': score,
        'label': int(score > 0.5)
    }

def score_directory(data_dir, model, features_to_use, verbose=True, cache_dir=None):
    """Score all bins in a directory using a trained model.
    
    Processes all IFCB bins found in the specified directory, scoring each
    one and returning results as a DataFrame.
    
    Args:
        data_dir (str): Directory containing IFCB bin files.
        model: Trained model with predict() method.
        features_to_use (list): List of feature names to extract and use.
        verbose (bool): Whether to print scoring details for each bin.
        cache_dir (str): Optional directory to cache extracted features.
        
    Returns:
        pd.DataFrame: DataFrame with columns 'pid', 'score', and 'label'.
    """
    dd = DataDirectory(data_dir)
    results = []

    for b in dd:
        results.append(process_single_bin(b, model, features_to_use, verbose=verbose, cache_dir=cache_dir))

    return pd.DataFrame.from_records(results)
