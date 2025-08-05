import os
import re
import argparse

import numpy as np
import pandas as pd
import joblib
from skimage.io import imread
from skimage.filters import sobel
from skimage.util import img_as_float
from scipy.ndimage import convolve
from skimage import exposure, color
from skimage.filters import gaussian
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

def list_images(folder):
    return sorted([os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.png')])

def load_images(image_paths):
    for path in tqdm(image_paths, desc="Loading images"):
        image = imread(path)
        image_id = os.path.splitext(os.path.basename(path))[0]
        yield image, image_id

def preprocess_image(image, target_size=None, enhance_contrast=True):
    """
    Preprocess the image: grayscale, normalize, denoise, enhance contrast, and resize.

    Parameters:
        image (ndarray): Input image (2D grayscale or 3D RGB).
        target_size (tuple): Optional (height, width) to resize the image.
        enhance_contrast (bool): Whether to apply adaptive histogram equalization.

    Returns:
        ndarray: Preprocessed 2D float32 image in [0, 1].
    """
    # Normalize to [0, 1] float
    image = img_as_float(image)

   # Contrast enhancement
    if enhance_contrast:
        image = exposure.equalize_adapthist(image, clip_limit=0.03)

    # Optional resizing
    if target_size is not None:
        image = resize(image, target_size, anti_aliasing=True, preserve_range=True)

    return image.astype(np.float32)

def sum_modified_laplacian(image):
    """Compute the Sum Modified Laplacian (SML) sharpness."""
    kernel_h = np.array([[0, 0, 0],
                         [-1, 2, -1],
                         [0, 0, 0]])
    kernel_v = np.array([[0, -1, 0],
                         [0,  2, 0],
                         [0, -1, 0]])
    lap_h = convolve(image, kernel_h, mode='reflect')
    lap_v = convolve(image, kernel_v, mode='reflect')
    return np.sum(np.abs(lap_h) + np.abs(lap_v))

def tenengrad_sharpness(image):
    """Compute Tenengrad sharpness using Sobel gradient magnitude."""
    sobel_grad = sobel(image)
    return np.mean(sobel_grad ** 2)

def brenner_gradient(image):
    """Brenner's gradient — measures vertical detail."""
    return np.sum((image[:, :-2] - image[:, 2:]) ** 2)

def dog_variance(image, sigma1=1, sigma2=2):
    """Difference of Gaussians (DoG) variance."""
    blur1 = gaussian_filter(image, sigma1)
    blur2 = gaussian_filter(image, sigma2)
    dog = blur1 - blur2
    return np.var(dog)

def edge_density(image, sigma=1.0):
    """Edge density via Canny edge detector."""
    edges = feature.canny(image, sigma=sigma)
    return np.sum(edges) / edges.size

def local_std(image, window=5):
    """Local standard deviation in a sliding window."""
    shape = (image.shape[0] - window + 1, image.shape[1] - window + 1, window, window)
    strides = image.strides + image.strides
    windows = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    local_stds = windows.std(axis=(-1, -2))
    return np.mean(local_stds)

def mad(image):
    median = np.median(image)
    return np.median(np.abs(image - median))

def compute_features(image):
    """
    Compute various sharpness features for the image.

    Parameters:
        image (ndarray): Preprocessed 2D float32 image.

    Returns:
        dict: Dictionary of computed features.
    """
    # use slim set of features for speed
    features = {
        'tenengrad': tenengrad_sharpness(image),
        'dog_var': dog_variance(image),
        'edge_density': edge_density(image),
        'local_std': local_std(image),
        'mad': mad(image),
        'sum_modified_laplacian': sum_modified_laplacian(image),
        'brenner_gradient': brenner_gradient(image)
    }
    return features

def skip_image(image):
    """
    Determine if an image should be skipped based on its dimensions.
    Skips images smaller than 50x50 pixels.
    
    Parameters:
        image (ndarray): Input image.
    
    Returns:
        bool: True if the image should be skipped, False otherwise.
    """
    return image.shape[0] < 50 or image.shape[1] < 50

def batch_compute_features(image_list):
    """Compute features for a batch of images.
    
    Processes a list of image paths, extracting features from each valid image
    (skipping images smaller than 50x50 pixels).
    
    Args:
        image_list (list): List of image file paths to process.
        
    Returns:
        tuple: A tuple containing:
            - features_list (list): List of feature value lists for each image.
            - bin_id_list (list): List of bin IDs extracted from image filenames.
            - feature_names (list): List of feature names.
    """
    feature_names = None
    features_list = []
    bin_id_list = []
    for image, image_id in load_images(image_list):
        # enforce minimum size of 50x50
        if skip_image(image):
            continue
        preprocessed_image = preprocess_image(image)
        bin_id = re.sub(r'_\d+$', '', image_id)  # Remove trailing digits
        features = compute_features(preprocessed_image)
        features_list.append(list(features.values()))
        if feature_names is None:
            feature_names = list(features.keys())
        bin_id_list.append(bin_id)
    return features_list, bin_id_list, feature_names

def do_all_compute_features(good_dir, bad_dir, blurred_dir):
    """Compute features for all images and save to CSV.
    
    Extracts features from good (in-focus), bad (out-of-focus), and blurred images,
    combining them into a single labeled dataset saved as 'features.csv'.
    
    Args:
        good_dir (str): Directory containing good (in-focus) images.
        bad_dir (str): Directory containing bad (out-of-focus) images.
        blurred_dir (str): Directory containing blurred images.
    """
    good_features, good_bin_ids, feature_names = batch_compute_features(list_images(good_dir))
    bad_features, bad_bin_ids, _ = batch_compute_features(list_images(bad_dir))
    blurred_features, blurred_bin_ids, _ = batch_compute_features(list_images(blurred_dir))

    bin_ids = good_bin_ids + bad_bin_ids + blurred_bin_ids

    # add blurred features to bad features
    bad_features.extend(blurred_features)
    #blurred_bin_ids = [f"{bid}_blurred" for bid in blurred_bin_ids]
    # Create a DataFrame for the combined features

    good_features = pd.DataFrame(np.array(good_features), columns=feature_names)
    bad_features = pd.DataFrame(np.array(bad_features), columns=feature_names)

    y = np.concatenate([np.ones(len(good_features)), np.zeros(len(bad_features))])

    combined_features = pd.concat([good_features, bad_features], ignore_index=True)
    combined_features['bin_id'] = bin_ids
    combined_features['label'] = y.astype(int)
    combined_features.to_csv('features.csv', index=False)

def train_model(good_dir, bad_dir, blurred_dir, output_dir):
    """Train a Random Forest classifier for IFCB focus classification.
    
    Loads or computes features from image directories, trains a Random Forest model,
    evaluates its performance, and saves the trained model.
    
    Args:
        good_dir (str): Directory containing good (in-focus) images.
        bad_dir (str): Directory containing bad (out-of-focus) images.
        blurred_dir (str): Directory containing blurred images.
        output_dir (str): Directory where the trained model will be saved.
    """
    if not os.path.exists('features.csv'):
        do_all_compute_features(good_dir, bad_dir, blurred_dir)

    combined_features = pd.read_csv('features.csv', index_col=None)

    # Combine features into a single DataFrame
    X = combined_features.drop(columns=['bin_id', 'label'])
    feature_names = X.columns.tolist()
    y = combined_features['label'].values

    # train a simple classifier (e.g., logistic regression)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit a random forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    importances = rf_model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    print("Feature Importances:")
    print(feature_importance.sort_values(by='importance', ascending=False))

    # Evaluate the random forest model
    rf_y_pred = rf_model.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_y_pred))
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_y_pred))

    # save the model
    model_path = os.path.join(output_dir, 'ifcb_focus_model.pkl')
    joblib.dump(rf_model, model_path)

    # now classify all images
    all_features = X
    all_bin_ids = combined_features['bin_id'].values

    # Predict using the trained model
    all_predictions = rf_model.predict(all_features)

    # Create a DataFrame with results
    results_df = pd.DataFrame({
        'bin_id': all_bin_ids,
        'prediction': all_predictions
    })

    scores = results_df.groupby('bin_id')['prediction'].mean().reset_index()
    print(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train IFCB focus classification model')
    parser.add_argument('good_dir', help='Directory containing good (in-focus) images')
    parser.add_argument('bad_dir', help='Directory containing bad (out-of-focus) images')
    parser.add_argument('blurred_dir', help='Directory containing blurred images')
    parser.add_argument('output_dir', help='Directory to save the trained model')
    
    args = parser.parse_args()
    train_model(args.good_dir, args.bad_dir, args.blurred_dir, args.output_dir)
