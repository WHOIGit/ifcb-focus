"""
Base training functions for IFCB focus detection.

This module provides the base training pipeline for creating a focus detection
model from labeled image data.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from ifcb_focus.features import preprocess_image
from .utils import list_images, load_images, skip_image, compute_features


def train_base_model(data_dir=None, model_path='ifcb_focus_model.pkl', force_recompute=False):
    """
    Train a base focus detection model from labeled data.

    This function expects a directory structure with 'good', 'bad', and optionally
    'blurred' subdirectories containing labeled images.

    Args:
        data_dir (str, optional): Base directory containing labeled images.
            If None, uses environment variable IFCB_DATA_DIR or defaults to './data'.
        model_path (str): Path where the trained model will be saved.
            Default is 'ifcb_focus_model.pkl'.
        force_recompute (bool): If True, recompute features even if cached.
            Default is False.

    Returns:
        RandomForestClassifier: Trained model.

    Example:
        >>> from ifcb_focus.training import train_base_model
        >>> model = train_base_model('/path/to/labeled/data')
        >>> # Model is saved to 'ifcb_focus_model.pkl'
    """
    if data_dir is None:
        data_dir = os.environ.get('IFCB_DATA_DIR', './data')

    def do_all_compute_features():
        good_dir = os.path.join(data_dir, 'good')
        bad_dir = os.path.join(data_dir, 'bad')
        blurred_dir = os.path.join(data_dir, 'blurred')

        def batch_compute_features(image_list):
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

        good_features, good_bin_ids, feature_names = batch_compute_features(list_images(good_dir))
        bad_features, bad_bin_ids, _ = batch_compute_features(list_images(bad_dir))

        # Process blurred images if directory exists
        blurred_features = []
        blurred_bin_ids = []
        if os.path.exists(blurred_dir):
            blurred_features, blurred_bin_ids, _ = batch_compute_features(list_images(blurred_dir))

        bin_ids = good_bin_ids + bad_bin_ids + blurred_bin_ids

        # add blurred features to bad features
        bad_features.extend(blurred_features)

        # Create a DataFrame for the combined features
        good_features = pd.DataFrame(np.array(good_features), columns=feature_names)
        bad_features = pd.DataFrame(np.array(bad_features), columns=feature_names)

        y = np.concatenate([np.ones(len(good_features)), np.zeros(len(bad_features))])

        combined_features = pd.concat([good_features, bad_features], ignore_index=True)
        combined_features['bin_id'] = bin_ids
        combined_features['label'] = y.astype(int)
        combined_features.to_csv('features.csv', index=False)

    if not os.path.exists('features.csv') or force_recompute:
        do_all_compute_features()

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
    joblib.dump(rf_model, model_path)
    print(f"Model saved to {model_path}")

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
    print("\nBin-level scores:")
    print(scores)

    return rf_model
