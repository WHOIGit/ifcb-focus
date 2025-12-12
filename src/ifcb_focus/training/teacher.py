"""
Teacher model training for IFCB focus detection.

This module provides functions for training a teacher model using the full
feature set and evaluating its performance.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def assemble_features(bin_set, data_dir=None):
    """
    Assemble features for a given bin set from cached feature files.

    Args:
        bin_set (pd.DataFrame): DataFrame with columns ['pid', 'true_label'].
        data_dir (str, optional): Base data directory. If None, uses
            environment variable IFCB_DATA_DIR or defaults to './data'.

    Returns:
        tuple: (features_array, feature_names)
            - features_array: numpy array of shape (n_samples, n_features + 1)
              where last column is the label
            - feature_names: list of feature names (excluding label)
    """
    if data_dir is None:
        data_dir = os.environ.get('IFCB_DATA_DIR', './data')

    bin_pids = bin_set['pid'].tolist()
    labels = bin_set['true_label'].tolist()
    features = []
    for pid, label in zip(bin_pids, labels):
        features_path = os.path.join(data_dir, 'features', f"{pid}_features.csv")
        bin_features = pd.read_csv(features_path)
        bin_features.pop('pid')
        bin_features['label'] = label
        features.append(bin_features)

    combined = pd.concat(features, ignore_index=True)
    return combined.values, combined.columns[:-1].tolist()


def features_and_labels(data_dir=None):
    """
    Load and prepare training and validation data.

    Args:
        data_dir (str, optional): Base data directory containing train.csv
            and validation.csv. If None, uses environment variable IFCB_DATA_DIR
            or defaults to './data'.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, feature_names)
            - X_train: Training features array
            - y_train: Training labels array
            - X_val: Validation features array
            - y_val: Validation labels array
            - feature_names: List of feature names

    Example:
        >>> from ifcb_focus.training import features_and_labels
        >>> X_train, y_train, X_val, y_val, names = features_and_labels()
    """
    if data_dir is None:
        data_dir = os.environ.get('IFCB_DATA_DIR', './data')

    training_set = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    validation_set = pd.read_csv(os.path.join(data_dir, 'validation.csv'))

    train, feature_names = assemble_features(training_set, data_dir)
    val, _ = assemble_features(validation_set, data_dir)

    X_train = train[:, :-1]  # All columns except the last one
    y_train = train[:, -1]   # The last column is the label

    X_val = val[:, :-1]      # All columns except the last one
    y_val = val[:, -1]       # The last column is the label

    return X_train, y_train, X_val, y_val, feature_names


def train_random_forest(data_dir=None, model_path=None):
    """
    Train a Random Forest teacher model using the full feature set.

    Args:
        data_dir (str, optional): Base data directory. If None, uses
            environment variable IFCB_DATA_DIR or defaults to './data'.
        model_path (str, optional): Path to save the trained model. If None,
            saves to '{data_dir}/teacher_model.pkl'.

    Returns:
        RandomForestClassifier: Trained teacher model.

    Example:
        >>> from ifcb_focus.training import train_random_forest
        >>> teacher_model = train_random_forest()
    """
    if data_dir is None:
        data_dir = os.environ.get('IFCB_DATA_DIR', './data')

    if model_path is None:
        model_path = os.path.join(data_dir, 'teacher_model.pkl')

    X_train, y_train, _, _, feature_names = features_and_labels(data_dir)

    # Split training data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    print("Test Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_pred))

    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    print("\nFeature Importances:")
    print(feature_importance.sort_values(by='importance', ascending=False))

    # Save the model
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    return model


def validate_model(model, data_dir=None):
    """
    Validate a trained model on the validation set.

    Args:
        model: Trained classifier model.
        data_dir (str, optional): Base data directory. If None, uses
            environment variable IFCB_DATA_DIR or defaults to './data'.

    Example:
        >>> import joblib
        >>> from ifcb_focus.training import validate_model
        >>> model = joblib.load('teacher_model.pkl')
        >>> validate_model(model)
    """
    if data_dir is None:
        data_dir = os.environ.get('IFCB_DATA_DIR', './data')

    _, _, X_val, y_val, _ = features_and_labels(data_dir)

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
