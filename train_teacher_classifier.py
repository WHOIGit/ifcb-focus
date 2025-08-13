import os
import argparse
import joblib

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def assemble_features(bin_set, features_dir):
    """Assemble features for a given bin set.
    
    Loads feature CSV files for each PID in the bin set and combines them
    into a single feature matrix with labels.
    
    Args:
        bin_set (pd.DataFrame): DataFrame containing 'pid' and 'true_label' columns.
        features_dir (str): Directory containing individual feature CSV files.
        
    Returns:
        tuple: A tuple containing:
            - features_array (np.ndarray): Combined feature matrix with labels.
            - feature_names (list): List of feature column names.
    """
    bin_pids = bin_set['pid'].tolist()
    labels = bin_set['true_label'].tolist()
    features = []
    for pid, label in zip(bin_pids, labels):
        features_path = os.path.join(features_dir, f"{pid}_features.csv")
        bin_features = pd.read_csv(features_path)
        bin_features.pop('pid')
        bin_features['label'] = label
        features.append(bin_features)

    return pd.concat(features, ignore_index=True).values, bin_features.columns[:-1].tolist()

def features_and_labels(training_set_path, validation_set_path, features_dir):
    """Load and prepare training and validation features and labels.
    
    Reads training and validation CSV files, assembles features from individual
    feature files, and separates features from labels for model training.
    
    Args:
        training_set_path (str): Path to training set CSV file.
        validation_set_path (str): Path to validation set CSV file.
        features_dir (str): Directory containing feature CSV files.
        
    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray): Training features.
            - y_train (np.ndarray): Training labels.
            - X_val (np.ndarray): Validation features.
            - y_val (np.ndarray): Validation labels.
            - feature_names (list): List of feature names.
    """
    training_set = pd.read_csv(training_set_path)
    validation_set = pd.read_csv(validation_set_path)
    
    train, feature_names = assemble_features(training_set, features_dir)
    val, _ = assemble_features(validation_set, features_dir)

    X_train = train[:, :-1]  # All columns except the last one
    y_train = train[:, -1]   # The last column is the label

    X_val = val[:, :-1]      # All columns except the last one
    y_val = val[:, -1]       # The last column is the label

    return X_train, y_train, X_val, y_val, feature_names

def train_and_validate_model(training_set_path, validation_set_path, features_dir, output_path):
    """Train and validate a Random Forest teacher model for IFCB focus classification.
    
    Loads training and validation data, trains a Random Forest classifier, 
    evaluates performance on both test split and validation set, and saves the model.
    
    Args:
        training_set_path (str): Path to training set CSV file.
        validation_set_path (str): Path to validation set CSV file.
        features_dir (str): Directory containing feature CSV files.
        output_path (str): Path where the trained model will be saved.
        
    Returns:
        RandomForestClassifier: The trained Random Forest model.
    """
    # Load all data once
    X_train, y_train, X_val, y_val, feature_names = features_and_labels(training_set_path, validation_set_path, features_dir)

    # Split training data into training and test sets for initial evaluation
    X_train_split, X_test, y_train_split, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_split, y_train_split)

    # Evaluate the model on the test split
    y_pred = model.predict(X_test)
    print("Test Split Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Test Split Accuracy:", accuracy_score(y_test, y_pred))

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    print("\nValidation Set Classification Report:")
    print(classification_report(y_val, y_val_pred))
    print("Validation Set Accuracy:", accuracy_score(y_val, y_val_pred))

    # Show feature importances
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    print("\nFeature Importances:")
    print(feature_importance.sort_values(by='importance', ascending=False))

    # Save the model
    joblib.dump(model, output_path)
    print(f"\nModel saved to {output_path}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train teacher model for IFCB focus classification')
    parser.add_argument('training_set_path', help='Path to the training set CSV file')
    parser.add_argument('validation_set_path', help='Path to the validation set CSV file')
    parser.add_argument('features_dir', help='Directory containing feature CSV files')
    parser.add_argument('output_path', help='Path where the teacher model will be saved')
    
    args = parser.parse_args()
    model = train_and_validate_model(args.training_set_path, args.validation_set_path, args.features_dir, args.output_path)
