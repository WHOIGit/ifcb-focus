import os
import argparse
import joblib

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def assemble_features(bin_set, features_dir):
    """
    Assemble features for a given bin set.
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
    training_set = pd.read_csv(training_set_path)
    validation_set = pd.read_csv(validation_set_path)
    
    train, feature_names = assemble_features(training_set, features_dir)
    val, _ = assemble_features(validation_set, features_dir)

    X_train = train[:, :-1]  # All columns except the last one
    y_train = train[:, -1]   # The last column is the label

    X_val = val[:, :-1]      # All columns except the last one
    y_val = val[:, -1]       # The last column is the label

    return X_train, y_train, X_val, y_val, feature_names

def train_random_forest(training_set_path, validation_set_path, features_dir, output_path):
    X_train, y_train, _, _, feature_names = features_and_labels(training_set_path, validation_set_path, features_dir)

    # Split training data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    print("Feature Importances:")
    print(feature_importance.sort_values(by='importance', ascending=False))

    # Save the model
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")

    return model

def validate_model(model, training_set_path, validation_set_path, features_dir):
    _, _, X_val, y_val, _ = features_and_labels(training_set_path, validation_set_path, features_dir)

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train teacher model for IFCB focus classification')
    parser.add_argument('training_set_path', help='Path to the training set CSV file')
    parser.add_argument('validation_set_path', help='Path to the validation set CSV file')
    parser.add_argument('features_dir', help='Directory containing feature CSV files')
    parser.add_argument('output_path', help='Path where the teacher model will be saved')
    
    args = parser.parse_args()
    model = train_random_forest(args.training_set_path, args.validation_set_path, args.features_dir, args.output_path)
    validate_model(model, args.training_set_path, args.validation_set_path, args.features_dir)
