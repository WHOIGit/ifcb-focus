import os
import joblib

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd

DATA_DIR = os.environ.get('IFCB_DATA_DIR', './data')

training_set = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
validation_set = pd.read_csv(os.path.join(DATA_DIR, 'validation.csv'))

def assemble_features(bin_set):
    """
    Assemble features for a given bin set.
    """
    bin_pids = bin_set['pid'].tolist()
    labels = bin_set['true_label'].tolist()
    features = []
    for pid, label in zip(bin_pids, labels):
        features_path = os.path.join(DATA_DIR, 'features', f"{pid}_features.csv")
        bin_features = pd.read_csv(features_path)
        bin_features.pop('pid')
        bin_features['label'] = label
        features.append(bin_features)

    return pd.concat(features, ignore_index=True).values, bin_features.columns[:-1].tolist()

def features_and_labels():
    train, feature_names = assemble_features(training_set)
    val, _ = assemble_features(validation_set)

    X_train = train[:, :-1]  # All columns except the last one
    y_train = train[:, -1]   # The last column is the label

    X_val = val[:, :-1]      # All columns except the last one
    y_val = val[:, -1]       # The last column is the label

    return X_train, y_train, X_val, y_val, feature_names

def train_random_forest():

    X_train, y_train, _, _, feature_names = features_and_labels()

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
    model_path = os.path.join(DATA_DIR, 'teacher_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model

def validate_model(model):
    _, _, X_val, y_val, _ = features_and_labels()

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, feature_names = features_and_labels()
    model = joblib.load(os.path.join(DATA_DIR, 'teacher_model.pkl'))
    validate_model(model)
