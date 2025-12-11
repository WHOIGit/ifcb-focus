"""
Student model training for IFCB focus detection.

This module provides functions for training a lightweight student model using
feature distillation from a teacher model.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score

from ifcb_focus.features import rank_feature_speed, feature_names as get_feature_names
from ifcb_focus.features.constants import STUDENT_MODEL_FEATURES
from .teacher import features_and_labels


def feature_ranking(model):
    """
    Rank features by a combination of importance and computation speed.

    Args:
        model: Trained teacher model with feature_importances_ attribute.

    Returns:
        list: Feature names ranked by combined importance and speed score.

    Example:
        >>> import joblib
        >>> from ifcb_focus.training import feature_ranking
        >>> teacher_model = joblib.load('teacher_model.pkl')
        >>> ranked_features = feature_ranking(teacher_model)
        >>> print(f'Top 5 features: {ranked_features[:5]}')
    """
    speed = rank_feature_speed()
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': get_feature_names(),
        'importance': importances
    })
    # Sort features by importance
    importance = feature_importance.sort_values(
        by='importance', ascending=False
    )['feature'].tolist()

    feature_df = pd.DataFrame({
        'feature': importance,
        'speed_rank': [speed.index(f) for f in importance]
    })
    importance_rank = feature_df.index.tolist()
    feature_df['importance_rank'] = importance_rank
    feature_df['rank'] = feature_df['importance_rank'] + feature_df['speed_rank']
    feature_df = feature_df.sort_values(by='rank', ascending=True)

    candidate_ranking = feature_df.sort_values(
        by='rank', ascending=True
    )['feature'].tolist()

    return candidate_ranking


def search_for_features(data_dir=None):
    """
    Search for optimal number of features for student model.

    Trains student models with 1 to 4 features and evaluates their accuracy.

    Args:
        data_dir (str, optional): Base data directory. If None, uses
            environment variable IFCB_DATA_DIR or defaults to './data'.

    Example:
        >>> from ifcb_focus.training import search_for_features
        >>> search_for_features()
        n_features,accuracy
        1,0.8234
        2,0.8567
        3,0.8789
        4,0.8891
    """
    if data_dir is None:
        data_dir = os.environ.get('IFCB_DATA_DIR', './data')

    X_train, y_train, X_val, y_val, feature_names = features_and_labels(data_dir)

    # Load the teacher model
    teacher_model = joblib.load(os.path.join(data_dir, 'teacher_model.pkl'))

    # Get the candidate ranking from the teacher model
    candidate_ranking = feature_ranking(teacher_model)

    y_train_proba = teacher_model.predict_proba(X_train)[:, 1]

    # top n features based on the ranking
    print('n_features,accuracy')
    for n in range(1, 5):
        ranked_indices = [feature_names.index(f) for f in candidate_ranking[:n]]
        X_train_ranked = X_train[:, ranked_indices]
        X_val_ranked = X_val[:, ranked_indices]

        # train a regressor to predict the teacher's probabilities
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_ranked, y_train_proba)

        # Evaluate the model on the validation set
        y_val_pred = model.predict(X_val_ranked)
        accuracy = accuracy_score(y_val, (y_val_pred > 0.5).astype(int))

        print(f"{n},{accuracy:.4f}")

    # save the model
    model_path = os.path.join(data_dir, 'student_model.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")


def train_student_model(data_dir=None, model_path=None):
    """
    Train a student model using knowledge distillation from teacher model.

    The student model uses only the features defined in STUDENT_MODEL_FEATURES
    and is trained to predict the teacher model's probability outputs.

    Args:
        data_dir (str, optional): Base data directory. If None, uses
            environment variable IFCB_DATA_DIR or defaults to './data'.
        model_path (str, optional): Path to save the trained model. If None,
            saves to '{data_dir}/slim_student_model.pkl'.

    Returns:
        RandomForestRegressor: Trained student model.

    Example:
        >>> from ifcb_focus.training import train_student_model
        >>> student_model = train_student_model()
    """
    if data_dir is None:
        data_dir = os.environ.get('IFCB_DATA_DIR', './data')

    if model_path is None:
        model_path = os.path.join(data_dir, 'slim_student_model.pkl')

    X_train, y_train, X_val, y_val, feature_names = features_and_labels(data_dir)

    # Load the teacher model
    teacher_model = joblib.load(os.path.join(data_dir, 'teacher_model.pkl'))

    y_train_proba = teacher_model.predict_proba(X_train)[:, 1]

    # Select indices for student model features
    ranked_indices = [feature_names.index(f) for f in STUDENT_MODEL_FEATURES]
    X_train_ranked = X_train[:, ranked_indices]
    X_val_ranked = X_val[:, ranked_indices]

    # Train a regressor to predict the teacher's probabilities
    # Using optimized hyperparameters for a slim model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        max_leaf_nodes=64
    )
    model.fit(X_train_ranked, y_train_proba)

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val_ranked)
    accuracy = accuracy_score(y_val, (y_val_pred > 0.5).astype(int))

    # print report
    print("Validation Classification Report:")
    print(classification_report(y_val, (y_val_pred > 0.5).astype(int)))
    print("Validation Accuracy:", accuracy)

    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': STUDENT_MODEL_FEATURES,
        'importance': importances
    })
    print("\nFeature Importances:")
    print(feature_importance.sort_values(by='importance', ascending=False))

    # save the model
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    return model
