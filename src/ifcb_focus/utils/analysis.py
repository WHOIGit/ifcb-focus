"""
Analysis utilities for IFCB focus detection.

This module provides utilities for model analysis, evaluation, and feature selection,
including knockout voting ensembles and pseudo-label analysis.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


class KnockoutVotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier that uses knockout voting with individual feature-based models.

    This ensemble trains individual decision tree classifiers on single features
    and combines their predictions through averaging.

    Args:
        models (list): List of trained classifier models.
        feature_indices (list): List of feature index lists for each model.

    Example:
        >>> from ifcb_focus.utils.analysis import KnockoutVotingEnsemble
        >>> # Create individual classifiers for each feature
        >>> classifiers = [DecisionTreeClassifier(max_depth=1) for _ in range(3)]
        >>> feature_indices = [[0], [1], [2]]  # Each model uses one feature
        >>> ensemble = KnockoutVotingEnsemble(classifiers, feature_indices)
    """

    def __init__(self, models, feature_indices):
        self.models = models
        self.feature_indices = feature_indices

    def predict_proba(self, X):
        """
        Predict class probabilities using ensemble voting.

        Args:
            X (ndarray): Feature matrix.

        Returns:
            ndarray: Average class probabilities across all models.
        """
        preds = []
        for model, idxs in zip(self.models, self.feature_indices):
            probs = model.predict_proba(X[:, idxs])
            preds.append(probs)
        avg_probs = np.mean(preds, axis=0)
        return avg_probs

    def predict(self, X):
        """
        Predict class labels using ensemble voting.

        Args:
            X (ndarray): Feature matrix.

        Returns:
            ndarray: Predicted class labels.
        """
        return np.argmax(self.predict_proba(X), axis=1)


def get_strongest_correlations(corr_matrix, threshold=0.7):
    """
    Find feature pairs with correlation above a threshold.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix of features.
        threshold (float): Minimum absolute correlation to include. Default 0.7.

    Returns:
        pd.DataFrame: DataFrame with columns ['Feature 1', 'Feature 2', 'Correlation', 'Abs Correlation']
            sorted by absolute correlation.

    Example:
        >>> import pandas as pd
        >>> from ifcb_focus.utils.analysis import get_strongest_correlations
        >>> features = pd.read_csv('features.csv')
        >>> corr_matrix = features.corr()
        >>> strong_corrs = get_strongest_correlations(corr_matrix, threshold=0.7)
    """
    corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
    corr_pairs['Abs Correlation'] = corr_pairs['Correlation'].abs()
    return corr_pairs.sort_values('Abs Correlation', ascending=False).query('`Abs Correlation` >= @threshold')


def create_knockout_ensemble(features_df, labels, threshold=0.7, max_depth=1):
    """
    Create a knockout voting ensemble based on feature correlations.

    This function:
    1. Finds strongly correlated features
    2. Trains individual classifiers for each knockout feature
    3. Builds an ensemble that combines predictions

    Args:
        features_df (pd.DataFrame): Feature matrix.
        labels (array-like): Target labels.
        threshold (float): Correlation threshold for knockout features. Default 0.7.
        max_depth (int): Maximum depth for decision tree classifiers. Default 1.

    Returns:
        tuple: (ensemble_model, knockout_features, classifiers_dict)
            - ensemble_model: KnockoutVotingEnsemble instance
            - knockout_features: List of feature names used
            - classifiers_dict: Dict mapping feature names to trained classifiers

    Example:
        >>> from ifcb_focus.utils.analysis import create_knockout_ensemble
        >>> import pandas as pd
        >>> features = pd.read_csv('features.csv')
        >>> labels = features['label'].values
        >>> features = features.drop(columns=['bin_id', 'label'])
        >>> ensemble, ko_features, classifiers = create_knockout_ensemble(features, labels)
    """
    # Compute correlation matrix
    correlation_matrix = features_df.corr()

    # Find strongly correlated features
    strong_corrs = get_strongest_correlations(correlation_matrix, threshold=threshold)
    knockout_features = strong_corrs['Feature 1'].unique()

    print(f"Found {len(knockout_features)} knockout features with correlation >= {threshold}")

    # Train individual classifiers
    classifiers = {}
    for knockout_feature in knockout_features:
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(features_df[[knockout_feature]], labels)
        classifiers[knockout_feature] = clf

        if hasattr(clf.tree_, 'threshold') and len(clf.tree_.threshold) > 0:
            threshold_val = clf.tree_.threshold[0]
            print(f"Threshold for {knockout_feature}: {threshold_val}")

    # Build ensemble
    feature_names = features_df.columns.tolist()
    feature_name_to_index = {name: i for i, name in enumerate(feature_names)}
    feature_indices = []
    for feature_name in classifiers.keys():
        feature_index = feature_name_to_index[feature_name]
        feature_indices.append([feature_index])

    ensemble = KnockoutVotingEnsemble(
        models=list(classifiers.values()),
        feature_indices=feature_indices
    )

    return ensemble, knockout_features, classifiers


def interpret_bias(y_true, y_pseudo, threshold_f1_gap=0.15, min_f1=0.7):
    """
    Interpret bias in pseudo-labels by analyzing class-wise performance.

    Args:
        y_true (array-like): True labels.
        y_pseudo (array-like): Pseudo (predicted) labels.
        threshold_f1_gap (float): Maximum acceptable F1 gap between classes. Default 0.15.
        min_f1 (float): Minimum acceptable F1 score for either class. Default 0.7.

    Returns:
        str: Interpretation result: "acceptable", "borderline", or "unacceptable".

    Example:
        >>> from ifcb_focus.utils.analysis import interpret_bias
        >>> y_true = [0, 0, 1, 1, 1]
        >>> y_pseudo = [0, 1, 1, 1, 0]
        >>> result = interpret_bias(y_true, y_pseudo)
        Interpretation Summary:
        F1 Score (Bad):  0.67
        F1 Score (Good): 0.67
        ...
    """
    report = classification_report(y_true, y_pseudo, target_names=["Bad", "Good"], output_dict=True)
    f1_bad = report["Bad"]["f1-score"]
    f1_good = report["Good"]["f1-score"]

    precision_bad = report["Bad"]["precision"]
    recall_bad = report["Bad"]["recall"]
    precision_good = report["Good"]["precision"]
    recall_good = report["Good"]["recall"]

    f1_gap = abs(f1_good - f1_bad)

    print("Interpretation Summary:")
    print(f"F1 Score (Bad):  {f1_bad:.2f}")
    print(f"F1 Score (Good): {f1_good:.2f}")
    print(f"F1 Gap: {f1_gap:.2f}")
    print(f"Precision/Recall (Bad): {precision_bad:.2f} / {recall_bad:.2f}")
    print(f"Precision/Recall (Good): {precision_good:.2f} / {recall_good:.2f}")

    if min(f1_bad, f1_good) < min_f1:
        print(f"\n❌ Unacceptable: At least one class has low F1 (< {min_f1:.2f}).")
        return "unacceptable"
    elif f1_gap > threshold_f1_gap:
        print(f"\n⚠️ Potential Bias: Large F1 gap between classes (> {threshold_f1_gap:.2f}).")
        return "borderline"
    else:
        print("\n✅ Acceptable: Balanced performance across classes.")
        return "acceptable"


def analyze_pseudo_labels(y_true, y_pseudo, y_prob=None):
    """
    Comprehensive analysis of pseudo-labels vs true labels.

    Args:
        y_true (array-like): True labels.
        y_pseudo (array-like): Pseudo (predicted) labels.
        y_prob (array-like, optional): Predicted probabilities for positive class.

    Returns:
        dict: Analysis results including confusion matrix, accuracy, and class-wise metrics.

    Example:
        >>> from ifcb_focus.utils.analysis import analyze_pseudo_labels
        >>> import pandas as pd
        >>> df = pd.read_csv('features_with_pseudo_labels.csv')
        >>> results = analyze_pseudo_labels(df['label'], df['pseudo_label'], df['pseudo_prob'])
    """
    # Compute distributions
    true_dist = np.bincount(y_true)
    pseudo_dist = np.bincount(y_pseudo)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pseudo)
    labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    flat_cm = cm.ravel()

    # Class-conditional accuracy
    good_mask = y_true == 1
    bad_mask = y_true == 0
    acc_good = accuracy_score(y_true[good_mask], y_pseudo[good_mask])
    acc_bad = accuracy_score(y_true[bad_mask], y_pseudo[bad_mask])

    # Classification report
    class_report = classification_report(
        y_true, y_pseudo,
        target_names=["Bad", "Good"],
        output_dict=True
    )

    # Print summary
    print("Confusion Matrix Details:")
    for label, count in zip(labels, flat_cm):
        print(f"{label}: {count}")

    print("\nClassification Report:")
    for cls in ["Bad", "Good"]:
        metrics = class_report[cls]
        print(f"{cls} - Precision: {metrics['precision']:.2f}, "
              f"Recall: {metrics['recall']:.2f}, "
              f"F1: {metrics['f1-score']:.2f}")

    print(f"\nOverall Accuracy: {accuracy_score(y_true, y_pseudo):.2f}")
    print(f"Class-conditional Accuracy (Bad): {acc_bad:.2f}")
    print(f"Class-conditional Accuracy (Good): {acc_good:.2f}")

    return {
        'confusion_matrix': cm,
        'true_distribution': true_dist,
        'pseudo_distribution': pseudo_dist,
        'accuracy': accuracy_score(y_true, y_pseudo),
        'acc_bad': acc_bad,
        'acc_good': acc_good,
        'classification_report': class_report
    }


def evaluate_model_on_pseudo_labels(model, features_df, pseudo_labels_df):
    """
    Evaluate a trained model on features with pseudo-labels.

    Args:
        model: Trained classifier with predict() and predict_proba() methods.
        features_df (pd.DataFrame): Feature matrix (without labels).
        pseudo_labels_df (pd.DataFrame): DataFrame containing pseudo-labels.

    Returns:
        pd.DataFrame: Input DataFrame with added 'rf_label' and 'rf_prob' columns.

    Example:
        >>> import joblib
        >>> from ifcb_focus.utils.analysis import evaluate_model_on_pseudo_labels
        >>> model = joblib.load('ifcb_focus_model.pkl')
        >>> features = pd.read_csv('features_with_pseudo_labels.csv')
        >>> feature_cols = features.drop(columns=['bin_id', 'label', 'pseudo_label', 'pseudo_prob', 'correct'])
        >>> result = evaluate_model_on_pseudo_labels(model, feature_cols, features)
    """
    y = model.predict(features_df)
    y_prob = model.predict_proba(features_df)[:, 1]  # Probability of being a 'good' ROI

    result_df = pseudo_labels_df.copy()
    result_df['rf_label'] = y.astype(int)
    result_df['rf_prob'] = y_prob

    return result_df
