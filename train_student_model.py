
import argparse
import joblib
import os

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score

from feature_extraction import rank_feature_speed, feature_names
from train_teacher_model import features_and_labels

from score_bins import STUDENT_MODEL_FEATURES

def feature_ranking(model):
    """Rank features by combining importance and speed metrics.
    
    Creates a composite ranking of features based on both their importance in the
    model and their computational speed, allowing selection of features that are
    both predictive and fast to compute.
    
    Args:
        model: Trained model with feature_importances_ attribute.
        
    Returns:
        list: List of feature names ranked by combined importance and speed score.
    """
    speed = rank_feature_speed()
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names(), 'importance': importances})
    # Sort features by importance
    importance = feature_importance.sort_values(by='importance', ascending=False)['feature'].tolist()

    feature_df = pd.DataFrame({
        'feature': importance,
        'speed_rank': [speed.index(f) for f in importance]
    })
    importance_rank = feature_df.index.tolist()
    feature_df['importance_rank'] = importance_rank
    feature_df['rank'] = feature_df['importance_rank'] + feature_df['speed_rank']
    feature_df = feature_df.sort_values(by='rank', ascending=True)

    candidate_ranking = feature_df.sort_values(by='rank', ascending=True)['feature'].tolist()

    return candidate_ranking

def search_for_features(teacher_model_path, student_model_path):
    """Search for optimal number of features for student model.
    
    Tests different numbers of top-ranked features to find the optimal subset
    for the student model, using knowledge distillation from the teacher model.
    
    Args:
        teacher_model_path (str): Path to the trained teacher model file.
        student_model_path (str): Path where the final student model will be saved.
    """
    X_train, y_train, X_val, y_val, feature_names = features_and_labels()

    # Load the teacher model
    teacher_model = joblib.load(teacher_model_path)

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
    joblib.dump(model, student_model_path)

def train_student_model(teacher_model_path, student_model_path):
    """Train a student model using knowledge distillation from teacher model.
    
    Creates a smaller, faster student model that learns to mimic the teacher's
    predictions using only a subset of the most important and fastest features.
    
    Args:
        teacher_model_path (str): Path to the trained teacher model file.
        student_model_path (str): Path where the student model will be saved.
    """
    X_train, y_train, X_val, y_val, feature_names = features_and_labels()

    # Load the teacher model
    teacher_model = joblib.load(teacher_model_path)
    
    y_train_proba = teacher_model.predict_proba(X_train)[:, 1]
    
    # top n features based on the ranking
    ranked_indices = [feature_names.index(f) for f in STUDENT_MODEL_FEATURES]
    X_train_ranked = X_train[:, ranked_indices]
    X_val_ranked = X_val[:, ranked_indices]

    # train a regressor to predict the teacher's probabilities
    model = RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=10,
        min_samples_split=4, min_samples_leaf=2, max_leaf_nodes=64
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
    feature_importance = pd.DataFrame({'feature': STUDENT_MODEL_FEATURES, 'importance': importances})
    print("Feature Importances:")
    print(feature_importance.sort_values(by='importance', ascending=False))

    # save the model
    joblib.dump(model, student_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train student model from teacher model')
    parser.add_argument('teacher_model_path', help='Path to the teacher model file')
    parser.add_argument('student_model_path', help='Path where the student model will be saved')
    
    args = parser.parse_args()
    train_student_model(args.teacher_model_path, args.student_model_path)
