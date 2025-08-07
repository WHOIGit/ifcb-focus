
import argparse
import joblib
import os

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

from feature_extraction import rank_feature_speed, feature_names
from train_teacher_model import features_and_labels

from score_bins import STUDENT_MODEL_FEATURES

def create_student_model(model_type):
    """Create a student model based on the specified type.
    
    Args:
        model_type (str): Type of model to create. Options are:
            - 'randomforest': RandomForestRegressor
            - 'logistic': Logistic Regression with L1 regularization
            - 'decision_stump': Decision Tree with max_depth=1
            - 'mlp_1layer': MLPClassifier with 1 hidden layer
            - 'mlp_10hidden': MLPRegressor with 10 hidden neurons
            - 'xgboost': XGBoost with minimal parameters
            
    Returns:
        sklearn model: Configured model instance
    """
    if model_type == 'randomforest':
        return RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10,
            min_samples_split=4, min_samples_leaf=2, max_leaf_nodes=64
        )
    elif model_type == 'logistic':
        return LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    elif model_type == 'decision_stump':
        return DecisionTreeRegressor(max_depth=1, random_state=42)
    elif model_type == 'mlp_1layer':
        return MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=1000)
    elif model_type == 'mlp_10hidden':
        return MLPRegressor(hidden_layer_sizes=(10,), random_state=42, max_iter=1000)
    elif model_type == 'xgboost':
        return xgb.XGBRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1, 
            random_state=42, verbosity=0
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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

def search_for_features(teacher_model_path, student_model_path, model_type='randomforest'):
    """Search for optimal number of features for student model.
    
    Tests different numbers of top-ranked features to find the optimal subset
    for the student model, using knowledge distillation from the teacher model.
    
    Args:
        teacher_model_path (str): Path to the trained teacher model file.
        student_model_path (str): Path where the final student model will be saved.
        model_type (str): Type of student model to use.
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

        # train a model to predict the teacher's probabilities
        model = create_student_model(model_type)
        
        # Handle classification models differently
        if model_type in ['logistic', 'mlp_1layer']:
            # For classification models, use discrete labels
            model.fit(X_train_ranked, y_train)
            y_val_pred_proba = model.predict_proba(X_val_ranked)[:, 1]
        else:
            # For regression models, use teacher probabilities
            model.fit(X_train_ranked, y_train_proba)
            y_val_pred_proba = model.predict(X_val_ranked)

        # Evaluate the model on the validation set
        accuracy = accuracy_score(y_val, (y_val_pred_proba > 0.5).astype(int))

        print(f"{n},{accuracy:.4f}")

    # save the model
    joblib.dump(model, student_model_path)

def train_student_model(teacher_model_path, student_model_path, model_type='randomforest'):
    """Train a student model using knowledge distillation from teacher model.
    
    Creates a smaller, faster student model that learns to mimic the teacher's
    predictions using only a subset of the most important and fastest features.
    
    Args:
        teacher_model_path (str): Path to the trained teacher model file.
        student_model_path (str): Path where the student model will be saved.
        model_type (str): Type of student model to use.
    """
    X_train, y_train, X_val, y_val, feature_names = features_and_labels()

    # Load the teacher model
    teacher_model = joblib.load(teacher_model_path)
    
    y_train_proba = teacher_model.predict_proba(X_train)[:, 1]
    
    # top n features based on the ranking
    ranked_indices = [feature_names.index(f) for f in STUDENT_MODEL_FEATURES]
    X_train_ranked = X_train[:, ranked_indices]
    X_val_ranked = X_val[:, ranked_indices]

    # train a model to predict the teacher's probabilities
    model = create_student_model(model_type)
    
    # Handle classification models differently
    if model_type in ['logistic', 'mlp_1layer']:
        # For classification models, use discrete labels
        model.fit(X_train_ranked, y_train)
        y_val_pred_proba = model.predict_proba(X_val_ranked)[:, 1]
    else:
        # For regression models, use teacher probabilities
        model.fit(X_train_ranked, y_train_proba)
        y_val_pred_proba = model.predict(X_val_ranked)

    # Evaluate the model on the validation set
    accuracy = accuracy_score(y_val, (y_val_pred_proba > 0.5).astype(int))
    # print report
    print("Validation Classification Report:")
    print(classification_report(y_val, (y_val_pred_proba > 0.5).astype(int)))
    print("Validation Accuracy:", accuracy)
    
    # Print feature importances if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({'feature': STUDENT_MODEL_FEATURES, 'importance': importances})
        print("Feature Importances:")
        print(feature_importance.sort_values(by='importance', ascending=False))
    elif hasattr(model, 'coef_'):
        coefficients = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
        feature_importance = pd.DataFrame({'feature': STUDENT_MODEL_FEATURES, 'coefficient': coefficients})
        print("Model Coefficients:")
        print(feature_importance.sort_values(by='coefficient', key=abs, ascending=False))
    else:
        print("Model does not provide feature importance or coefficients")

    # save the model
    joblib.dump(model, student_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train student model from teacher model')
    parser.add_argument('teacher_model_path', help='Path to the teacher model file')
    parser.add_argument('student_model_path', help='Path where the student model will be saved')
    parser.add_argument('--model_type', 
                        choices=['randomforest', 'logistic', 'decision_stump', 'mlp_1layer', 'mlp_10hidden', 'xgboost'],
                        default='randomforest',
                        help='Type of student model to train (default: randomforest)')
    parser.add_argument('--search_features', action='store_true',
                        help='Search for optimal number of features instead of using predefined features')
    
    args = parser.parse_args()
    
    if args.search_features:
        search_for_features(args.teacher_model_path, args.student_model_path, args.model_type)
    else:
        train_student_model(args.teacher_model_path, args.student_model_path, args.model_type)
