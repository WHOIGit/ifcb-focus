
import joblib
import os

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score

from feature_extraction import rank_feature_speed, feature_names
from train_teacher_model import features_and_labels

DATA_DIR = '/Users/jfutrelle/Data/ifcb-data/focus'

def feature_ranking(model):
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

def train_student_model():
    X_train, y_train, X_val, y_val, feature_names = features_and_labels()

    # Load the teacher model
    teacher_model = joblib.load(os.path.join(DATA_DIR, 'teacher_model.pkl'))

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
    model_path = os.path.join(DATA_DIR, 'student_model.pkl')
    joblib.dump(model, model_path)

if __name__ == "__main__":
    teacher_model = joblib.load(os.path.join(DATA_DIR, 'teacher_model.pkl'))
    candidate_ranking = feature_ranking(teacher_model)
    selected_features = candidate_ranking[:4]  # Select top 4 features
    print("Selected features for student model:", selected_features)
