import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('features.csv')
df['label'] = df['label'].astype(int)
features = df.drop(columns=['bin_id', 'label'])
feature_names = features.columns.tolist()

# Compute correlation matrix
correlation_matrix = features.corr()

# Plot the heatmap
"""
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
"""

def get_strongest_correlations(corr_matrix, threshold=0.7):
    corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
    corr_pairs['Abs Correlation'] = corr_pairs['Correlation'].abs()
    return corr_pairs.sort_values('Abs Correlation', ascending=False).query('`Abs Correlation` >= @threshold')

strong_corrs = get_strongest_correlations(correlation_matrix, threshold=0.7)
print(strong_corrs)

# now "knock out" each of the features with strong correlations
knockout_features = strong_corrs['Feature 1'].unique()

print("Knockout Features:")
print(knockout_features)

classifiers = {}

for knockout_feature in knockout_features:

    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(features[[knockout_feature]], df['label'])

    classifiers[knockout_feature] = clf

    threshold = clf.tree_.threshold[0]

    print(f"Threshold for {knockout_feature}: {threshold}")

    # now generate pseudo-labels based on this threshold
    #pseudo_labels = (features[knockout_feature] > threshold).astype(int)
    #features[f'{knockout_feature}_label'] = pseudo_labels

feature_name_to_index = {name: i for i, name in enumerate(feature_names)}
feature_indices = []
for feature_name, clf in classifiers.items():
    feature_index = feature_name_to_index[feature_name]
    feature_indices.append([feature_index])

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class KnockoutVotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models, feature_indices):
        self.models = models
        self.feature_indices = feature_indices
        print(feature_indices)

    def predict_proba(self, X):
        preds = []
        for model, idxs in zip(self.models, self.feature_indices):
            probs = model.predict_proba(X[:, idxs])
            preds.append(probs)
        avg_probs = np.mean(preds, axis=0)
        return avg_probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

pseudo_labeler = KnockoutVotingEnsemble(
    models=list(classifiers.values()),
    feature_indices=feature_indices
)

df['pseudo_label'] = pseudo_labeler.predict(features.values)
df['pseudo_prob'] = pseudo_labeler.predict_proba(features.values)[:, 1]
# now compute stats comparing pseudo-labels to actual labels
df['correct'] = df['pseudo_label'] == df['label']
accuracy = df['correct'].mean()
print(f"Accuracy of pseudo-labeler: {accuracy:.2f}")

df.to_csv('features_with_pseudo_labels.csv', index=False)

# now save the pseudo-labeler model
import joblib
model_path = 'pseudo_labeler_model.pkl'
joblib.dump(pseudo_labeler, model_path)

print(f"Pseudo-labeler model saved to {model_path}")