import joblib

model = joblib.load('ifcb_focus_model.pkl')
print("Model loaded successfully.")

import pandas as pd
import numpy as np

features_with_pseudo_labels = pd.read_csv('features_with_pseudo_labels.csv')
print("Features with pseudo-labels loaded successfully.")

# Assuming the model expects the same features as used during training
features = features_with_pseudo_labels.drop(columns=['bin_id', 'label', 'pseudo_label', 'pseudo_prob', 'correct'])
y = model.predict(features)
y_prob = model.predict_proba(features)[:, 1]  # Probability of being a 'good' ROI

features_with_pseudo_labels['rf_label'] = y.astype(int)
features_with_pseudo_labels['rf_prob'] = y_prob

features_with_pseudo_labels.to_csv('features_with_all_labels.csv', index=False)