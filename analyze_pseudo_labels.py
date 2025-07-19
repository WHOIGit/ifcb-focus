import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv('features_with_all_labels.csv')
y_prob = df['pseudo_prob'].values
y_true= df['label'].values
y_pseudo = df['pseudo_label'].values

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
class_report = classification_report(y_true, y_pseudo, target_names=["Bad", "Good"], output_dict=True)

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Label Distribution
axes[0].bar(["True Bad", "True Good"], true_dist, alpha=0.7, label="True")
axes[0].bar(["Pseudo Bad", "Pseudo Good"], pseudo_dist, alpha=0.7, label="Pseudo")
axes[0].set_title("Label Distribution")
axes[0].legend()

# 2. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_xlabel("Pseudo Label")
axes[1].set_ylabel("True Label")
axes[1].set_title("Confusion Matrix")

# 3. Class-wise Accuracy
axes[2].bar(["Bad", "Good"], [acc_bad, acc_good], color=["#e377c2", "#1f77b4"])
axes[2].set_ylim(0, 1)
axes[2].set_title("Class-Conditional Accuracy")

plt.tight_layout()
plt.show()

# Print report
print("Confusion Matrix Details:")
for label, count in zip(labels, flat_cm):
    print(f"{label}: {count}")

print("\nClassification Report:")
for cls in ["Bad", "Good"]:
    metrics = class_report[cls]
    print(f"{cls} - Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1: {metrics['f1-score']:.2f}")

from sklearn.metrics import precision_score, recall_score, f1_score

def interpret_bias(y_true, y_pseudo, threshold_f1_gap=0.15, min_f1=0.7):
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
        print("\n❌ Unacceptable: At least one class has low F1 (< {:.2f}).".format(min_f1))
        return "unacceptable"
    elif f1_gap > threshold_f1_gap:
        print("\n⚠️ Potential Bias: Large F1 gap between classes (> {:.2f}).".format(threshold_f1_gap))
        return "borderline"
    else:
        print("\n✅ Acceptable: Balanced performance across classes.")
        return "acceptable"

# Run interpretation on current pseudo-labels
interpret_bias(y_true, y_pseudo)
