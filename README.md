# IFCB Focus Detection

This repository provides a machine learning framework for detecting blurry images captured by Imaging FlowCytobot (IFCB) instruments. The framework leverages a teacher-student pseudo-labeling approach and ensemble learning to improve the quality of data used in downstream analysis.

## Features

- **Feature Extractor**: Extracts informative features from IFCB images to characterize their focus quality.
- **Teacher-Student Pseudo-Labeling**: Uses a teacher model to generate pseudo-labels for unlabeled images, enabling the student model to learn from a larger dataset.
- **Knockout Voting Ensemble**: Implements an ensemble of models trained on subsets of features, improving robustness and generalization.
- **Augmentation Pipeline**: A pipeline for blurring images to augment training sets, enhancing the model's generalization.

## Installation

```bash
pip install -e .
```

## Usage

1. **Feature Extraction**: Use the feature extraction module to process IFCB images and generate feature sets.
2. **Teacher-Student Training**: Train the teacher model with labeled data, generate pseudo-labels, and train the student model with the expanded dataset.
3. **Inference**: Apply the trained ensemble model to classify new IFCB images as blurry or non-blurry.
4. **Pseudo-Labeling**: Use the teacher model to generate additional labeled data from unlabeled datasets.
5. **Data Augmentation**: Leverage the augmentation pipeline to create diverse training datasets with simulated blurry images.

### Running Inference

```python
from ifcb import DataDirectory
import joblib
from ifcb_focus import score_bin

# Load bin data
dd = DataDirectory('/path/to/raw_data')
bin_data = dd['D20130823T160901_IFCB010']

# Load model and score
model = joblib.load('slim_student_model.pkl')
score = score_bin(bin_data, model)

print(f'Bin focus score: {score:.4f}')
```

### Running the Model

1. Prepare your IFCB images and extract features using the feature extractor.
2. Train the teacher model with labeled data.
3. Generate pseudo-labels using the teacher model and train the student model.
4. Run the inference pipeline to classify new images using the ensemble model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Much of the code is AI generated.
