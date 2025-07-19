# IFCB Focus Detection

This repository provides a machine learning model for detecting blurry images captured by Imaging FlowCytobot (IFCB) instruments. The model is designed to assist in identifying and filtering out blurry images, improving the quality of data used in downstream analysis.

## Features

- **Feature Extractor**: Extracts informative features from IFCB images to characterize their focus quality.
- **Random Forest Classifier**: A robust classifier trained to distinguish between blurry and non-blurry images.
- **Pseudo-Labeler**: Generates pseudo-labels for unlabeled images using knockout features to expand training datasets.
- **Augmentation Pipeline**: A pipeline for blurring images to augment training sets, enhancing the model's generalization.

## Usage

1. **Feature Extraction**: Use the feature extraction module to process IFCB images and generate feature sets.
2. **Training**: Train the Random Forest classifier using the extracted features and labels.
3. **Inference**: Apply the trained model to classify new IFCB images as blurry or non-blurry.
4. **Pseudo-Labeling**: Use the pseudo-labeler to generate additional labeled data from unlabeled datasets.
5. **Data Augmentation**: Leverage the augmentation pipeline to create diverse training datasets with simulated blurry images.

## TODOs

- **Feature Pruning**: Optimize the feature set by removing less informative features to improve computational efficiency.
- **Sample-Bin-Level Scoring**: Develop a pipeline for scoring bins of images to assess focus quality at a higher level.
- **Enhanced Training Set Construction**: Use pseudo-labeling and data augmentation to build larger and more diverse training sets.

### Running the Model

1. Prepare your IFCB images and extract features using the feature extractor.
2. Train the Random Forest classifier with labeled data.
3. Run the inference pipeline to classify new images.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Much of the code is AI generated.
