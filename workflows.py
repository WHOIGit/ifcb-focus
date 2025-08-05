import os
import argparse

import pandas as pd
from tqdm import tqdm
from ifcb import DataDirectory

from feature_extraction import preprocess_image, extract_features


def feature_extraction(ground_truth_csv, raw_data_dir, output_features_dir):
    validation_set = pd.read_csv(ground_truth_csv)
    dd = DataDirectory(os.path.dirname(raw_data_dir), os.path.basename(raw_data_dir))
    pids = validation_set['pid'].tolist()
    for pid in pids:
        b = dd[pid]
        roi_numbers = list(b.images.keys())
        features = []
        for roi in tqdm(roi_numbers, desc=pid):
            roi_pid = f"{pid}_{roi:05d}"
            image = b.images[roi]
            image = preprocess_image(image)
            image_features = { 'pid': roi_pid }
            image_features.update(extract_features(image))
            features.append(image_features)
        features_df = pd.DataFrame.from_records(features)
        features_df.to_csv(os.path.join(output_features_dir, f"{pid}_features.csv"), float_format='%.5f', index=False)

def split_true_labels(ground_truth_csv, train_csv, val_csv):
    truth = pd.read_csv(ground_truth_csv)
    # split 66/34 into training and validation
    train_set = truth.sample(frac=0.66, random_state=42)
    validation_set = truth.drop(train_set.index)
    train_set.to_csv(train_csv, index=False)
    validation_set.to_csv(val_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IFCB focus classification workflows')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Feature extraction command
    extract_parser = subparsers.add_parser('extract', help='Extract features from images')
    extract_parser.add_argument('ground_truth_csv', help='Path to ground truth CSV file')
    extract_parser.add_argument('raw_data_dir', help='Directory containing raw IFCB data')
    extract_parser.add_argument('output_features_dir', help='Directory to save extracted features')
    
    # Split labels command
    split_parser = subparsers.add_parser('split', help='Split ground truth into train/validation sets')
    split_parser.add_argument('ground_truth_csv', help='Path to ground truth CSV file')
    split_parser.add_argument('train_csv', help='Path to save training set CSV')
    split_parser.add_argument('val_csv', help='Path to save validation set CSV')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        feature_extraction(args.ground_truth_csv, args.raw_data_dir, args.output_features_dir)
    elif args.command == 'split':
        split_true_labels(args.ground_truth_csv, args.train_csv, args.val_csv)
    else:
        parser.print_help()
