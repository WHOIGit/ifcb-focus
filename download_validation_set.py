import requests
import pandas as pd
import os
import argparse

def download_bin(pid, url_prefix, raw_data_dir):
    for ext in ['.hdr', '.adc', '.roi']:
        path = os.path.join(raw_data_dir, f"{pid}{ext}")
        if os.path.exists(path):
            #print(f"File {path} already exists, skipping download.")
            continue
        url = f"{url_prefix}{pid}{ext}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download {url}")

def download_validation_set(validation_csv, url_prefix, raw_data_dir):
    validation_set = pd.read_csv(validation_csv)
    
    for index, row in validation_set.iterrows():
        pid = row['pid']
        print(pid)
        download_bin(pid, url_prefix, raw_data_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download validation set files')
    parser.add_argument('validation_csv', help='Path to validation CSV file')
    parser.add_argument('url_prefix', help='URL prefix for downloading files')
    parser.add_argument('raw_data_dir', help='Directory to save raw data files')
    
    args = parser.parse_args()
    download_validation_set(args.validation_csv, args.url_prefix, args.raw_data_dir)
