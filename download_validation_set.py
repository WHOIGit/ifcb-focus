import requests
import pandas as pd
import os

DATA_DIR = '/Users/jfutrelle/Data/ifcb-data/focus'
URL_PREFIX = 'https://ifcb-data.whoi.edu/data/'

def download_bin(pid):
    for ext in ['.hdr', '.adc', '.roi']:
        path = os.path.join(DATA_DIR, 'raw_data', f"{pid}{ext}")
        if os.path.exists(path):
            #print(f"File {path} already exists, skipping download.")
            continue
        url = f"{URL_PREFIX}{pid}{ext}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download {url}")

validation_set = pd.read_csv(os.path.join(DATA_DIR, 'validation.csv'))

for index, row in validation_set.iterrows():
    pid = row['pid']
    print(pid)
    download_bin(pid)
