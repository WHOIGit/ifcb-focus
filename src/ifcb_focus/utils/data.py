"""
Data downloading utilities for IFCB focus detection.

This module provides functions for downloading IFCB bin data from remote servers.
"""

import os
import requests
import pandas as pd


def download_bin(pid, data_dir=None, url_prefix=None):
    """
    Download an IFCB bin (.hdr, .adc, .roi files) from a remote server.

    Args:
        pid (str): Bin ID (e.g., 'D20230101T120000_IFCB123').
        data_dir (str, optional): Directory to save downloaded files.
            If None, uses environment variable IFCB_DATA_DIR or defaults to './data'.
        url_prefix (str, optional): URL prefix for downloading bins.
            If None, constructs from IFCB_DASHBOARD_HOST environment variable
            or defaults to 'https://localhost:8000/data/'.

    Returns:
        bool: True if all files downloaded successfully, False otherwise.

    Example:
        >>> from ifcb_focus.utils import download_bin
        >>> download_bin('D20230101T120000_IFCB123', data_dir='/path/to/data')
    """
    if data_dir is None:
        data_dir = os.environ.get('IFCB_DATA_DIR', './data')

    if url_prefix is None:
        ifcb_host = os.environ.get('IFCB_DASHBOARD_HOST', 'localhost:8000')
        url_prefix = f'https://{ifcb_host}/data/'

    # Create raw_data directory if it doesn't exist
    raw_data_dir = os.path.join(data_dir, 'raw_data')
    os.makedirs(raw_data_dir, exist_ok=True)

    success = True
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
            print(f"Downloaded {path}")
        else:
            print(f"Failed to download {url}")
            success = False

    return success


def download_validation_dataset(data_dir=None, url_prefix=None):
    """
    Download all bins listed in the validation set.

    Args:
        data_dir (str, optional): Directory containing validation.csv.
            If None, uses environment variable IFCB_DATA_DIR or defaults to './data'.
        url_prefix (str, optional): URL prefix for downloading bins.

    Example:
        >>> from ifcb_focus.utils import download_validation_dataset
        >>> download_validation_dataset('/path/to/data')
    """
    if data_dir is None:
        data_dir = os.environ.get('IFCB_DATA_DIR', './data')

    validation_set = pd.read_csv(os.path.join(data_dir, 'validation.csv'))

    for index, row in validation_set.iterrows():
        pid = row['pid']
        print(f"Downloading {pid}...")
        download_bin(pid, data_dir=data_dir, url_prefix=url_prefix)
