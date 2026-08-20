"""
IFCB data access for focus detection, backed by ifcbkit.

ifcbkit is PIL-only, while the feature extraction code in this package works
on numpy arrays. This module is the single place that bridges the two: it
wraps a bin's raw ``.adc``/``.roi`` bytes in a :class:`FocusBin`, whose
``images`` mapping yields grayscale numpy arrays.

Example:
    >>> from ifcb_focus.data import data_directory, open_bin
    >>>
    >>> dd = data_directory('/path/to/raw_data')
    >>> b = open_bin(dd, 'D20130823T160901_IFCB010')
    >>> image = b.images[1]
"""

import os
from collections.abc import Mapping

import numpy as np
import requests
from ifcbkit import SyncIfcbDataDirectory, bin_images, parse_adc_bytes

__all__ = [
    'FocusBin',
    'data_directory',
    'open_bin',
    'open_remote_bin',
    'iter_bins',
]


class NumpyImages(Mapping):
    """
    Mapping of ROI number to grayscale numpy array.

    Wraps an ifcbkit image mapping (which yields PIL Images) and converts
    each image to a numpy array on access.

    Args:
        images (Mapping): Mapping of ROI number to PIL Image.
    """

    def __init__(self, images):
        self._images = images

    def __getitem__(self, roi_number):
        return np.asarray(self._images[roi_number])

    def __iter__(self):
        return iter(self._images)

    def __len__(self):
        return len(self._images)


class FocusBin:
    """
    A single IFCB bin, backed by ifcbkit.

    Args:
        pid (str): Bin ID (e.g., 'D20230101T120000_IFCB123').
        adc_bytes (bytes): Raw contents of the bin's .adc file.
        roi_bytes (bytes): Raw contents of the bin's .roi file.

    Attributes:
        lid (str): Bin ID.
        images (Mapping): ROI number -> grayscale numpy array. I-style
            stitched pairs are composited by ifcbkit.
        adc (dict): ROI number -> {'roi_id', 'x', 'y', 'width', 'height'}.

    Example:
        >>> from ifcb_focus.data import FocusBin
        >>>
        >>> b = FocusBin.from_paths('D20230101T120000_IFCB123', 'bin.adc', 'bin.roi')
        >>> print(len(b.images))
    """

    def __init__(self, pid, adc_bytes, roi_bytes):
        self.lid = pid
        self._adc_bytes = adc_bytes
        self._roi_bytes = roi_bytes
        self._images = None
        self._adc = None

    @classmethod
    def from_paths(cls, pid, adc_path, roi_path):
        """
        Construct a bin from .adc and .roi file paths.

        Args:
            pid (str): Bin ID.
            adc_path (str): Path to the .adc file.
            roi_path (str): Path to the .roi file.

        Returns:
            FocusBin: The bin.
        """
        with open(adc_path, 'rb') as f:
            adc_bytes = f.read()
        with open(roi_path, 'rb') as f:
            roi_bytes = f.read()
        return cls(pid, adc_bytes, roi_bytes)

    @property
    def images(self):
        """Mapping of ROI number to grayscale numpy array."""
        if self._images is None:
            self._images = NumpyImages(
                bin_images(self.lid, self._adc_bytes, self._roi_bytes)
            )
        return self._images

    @property
    def adc(self):
        """Dict of ROI number to ROI metadata (roi_id, x, y, width, height)."""
        if self._adc is None:
            self._adc = parse_adc_bytes(self.lid, self._adc_bytes)
        return self._adc

    def __repr__(self):
        return f'FocusBin({self.lid!r})'


def data_directory(data_dir, include=None):
    """
    Open an IFCB raw data directory.

    Args:
        data_dir (str): Path to the directory containing IFCB filesets.
        include (list, optional): Directory names to always descend into.
            If None, uses ifcbkit's default (['data']).

    Returns:
        ifcbkit.SyncIfcbDataDirectory: The data directory.

    Example:
        >>> from ifcb_focus.data import data_directory
        >>> dd = data_directory('/path/to/raw_data')
    """
    if include is None:
        return SyncIfcbDataDirectory(data_dir)
    return SyncIfcbDataDirectory(data_dir, include=include)


def open_bin(dd, pid):
    """
    Open a single bin from a data directory.

    Args:
        dd (SyncIfcbDataDirectory or str): Data directory, or a path to one.
        pid (str): Bin ID.

    Returns:
        FocusBin: The bin.

    Raises:
        KeyError: If the bin is not present in the data directory.

    Example:
        >>> from ifcb_focus.data import open_bin
        >>> b = open_bin('/path/to/raw_data', 'D20230101T120000_IFCB123')
    """
    if isinstance(dd, str):
        dd = data_directory(dd)
    paths = dd.paths(pid)
    return FocusBin.from_paths(pid, paths['adc'], paths['roi'])


def iter_bins(dd):
    """
    Iterate over all bins in a data directory.

    Args:
        dd (SyncIfcbDataDirectory or str): Data directory, or a path to one.

    Yields:
        FocusBin: Each bin in the directory.

    Example:
        >>> from ifcb_focus.data import iter_bins
        >>> for b in iter_bins('/path/to/raw_data'):
        ...     print(b.lid)
    """
    if isinstance(dd, str):
        dd = data_directory(dd)
    for fileset in dd.list():
        yield FocusBin.from_paths(fileset['pid'], fileset['adc'], fileset['roi'])


def open_remote_bin(host, pid, timeout=60):
    """
    Fetch a bin's .adc and .roi files over HTTPS.

    Args:
        host (str): Hostname of the IFCB data server.
        pid (str): Bin ID.
        timeout (float): Per-request timeout in seconds. Default is 60.

    Returns:
        FocusBin: The bin, with its raw data held in memory.

    Example:
        >>> from ifcb_focus.data import open_remote_bin
        >>> b = open_remote_bin('ifcb-data.whoi.edu', 'D20230101T120000_IFCB123')
    """
    base_url = f'https://{host}/data/{pid}'

    def fetch(ext):
        response = requests.get(f'{base_url}.{ext}', timeout=timeout)
        response.raise_for_status()
        return response.content

    return FocusBin(pid, fetch('adc'), fetch('roi'))
