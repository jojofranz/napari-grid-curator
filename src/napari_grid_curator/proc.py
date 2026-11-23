"""
============================================================================
Napari RGC Plugin - Processing Utilities Module
============================================================================

This module provides utility functions for image processing and grid layout
generation used by the napari RGC plugin.

Key Functions:
--------------
- partition_images: Split labels into included/excluded based on DataFrame
- new_partitions_to_grid: Arrange cropped nucleus images in grid layouts
- get_new_file_path: Generate versioned filenames to prevent overwriting

These functions support the main widget's visualization and data management
functionality.

============================================================================
"""

import sys
import os
import re
import pickle
import copy
import pandas as pd
import numpy as np
import skimage


def as_channel_last_image(image):
    """
    Return an image as (Y, X, C) without duplicating single-channel data.

    Raw 2D crops become (Y, X, 1) via ``[..., np.newaxis]``. This adds a
    channel axis as a view, unlike repeating a grayscale crop into RGB.
    Channel-first images with a small leading channel axis, e.g. (1, Y, X),
    are moved to channel-last for skimage's montage function.
    """
    image = np.asarray(image)

    if image.ndim == 2:
        return image[..., np.newaxis]

    if image.ndim == 3:
        if image.shape[-1] in (1, 2, 3, 4):
            return image
        if image.shape[0] in (1, 2, 3, 4):
            return np.moveaxis(image, 0, -1)

    return image


def prepare_images_for_grid(images, labels=None, pad_value=0):
    """
    Normalize image arrays for 2D montage display.

    This keeps datasets on disk in their native crop sizes and pads only the
    currently displayed batch/scene immediately before visualization.
    """
    if labels is None:
        labels = list(images.keys())

    prepared = {}
    max_y = 0
    max_x = 0
    max_c = 0

    for label in labels:
        if label not in images:
            continue
        image = as_channel_last_image(images[label])
        if image.ndim != 3:
            raise ValueError(
                f"Expected a 2D image or 3D channel image for label {label}, "
                f"got shape {image.shape}"
            )
        prepared[label] = image
        max_y = max(max_y, image.shape[0])
        max_x = max(max_x, image.shape[1])
        max_c = max(max_c, image.shape[2])

    padded = {}
    for label, image in prepared.items():
        pad_y = max_y - image.shape[0]
        pad_x = max_x - image.shape[1]
        pad_c = max_c - image.shape[2]
        pad_width = (
            (pad_y // 2, pad_y - pad_y // 2),
            (pad_x // 2, pad_x - pad_x // 2),
            (0, pad_c),
        )
        if any(before or after for before, after in pad_width):
            image = np.pad(image, pad_width, mode='constant', constant_values=pad_value)
        padded[label] = image

    return padded


def _montage_fill_value(image):
    """Choose a fill value that does not compress raw grayscale contrast.

    Returned as a plain list (one value per channel): scikit-image >= 0.25 compares
    ``fill == 'mean'``, which raises for a numpy array.
    """
    if image.shape[-1] == 1 and np.issubdtype(image.dtype, np.integer):
        return np.zeros(1, dtype=image.dtype).tolist()
    return np.ones(image.shape[-1], dtype=image.dtype).tolist()


def partition_images(df):
    """
    Partition labels from a DataFrame into included and excluded images.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column 'label' and a column 'exclude' indicating whether the label is excluded.

    Returns
    -------
    included_labels : np.ndarray
        Array of included labels (as integers).
    excluded_labels : np.ndarray
        Array of excluded labels (as integers).
    """

    # Convert to int to ensure type compatibility with dict keys (fixes float64 mismatch)
    included_labels = df.loc[~df['exclude'], 'label'].values.astype(int)
    excluded_labels = df.loc[df['exclude'], 'label'].values.astype(int)
    return included_labels, excluded_labels


def partitions_to_grid(images, labels, included_labels, excluded_labels, cols = 5, padding_width = 5):
    """
    Creates grids of images and labels.

    Parameters
    ----------
    images : dict
        Dictionary of images, where the keys are the labels, and images are 3D numpy arrays with shape (height, width, channels).
    labels : np.ndarray
        Array of labels.
    included_labels : np.ndarray
        Array of included labels.
    excluded_labels : np.ndarray
        Array of excluded labels.
    cols : int
        Number of columns in the grid.
    padding_width : int
        Width of the padding between images (in pixels).

    Returns
    -------
    all_images_grid : np.ndarray
        Grid of all images.
    all_labels : np.ndarray
        Grid of all labels.
    included_images : np.ndarray
        Grid of included images.
    excluded_images : np.ndarray
        Grid of excluded images.
    """
    #labels = np.unique(labels, sorted=False)
    # unique values accorting to old order
    # np.unique(labels, sorted=False) only available from numpy 2.6
    labels_unique = []
    for label in labels:
        if not label in labels_unique:
            labels_unique.append(label)
    labels = np.array(labels_unique)
    # included_labels = np.unique(included_labels, sorted=False)
    # excluded_labels = np.unique(excluded_labels, sorted=False)

    labels = np.array([label for label in labels if label in images])
    images = prepare_images_for_grid(images, labels)

    rows = int(np.ceil(len(images) / cols))
    # print(f'images keys: {np.sort(list(images.keys()))}')
    # print(f'labels: {np.sort(list(labels))}')
    # print(f'union of included and excluded labels: {np.sort(np.union1d(included_labels, excluded_labels))}')
    all_images = [images[label] for label in labels]
    first_image = images[list(images.keys())[0]]
    single_tile = np.ones(first_image.shape[:2], dtype=np.uint16)
    all_labels = [single_tile*label for label in labels]
    fill = _montage_fill_value(first_image)
    all_images_grid = skimage.util.montage(all_images, grid_shape=(rows, cols), channel_axis=3,
                                           fill = fill, padding_width = padding_width)
    all_labels = skimage.util.montage(all_labels, grid_shape=(rows, cols),
                                      fill=0, padding_width = padding_width)
    all_labels = all_labels.astype(np.uint16)

    # ``skimage.util.montage`` requires a channel axis, but napari displays raw
    # single-channel microscopy data more naturally as scalar 2D data. Squeeze
    # the synthetic singleton channel back out after montage so uint16 contrast
    # is preserved and the layer is not interpreted as a one-slice volume.
    if all_images_grid.ndim == 3 and all_images_grid.shape[-1] == 1:
        all_images_grid = all_images_grid[..., 0]

    coords = np.array(np.meshgrid(np.arange(rows), np.arange(cols))).T.reshape(-1, 2) + .5
    coords = coords[:len(labels)]
    coords[:, 0] *= first_image.shape[0] + padding_width
    coords[:, 1] *= first_image.shape[1] + padding_width

    included_images = all_images_grid.copy()
    included_images[np.isin(all_labels, excluded_labels)] = 0

    excluded_images = all_images_grid.copy()
    excluded_images[np.isin(all_labels, included_labels)] = 0

    return all_images_grid, all_labels, coords, included_images, excluded_images


def get_new_file_path(file_path, search_endings=[r'_v(\d+)\.pkl', r'_v(\d+)\.xlsx', r'_v(\d+)\.csv', r'_v(\d+)_summary\.csv', r'_v(\d+)_summary\.xlsx'], file_ending='.pkl'):
    """
    Get a new file path by adding a version number to the file name.
    e.g. if the file path is 'C:\\Users\\johannes.franz\\Downloads\\birke_napari_test\\dataset_for_napari.pkl'
    'C:\\Users\\johannes.franz\\Downloads\\birke_napari_test\\dataset_for_napari_v1.pkl'

    Parameters
    ----------
    file_path : str
        File path to the file that should be saved.
    file_ending : str
        File ending of the file that should be saved.
    
    Returns
    -------
    new_file_path : str
        New file path with version number.
    """
    import re
    file_path = str(file_path)

    # new_file_path = copy.deepcopy(file_path)
    if len(re.findall(r'_v(\d+)' + os.path.sep + file_ending, str(file_path))):
        file_path = str(file_path).rsplit(r'_v(\d+)' + file_ending, 1)[0]
    else:
        file_path = str(file_path).rsplit(file_ending, 1)[0]

    max_version = 1
    for ending in search_endings:
        ending = r'_v(\d+)\.pkl'
        pattern = re.escape(os.path.basename(file_path)) + ending
        versions = [re.findall(pattern, file) for file in os.listdir(os.path.dirname(file_path))]
        versions = [int(v[0]) for v in versions if v]
        version = np.max(versions) + 1 if versions else 1
        if version > max_version:
            max_version = version

    new_file_path = file_path + f'_v{max_version}{file_ending}'

    return new_file_path
