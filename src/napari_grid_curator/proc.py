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


def new_partitions_to_grid(images, labels, included_labels, excluded_labels, cols = 5, padding_width = 5):
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

    rows = int(np.ceil(len(images) / cols))
    # print(f'images keys: {np.sort(list(images.keys()))}')
    # print(f'labels: {np.sort(list(labels))}')
    # print(f'union of included and excluded labels: {np.sort(np.union1d(included_labels, excluded_labels))}')
    all_images = [images[label] for label in labels]
    single_tile = np.ones_like(images[list(images.keys())[0]])[..., 0].astype(np.uint16)
    all_labels = [single_tile*label for label in labels]
    all_images_grid = skimage.util.montage(all_images, grid_shape=(rows, cols), channel_axis=3,
                                           fill = [1, 1, 1, 1], padding_width = padding_width)
    all_labels = skimage.util.montage(all_labels, grid_shape=(rows, cols),
                                      fill=0, padding_width = padding_width)
    all_labels = all_labels.astype(np.uint16)

    coords = np.array(np.meshgrid(np.arange(rows), np.arange(cols))).T.reshape(-1, 2) + .5
    coords = coords[:len(labels)]
    coords[:, 0] *= list(images.values())[0].shape[0] + padding_width
    coords[:, 1] *= list(images.values())[0].shape[1] + padding_width

    included_images = all_images_grid.copy()
    included_images[np.isin(all_labels, excluded_labels)] = 0

    excluded_images = all_images_grid.copy()
    excluded_images[np.isin(all_labels, included_labels)] = 0

    return all_images_grid, all_labels, coords, included_images, excluded_images


# def partitions_to_grid(images, included_labels, excluded_labels, cols = 5, padding_width = 5):
#     """
#     Creates grids of images and labels.

#     Parameters
#     ----------
#     images : dict
#         Dictionary of images, where the keys are the labels, and images are 3D numpy arrays with shape (height, width, channels).
#     included_labels : np.ndarray
#         Array of included labels.
#     excluded_labels : np.ndarray
#         Array of excluded labels.
#     cols : int
#         Number of columns in the grid.
#     padding_width : int
#         Width of the padding between images (in pixels).

#     Returns
#     -------
#     all_images_grid : np.ndarray
#         Grid of all images.
#     all_labels : np.ndarray
#         Grid of all labels.
#     included_images : np.ndarray
#         Grid of included images.
#     excluded_images : np.ndarray
#         Grid of excluded images.
#     """
#     rows = int(np.ceil(len(images) / cols))
#     labels = np.union1d(included_labels, excluded_labels)
#     all_images = [images[label] for label in labels]
#     single_tile = np.ones_like(images[list(images.keys())[0]])[..., 0].astype(np.uint16)
#     all_labels = [single_tile*label for label in labels]
#     all_images_grid = skimage.util.montage(all_images, grid_shape=(rows, cols), channel_axis=3,
#                                            fill = [1, 1, 1, 1], padding_width = padding_width)
#     all_labels = skimage.util.montage(all_labels, grid_shape=(rows, cols),
#                                       fill=0, padding_width = padding_width)
#     all_labels = all_labels.astype(np.uint16)

#     coords = np.array(np.meshgrid(np.arange(rows), np.arange(cols))).T.reshape(-1, 2) + .5
#     coords = coords[:len(labels)]
#     coords[:, 0] *= list(images.values())[0].shape[0] + padding_width
#     coords[:, 1] *= list(images.values())[0].shape[1] + padding_width

#     included_images = all_images_grid.copy()
#     included_images[np.isin(all_labels, excluded_labels)] = 0

#     excluded_images = all_images_grid.copy()
#     excluded_images[np.isin(all_labels, included_labels)] = 0

#     return all_images_grid, all_labels, coords, included_images, excluded_images


# def get_new_file_path(file_path, search_endings=['.pkl', '.csv', '_summary.csv'], file_ending='.pkl'):
#     """
#     Get a new file path by adding a version number to the file name.
#     e.g. if the file path is 'C:\\Users\\johannes.franz\\Downloads\\birke_napari_test\\dataset_for_napari.pkl'
#     'C:\\Users\\johannes.franz\\Downloads\\birke_napari_test\\dataset_for_napari_v1.pkl'

#     Parameters
#     ----------
#     file_path : str
#         File path to the file that should be saved.
#     file_ending : str
#         File ending of the file that should be saved.
    
#     Returns
#     -------
#     new_file_path : str
#         New file path with version number.
#     """
#     # new_file_path = copy.deepcopy(file_path)
#     file_path = str(file_path).rsplit(file_ending, 1)[0]
#     options = []
#     for ending in search_endings:
#         new_file_path = file_path + ending
#         while os.path.exists(new_file_path):
#             new_file_path = str(new_file_path).rsplit(ending, 1)[0]
#             if new_file_path.rsplit('_', 1)[-1].startswith('v'):
#                 if new_file_path.rsplit('_v', 1)[-1].isnumeric():
#                     version = int(new_file_path.rsplit('_v', 1)[-1]) + 1
#                     new_file_path = new_file_path.rsplit('_v', 1)[0] + f'_v{version}{file_ending}'
#                 else:
#                     version = 1
#                     new_file_path = new_file_path.rsplit('_v', 1)[0] + f'_v{version}{file_ending}'
#             else:
#                 version = 1
#                 new_file_path = new_file_path.rsplit(f'{file_ending}', 1)[0] + f'_v{version}{file_ending}'
#         new_file_path = new_file_path[:-len(ending)]
#         options.append([version, new_file_path])

#     max_version = np.argmax([option[0] for option in options])
#     new_file_path = options[max_version][1] + file_ending

#     return new_file_path


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
