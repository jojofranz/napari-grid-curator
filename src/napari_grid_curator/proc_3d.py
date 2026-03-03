"""
3D processing utilities for napari RGC plugin.

Functions for handling 3D volumes in mosaic mode.
"""

import numpy as np


def create_3d_montage(images_dict, labels_list, cols=5, padding_width=5):
    """
    Create a 3D montage of volumetric images arranged in a grid.
    
    Parameters:
        images_dict (dict): {label: ndarray} where each array has shape (C, Z, Y, X) or (Z, Y, X, C)
        labels_list (list/array): Ordered list of labels to montage
        cols (int): Number of columns in grid
        padding_width (int): Padding between tiles in pixels
        
    Returns:
        montage_volume: (Z, grid_Y, grid_X, C) - montaged volume
        montage_coords: (n_images, 2) - grid coordinates for each image
    """
    if len(images_dict) == 0:
        return None, None
    
    # Get first image to determine dimensions
    first_label = labels_list[0]
    first_img = images_dict[first_label]
    
    # Handle both (C, Z, Y, X) and (Z, Y, X, C) formats
    if first_img.ndim == 4:
        if first_img.shape[0] <= 4:  # Likely (C, Z, Y, X)
            # Convert to (Z, Y, X, C)
            images_dict = {label: img.transpose(1, 2, 3, 0) for label, img in images_dict.items()}
            first_img = images_dict[first_label]
    
    # Find maximum Z dimension across all images (they may vary due to edge cropping)
    max_Z = max(img.shape[0] for img in images_dict.values())
    tile_h = first_img.shape[1]
    tile_w = first_img.shape[2]
    n_channels = first_img.shape[3]
    
    print(f"  3D montage: max_Z={max_Z}, tile_h={tile_h}, tile_w={tile_w}, n_channels={n_channels}")
    
    n_images = len(labels_list)
    rows = int(np.ceil(n_images / cols))
    
    # Calculate grid dimensions with initial padding to match skimage.util.montage behavior
    grid_h = rows * (tile_h + padding_width) + padding_width
    grid_w = cols * (tile_w + padding_width) + padding_width
    
    # TODO: Add support for multi-channel 3D images
    # For now, squeeze out singleton channel dimension for single-channel data
    if n_channels == 1:
        montage = np.zeros((max_Z, grid_h, grid_w), dtype=first_img.dtype)
    else:
        montage = np.zeros((max_Z, grid_h, grid_w, n_channels), dtype=first_img.dtype)
    
    # Calculate coordinates for each tile
    coords = []
    
    # Fill montage
    for idx, label in enumerate(labels_list):
        if label not in images_dict:
            continue
            
        row = idx // cols
        col = idx % cols
        
        # Add initial padding offset to match grid layout
        y_start = row * (tile_h + padding_width) + padding_width
        y_end = y_start + tile_h
        x_start = col * (tile_w + padding_width) + padding_width
        x_end = x_start + tile_w
        
        # Get image and handle variable Z dimension
        img = images_dict[label]
        img_Z = img.shape[0]
        
        # Squeeze singleton channel dimension if single-channel
        if n_channels == 1 and img.ndim == 4:
            img = img.squeeze(axis=-1)
        
        # Place image in grid (center in Z if smaller than max_Z)
        z_offset = (max_Z - img_Z) // 2
        if n_channels == 1:
            montage[z_offset:z_offset+img_Z, y_start:y_end, x_start:x_end] = img
        else:
            montage[z_offset:z_offset+img_Z, y_start:y_end, x_start:x_end, :] = img
        
        # Store center coordinates for this tile
        coord_y = y_start + tile_h / 2
        coord_x = x_start + tile_w / 2
        coords.append([coord_y, coord_x])
    
    coords = np.array(coords)
    
    return montage, coords


def create_3d_label_montage(segmentations_dict, labels_list, cols=5, padding_width=5):
    """
    Create a 3D montage of binary segmentation masks arranged in a grid.
    
    Parameters:
        segmentations_dict (dict): {label: ndarray} where each array has shape (Z, Y, X)
        labels_list (list/array): Ordered list of labels to montage
        cols (int): Number of columns in grid
        padding_width (int): Padding between tiles in pixels
        
    Returns:
        montage_labels: (Z, grid_Y, grid_X) - montaged labels with proper label IDs
    """
    if len(segmentations_dict) == 0:
        return None
    
    # Get first segmentation to determine dimensions
    first_label = labels_list[0]
    first_seg = segmentations_dict[first_label]
    
    # Find maximum Z dimension across all segmentations (they may vary due to edge cropping)
    max_Z = max(seg.shape[0] for seg in segmentations_dict.values())
    tile_h = first_seg.shape[1]
    tile_w = first_seg.shape[2]
    
    print(f"  3D label montage: max_Z={max_Z}, tile_h={tile_h}, tile_w={tile_w}")
    
    n_images = len(labels_list)
    rows = int(np.ceil(n_images / cols))
    
    # Calculate grid dimensions with initial padding to match skimage.util.montage behavior
    grid_h = rows * (tile_h + padding_width) + padding_width
    grid_w = cols * (tile_w + padding_width) + padding_width
    
    # Create output volume initialized with zeros (background)
    montage = np.zeros((max_Z, grid_h, grid_w), dtype=np.uint32)
    
    # Fill montage with labeled masks
    for idx, label in enumerate(labels_list):
        if label not in segmentations_dict:
            continue
            
        row = idx // cols
        col = idx % cols
        
        # Add initial padding offset to match grid layout
        y_start = row * (tile_h + padding_width) + padding_width
        y_end = y_start + tile_h
        x_start = col * (tile_w + padding_width) + padding_width
        x_end = x_start + tile_w
        
        # Get binary mask for this label and handle variable Z dimension
        mask = segmentations_dict[label]
        mask_Z = mask.shape[0]
        
        # Place mask with proper label ID in grid (center in Z if smaller than max_Z)
        # Where mask is 1, set the label ID
        z_offset = (max_Z - mask_Z) // 2
        montage[z_offset:z_offset+mask_Z, y_start:y_end, x_start:x_end][mask > 0] = label
    
    return montage


def new_partitions_to_grid_3d(images, segmentations, labels, included_labels, excluded_labels, 
                                cols=5, padding_width=5):
    """
    Create 3D grids of volumetric images and segmentations with included/excluded partitions.
    
    Parameters:
        images (dict): {label: ndarray} - 3D volumes with shape (C, Z, Y, X) or (Z, Y, X, C)
        segmentations (dict): {label: ndarray} - 3D binary masks with shape (Z, Y, X)
        labels (array): Ordered labels to include in grid
        included_labels (array): Labels that pass exclusion criteria
        excluded_labels (array): Labels that fail exclusion criteria
        cols (int): Number of columns in grid
        padding_width (int): Padding between tiles
        
    Returns:
        all_images_grid: (Z, grid_Y, grid_X, C) - All images montaged
        all_labels_grid: (Z, grid_Y, grid_X) - All segmentation labels montaged
        coords: (n_labels, 2) - Grid coordinates
        included_images: (Z, grid_Y, grid_X, C) - Only included images
        excluded_images: (Z, grid_Y, grid_X, C) - Only excluded images
        included_labels_grid: (Z, grid_Y, grid_X) - Only included labels
        excluded_labels_grid: (Z, grid_Y, grid_X) - Only excluded labels
    """
    # Ensure unique labels in original order
    labels_unique = []
    for label in labels:
        if label not in labels_unique:
            labels_unique.append(label)
    labels = np.array(labels_unique)
    
    # Create full montage
    all_images_grid, coords = create_3d_montage(images, labels, cols, padding_width)
    
    # Create label montage if segmentations provided
    if segmentations is not None and len(segmentations) > 0:
        all_labels_grid = create_3d_label_montage(segmentations, labels, cols, padding_width)
    else:
        all_labels_grid = None
    
    # Create included/excluded partitions using TILE-level masking (not pixel-level)
    # This ensures entire cell tiles are shown/hidden, not just the segmented pixels
    if all_labels_grid is not None:
        # Create tile-level masks by checking which labels exist in each tile
        # For each tile position in the grid, check if the label is included/excluded
        included_images = all_images_grid.copy()
        excluded_images = all_images_grid.copy()
        included_labels_grid = all_labels_grid.copy()
        excluded_labels_grid = all_labels_grid.copy()
        
        # Get tile dimensions from first image
        first_label = labels[0]
        first_seg = segmentations[first_label]
        tile_h = first_seg.shape[1]
        tile_w = first_seg.shape[2]
        
        # Check if images have channel dimension
        has_channels = all_images_grid.ndim == 4
        
        print(f"  Masking tiles: images shape={all_images_grid.shape}, has_channels={has_channels}")
        
        # Zero out tiles based on inclusion/exclusion
        for idx, label in enumerate(labels):
            row = idx // cols
            col = idx % cols
            
            # Add initial padding offset to match grid layout
            y_start = row * (tile_h + padding_width) + padding_width
            y_end = y_start + tile_h
            x_start = col * (tile_w + padding_width) + padding_width
            x_end = x_start + tile_w
            
            # Zero out the tile in the appropriate grid
            # Handle both 3D (Z,Y,X) and 4D (Z,Y,X,C) cases
            if label in excluded_labels:
                # This label is excluded, so zero it in included grids
                if has_channels:
                    included_images[:, y_start:y_end, x_start:x_end, :] = 0
                else:
                    included_images[:, y_start:y_end, x_start:x_end] = 0
                included_labels_grid[:, y_start:y_end, x_start:x_end] = 0
            elif label in included_labels:
                # This label is included, so zero it in excluded grids
                if has_channels:
                    excluded_images[:, y_start:y_end, x_start:x_end, :] = 0
                else:
                    excluded_images[:, y_start:y_end, x_start:x_end] = 0
                excluded_labels_grid[:, y_start:y_end, x_start:x_end] = 0
    else:
        # No segmentations - just return images as-is (can't properly mask without labels)
        included_images = all_images_grid
        excluded_images = np.zeros_like(all_images_grid)
        included_labels_grid = None
        excluded_labels_grid = None
    
    return (all_images_grid, all_labels_grid, coords, 
            included_images, excluded_images, 
            included_labels_grid, excluded_labels_grid)
