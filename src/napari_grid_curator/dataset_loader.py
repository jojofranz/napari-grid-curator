"""
Dataset loading utilities for napari RGC plugin.

Supports three dataset formats:
1. Legacy: All images in one .pkl file
2. Lazy: Images split into per-scene .pkl files
3. Minimal: References to original .lif images + label files

Handles path validation and resolution for moved/transferred datasets.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import Dict, Tuple, Optional, List
import warnings

try:
    from microfilm import colorify
    COLORIFY_AVAILABLE = True
except ImportError:
    COLORIFY_AVAILABLE = False

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

try:
    from aicsimageio import AICSImage
    AICSIMAGEIO_AVAILABLE = True
except ImportError:
    AICSIMAGEIO_AVAILABLE = False


class DatasetLoader:
    """
    Multi-format dataset loader with path validation.
    
    Attributes:
        file_path (str): Path to main .pkl file
        dataset_mode (str): 'legacy', 'lazy', or 'minimal'
        base_dir (str): Base directory for resolving relative paths
        metadata (dict): Format-specific metadata
        ndim (int): Number of spatial dimensions (2 or 3)
    """
    
    def __init__(self, file_path: str):
        """
        Initialize loader and detect dataset format.
        
        Parameters:
            file_path: Path to .pkl file
        """
        self.file_path = file_path
        self.base_dir = os.path.dirname(file_path)
        self.dataset_mode = None
        self.metadata = {}
        self.ndim = 2  # Default to 2D, will be detected
        
        # Load and detect format
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self._detect_format(data)
        self._initialize_dataset(data)
        self._detect_ndim(data)
    
    def _detect_format(self, data: dict):
        """Detect which format the dataset uses."""
        if isinstance(data, dict):
            if 'format_version' in data:
                version = data['format_version']
                if version == 'lazy_v1':
                    self.dataset_mode = 'lazy'
                elif version == 'minimal_v1':
                    self.dataset_mode = 'minimal'
                else:
                    raise ValueError(f"Unknown format version: {version}")
            else:
                # Legacy format: dict of {scene_name: {'images': ..., 'df': ...}}
                first_key = list(data.keys())[0]
                if isinstance(data[first_key], dict) and 'images' in data[first_key]:
                    self.dataset_mode = 'legacy'
                else:
                    raise ValueError("Unrecognized dataset format")
        else:
            raise ValueError("Dataset must be a dictionary")
    
    def _initialize_dataset(self, data: dict):
        """Initialize dataset based on detected format."""
        if self.dataset_mode == 'legacy':
            self.dataset = data
            self.scene_names = list(data.keys())
        
        elif self.dataset_mode == 'lazy':
            # Check if this is 3D tiff-based lazy format
            self._is_tiff_based = '_source_image' in data and '_source_segmentation' in data
            
            self.metadata = {
                'scene_names': data['scene_names'],
                'image_files': data.get('image_files', {}),
                'base_dir': data.get('base_dir', self.base_dir)
            }
            
            # Store 3D tiff sources if present
            if self._is_tiff_based:
                self.metadata['_source_image'] = data['_source_image']
                self.metadata['_source_segmentation'] = data['_source_segmentation']
                # Separate metadata for each file
                self.metadata['image_filetype'] = data.get('image_filetype', 'tiff')
                self.metadata['image_dims'] = data.get('image_dims', 'ZYX')
                self.metadata['segmentation_filetype'] = data.get('segmentation_filetype', 'tiff')
                self.metadata['segmentation_dims'] = data.get('segmentation_dims', 'ZYX')
                # Store relabel flag for segmentation loading
                self.metadata['relabel'] = data.get('relabel', False)
            
            self.scene_names = data['scene_names']
            
            # Initialize dataset structure
            self.dataset = {}
            for scene_name in self.scene_names:
                self.dataset[scene_name] = {
                    'df': data['dataframes'][scene_name],
                    'images': None,  # Load on demand
                    'segmentations': None,  # Load on demand (separate from images)
                    '_image_file': data['image_files'].get(scene_name) if not self._is_tiff_based else None
                }
            
            # Validate image file paths (only for non-tiff-based)
            if not self._is_tiff_based:
                self._validate_paths_lazy()
        
        elif self.dataset_mode == 'minimal':
            # Check if this is 3D file-based minimal format
            self._is_3d_file_based = (
                'image_file' in data and 
                'image_filetype' in data and 
                'segmentation_filetype' in data
            )
            
            if self._is_3d_file_based:
                # 3D minimal format with TIFF/NIfTI/Zarr files
                self.metadata = {
                    'scene_names': data['scene_names'],
                    'label_files': data['label_files'],
                    'image_file': data['image_file'],  # Path to 3D image
                    'base_dir': data.get('base_dir', self.base_dir),
                    'save_format': data['save_format'],
                    # 3D-specific metadata
                    'image_filetype': data['image_filetype'],
                    'image_dims': data['image_dims'],
                    'segmentation_filetype': data['segmentation_filetype'],
                    'segmentation_dims': data['segmentation_dims'],
                    'relabel': data.get('relabel', False),
                    'channel_names': data.get('channel_names', [])
                }
            else:
                # 2D minimal format with .lif files
                self.metadata = {
                    'scene_names': data['scene_names'],
                    'label_files': data['label_files'],
                    'lif_file': data['lif_file'],
                    'base_dir': data.get('base_dir', self.base_dir),
                    'save_format': data['save_format'],
                    'emission_to_fluorophore_map': data.get('emission_to_fluorophore_map', {}),
                    'cmaps': data.get('cmaps', {})
                }
            
            self.scene_names = data['scene_names']
            
            # Initialize dataset structure
            self.dataset = {}
            for scene_name in self.scene_names:
                df = data['dataframes'][scene_name]
                
                # Handle relabel mode: rename instance_label → label for consistency
                if self._is_3d_file_based and data.get('relabel', False):
                    if 'instance_label' in df.columns:
                        df = df.rename(columns={'instance_label': 'label'})
                        print(f"INFO: Renamed 'instance_label' → 'label' for scene '{scene_name}' (relabel mode)")
                
                if self._is_3d_file_based:
                    # 3D file-based minimal format
                    self.dataset[scene_name] = {
                        'df': df,
                        'images': None,  # Crop on demand
                        'segmentations': None,  # Crop on demand (separate from images)
                        '_label_file': data['label_files'][scene_name],
                        '_image_file': data['image_file'],
                        '_labels': None,  # Load on demand
                        '_full_image': None  # Load on demand
                    }
                else:
                    # 2D .lif-based minimal format
                    self.dataset[scene_name] = {
                        'df': df,
                        'images': None,  # Crop on demand
                        'segmentations': None,  # Crop on demand (separate from images)
                        '_label_file': data['label_files'][scene_name],
                        '_lif_file': df['_source_lif_file'].iloc[0] if '_source_lif_file' in df.columns else data['lif_file'],
                        '_scene_name': df['_source_scene_name'].iloc[0] if '_source_scene_name' in df.columns else scene_name,
                        '_labels': None  # Load on demand
                    }
            
            # Validate paths
            missing_paths = self._validate_paths_minimal()
            if missing_paths:
                warnings.warn(f"Some paths are missing: {missing_paths}\n"
                            "Use fix_paths() to attempt automatic resolution.")
    
    def _detect_ndim(self, data: dict):
        """
        Detect whether dataset is 2D or 3D based on bbox columns in DataFrame.
        
        2D datasets have bbox-0 to bbox-3 (4 values: y_min, x_min, y_max, x_max)
        3D datasets have bbox-0 to bbox-5 (6 values: z_min, y_min, x_min, z_max, y_max, x_max)
        
        Also checks for explicit 'ndim' field in data.
        """
        # First check if ndim is explicitly stored in the data
        if 'ndim' in data:
            self.ndim = data['ndim']
            return
        
        # Otherwise, detect from DataFrame columns
        df = None
        if self.dataset_mode == 'legacy':
            first_scene = list(self.dataset.keys())[0]
            df = self.dataset[first_scene].get('df')
        elif self.dataset_mode in ['lazy', 'minimal']:
            first_scene = self.scene_names[0]
            df = self.dataset[first_scene].get('df')
        
        if df is not None:
            # Check for 3D bbox columns
            has_bbox_5 = 'bbox-5' in df.columns
            has_centroid_2 = 'centroid-2' in df.columns
            
            if has_bbox_5 or has_centroid_2:
                self.ndim = 3
            else:
                self.ndim = 2
    
    def is_3d(self) -> bool:
        """Return True if dataset is 3D."""
        return self.ndim == 3
    
    def _validate_paths_lazy(self) -> List[str]:
        """Validate that image files exist for lazy loading."""
        missing = []
        for scene_name in self.scene_names:
            rel_path = self.dataset[scene_name]['_image_file']
            full_path = os.path.join(self.base_dir, rel_path)
            if not os.path.exists(full_path):
                missing.append(full_path)
                # Try alternative: relative to .pkl file location
                alt_path = os.path.join(os.path.dirname(self.file_path), rel_path)
                if os.path.exists(alt_path):
                    self.dataset[scene_name]['_image_file'] = os.path.relpath(alt_path, self.base_dir)
                    missing.remove(full_path)
        return missing
    
    def _validate_paths_minimal(self) -> List[str]:
        """Validate that source image and label files exist."""
        missing = []
        
        # Check source image file (either .lif or 3D image file)
        if hasattr(self, '_is_3d_file_based') and self._is_3d_file_based:
            # 3D file-based minimal format
            image_rel = self.metadata['image_file']
            image_full = os.path.join(self.base_dir, image_rel)
            if not os.path.exists(image_full):
                missing.append(image_full)
        else:
            # 2D .lif-based minimal format
            lif_rel = self.metadata['lif_file']
            lif_full = os.path.join(self.base_dir, lif_rel)
            if not os.path.exists(lif_full):
                missing.append(lif_full)
        
        # Check label files
        for scene_name in self.scene_names:
            label_rel = self.dataset[scene_name]['_label_file']
            label_full = os.path.join(self.base_dir, label_rel)
            if not os.path.exists(label_full):
                missing.append(label_full)
        
        return missing
    
    def fix_paths(self, new_base_dir: Optional[str] = None, 
                   lif_path: Optional[str] = None,
                   labels_dir: Optional[str] = None) -> bool:
        """
        Attempt to fix broken paths after dataset transfer.
        
        Parameters:
            new_base_dir: New base directory for relative paths
            lif_path: New path to .lif file
            labels_dir: New directory containing label files
        
        Returns:
            True if all paths resolved successfully
        """
        if self.dataset_mode == 'legacy':
            return True  # No external paths to fix
        
        if new_base_dir:
            self.base_dir = new_base_dir
            self.metadata['base_dir'] = new_base_dir
        
        if self.dataset_mode == 'lazy':
            return len(self._validate_paths_lazy()) == 0
        
        elif self.dataset_mode == 'minimal':
            # Fix .lif path
            if lif_path:
                if os.path.exists(lif_path):
                    self.metadata['lif_file'] = os.path.relpath(lif_path, self.base_dir)
                    for scene_name in self.scene_names:
                        self.dataset[scene_name]['_lif_file'] = self.metadata['lif_file']
            
            # Fix label paths
            if labels_dir:
                for scene_name in self.scene_names:
                    label_file = self.dataset[scene_name]['_label_file']
                    label_basename = os.path.basename(label_file)
                    new_label_path = os.path.join(labels_dir, label_basename)
                    if os.path.exists(new_label_path):
                        self.dataset[scene_name]['_label_file'] = os.path.relpath(new_label_path, self.base_dir)
            
            return len(self._validate_paths_minimal()) == 0
        
        return False
    
    def load_scene_images(self, scene_name: str, crop_size: int = 60, display_mode: str = 'overlay',
                           projection: str = 'mip', overlay_segmentation: bool = False, zero_out_labels: list = None,
                           labels_to_load: Optional[List[int]] = None):
        """
        Load images for a scene based on dataset mode.
        
        Parameters:
            scene_name: Scene to load
            crop_size: Size of cropped images (minimal/tiff-based mode)
            display_mode: 'overlay' or 'montage' (minimal mode only)
            projection: Z-projection for 3D data - 'mip', 'middle', 'sum', 'volume'
            overlay_segmentation: If True, overlay segmentation masks on images (3D only)
            zero_out_labels: List of label IDs to exclude from loading
            labels_to_load: Optional list of specific label IDs to load (for batch processing)
        
        Returns:
            dict: {label: ndarray} mapping of nucleus images
        """
        if scene_name not in self.dataset:
            raise ValueError(f"Scene '{scene_name}' not found")
        
        # Check if already loaded (for non-minimal modes)
        if self.dataset_mode != 'minimal' and self.dataset[scene_name]['images'] is not None:
            print(f"Scene '{scene_name}' already loaded, returning cached images...")
            return self.dataset[scene_name]['images']
        
        if self.dataset_mode == 'legacy':
            # Already loaded
            return self.dataset[scene_name]['images']
        
        elif self.dataset_mode == 'lazy':
            # Check if tiff-based (3D on-the-fly cropping)
            print(f"Loading scene '{scene_name}' in lazy mode (tiff-based: {getattr(self, '_is_tiff_based', False)})...")
            if getattr(self, '_is_tiff_based', False):
                raise DeprecationWarning()
                # print("Detected tiff-based lazy format. Loading from 3D files with on-the-fly cropping...")
                # print(f"Cropping 3D images for scene '{scene_name}' on-the-fly...")
                # images = self._crop_images_from_tiff(scene_name, crop_size=crop_size, projection=projection,
                #                                      overlay_segmentation=overlay_segmentation, 
                #                                      zero_out_labels=zero_out_labels)
                # self.dataset[scene_name]['images'] = images
                # return images
            
            # Standard lazy: load from separate .pkl file
            rel_path = self.dataset[scene_name]['_image_file']
            full_path = os.path.join(self.base_dir, rel_path)
            
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Image file not found: {full_path}")
            
            print(f"Loading images for scene '{scene_name}' from {full_path}...")
            with open(full_path, 'rb') as f:
                images = pickle.load(f)
            
            self.dataset[scene_name]['images'] = images
            return images
        
        elif self.dataset_mode == 'minimal':
            # Crop on-the-fly from original images
            if hasattr(self, '_is_3d_file_based') and self._is_3d_file_based:
                # 3D file-based minimal format (TIFF/NIfTI/Zarr)
                print(f"Cropping 3D images for scene '{scene_name}' on-the-fly from file {self.dataset[scene_name]['_image_file']}...")
                images = self._crop_images_from_3d_files(scene_name, crop_size=crop_size, projection=projection,
                                                         overlay_segmentation=overlay_segmentation, 
                                                         zero_out_labels=zero_out_labels,
                                                         labels_to_load=labels_to_load)
                return images
            else:
                # 2D .lif-based minimal format
                print(f"Cropping images for scene '{scene_name}' on-the-fly from file {self.dataset[scene_name]['_lif_file']}...")
                return self._crop_images_minimal(scene_name, batch_labels=labels_to_load, crop_size=crop_size, display_mode=display_mode)
        
        raise ValueError(f"Unknown dataset mode: {self.dataset_mode}")
    
    def load_scene_segmentations(self, scene_name: str, crop_size: int = 60,
                                  projection: str = 'mip', zero_out_labels: list = None):
        """
        Load segmentation masks for a scene (3D datasets only).
        
        Parameters:
            scene_name: Scene to load
            crop_size: Size of cropped segmentations (tiff-based mode)
            projection: Z-projection for 3D data - 'mip', 'middle', 'sum', 'volume'
            zero_out_labels: List of label IDs to exclude from loading
        
        Returns:
            dict: {label: ndarray} mapping of binary segmentation masks
                  - For projections: 2D binary masks (Y, X)
                  - For 'volume': 3D binary masks (Z, Y, X)
        """
        if scene_name not in self.dataset:
            raise ValueError(f"Scene '{scene_name}' not found")
        
        # Check if already loaded
        if self.dataset[scene_name]['segmentations'] is not None:
            print(f"Segmentations for scene '{scene_name}' already loaded, returning cached segmentations...")
            return self.dataset[scene_name]['segmentations']
        
        if self.dataset_mode == 'legacy':
            raise NotImplementedError("Segmentation loading not supported for legacy format")
        
        elif self.dataset_mode == 'lazy':
            # Check if tiff-based (3D on-the-fly cropping)
            print(f"Loading segmentations for scene '{scene_name}' in lazy mode (tiff-based: {getattr(self, '_is_tiff_based', False)})...")
            if getattr(self, '_is_tiff_based', False):
                raise DeprecationWarning()
                # print("Detected tiff-based lazy format. Loading segmentations from 3D files with on-the-fly cropping...")
                # segmentations = self._crop_segmentations_from_tiff(scene_name, crop_size=crop_size, 
                #                                                     projection=projection,
                #                                                     zero_out_labels=zero_out_labels)
                # self.dataset[scene_name]['segmentations'] = segmentations
                # return segmentations
            else:
                raise NotImplementedError("Segmentation loading only supported for tiff-based lazy format")
        
        elif self.dataset_mode == 'minimal':
            # Check if this is 3D file-based minimal format
            if hasattr(self, '_is_3d_file_based') and self._is_3d_file_based:
                print(f"Cropping 3D segmentations for scene '{scene_name}' on-the-fly...")
                segmentations = self._crop_segmentations_from_3d_files(scene_name, crop_size=crop_size,
                                                                        projection=projection,
                                                                        zero_out_labels=zero_out_labels)
                self.dataset[scene_name]['segmentations'] = segmentations
                return segmentations
            else:
                raise NotImplementedError("Segmentation loading not supported for 2D minimal format")
        
        raise ValueError(f"Unknown dataset mode: {self.dataset_mode}")
    
    def _crop_images_minimal(self, scene_name: str, batch_labels: Optional[List[int]] = None, 
                               crop_size: int = 60, display_mode: str = 'overlay',
                               z_mode: str = 'middle') -> dict:
        """
        Crop nucleus images on-the-fly from original .lif file.
        
        Parameters:
            scene_name: Scene to load
            batch_labels: Optional list of specific labels to crop (for efficiency)
            crop_size: Size of cropped images in pixels (default: 60)
            display_mode: 'overlay' (all channels combined) or 'montage' (channel pairs in grid)
            z_mode: How to handle 3D data - 'middle' (middle slice), 'max' (max projection), 
                   'sum' (sum projection), or int (specific z-index)
        
        Returns:
            dict: {label: ndarray} mapping of cropped nucleus images
        """
        if not AICSIMAGEIO_AVAILABLE:
            raise ImportError("aicsimageio is required for minimal format. Install with: pip install aicsimageio")
        
        # Load .lif file
        lif_path = os.path.join(self.base_dir, self.dataset[scene_name]['_lif_file'])
        if not os.path.exists(lif_path):
            raise FileNotFoundError(f".lif file not found: {lif_path}")
        
        lif = AICSImage(lif_path)
        scene_name_in_lif = self.dataset[scene_name]['_scene_name']
        lif.set_scene(scene_name_in_lif)
        
        # Get full scene image
        full_image = lif.data.squeeze()  # (C, Y, X) or (C, Z, Y, X)
        
        # Load labels
        if self.dataset[scene_name]['_labels'] is None:
            label_path = os.path.join(self.base_dir, self.dataset[scene_name]['_label_file'])
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: {label_path}")
            
            if self.metadata['save_format'] == 'zarr':
                labels = zarr.load(label_path)
            else:  # npy
                labels = np.load(label_path)
            
            self.dataset[scene_name]['_labels'] = labels
        else:
            labels = self.dataset[scene_name]['_labels']
        
        # Get DataFrame for bounding boxes
        df = self.dataset[scene_name]['df']
        
        # Filter to unique labels
        if batch_labels is not None:
            df_labels = df[df['label'].isin(batch_labels)].drop_duplicates('label')
        else:
            df_labels = df.drop_duplicates('label')
        
        # Crop each nucleus
        images = {}
        size = crop_size
        
        for _, row in df_labels.iterrows():
            label = row['label']
            bbox = [int(row['bbox-0']), int(row['bbox-2']), 
                   int(row['bbox-1']), int(row['bbox-3'])]
            
            # Add padding
            dx = (size - (bbox[1] - bbox[0])) / 2
            dy = (size - (bbox[3] - bbox[2])) / 2
            
            # Ensure bounds are valid
            y_start = max(0, int(bbox[0] - np.floor(dx)))
            y_end = min(full_image.shape[-2], int(bbox[1] + np.ceil(dx)))
            x_start = max(0, int(bbox[2] - np.floor(dy)))
            x_end = min(full_image.shape[-1], int(bbox[3] + np.ceil(dy)))
            
            if len(full_image.shape) == 3:
                # 2D: (C, Y, X)
                cropped = full_image[:, y_start:y_end, x_start:x_end]
            else:
                # 3D: (C, Z, Y, X) - handle based on z_mode
                cropped_3d = full_image[:, :, y_start:y_end, x_start:x_end]
                
                if z_mode == 'middle':
                    z_mid = cropped_3d.shape[1] // 2
                    cropped = cropped_3d[:, z_mid, :, :]
                elif z_mode == 'max':
                    cropped = np.max(cropped_3d, axis=1)
                elif z_mode == 'sum':
                    cropped = np.sum(cropped_3d, axis=1).astype(cropped_3d.dtype)
                elif z_mode == 'mean':
                    cropped = np.mean(cropped_3d, axis=1).astype(cropped_3d.dtype)
                elif isinstance(z_mode, int):
                    z_idx = min(max(0, z_mode), cropped_3d.shape[1] - 1)
                    cropped = cropped_3d[:, z_idx, :, :]
                else:
                    # Default to middle slice
                    z_mid = cropped_3d.shape[1] // 2
                    cropped = cropped_3d[:, z_mid, :, :]
            
            # Resize to exact size if needed
            if cropped.shape[-2:] != (size, size):
                from skimage.transform import resize
                # Resize each channel
                channels = []
                for c in range(cropped.shape[0]):
                    resized = resize(cropped[c], (size, size), 
                                   preserve_range=True, order=1)
                    channels.append(resized)
                cropped = np.stack(channels, axis=0)
            
            # Keep as (C, Y, X) for colorify processing
            images[label] = cropped.astype(full_image.dtype)
        
        # Convert to RGB using colorify if available
        if COLORIFY_AVAILABLE and len(images) > 0:
            # Get channel colormaps from metadata if available
            cmaps = self.metadata.get('channel_cmaps', None)
            if cmaps is None:
                # Default colormaps for common channels
                n_channels = next(iter(images.values())).shape[0]
                cmaps = ['pure_blue', 'pure_green', 'pure_red', 'pure_cyan', 'pure_magenta', 'pure_yellow'][:n_channels]
            
            # Process each image
            rgba_images = {}
            for label, img in images.items():
                # img shape: (C, Y, X)
                n_channels = img.shape[0]
                
                if display_mode == 'montage' and n_channels > 1:
                    # Montage mode: create grid of channel pairs (DAPI + each other channel)
                    # Grid layout: ceil(sqrt(n_pairs)) x ceil(sqrt(n_pairs))
                    n_pairs = n_channels - 1  # DAPI + each other channel
                    grid_cols = int(np.ceil(np.sqrt(n_pairs)))
                    grid_rows = int(np.ceil(n_pairs / grid_cols))
                    
                    # Create output grid (Y, X, 4)
                    grid = np.zeros((grid_rows * size, grid_cols * size, 4), dtype=np.float32)
                    
                    for i in range(n_pairs):
                        c = i % grid_cols
                        r = i // grid_cols
                        
                        # Extract DAPI (channel 0) and current channel (i+1)
                        pair_channels = [0, i + 1]
                        pair_img = img[pair_channels]  # (2, Y, X)
                        pair_cmaps = [cmaps[0], cmaps[i + 1]]
                        
                        # Compute limits
                        pair_limits = [[np.percentile(pair_img[ci], 0), np.percentile(pair_img[ci], 100)] 
                                      for ci in range(2)]
                        
                        try:
                            converted, _, _, _ = colorify.multichannel_to_rgb(
                                pair_img,
                                pair_cmaps,
                                rescale_type='limits',
                                limits=pair_limits,
                                proj_type='sum'
                            )
                            # Place in grid
                            y_start, y_end = r * size, (r + 1) * size
                            x_start, x_end = c * size, (c + 1) * size
                            grid[y_start:y_end, x_start:x_end, :] = converted
                        except Exception as e:
                            warnings.warn(f"Failed to convert montage pair for label {label}: {e}")
                    
                    rgba_images[label] = grid.astype(np.float32)
                else:
                    # Overlay mode: all channels combined
                    # Compute limits per channel (min/max)
                    limits = [[np.percentile(img[c], 0), np.percentile(img[c], 100)] for c in range(img.shape[0])]
                    
                    # Convert to RGBA using colorify (matches legacy format)
                    try:
                        converted, _, _, _ = colorify.multichannel_to_rgb(
                            img,
                            cmaps[:n_channels],
                            rescale_type='limits',
                            limits=limits,
                            proj_type='sum'
                        )
                        # converted shape: (Y, X, 4) with RGBA in [0, 1] float32
                        rgba_images[label] = converted.astype(np.float32)
                    except Exception as e:
                        warnings.warn(f"Failed to convert image {label} to RGBA: {e}. Using raw data.")
                        # Fallback: transpose, normalize to [0, 1], and add alpha channel
                        img_transposed = img.transpose(1, 2, 0)  # (Y, X, C)
                        # Normalize to [0, 1]
                        img_normalized = (img_transposed - img_transposed.min()) / (img_transposed.max() - img_transposed.min() + 1e-8)
                        # Ensure we have 3 channels
                        if img_normalized.shape[-1] == 1:
                            img_normalized = np.repeat(img_normalized, 3, axis=-1)
                        elif img_normalized.shape[-1] > 3:
                            img_normalized = img_normalized[:, :, :3]
                        # Add alpha channel (fully opaque)
                        alpha = np.ones((*img_normalized.shape[:2], 1), dtype=np.float32)
                        rgba_images[label] = np.concatenate([img_normalized, alpha], axis=-1).astype(np.float32)
            
            images = rgba_images
        else:
            # Fallback if colorify not available: transpose to (Y, X, C) and normalize to [0, 1] with alpha
            rgba_images = {}
            for label, img in images.items():
                img_transposed = img.transpose(1, 2, 0)  # (Y, X, C)
                # Normalize to [0, 1]
                img_normalized = (img_transposed - img_transposed.min()) / (img_transposed.max() - img_transposed.min() + 1e-8)
                # Ensure 3 channels
                if img_normalized.shape[-1] == 1:
                    img_normalized = np.repeat(img_normalized, 3, axis=-1)
                elif img_normalized.shape[-1] > 3:
                    img_normalized = img_normalized[:, :, :3]
                # Add alpha channel
                alpha = np.ones((*img_normalized.shape[:2], 1), dtype=np.float32)
                rgba_images[label] = np.concatenate([img_normalized, alpha], axis=-1).astype(np.float32)
            images = rgba_images

        print(f"Cropped {len(images)} nuclei for scene '{scene_name}' (display_mode={display_mode}).")
        if len(images) > 0:
            first_img = next(iter(images.values()))
            print(f"First image shape: {first_img.shape}, dtype: {first_img.dtype}, range: [{first_img.min()}, {first_img.max()}]")
        
        return images
    
    def _load_3d_file(self, file_path: str, filetype: str, dims: str, as_int: bool = False):
        """
        Load a 3D volume from file with axis reordering.
        
        Parameters:
            file_path: Path to file
            filetype: 'tiff', 'nifti', or 'zarr'
            dims: Axis order (e.g., 'XYZ', 'ZYX')
            as_int: Convert to int32 (for segmentation)
            
        Returns:
            np.ndarray: Loaded volume in ZYX order
        """
        # Load based on file type
        if filetype == 'nifti':
            try:
                import nibabel as nib
            except ImportError:
                raise ImportError("nibabel is required for NIfTI files. Install with: pip install nibabel")
            nii = nib.load(file_path)
            data = nii.get_fdata()
            if as_int:
                data = data.astype(np.int32)
        elif filetype == 'zarr':
            try:
                import zarr
            except ImportError:
                raise ImportError("zarr is required for Zarr files. Install with: pip install zarr")
            data = zarr.load(file_path)
            if as_int:
                data = data.astype(np.int32)
        else:  # tiff
            import tifffile
            data = tifffile.imread(file_path)
            if as_int:
                data = data.astype(np.int32)
        
        # Transpose if needed
        if dims.upper() != 'ZYX':
            dims_upper = dims.upper()
            transpose_order = tuple([dims_upper.index(d) for d in 'ZYX'])
            data = np.transpose(data, transpose_order)
        
        return data
    
    # def _crop_segmentations_from_tiff(self, scene_name: str, crop_size: int = 60,
    #                                    projection: str = 'mip', zero_out_labels: list = None) -> dict:
    #     """
    #     Crop segmentation masks on-the-fly from 3D segmentation files.
        
    #     Parameters:
    #         scene_name: Scene to load
    #         crop_size: Size of cropped segmentations in Y, X (default: 60)
    #         projection: Z-projection - 'mip', 'middle', 'sum', 'mean', or 'volume' for full 3D
    #         zero_out_labels: List of label IDs to exclude from loading
            
    #     Returns:
    #         dict: {label: ndarray} mapping of binary segmentation masks
    #               - For projections: 2D binary masks (Y, X)
    #               - For 'volume': 3D binary masks (Z, Y, X)
    #     """
    #     import tifffile
    #     import cc3d
    #     from fastremap import remap as fr

        
    #     # Get file metadata
    #     segmentation_filetype = self.metadata.get('segmentation_filetype', 'tiff')
    #     segmentation_dims = self.metadata.get('segmentation_dims', 'ZYX')
        
    #     # Load segmentation file
    #     seg_path = os.path.join(self.base_dir, self.metadata['_source_segmentation'])
        
    #     if not os.path.exists(seg_path):
    #         raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
        
    #     print(f"Loading segmentation from {seg_path} ({segmentation_filetype}, dims={segmentation_dims})...")
    #     segmentation = self._load_3d_file(seg_path, segmentation_filetype, segmentation_dims, as_int=True)
    #     # segmentation = segmentation[:500, :500, :500]
    #     print(f"Loaded segmentation with shape {segmentation.shape}")

    #     if zero_out_labels is not None and len(zero_out_labels) > 0:
    #         print(f'Zeroing out labels {zero_out_labels}')
    #         segmentation = fr(segmentation, {label:0 for label in zero_out_labels}, preserve_missing_labels=True)

    #     # Get dimensions (Z, Y, X)
    #     depth, height, width = segmentation.shape
        
    #     # Get DataFrame
    #     df = self.dataset[scene_name]['df']
    #     # df_unique = df.drop_duplicates('label')
        
    #     # Check if relabel mode is enabled
    #     relabel = self.metadata.get('relabel', False)
    #     if relabel:
    #         # When relabel is enabled, we need to get unique instance labels
    #         # and use relabel_label for masking (semantic -> instance separation)
    #         if 'relabel_label' not in df.columns:
    #             raise ValueError("Dataset was created with relabel=True but 'relabel_label' column not found in DataFrame")
    #         # df_unique = df.drop_duplicates('relabel_label')

    #         print(f'Relabeling labels')
    #         segmentation = cc3d.connected_components(segmentation, connectivity=26)
        
    #     # # Filter out zero_out_labels if specified
    #     # if zero_out_labels is not None and len(zero_out_labels) > 0:
    #     #     df_unique = df_unique[~df_unique['label'].isin(zero_out_labels)]
    #     #     if len(df_unique) == 0:
    #     #         print("Warning: All labels were filtered out by zero_out_labels")
    #     #         return {}
        
    #     segmentations = {}
    #     size = crop_size
        
    #     for _, row in df.iterrows():
    #         # Use relabel_label for instance masks, or label for semantic masks
    #         if relabel:
    #             print(f"Relabel is True: taking relabel_label {row['relabel_label']} as label")
    #             label = row['relabel_label']  # Instance label from cc3d.connected_components
    #             original_label = row['label']  # Original semantic label (for display/tracking)
    #         else:
    #             label = row['label']  # Semantic label
    #             original_label = label
            
    #         # Get 3D bounding box from regionprops (z_min, y_min, x_min, z_max, y_max, x_max)
    #         z_min, y_min, x_min = int(row['bbox-0']), int(row['bbox-1']), int(row['bbox-2'])
    #         z_max, y_max, x_max = int(row['bbox-3']), int(row['bbox-4']), int(row['bbox-5'])
            
    #         # Calculate center
    #         z_center = (z_min + z_max) // 2
    #         y_center = (y_min + y_max) // 2
    #         x_center = (x_min + x_max) // 2
            
    #         # Calculate crop bounds with padding to reach crop_size
    #         half_size = size // 2
    #         z_start = max(0, z_center - half_size)
    #         z_end = min(depth, z_center + half_size)
    #         y_start = max(0, y_center - half_size)
    #         y_end = min(height, y_center + half_size)
    #         x_start = max(0, x_center - half_size)
    #         x_end = min(width, x_center + half_size)
            
    #         # Crop segmentation and create binary mask for this label
    #         seg_cropped = segmentation[z_start:z_end, y_start:y_end, x_start:x_end]
    #         print(f'Unique values in seg_cropped {np.unique(seg_cropped)}')
    #         mask_3d = (seg_cropped == label).astype(np.uint8)
    #         print(f'Sum {np.sum(mask_3d)}')
            
    #         # Handle segmentation volume/projection
    #         if projection == 'volume':
    #             # Return full 3D segmentation (no projection)
    #             # Pad to exact crop_size in Y, X if needed
    #             if mask_3d.shape[1] != size or mask_3d.shape[2] != size:
    #                 z_depth = mask_3d.shape[0]
    #                 padded = np.zeros((z_depth, size, size), dtype=mask_3d.dtype)
    #                 y_offset = (size - mask_3d.shape[1]) // 2
    #                 x_offset = (size - mask_3d.shape[2]) // 2
    #                 padded[:, y_offset:y_offset+mask_3d.shape[1], 
    #                        x_offset:x_offset+mask_3d.shape[2]] = mask_3d
    #                 mask_3d = padded
                
    #             segmentations[label] = mask_3d  # (Z, Y, X)
    #         else:
    #             # Apply Z-projection to segmentation
    #             if projection == 'mip':
    #                 projected_seg = np.max(mask_3d, axis=0)
    #             elif projection == 'middle':
    #                 z_mid = mask_3d.shape[0] // 2
    #                 projected_seg = mask_3d[z_mid, :, :]
    #             elif projection in ['sum', 'mean']:
    #                 projected_seg = (np.sum(mask_3d, axis=0) > 0).astype(np.uint8)
    #             else:
    #                 projected_seg = np.max(mask_3d, axis=0)
                
    #             # Pad to exact crop_size if needed
    #             if projected_seg.shape[0] != size or projected_seg.shape[1] != size:
    #                 padded = np.zeros((size, size), dtype=projected_seg.dtype)
    #                 y_offset = (size - projected_seg.shape[0]) // 2
    #                 x_offset = (size - projected_seg.shape[1]) // 2
    #                 padded[y_offset:y_offset+projected_seg.shape[0], 
    #                        x_offset:x_offset+projected_seg.shape[1]] = projected_seg
    #                 projected_seg = padded
                
    #             segmentations[label] = projected_seg  # (Y, X)
        
    #     print(f"Cropped {len(segmentations)} segmentations from 3D file (projection={projection}).")
    #     if len(segmentations) > 0:
    #         first_seg = next(iter(segmentations.values()))
    #         print(f"First segmentation shape: {first_seg.shape}, dtype: {first_seg.dtype}")
        
    #     return segmentations
    
    # def _crop_images_from_tiff(self, scene_name: str, crop_size: int = 60, 
    #                             projection: str = 'mip', overlay_segmentation: bool = False, zero_out_labels: list = None) -> dict:
    #     """
    #     Crop images on-the-fly from 3D files (TIFF, NIfTI, Zarr).
        
    #     Parameters:
    #         scene_name: Scene to load
    #         crop_size: Size of cropped images in Y, X (default: 60)
    #         projection: Z-projection - 'mip', 'middle', 'sum', or 'volume' for full 3D
    #         overlay_segmentation: If True, overlay segmentation masks on images
            
    #     Returns:
    #         dict: {label: ndarray} mapping of cropped images
    #               - For projections: RGBA images (Y, X, 4)
    #               - For 'volume': 3D volumes (C, Z, Y, X)
    #     """
    #     import tifffile
    #     import cc3d
    #     from fastremap import remap as fr

        
    #     # Get file metadata
    #     image_filetype = self.metadata.get('image_filetype', 'tiff')
    #     image_dims = self.metadata.get('image_dims', 'ZYX')
    #     segmentation_filetype = self.metadata.get('segmentation_filetype', 'tiff')
    #     segmentation_dims = self.metadata.get('segmentation_dims', 'ZYX')
        
    #     # Load source files
    #     img_path = os.path.join(self.base_dir, self.metadata['_source_image'])
    #     seg_path = os.path.join(self.base_dir, self.metadata['_source_segmentation'])
        
    #     if not os.path.exists(img_path):
    #         raise FileNotFoundError(f"Image file not found: {img_path}")
    #     if not os.path.exists(seg_path):
    #         raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
        
    #     # Load image
    #     print(f"Loading image from {img_path} ({image_filetype}, dims={image_dims})...")
    #     image = self._load_3d_file(img_path, image_filetype, image_dims, as_int=False)
    #     print(f"Loaded image with shape {image.shape}")

    #     # Check if relabel mode is enabled
    #     relabel = self.metadata.get('relabel', False)
        
    #     if overlay_segmentation:
    #         print(f"Loading segmentation from {seg_path} ({segmentation_filetype}, dims={segmentation_dims})...")
    #         segmentation = self._load_3d_file(seg_path, segmentation_filetype, segmentation_dims, as_int=True)
    #         # segmentation = segmentation[:500, :500, :500]
    #         print(f"Loaded segmentation with shape {segmentation.shape}")

    #         if zero_out_labels:
    #             segmentation = fr(segmentation, {label:0 for label in zero_out_labels}, preserve_missing_labels=True)

    #         if relabel:
    #             # When relabel is enabled, we need to get unique instance labels
    #             # and use relabel_label for masking (semantic -> instance separation)
    #             if 'relabel_label' not in df.columns:
    #                 raise ValueError("Dataset was created with relabel=True but 'relabel_label' column not found in DataFrame")
    #             # df_unique = df.drop_duplicates('relabel_label')

    #             segmentation = cc3d.connected_components(segmentation, connectivity=26)
    #     else:
    #         segmentation = None


        
    #     # Handle image shape: ensure (C, Z, Y, X)
    #     if len(image.shape) == 3:
    #         image = image[np.newaxis, ...]  # (Z, Y, X) -> (C, Z, Y, X)
    #     n_channels, depth, height, width = image.shape
        
    #     # Get DataFrame
    #     df = self.dataset[scene_name]['df']
    #     # df_unique = df.drop_duplicates('label')
        
    #     images = {}
    #     size = crop_size
        
    #     for _, row in df.iterrows():
    #         # Use relabel_label for instance masks, or label for semantic masks
    #         if relabel:
    #             label = row['relabel_label']  # Instance label from cc3d.connected_components
    #             original_label = row['label']  # Original semantic label (for display/tracking)
    #         else:
    #             label = row['label']  # Semantic label
    #             original_label = label
            
    #         # Get 3D bounding box from regionprops (z_min, y_min, x_min, z_max, y_max, x_max)
    #         z_min, y_min, x_min = int(row['bbox-0']), int(row['bbox-1']), int(row['bbox-2'])
    #         z_max, y_max, x_max = int(row['bbox-3']), int(row['bbox-4']), int(row['bbox-5'])
            
    #         # Calculate center for Y, X
    #         z_center = (z_min + z_max) // 2
    #         y_center = (y_min + y_max) // 2
    #         x_center = (x_min + x_max) // 2
            
    #         # Calculate crop bounds with padding for Y, X to reach crop_size
    #         half_size = size // 2
    #         z_start = max(0, z_center - half_size)
    #         z_end = min(depth, z_center + half_size)
    #         y_start = max(0, y_center - half_size)
    #         y_end = min(height, y_center + half_size)
    #         x_start = max(0, x_center - half_size)
    #         x_end = min(width, x_center + half_size)
            
    #         # Crop image
    #         cropped = image[:, z_start:z_end, y_start:y_end, x_start:x_end]  # (C, Z, Y, X)
            
    #         # Optionally crop segmentation mask for this label
    #         if overlay_segmentation:
    #             seg_cropped = segmentation[z_start:z_end, y_start:y_end, x_start:x_end]
    #             # Create binary mask for this specific label
    #             mask_3d = (seg_cropped == label).astype(np.uint8)
    #         else:
    #             mask_3d = None
            
    #         # Handle image projection/volume
    #         if projection == 'volume':
    #             # Return full 3D volume (no projection)
    #             # Pad to exact crop_size in Y, X if needed
    #             if cropped.shape[2] != size or cropped.shape[3] != size:
    #                 z_depth = cropped.shape[1]
    #                 padded = np.zeros((n_channels, z_depth, size, size), dtype=cropped.dtype)
    #                 y_offset = (size - cropped.shape[2]) // 2
    #                 x_offset = (size - cropped.shape[3]) // 2
    #                 padded[:, :, y_offset:y_offset+cropped.shape[2], 
    #                        x_offset:x_offset+cropped.shape[3]] = cropped
    #                 cropped = padded
                
    #             images[label] = cropped  # (C, Z, Y, X)
    #         else:
    #             # Apply Z-projection (only on cropped Z range)
    #             if projection == 'mip':
    #                 projected = np.max(cropped, axis=1)  # (C, Y, X)
    #             elif projection == 'middle':
    #                 z_mid = cropped.shape[1] // 2
    #                 projected = cropped[:, z_mid, :, :]
    #             elif projection == 'sum':
    #                 projected = np.sum(cropped, axis=1)
    #             elif projection == 'mean':
    #                 projected = np.mean(cropped, axis=1)
    #             else:
    #                 projected = np.max(cropped, axis=1)  # default: MIP
                
    #             # Pad to exact crop_size if needed
    #             if projected.shape[1] != size or projected.shape[2] != size:
    #                 padded = np.zeros((n_channels, size, size), dtype=projected.dtype)
    #                 y_offset = (size - projected.shape[1]) // 2
    #                 x_offset = (size - projected.shape[2]) // 2
    #                 padded[:, y_offset:y_offset+projected.shape[1], 
    #                        x_offset:x_offset+projected.shape[2]] = projected
    #                 projected = padded
                
    #             # Convert to RGBA for display
    #             # Transpose to (Y, X, C)
    #             img_t = projected.transpose(1, 2, 0).astype(np.float32)
                
    #             # Normalize each channel
    #             for c in range(img_t.shape[-1]):
    #                 ch = img_t[..., c]
    #                 img_t[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
                
    #             # Create RGB
    #             if n_channels == 1:
    #                 rgb = np.repeat(img_t, 3, axis=-1)
    #             elif n_channels == 2:
    #                 rgb = np.zeros((size, size, 3), dtype=np.float32)
    #                 rgb[..., 2] = img_t[..., 0]  # Blue
    #                 rgb[..., 1] = img_t[..., 1]  # Green
    #             else:
    #                 rgb = img_t[..., :3]
                
    #             # Overlay segmentation mask if requested
    #             if overlay_segmentation:
    #                 # Project mask using same method
    #                 if projection == 'mip':
    #                     mask_2d = np.max(mask_3d, axis=0)
    #                 elif projection == 'middle':
    #                     z_mid = mask_3d.shape[0] // 2
    #                     mask_2d = mask_3d[z_mid, :, :]
    #                 elif projection in ['sum', 'mean']:
    #                     mask_2d = (np.sum(mask_3d, axis=0) > 0).astype(np.uint8)
    #                 else:
    #                     mask_2d = np.max(mask_3d, axis=0)
                    
    #                 # Pad mask to match image size
    #                 if mask_2d.shape[0] != size or mask_2d.shape[1] != size:
    #                     mask_padded = np.zeros((size, size), dtype=np.uint8)
    #                     y_offset = (size - mask_2d.shape[0]) // 2
    #                     x_offset = (size - mask_2d.shape[1]) // 2
    #                     mask_padded[y_offset:y_offset+mask_2d.shape[0], 
    #                                x_offset:x_offset+mask_2d.shape[1]] = mask_2d
    #                     mask_2d = mask_padded
                    
    #                 # Overlay mask as yellow outline (boundary)
    #                 from scipy.ndimage import binary_erosion
    #                 eroded = binary_erosion(mask_2d)
    #                 boundary = mask_2d.astype(bool) & ~eroded
                    
    #                 # Add yellow outline to RGB (yellow = red + green)
    #                 rgb[boundary, 0] = 1.0  # Red
    #                 rgb[boundary, 1] = 1.0  # Green
    #                 rgb[boundary, 2] = 0.0  # No blue
                
    #             # Add alpha channel
    #             alpha = np.ones((size, size, 1), dtype=np.float32)
    #             img_rgba = np.concatenate([rgb, alpha], axis=-1)
                
    #             images[label] = img_rgba
        
    #     # Summary message
    #     mode_str = "images with segmentation overlay" if overlay_segmentation else "images"
    #     print(f"Cropped {len(images)} {mode_str} from 3D files (projection={projection}).")
    #     if len(images) > 0:
    #         first_img = next(iter(images.values()))
    #         print(f"First output shape: {first_img.shape}, dtype: {first_img.dtype}")
        
    #     return images
    
    def _crop_images_from_3d_files(self, scene_name: str, crop_size: int = 60, 
                                    projection: str = 'mip', overlay_segmentation: bool = False, 
                                    zero_out_labels: list = None, labels_to_load: Optional[List[int]] = None) -> dict:
        """
        Crop images on-the-fly from 3D files for minimal format (TIFF, NIfTI, Zarr).
        
        This is the minimal format version that uses image_file instead of _source_image.
        
        Parameters:
            scene_name: Scene to load
            crop_size: Size of cropped images in Y, X (default: 60)
            projection: Z-projection - 'mip', 'middle', 'sum', or 'volume' for full 3D
            overlay_segmentation: If True, overlay segmentation masks on images
            zero_out_labels: List of label IDs to exclude
            labels_to_load: Optional list of specific label IDs to crop (for batch processing)
            
        Returns:
            dict: {label: ndarray} mapping of cropped images
                  - For projections: RGBA images (Y, X, 4)
                  - For 'volume': 3D volumes (C, Z, Y, X)
        """
        try:
            import tifffile
        except ImportError:
            raise ImportError("tifffile is required. Install with: pip install tifffile")
        
        try:
            import cc3d
        except ImportError:
            raise ImportError("cc3d is required. Install with: pip install connected-components-3d")
        
        from fastremap import remap as fr
        
        # Get file metadata
        image_filetype = self.metadata.get('image_filetype', 'tiff')
        image_dims = self.metadata.get('image_dims', 'ZYX')
        
        # Load source image file - WITH CACHING to avoid re-loading on every batch!
        img_path = os.path.join(self.base_dir, self.metadata['image_file'])
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        # OPTIMIZATION: Cache the loaded image to avoid re-loading TIFF on every batch change
        cache_key = (img_path, image_filetype, image_dims)
        if not hasattr(self, '_cached_3d_images'):
            self._cached_3d_images = {}
        
        if cache_key in self._cached_3d_images:
            print(f"Using cached image for {img_path} (shape: {self._cached_3d_images[cache_key].shape})")
            image = self._cached_3d_images[cache_key]
        else:
            print(f"Loading image from {img_path} ({image_filetype}, dims={image_dims})...")
            image = self._load_3d_file(img_path, image_filetype, image_dims, as_int=False)
            print(f"Loaded image with shape {image.shape}")
            # Cache for future batches
            self._cached_3d_images[cache_key] = image

        # Check if relabel mode is enabled
        relabel = self.metadata.get('relabel', False)
        
        # Load segmentation if overlay requested
        if overlay_segmentation:
            segmentation_filetype = self.metadata.get('segmentation_filetype', 'tiff')
            segmentation_dims = self.metadata.get('segmentation_dims', 'ZYX')
            
            # Load labels from stored label file (same as cropping from)
            label_path = os.path.join(self.base_dir, self.dataset[scene_name]['_label_file'])
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: {label_path}")
            
            print(f"Loading segmentation from {label_path} ({self.metadata['save_format']})...")
            if self.metadata['save_format'] == 'zarr':
                segmentation = zarr.load(label_path)
            else:  # npy
                segmentation = np.load(label_path)
            print(f"Loaded segmentation with shape {segmentation.shape}")

            if zero_out_labels:
                segmentation = fr(segmentation, {label:0 for label in zero_out_labels}, preserve_missing_labels=True)
        else:
            segmentation = None

        # Handle image shape: ensure (C, Z, Y, X)
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]  # (Z, Y, X) -> (C, Z, Y, X)
        n_channels, depth, height, width = image.shape
        
        # Get DataFrame
        df = self.dataset[scene_name]['df']
        
        # OPTIMIZATION: Filter DataFrame to only labels we need BEFORE cropping loop
        if labels_to_load is not None:
            df = df[df['label'].isin(labels_to_load)]
            print(f"Filtering to {len(labels_to_load)} requested labels (batch processing)")
        
        images = {}
        size = crop_size
        
        for _, row in df.iterrows():
            # After renaming in _initialize_dataset, 'label' is always the correct column to use
            label = row['label']
            
            # Get 3D bounding box from regionprops (z_min, y_min, x_min, z_max, y_max, x_max)
            z_min, y_min, x_min = int(row['z_min']), int(row['y_min']), int(row['x_min'])
            z_max, y_max, x_max = int(row['z_max']), int(row['y_max']), int(row['x_max'])
            
            # Calculate center for Y, X
            z_center = (z_min + z_max) // 2
            y_center = (y_min + y_max) // 2
            x_center = (x_min + x_max) // 2
            
            # Calculate crop bounds with padding for Y, X to reach crop_size
            half_size = size // 2
            z_start = max(0, z_center - half_size)
            z_end = min(depth, z_center + half_size)
            y_start = max(0, y_center - half_size)
            y_end = min(height, y_center + half_size)
            x_start = max(0, x_center - half_size)
            x_end = min(width, x_center + half_size)
            
            # Crop image
            cropped = image[:, z_start:z_end, y_start:y_end, x_start:x_end]  # (C, Z, Y, X)
            
            # Optionally crop segmentation mask for this label
            if overlay_segmentation:
                seg_cropped = segmentation[z_start:z_end, y_start:y_end, x_start:x_end]
                # Create binary mask for this specific label
                mask_3d = (seg_cropped == label).astype(np.uint8)
            else:
                mask_3d = None
            
            # Handle image projection/volume
            if projection == 'volume':
                # Return full 3D volume (no projection)
                # Pad to exact crop_size in Y, X if needed
                if cropped.shape[2] != size or cropped.shape[3] != size:
                    z_depth = cropped.shape[1]
                    padded = np.zeros((n_channels, z_depth, size, size), dtype=cropped.dtype)
                    y_offset = (size - cropped.shape[2]) // 2
                    x_offset = (size - cropped.shape[3]) // 2
                    padded[:, :, y_offset:y_offset+cropped.shape[2], 
                           x_offset:x_offset+cropped.shape[3]] = cropped
                    cropped = padded
                
                images[label] = cropped  # (C, Z, Y, X)
            else:
                # Apply Z-projection (only on cropped Z range)
                if projection == 'mip':
                    projected = np.max(cropped, axis=1)  # (C, Y, X)
                elif projection == 'middle':
                    z_mid = cropped.shape[1] // 2
                    projected = cropped[:, z_mid, :, :]
                elif projection == 'sum':
                    projected = np.sum(cropped, axis=1)
                elif projection == 'mean':
                    projected = np.mean(cropped, axis=1)
                else:
                    projected = np.max(cropped, axis=1)  # default: MIP
                
                # Pad to exact crop_size if needed
                if projected.shape[1] != size or projected.shape[2] != size:
                    padded = np.zeros((n_channels, size, size), dtype=projected.dtype)
                    y_offset = (size - projected.shape[1]) // 2
                    x_offset = (size - projected.shape[2]) // 2
                    padded[:, y_offset:y_offset+projected.shape[1], 
                           x_offset:x_offset+projected.shape[2]] = projected
                    projected = padded
                
                # Convert to RGBA for display
                # Transpose to (Y, X, C)
                img_t = projected.transpose(1, 2, 0).astype(np.float32)
                
                # Normalize each channel
                for c in range(img_t.shape[-1]):
                    ch = img_t[..., c]
                    img_t[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
                
                # Create RGB
                if n_channels == 1:
                    rgb = np.repeat(img_t, 3, axis=-1)
                elif n_channels == 2:
                    rgb = np.zeros((size, size, 3), dtype=np.float32)
                    rgb[..., 2] = img_t[..., 0]  # Blue
                    rgb[..., 1] = img_t[..., 1]  # Green
                else:
                    rgb = img_t[..., :3]
                
                # Overlay segmentation mask if requested
                if overlay_segmentation:
                    # Project mask using same method
                    if projection == 'mip':
                        mask_2d = np.max(mask_3d, axis=0)
                    elif projection == 'middle':
                        z_mid = mask_3d.shape[0] // 2
                        mask_2d = mask_3d[z_mid, :, :]
                    elif projection in ['sum', 'mean']:
                        mask_2d = (np.sum(mask_3d, axis=0) > 0).astype(np.uint8)
                    else:
                        mask_2d = np.max(mask_3d, axis=0)
                    
                    # Pad mask to match image size
                    if mask_2d.shape[0] != size or mask_2d.shape[1] != size:
                        mask_padded = np.zeros((size, size), dtype=np.uint8)
                        y_offset = (size - mask_2d.shape[0]) // 2
                        x_offset = (size - mask_2d.shape[1]) // 2
                        mask_padded[y_offset:y_offset+mask_2d.shape[0], 
                                   x_offset:x_offset+mask_2d.shape[1]] = mask_2d
                        mask_2d = mask_padded
                    
                    # Overlay mask as yellow outline (boundary)
                    from scipy.ndimage import binary_erosion
                    eroded = binary_erosion(mask_2d)
                    boundary = mask_2d.astype(bool) & ~eroded
                    
                    # Add yellow outline to RGB (yellow = red + green)
                    rgb[boundary, 0] = 1.0  # Red
                    rgb[boundary, 1] = 1.0  # Green
                    rgb[boundary, 2] = 0.0  # No blue
                
                # Add alpha channel
                alpha = np.ones((size, size, 1), dtype=np.float32)
                img_rgba = np.concatenate([rgb, alpha], axis=-1)
                
                images[label] = img_rgba
        
        # Summary message
        mode_str = "images with segmentation overlay" if overlay_segmentation else "images"
        print(f"Cropped {len(images)} {mode_str} from 3D files (projection={projection}).")
        if len(images) > 0:
            first_img = next(iter(images.values()))
            print(f"First output shape: {first_img.shape}, dtype: {first_img.dtype}")
        
        return images
    
    def _crop_segmentations_from_3d_files(self, scene_name: str, crop_size: int = 60,
                                           projection: str = 'mip', zero_out_labels: list = None) -> dict:
        """
        Crop segmentation masks on-the-fly from 3D label files for minimal format.
        
        This is the minimal format version that uses stored label files.
        
        Parameters:
            scene_name: Scene to load
            crop_size: Size of cropped segmentations in Y, X
            projection: Z-projection - 'mip', 'middle', 'sum', or 'volume' for full 3D
            zero_out_labels: List of label IDs to exclude
        
        Returns:
            dict: {label: ndarray} mapping of binary segmentation masks
                  - For projections: 2D binary masks (Y, X)
                  - For 'volume': 3D binary masks (Z, Y, X)
        """
        from fastremap import remap as fr
        
        # Load labels from stored label file
        label_path = os.path.join(self.base_dir, self.dataset[scene_name]['_label_file'])
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        print(f"Loading segmentation from {label_path} ({self.metadata['save_format']})...")
        if self.metadata['save_format'] == 'zarr':
            segmentation = zarr.load(label_path)
        else:  # npy
            segmentation = np.load(label_path)
        print(f"Loaded segmentation with shape {segmentation.shape}")
        
        if zero_out_labels:
            segmentation = fr(segmentation, {label:0 for label in zero_out_labels}, preserve_missing_labels=True)
        
        # Get DataFrame
        df = self.dataset[scene_name]['df']
        relabel = self.metadata.get('relabel', False)
        
        segmentations = {}
        size = crop_size
        depth, height, width = segmentation.shape 
        
        for _, row in df.iterrows():
            # After renaming in _initialize_dataset, 'label' is always the correct column to use
            label = row['label']
            
            # Get 3D bounding box
            z_min, y_min, x_min = int(row['z_min']), int(row['y_min']), int(row['x_min'])
            z_max, y_max, x_max = int(row['z_max']), int(row['y_max']), int(row['x_max'])
            
            # Calculate center
            z_center = (z_min + z_max) // 2
            y_center = (y_min + y_max) // 2
            x_center = (x_min + x_max) // 2
            
            # Calculate crop bounds
            half_size = size // 2
            z_start = max(0, z_center - half_size)
            z_end = min(depth, z_center + half_size)
            y_start = max(0, y_center - half_size)
            y_end = min(height, y_center + half_size)
            x_start = max(0, x_center - half_size)
            x_end = min(width, x_center + half_size)
            
            # Crop segmentation
            seg_cropped = segmentation[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Create binary mask for this specific label
            mask_3d = (seg_cropped == label).astype(np.uint8)
            
            if projection == 'volume':
                # Return full 3D volume
                if mask_3d.shape[1] != size or mask_3d.shape[2] != size:
                    z_depth = mask_3d.shape[0]
                    padded = np.zeros((z_depth, size, size), dtype=np.uint8)
                    y_offset = (size - mask_3d.shape[1]) // 2
                    x_offset = (size - mask_3d.shape[2]) // 2
                    padded[:, y_offset:y_offset+mask_3d.shape[1], 
                           x_offset:x_offset+mask_3d.shape[2]] = mask_3d
                    mask_3d = padded
                
                segmentations[label] = mask_3d  # (Z, Y, X)
            else:
                # Apply projection
                if projection == 'mip':
                    projected_seg = np.max(mask_3d, axis=0)
                elif projection == 'middle':
                    z_mid = mask_3d.shape[0] // 2
                    projected_seg = mask_3d[z_mid, :, :]
                elif projection in ['sum', 'mean']:
                    projected_seg = (np.sum(mask_3d, axis=0) > 0).astype(np.uint8)
                else:
                    projected_seg = np.max(mask_3d, axis=0)
                
                # Pad to exact crop_size if needed
                if projected_seg.shape[0] != size or projected_seg.shape[1] != size:
                    padded = np.zeros((size, size), dtype=projected_seg.dtype)
                    y_offset = (size - projected_seg.shape[0]) // 2
                    x_offset = (size - projected_seg.shape[1]) // 2
                    padded[y_offset:y_offset+projected_seg.shape[0], 
                           x_offset:x_offset+projected_seg.shape[1]] = projected_seg
                    projected_seg = padded
                
                segmentations[label] = projected_seg  # (Y, X)
        
        print(f"Cropped {len(segmentations)} segmentations from 3D file (projection={projection}).")
        if len(segmentations) > 0:
            first_seg = next(iter(segmentations.values()))
            print(f"First segmentation shape: {first_seg.shape}, dtype: {first_seg.dtype}")
        
        return segmentations
    
    def get_dataset_info(self) -> dict:
        """Get information about the loaded dataset."""
        return {
            'mode': self.dataset_mode,
            'n_scenes': len(self.scene_names),
            'scene_names': self.scene_names,
            'base_dir': self.base_dir,
            'metadata': self.metadata,
            'ndim': self.ndim,
            'is_3d': self.is_3d(),
            'is_tiff_based': getattr(self, '_is_tiff_based', False)
        }
    
    def get_image_size(self, scene_name: str) -> Optional[Tuple[int, int]]:
        """
        Get the image dimensions (height, width) for a scene.
        
        For minimal mode, reads from the .lif file.
        For legacy/lazy modes, gets from first loaded image.
        
        Parameters:
            scene_name: Scene to get dimensions for
            
        Returns:
            Tuple of (height, width) or None if not determinable.
        """
        if self.dataset_mode == 'minimal':
            if not AICSIMAGEIO_AVAILABLE:
                return None
            try:
                lif_path = os.path.join(self.base_dir, self.dataset[scene_name]['_lif_file'])
                if not os.path.exists(lif_path):
                    return None
                lif = AICSImage(lif_path)
                scene_name_in_lif = self.dataset[scene_name]['_scene_name']
                lif.set_scene(scene_name_in_lif)
                # Get shape: could be (C, Y, X) or (C, Z, Y, X)
                shape = lif.data.squeeze().shape
                if len(shape) == 3:
                    return (shape[1], shape[2])  # (Y, X)
                else:
                    return (shape[2], shape[3])  # (Y, X) from 3D
            except Exception:
                return None
        else:
            # For legacy/lazy, try to get from first image
            if self.dataset[scene_name]['images'] is not None:
                first_img = next(iter(self.dataset[scene_name]['images'].values()), None)
                if first_img is not None:
                    return first_img.shape[:2]
            return None
    
    def set_channel_cmaps(self, cmaps: List[str]):
        """
        Set colormaps for each channel in minimal mode.
        
        Parameters:
            cmaps: List of colormap names (e.g., ['pure_blue', 'pure_green', 'pure_red'])
                   Must be valid microfilm colormaps.
        """
        self.metadata['channel_cmaps'] = cmaps
    
    def get_channel_cmaps(self) -> Optional[List[str]]:
        """
        Get the current channel colormaps.
        
        Returns:
            List of colormap names, or None if not set.
        """
        return self.metadata.get('channel_cmaps', None)
    
    def get_available_cmaps(self) -> List[str]:
        """
        Get list of available microfilm colormaps.
        
        Returns:
            List of available colormap names.
        """
        if COLORIFY_AVAILABLE:
            # Common microfilm colormaps
            return [
                'pure_blue', 'pure_green', 'pure_red', 
                'pure_cyan', 'pure_magenta', 'pure_yellow',
                'gray', 'hot', 'viridis', 'plasma', 'inferno', 'magma',
                'bone', 'copper', 'cool', 'spring', 'summer', 'autumn', 'winter'
            ]
        return []
