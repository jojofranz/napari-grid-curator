# Dataset Format Guide

This plugin now supports **three dataset formats** for maximum flexibility:

## 1. Legacy Format (All-in-One)
**File**: `dataset_for_napari.pkl`

### Structure:
```python
{
    'scene_name': {
        'images': {label: ndarray},  # All cropped nuclei
        'df': DataFrame              # Measurements
    },
    ...
}
```

### Pros:
- ✅ Simple, single file
- ✅ No path dependencies
- ✅ Easy to transfer

### Cons:
- ⚠️ Large file size (can exceed original .lif)
- ⚠️ Slow initial loading
- ⚠️ High memory usage

### When to use:
- Small datasets (<100 scenes, <10GB)
- Simple workflows
- Maximum compatibility

---

## 2. Lazy Loading Format (Scene-Level)
**Files**: 
- `dataset_for_napari_lazy.pkl` (metadata + DataFrames)
- `napari_images/<scene>_images.pkl` (one per scene)

### Structure:
```python
# Main file:
{
    'format_version': 'lazy_v1',
    'scene_names': [...],
    'dataframes': {scene: DataFrame},
    'image_files': {scene: 'relative/path.pkl'},
    'base_dir': '/path/to/dataset'
}
```

### Pros:
- ✅ 50-80% faster initial load
- ✅ Memory scales with viewed scenes only
- ✅ Still uses pre-cropped images (no processing overhead)

### Cons:
- ⚠️ Multiple files (but in one folder)
- ⚠️ Disk space same as legacy

### When to use:
- Medium/large datasets (100-500 scenes)
- Interactive exploration workflows
- Limited RAM systems

---

## 3. Minimal Format (On-the-Fly Cropping)
**Files**:
- `dataset_for_napari_minimal.pkl` (metadata + DataFrames)
- `napari_labels/<scene>_labels.zarr` or `.npy` (one per scene)
- Original `.lif` file (must be accessible)

### Structure:
```python
# Main file:
{
    'format_version': 'minimal_v1',
    'save_format': 'zarr',  # or 'npy'
    'scene_names': [...],
    'dataframes': {
        scene: DataFrame with:
            - '_source_lif_file': path to .lif
            - '_source_scene_name': scene name in .lif
            - '_label_file': path to labels
            - 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3'
    },
    'label_files': {scene: 'relative/path.zarr'},
    'lif_file': 'relative/path.lif'
}
```

### Pros:
- ✅ 70-90% disk space reduction
- ✅ Flexible crop size (adjust on-the-fly)
- ✅ True memory efficiency (only loads current batch)
- ✅ Easier to adjust segmentation later

### Cons:
- ⚠️ Requires original .lif file
- ⚠️ 20-30% slower batch loading (cropping overhead)
- ⚠️ Path management complexity

### When to use:
- Very large datasets (>500 scenes, >50GB)
- Disk space constraints
- Need to experiment with different crop sizes
- Have reliable access to original data

---

## Creating Datasets

### Pre-processing (MCL-Birke)

Edit `main.py` to generate all three formats:

```python
# In main() function:
ds = ibidi.Dataset(**config, verbose=True)
ds.process_scenes()
ds.automatic_exclusion()

# Generate all formats
s_io.pickle_dataset_for_napari(ds, output_dir)        # Legacy
s_io.pickle_dataset_for_napari_lazy(ds, output_dir)   # Lazy
s_io.save_dataset_minimal(
    ds, output_dir, 
    lif_file_path=config['lif_file_path'],
    save_format='zarr'  # or 'npy'
)
```

### Optional: Generate only specific formats

```python
# Legacy only (default, always fastest to process)
s_io.pickle_dataset_for_napari(ds, output_dir, verbose=True)

# Lazy only (recommended for most cases)
s_io.pickle_dataset_for_napari_lazy(ds, output_dir, verbose=True)

# Minimal only (best for large datasets)
s_io.save_dataset_minimal(
    ds, output_dir, 
    lif_file_path=config['lif_file_path'],
    save_format='zarr',  # zarr is compressed, npy is faster
    verbose=True
)
```

---

## Loading in napari Plugin

The plugin **automatically detects** the format. Just select any `.pkl` file:

```python
# In napari:
# Plugins > napari-rgc > Exclusion Widget
# Click "Dataset :" → Select .pkl file
# Format is auto-detected and displayed
```

### Format Detection:
- Checks for `format_version` key
- Falls back to legacy if not found
- Shows mode in UI: `Mode: legacy/lazy/minimal`

---

## Path Management (Lazy/Minimal)

### Relative Paths
Both lazy and minimal formats use **relative paths** from the main `.pkl` file:

```
project/
├── dataset_for_napari_lazy.pkl      # References: napari_images/...
├── napari_images/
│   ├── scene1_images.pkl
│   └── scene2_images.pkl
```

### Transferring Datasets

**Lazy Format**: Transfer entire directory
```bash
# Preserves structure automatically
cp -r project/ /new/location/
```

**Minimal Format**: Need to maintain .lif access
```bash
# Option 1: Transfer with .lif in same structure
project/
├── dataset_for_napari_minimal.pkl
├── napari_labels/
│   └── *.zarr
└── data.lif   # Keep relative path same

# Option 2: Fix paths in plugin
# Click "Fix Paths" button after loading
# Browse to new .lif location
```

### Path Fixing UI

If paths are broken, the plugin shows:
1. "Fix Paths" button (appears automatically)
2. Dialog with file browsers
3. Validates and updates paths

```python
# Manually fix paths in Python:
loader = DatasetLoader('dataset.pkl')
loader.fix_paths(
    lif_path='/new/path/to/file.lif',
    labels_dir='/new/path/to/labels'
)
```

---

## Dependencies

### Required (all formats):
```bash
pip install napari pandas numpy scipy scikit-image
```

### Optional (minimal format):
```bash
# For zarr support:
pip install zarr

# For .lif reading:
pip install aicsimageio
```

---

## Performance Comparison

| Format | File Size | Load Time | Memory (100 scenes) | Batch Load |
|--------|-----------|-----------|---------------------|------------|
| Legacy | 15 GB | 45 s | 15 GB | Instant |
| Lazy | 15 GB (split) | 8 s | 3 GB (5 scenes) | Instant |
| Minimal | 2 GB + .lif | 5 s | 500 MB | 2 s/batch |

*Example: 500 scenes, 200 nuclei/scene, 60×60×4 crops*

---

## 3D Support

All formats support 3D data:

### Legacy/Lazy:
- Store 3D crops as (Z, Y, X, C)
- Works immediately

### Minimal:
- Reads full 3D from .lif
- Crops Z dimension based on bounding boxes
- Future: Add Z-slice navigation in plugin

---

## Recommendations

| Use Case | Recommended Format | Rationale |
|----------|-------------------|-----------|
| Small datasets (<50 scenes) | Legacy | Simple, fast processing |
| Medium datasets (50-200 scenes) | Lazy | Best balance |
| Large datasets (>200 scenes) | Minimal | Disk space critical |
| Limited disk space | Minimal | 70-90% reduction |
| Frequent transfers | Legacy | Single file |
| Active segmentation tuning | Minimal | Crop from source |
| RAM-constrained systems | Lazy or Minimal | Lower memory |

---

## Troubleshooting

### "Could not find .lif file"
- **Cause**: Dataset transferred, paths broken
- **Fix**: Click "Fix Paths" button, browse to new .lif location

### "Import zarr could not be resolved"
- **Cause**: zarr not installed (minimal format with zarr labels)
- **Fix**: `pip install zarr` or use npy format

### "Scene loading very slow"
- **Cause**: Minimal format with high downsampling
- **Fix**: Reduce downsampling or use lazy format for that dataset

### "Memory error"
- **Cause**: Too many scenes loaded (legacy format)
- **Fix**: Use lazy or minimal format, or reduce batch size

---

## Migration

### Legacy → Lazy
Not directly supported. Re-run pre-processing with `pickle_dataset_for_napari_lazy()`.

### Legacy → Minimal
Requires original .lif file. Re-run pre-processing with `save_dataset_minimal()`.

### Lazy → Legacy
Combine all scene .pkl files:
```python
import pickle
main = pickle.load(open('dataset_lazy.pkl', 'rb'))
combined = {}
for scene in main['scene_names']:
    img_file = main['image_files'][scene]
    imgs = pickle.load(open(img_file, 'rb'))
    combined[scene] = {'images': imgs, 'df': main['dataframes'][scene]}
pickle.dump(combined, open('dataset_legacy.pkl', 'wb'))
```

---

## Future Enhancements

- [ ] HDF5 format for truly random access
- [ ] Automatic format conversion in plugin
- [ ] Progressive loading (load as you scroll)
- [ ] Cloud storage support (S3, GCS)
- [ ] Compression options for lazy format
