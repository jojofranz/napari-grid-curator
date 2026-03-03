# Grid Curator

**Interactive batch annotation for microscopy datasets with tiled grid visualization**

A napari plugin for manual curation of microscopy datasets. The key feature is a **tiled grid layout** that displays many segmented objects simultaneously, enabling quality control and annotation.

## Key Features

### Two View Modes
- **Mosaic Mode**: Grid of all segmented objects in current batch (default)
- **Slide Mode**: Navigate object-by-object for inspection in the context of the whole image

### Interactive Annotation
- **Alt + Click**: Toggle cell inclusion/exclusion
- **Shift + Click**: Mark cells as marker-positive/negative (for thresholding)
- **Quick Filters**: DAPI intensity and nearest-neighbor distance sliders

### Flexible Data Loading
Three dataset format modes to fit your workflow:
- **Legacy**: pre-cropped images in single `.pkl` file (containing images and dataframe)
- **Lazy**: Images split per-scene (moderate memory usage)
- **Minimal**: On-the-fly cropping from original `.lif`/`.tif` files (lowest memory footprint)

### Conveniences
- Configurable batch sizes (default: 100 cells)
- Downsampling for faster rendering
- Configurable bbox padding
- 3D and multi-channel support

## Status

Early release - stay tuned for updates. We are currently working on generalizing this plugin for broader use.

## Get Started

Install from PyPI:
```bash
pip install napari-grid-curator
```


Then open napari and find the plugin under **Plugins > Grid Curator (Grid Curator)**.

See the [README](https://github.com/jojofranz/napari-grid-curator#readme) for detailed usage instructions and dataset format specifications.

## Questions or Issues?

- [Open an issue](https://github.com/jojofranz/napari-grid-curator/issues)
- [View documentation](https://github.com/jojofranz/napari-grid-curator/blob/main/DATASET_FORMATS.md)

---

Development of this plugin was funded by [NL-BioImaging AM](https://www.nwo.nl/projecten/184036012) and is happening in at the [MCL](https://www.maastrichtuniversity.nl/research/microscopy-core-lab).
