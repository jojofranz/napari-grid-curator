# napari-grid-curator

[![License MIT](https://img.shields.io/pypi/l/napari-grid-curator.svg?color=green)](https://github.com/jojofranz/napari-grid-curator/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-grid-curator.svg?color=green)](https://pypi.org/project/napari-grid-curator)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-grid-curator.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-grid-curator)](https://napari-hub.org/plugins/napari-grid-curator)

Interactive batch annotation tool for microscopy datasets with tiled grid visualization.

----------------------------------

## Overview

**napari-grid-curator** is a [napari] plugin for efficient manual curation of large microscopy datasets. The key feature is a **tiled grid layout** that displays many segmented objects simultaneously, allowing rapid quality control and annotation.

### Key Features

- **Tiled Grid Layout**: View 100+ cells at once in an organized montage
- **Batch Processing**: Efficiently navigate through thousands of segmented objects
- **2D & 3D Support**: Works with both 2D image stacks and 3D volumetric data with scrollable Z-stacks
- **Dual View Modes**:
  - **Mosaic Mode**: See all cells in batch simultaneously
  - **Slide Mode**: Navigate cell-by-cell with arrow keys
- **Interactive Annotation**:
  - **Alt + Click**: Exclude/include individual cells
  - **Shift + Click**: Mark cells as positive (thresholding)
  - **Quick Filters**: DAPI intensity and nearest-neighbor distance sliders
- **Flexible Data Loading**: 
  - Legacy mode (all data in single `.pkl`)
  - Lazy mode (split per-scene `.pkl` files)
  - Minimal mode (on-the-fly cropping from original `.lif`/`.tif` files)
- **Multi-channel Thresholding**: Set per-channel thresholds with interactive histograms
- **Smart Caching**: Efficient caching for large 3D wholebrain datasets

## Installation

Install directly from PyPI:

```bash
pip install napari-grid-curator
```

Or install latest development version:

```bash
pip install git+https://github.com/jojofranz/napari-grid-curator.git
```

## Usage

### From napari GUI

1. Open napari
2. Go to **Plugins > Grid Curator**
3. Load your dataset (`.pkl` file)
4. Use mouse interactions to annotate:
   - **Alt + Click**: Toggle cell inclusion/exclusion
   - **Shift + Click**: Mark cell as positive
5. Set thresholds per channel using histogram widget
6. Export annotated dataset when done

### Supported Dataset Formats

The plugin works with three dataset modes:

1. **Legacy Mode**: All images stored in single `.pkl` file  
2. **Lazy Mode**: Images split across per-scene `.pkl` files  
3. **Minimal Mode**: References to original `.lif` or `.tif` files with on-the-fly cropping

See [DATASET_FORMATS.md](DATASET_FORMATS.md) for detailed format specifications.

### Key Bindings

- **Alt + Click**: Toggle cell exclusion (include/exclude)
- **Shift + Click**: Toggle positive marker (for thresholding)
- **Arrow Keys**: Navigate between cells in slide mode (Left/Right)
- **Mouse Wheel**: Scroll through Z-slices (3D mode)

## Use Cases

This plugin was originally developed for retinal ganglion cell (RGC) analysis but is applicable to:

- Quality control of automated segmentations
- Manual classification of cell types
- Thresholding based on marker expression
- Excluding edge artifacts or missegmented objects
- Any workflow requiring rapid inspection of many segmented objects

## Development

This plugin was generated using the [napari-plugin-template] and follows napari plugin best practices.

### Dataset Preparation

Your dataset should include:
- Images (2D or 3D, multi-channel)
- Segmentation labels
- Metadata table with regionprops (e.g., from `skimage.measure.regionprops_table`)

The plugin uses bounding boxes to crop regions around each segmented object for efficient visualization.

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/jojofranz/napari-grid-curator).

## License

Distributed under the terms of the [MIT] license. "napari-grid-curator" is free and open source software.

## Issues

If you encounter any problems, please [file an issue](https://github.com/jojofranz/napari-grid-curator/issues) with:
- Dataset format and size
- Full error traceback
- napari and plugin versions (`napari --info`)

## Acknowledgements

This plugin was developed for wholebrain RGC analysis and uses (amongst others):
- [napari] for visualization
- [magicgui] for GUI widgets
- [napari-clusters-plotter] for histogram visualization

Development of this plugin was funded by [NL-BioImaging AM](https://www.nwo.nl/projecten/184036012) and is happening in at the [MCL](https://www.maastrichtuniversity.nl/research/microscopy-core-lab).

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[magicgui]: https://github.com/pyapp-kit/magicgui
[napari-clusters-plotter]: https://github.com/BiAPoL/napari-clusters-plotter
[MIT]: http://opensource.org/licenses/MIT
