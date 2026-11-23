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
  - **Quick Filters**: DAPI intensity, nearest-neighbor distance and minimum nucleus area sliders
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

Or create a fresh conda/mamba environment with napari and the plugin, using
[`environment.yml`](https://github.com/jojofranz/napari-grid-curator/blob/main/environment.yml)
(download it, then run in the same folder):

```bash
mamba env create -f environment.yml      # or: conda env create -f environment.yml
mamba activate napari-grid-curator
napari
```

Or create the same environment by hand, without the `.yml` file:

```bash
mamba create -n napari_rgc python=3.11 "napari=0.6" "pyqt=5" "libblas=*=*openblas" "numcodecs<0.16" pip
mamba activate napari_rgc
python -m pip install napari-grid-curator      # or e.g. napari-grid-curator==0.1.1
python -m napari
```

Notes for conda/mamba environments:
- Make sure the environment is **activated** before installing with pip. `python -m pip` and
  `python -m napari` always use the active environment's Python.
- Don't run `pip install -U napari` or `pip install "napari[all]"` in a conda environment.
  That replaces conda's napari/Qt with pip's and breaks Qt on Windows
  (`DLL load failed while importing QtWidgets`).
- The version pins avoid known problems: the plugin currently needs Qt5
  (newer napari versions, e.g. 0.10, require Qt6). `aicsimageio` needs `zarr<2.16`, which only works with
  `numcodecs<0.16`. On some Windows machines the MKL maths library crashes, so OpenBLAS is used.

Or install latest development version:

```bash
pip install git+https://github.com/jojofranz/napari-grid-curator.git
```

### Installing a specific version (upgrade / downgrade)

Run these in the environment that runs napari, then restart napari:

```bash
pip install --upgrade napari-grid-curator          # latest release
pip install napari-grid-curator==0.1.1             # a specific version (also downgrades)
pip index versions napari-grid-curator             # list available versions
pip show napari-grid-curator                       # show the installed version
```

Any tagged version can also be installed directly from GitHub:

```bash
pip install git+https://github.com/jojofranz/napari-grid-curator.git@v0.1.1
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

> ⚠️ **Warning: changing exclusion criteria resets manual exclusions.** Whenever you change the
> *exclusion metrics* or any *quick filter* (DAPI intensity, min distance, min area: checkbox
> or slider), the `exclude` status of **all** nuclei in **all** scenes is recalculated from
> those criteria alone. All **Alt + Click** include/exclude decisions made before are lost.
> Changing the exclusion metrics also drops the quick-filter exclusions until a quick filter
> is changed again.
>
> **Recommended workflow:** first set the exclusion metrics and quick filters, then do the
> manual Alt + Click curation, then save. Don't touch the criteria after manual curation.
> Marker positivity (**Shift + Click**, thresholds) is not affected.

### Quick filters

- **DAPI intensity min**: exclude nuclei with a mean DAPI intensity (`DAPI_inside`, `intensity_mean`) below the threshold.
- **Min distance (px)**: exclude nuclei closer than the threshold to their nearest neighbour or to the image border.
- **Min area (px)**: exclude nuclei with an area (`DAPI_inside`, `area`, in pixels) below the threshold.

Each filter is switched on with its checkbox. A nucleus excluded by a filter is excluded in all channels.

### Small screens: smaller napari menus

On small laptop screens the napari menus and the Grid Curator widget may not fit.
[`scripts/napari_scaled.py`](https://github.com/jojofranz/napari-grid-curator/blob/main/scripts/napari_scaled.py) starts napari with a custom Qt scale factor
(`QT_SCALE_FACTOR`), which shrinks all menus and widgets:

```bash
python scripts/napari_scaled.py                  # scale 0.7 (default)
python scripts/napari_scaled.py --scale 0.8      # 1.0 = normal size
python scripts/napari_scaled.py --plugin         # also open the Grid Curator widget
```

Alternatively, set the environment variable yourself before starting napari, e.g.
`set QT_SCALE_FACTOR=0.7` (Windows cmd) or `export QT_SCALE_FACTOR=0.7` (Linux/macOS), then run `napari`.

### Supported Dataset Formats

The plugin works with three dataset modes:

1. **Legacy Mode**: All images stored in single `.pkl` file  
2. **Lazy Mode**: Images split across per-scene `.pkl` files  
3. **Minimal Mode**: References to original `.lif` or `.tif` files with on-the-fly cropping

See [DATASET_FORMATS.md](DATASET_FORMATS.md) for detailed format specifications.

### Key Bindings

- **Alt + Click**: Toggle cell exclusion (include/exclude)
- **Shift + Click**: Toggle positive marker (for thresholding)
- **C**: Toggle the segmentation boundary overlay on DAPI (same as the checkbox; the viewer must have focus)
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
- [matplotlib] for histogram visualization

Development of this plugin was funded by [NL-BioImaging AM](https://www.nwo.nl/projecten/184036012) and is happening at the [MCL](https://www.maastrichtuniversity.nl/research/microscopy-core-lab).

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
[matplotlib]: https://matplotlib.org/
[MIT]: http://opensource.org/licenses/MIT
