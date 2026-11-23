"""Grid montage: RGBA overlay crops and raw single-channel crops of varying size."""
import numpy as np
from napari_grid_curator.proc import partitions_to_grid

def _run(images):
    labels = np.array(list(images))
    out = partitions_to_grid(images, labels, labels[:3], labels[3:])
    return out[0]

def test_rgba_float():          # the case that crashed (overlay mode, (60,60,4) float32)
    imgs = {i: np.random.rand(60, 60, 4).astype(np.float32) for i in range(1, 8)}
    g = _run(imgs); assert g.ndim == 3 and g.shape[-1] == 4

def test_raw_uint16_varying_size():   # Virginia case
    imgs = {i: np.random.randint(0, 4000, (50 + i, 40 + 2 * i), dtype=np.uint16) for i in range(1, 8)}
    g = _run(imgs); assert g.ndim == 2 and g.dtype == np.uint16
