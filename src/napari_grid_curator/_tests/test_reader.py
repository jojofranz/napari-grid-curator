from napari_grid_curator import napari_get_reader


def test_get_reader_accepts_pkl():
    assert callable(napari_get_reader("dataset.pkl"))
    assert callable(napari_get_reader(["dataset.pkl"]))


def test_get_reader_rejects_other_files():
    assert napari_get_reader("image.tif") is None
