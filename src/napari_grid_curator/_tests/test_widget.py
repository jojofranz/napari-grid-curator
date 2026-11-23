from napari_grid_curator import ExclusionWidget


def test_widget_can_be_created(make_napari_viewer):
    """Smoke test: the widget builds without a dataset loaded."""
    viewer = make_napari_viewer()
    widget = ExclusionWidget(viewer)
    assert widget.layout() is not None
