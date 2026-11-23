"""
Start napari with a custom Qt scale factor, e.g. to make menus and widgets
smaller on small laptop screens.

Usage:
    python napari_scaled.py                 # scale 0.7
    python napari_scaled.py --scale 0.8
    python napari_scaled.py --plugin        # also open the Grid Curator widget
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scale', type=float, default=0.7,
                        help='Qt scale factor (1.0 = normal size, <1 = smaller). Default: 0.7')
    parser.add_argument('--plugin', action='store_true',
                        help='Open the Grid Curator widget on startup.')
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error('--scale must be > 0')

    # Must be set before the Qt application is created (i.e. before napari starts Qt).
    os.environ['QT_SCALE_FACTOR'] = str(args.scale)

    import napari

    viewer = napari.Viewer()
    if args.plugin:
        viewer.window.add_plugin_dock_widget('napari-grid-curator', 'Grid Curator')
    napari.run()


if __name__ == '__main__':
    main()
