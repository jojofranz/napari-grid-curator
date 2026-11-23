"""
Start napari with a smaller UI, e.g. for small laptop screens.

Two independent knobs:
  --font-size  napari's own font size (default 7; napari's default is 9). Safe on any screen.
               Only applies to this session; saved napari settings are not changed.
  --scale      Qt scale factor for the whole UI (QT_SCALE_FACTOR; default 1.0 = off).
               Only works if the screen's effective pixel ratio stays >= 1, i.e. on screens
               with OS display scaling (e.g. Windows 150% + --scale 0.7 -> 1.05). On a 100%
               screen, values < 1 make Qt5 render menus and widgets incompletely.

Usage:
    python napari_scaled.py                      # font size 7
    python napari_scaled.py --font-size 8
    python napari_scaled.py --scale 0.8          # on a high-DPI laptop screen
    python napari_scaled.py --plugin             # also open the Grid Curator widget
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--font-size', type=int, default=7,
                        help="napari font size (napari default: 9). Default: 7")
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Qt scale factor (1.0 = off). See notes above. Default: 1.0')
    parser.add_argument('--plugin', action='store_true',
                        help='Open the Grid Curator widget on startup.')
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error('--scale must be > 0')
    if args.font_size <= 0:
        parser.error('--font-size must be > 0')

    # Both must be set before napari/Qt start. The env var overrides napari's
    # appearance.font_size for this process only (nothing is saved).
    os.environ['NAPARI_APPEARANCE_FONT_SIZE'] = str(args.font_size)
    if args.scale != 1.0:
        os.environ['QT_SCALE_FACTOR'] = str(args.scale)

    import napari
    from qtpy.QtWidgets import QApplication

    viewer = napari.Viewer()

    ratio = QApplication.instance().primaryScreen().devicePixelRatio()
    if ratio < 1:
        print(f"WARNING: --scale {args.scale} gives a device pixel ratio of {ratio:.2f} (< 1) on this "
              f"screen. Qt5 does not render this correctly (missing text/widgets). Use a larger "
              f"--scale (>= {args.scale / ratio:.2f}) or only --font-size.")

    if args.plugin:
        viewer.window.add_plugin_dock_widget('napari-grid-curator', 'Grid Curator')
    napari.run()


if __name__ == '__main__':
    main()
