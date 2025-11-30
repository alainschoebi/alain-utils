# alain-utils
Various Python utilities including loggers, SE(3) poses, colors, 2D bounding boxes, function decorators, Kalman filter, Plotly functions etc.

## Installation
Clone and install the package via `pip`:
```sh
git clone git@github.com:alainschoebi/alain-utils.git
pip install ./alain-utils
```

To install the package in editable mode for development:
```sh
git clone git@github.com:alainschoebi/alain-utils.git
pip install -e ./alain-utils
```

## Dependencies and additional packages
The `alain-utils` package depends on `pre-commit`, `pytest`, `numpy`, `colorama`, `scipy`, and `matplotlib`. These dependencies are installed automatically when running the installation command above.

Some specialized utilities in `alain-utils` rely on additional packages and are only available if the corresponding packages are installed. Optional dependencies include `plotly`, `cv2`, `tqdm`, `shapely`, `pycolmap`, `cython-bbox`, `imageio[ffmpeg]`, and ROS.

## Tests
To run tests:
```sh
cd alain-utils
python -m pytest
```
Note that simply running `pytest` might not work and lead to `ImportError`.
