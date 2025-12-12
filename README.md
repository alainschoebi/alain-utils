# alain-utils
A collection of Python utilities for various tasks, including:
- SE(3) poses
- 2D bounding boxes
- colors
- loggers
- function decorators
- Kalman filtering
- Plotly visualization functions
- and more...

## Installation
### Clone and install
Clone and install the package via `pip`:
```sh
git clone git@github.com:alainschoebi/alain-utils.git
cd alain-utils
pip install .
```

### Editable mode
To install the package in **editable mode** for development, use the `-e` option:
```sh
pip install -e .
```

### Install with all optional dependencies
To leverage **all utilities**, use `[all]` to install all optional dependencies:
```sh
pip install .[all]
```

> [!NOTE]
> The `alain-utils` package depends on `pre-commit`, `pytest`, `numpy`, `colorama`, `scipy`, and `matplotlib`. These are installed automatically.
> Some specialized utilities are optional and rely on additional packages. Optional dependencies include `plotly`, `opencv`, `tqdm`, `shapely`, `pycolmap`, `cython-bbox`, `imageio[ffmpeg]`, and ROS.

## Examples
### SE(3) poses
```python
from alutils import Pose
poses = [Pose.random() for _ in range(5)]
Pose.show(poses)
```

### 2D bounding boxes
```python
from alutils import BBox
bboxes = [BBox.random() for _ in range(5)]
BBox.show(bboxes)
```

### Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
from alutils import get_logger
logger = get_logger(__name__)
logger.debug("This is a debug log.")
logger.info("This is an information log.")
logger.info("**This is a bold information log.**")
logger.info("This is a success.")
logger.warning("This is a warning.")
```

## Tests
To run tests:
```sh
cd alain-utils
python -m pytest
```
> [!WARNING]
> Simply running `pytest` might not work and lead to `ImportError`.
