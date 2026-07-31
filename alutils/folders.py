# Python
import os, os.path, shutil
from pathlib import Path, PosixPath, WindowsPath

# Logging
from .loggers import get_logger
logger = get_logger(__name__)

def create_folders(*args: str | Path) -> None:
    """
    Create folders at all the given locations.

    Warning:
        If some of the folders already exist, thes folder and their content are
        deleted before them being created.
    """
    for folder in args:
        folder = Path(folder)
        assert type(folder) == PosixPath or type(folder) == WindowsPath
        create_folder(folder)

def create_folder(folder: str | Path) -> None:
    """
    Create a folder at a specifc location.

    Warning:
        If the folder already exists, the folder and its content are deleted
        before the folder is created.
    """

    # Turn into Path
    folder = Path(folder)

    # If the folder does not exist, create it
    if not folder.exists():
        folder.mkdir()

    # If it already exists, empty its content
    else:
        for filename in os.listdir(str(folder)):
            file_path = os.path.join(str(folder), filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(f"Error deleting {file_path}: {e}")
