# Typing
from typing import Optional, Literal

# NumPy
import numpy as np

# Python
import math
from pathlib import Path
import argparse
try:
    from tqdm import tqdm
except:
    pass

try:
    import natsort
except:
    natsort = None

# ImageIO
try:
    import imageio.v2 as imageio
except:
    pass

# Utils
from .decorators import requires_package
from .color import Color

# Logging
import logging
from .loggers import get_logger
logger = get_logger(__name__)

@requires_package('imageio', 'tqdm')
def generate_video(
    images: list[Path] | list[str],
    video_filename: str | Path,
    fps: Optional[int] = None,
    duration: Optional[float] = None,
    padding_color: Color | None = None,
    no_resize_warning: bool = False,
    ) -> None:
    """
    Generate a video from a list of images using imageio.

    Inputs
    - images: `list[Path | str] list of images to be included in the video.
    - video_filename: `str | Path` name of the output video file.

    Partially Required Inputs
    - fps: `int` frame rate per seconds of the video.
    - duration: `float` duration of the video in seconds.

    Optional Inputs
    - padding_color: `Color` color to be used for padding the images. If None,
                     the images will be padded with the edge color of the image.
    - no_resize_warning: `bool` if True, no warning will be raised if the image
                         dimensions are not divisible by the macro block size.

    Note: ImageIO automatically resizes the images such that the dimensions are
          divisible by the `macro_block_size`. This is done to ensure that the
          videos are properly encoded and compatible with most video players.
    """

    if len(images) == 0:
        logger.warning("The images list contains zero images. " +
                       "Not generating any video.")
        return

    # Get the image dimensions
    frame = imageio.imread(str(images[0]))
    if not frame.ndim == 3 or not frame.shape[2] in (3, 4):
        logger.error("The images must be RGB or RGBA images.")
        raise ValueError("The images must be RGB or RGBA images.")
    H, W, C = frame.shape

    if padding_color is not None and \
       not padding_color.has_alpha == (frame.shape[2] == 4):
        logger.error("The padding color must have the same number of " +
                     "channels as the images.")
        raise ValueError("The padding color must have the same number of " +
                         "channels as the images.")

    # Handle macro_block_size via padding if necessary
    macro_block_size = 16
    def pad_fct(frame): return frame
    if not W % macro_block_size == 0 or not H % macro_block_size == 0:
        W_new = math.ceil(W / macro_block_size) * macro_block_size
        H_new = math.ceil(H / macro_block_size) * macro_block_size
        if not no_resize_warning:
            logger.warning(
                f"The image dimensions ({W} x {H}) are not divisible by the " +
                f"macro block size ({macro_block_size}). The images will be " +
                f"resized to ({W_new} x {H_new}) to ensure proper encoding " +
                f"compatibility."
            )
        def pad_fct(frame):
            return np.pad(
                frame,
                ((0, H_new - H), (0, W_new - W), (0, 0)),
                **({"mode": "edge"} if padding_color is None else \
                   {"constant_values": padding_color})
            )

    # Frame rate
    if fps is None and duration is None:
        logger.error("Either `fps` or `duration` must be provided.")
        raise ValueError("Either `fps` or `duration` must be provided.")
    if fps is None:
        fps = len(images) / duration

    # Generate video
    with imageio.get_writer(
            str(video_filename), fps=fps, ffmpeg_log_level="error",
            ffmpeg_params=["-probesize", "5000000"],
            macro_block_size=macro_block_size,
        ) as writer:

        for image in tqdm(images, total=len(images),
                          desc="Generating video", unit="frame"):
            frame = imageio.imread(str(image))

            if not frame.shape == (H, W, C):
                logger.error(f"Image `{image}` has different dimensions than " +
                             f"the first image.")
                raise ValueError(f"Image `{image}` has different dimensions " +
                                 f"than the first image.")
            writer.append_data(pad_fct(frame))

    # Log success
    logger.info(f"Successfully generated video with {len(images)} frames to " +
                f"`{video_filename}`.")

@requires_package('imageio', 'tqdm')
def generate_video_from_folder(
    folder: str | Path,
    video_filename: str | Path,
    images_prefix: str = "",
    fps: Optional[int] = None,
    duration: Optional[float] = None,
    padding_color: Color | None = None,
    no_resize_warning: bool = False,
    sorting: Literal["lexicographic", "natural"] = "natural",
    ) -> None:
    """
    Generate a video from all images within a given folder using imageio.

    Inputs
    - folder:         `str | Path` path to the folder containing images to be
                      included in the video.
    - video_filename: `str | Path` name of the output video file.
    - images_prefix:  `str` prefix of the images to be included in the video.

    Partially Required Inputs
    - fps: `int` frame rate per seconds of the video.
    - duration: `float` duration of the video in seconds.

    Optional Inputs
    - padding_color: `Color` color to be used for padding the images. If None,
                     the images will be padded with the edge color of the image.
    - no_resize_warning: `bool` if True, no warning will be raised if the image
                         dimensions are not divisible by the macro block size.
    - sorting:          `lexicographic | natural` sorting method for the images.
                        Default is "natural", which sorts the images in a
                        human-friendly way (e.g., "image1", "image2", ...,
                        "image10") instead of lexicographic order
                        ("image1", "image10", "image2", ...).

    Note: ImageIO automatically resizes the images such that the dimensions are
          divisible by the `macro_block_size`. This is done to ensure that the
          videos are properly encoded and compatible with most video players.
    """
    # Folder validation
    folder = Path(folder)
    if not folder.is_dir():
        logger.error(
            f"The folder `{folder}` does not exist or is not a directory."
        )
        raise ValueError(
            f"The folder `{folder}` does not exist or is not a directory."
        )

    # Get all images in the folder and sort them
    images = sorted(folder.glob(f"{images_prefix}*.png")) + \
             sorted(folder.glob(f"{images_prefix}*.jpg")) + \
             sorted(folder.glob(f"{images_prefix}*.jpeg"))

    if sorting == "natural":
        if not natsort:
            logger.error(
                "Natural sorting is requested, but natsort is not installed. " +
                "Please install natsort to use this feature."
            )
            raise ImportError("natsort is required for natural sorting.")
        images = natsort.natsorted(images, key=lambda x: str(x))

    # Check if the folder contains any images
    if len(images) == 0:
        logger.warning(
            f"The folder `{folder}` does not contain any images with the " +
            f"prefix `{images_prefix}`. Not generating any video."
        )
        return

    # Generate video
    generate_video(
        images=images,
        video_filename=video_filename,
        fps=fps,
        duration=duration,
        padding_color=padding_color,
        no_resize_warning=no_resize_warning
    )

@requires_package('imageio', 'tqdm')
def main():
    parser = argparse.ArgumentParser(description="Generate a video from images.")
    parser.add_argument(
        "i", type=str, help="Folder containing images."
    )
    parser.add_argument(
        "o", type=str, help="Output video filename."
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Prefix of the images to be included in the video."
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second."
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Video duration in seconds."
    )
    parser.add_argument(
        "--padding_color", type=str, default=None,
        help="Padding color (e.g., 'white')."
    )
    args = parser.parse_args()

    padding_color = Color(args.padding_color) if args.padding_color else None

    generate_video_from_folder(
        folder=args.i,
        video_filename=args.o,
        images_prefix=args.prefix,
        fps=args.fps,
        duration=args.duration,
        padding_color=padding_color
    )

if __name__ == "__main__":

    # Logging
    logging.basicConfig(level=logging.INFO)

    main()
