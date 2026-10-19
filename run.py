#!/usr/bin/env python3
"""
Car Parking Detection System - Main Entry Point

Usage:

    Create or edit parking boxes:
        python run.py --image carParkImg.jpg

    Run video using saved parking boxes:
        python run.py --video carPark.mp4

    Edit boxes and then automatically run video:
        python run.py --image carParkImg.jpg --video carPark.mp4
"""

import argparse
import logging
import os
import sys
from typing import Optional

import config
from enhanced_parking_detector import EnhancedParkingDetector


# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            "parking_detection.log",
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def print_banner() -> None:
    """Print application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║       Car Parking Space Detection System                  ║
║                                                           ║
║       Powered by YOLOv8 and OpenCV                       ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)


def validate_file(
    file_path: Optional[str],
    file_type: str = "file",
) -> bool:
    """Check whether a file exists and is readable."""
    if not file_path:
        return False

    if not os.path.exists(file_path):
        logger.error(
            "%s not found: %s",
            file_type.capitalize(),
            file_path,
        )

        print(
            "Error: {} not found at '{}'".format(
                file_type.capitalize(),
                file_path,
            )
        )

        return False

    if not os.path.isfile(file_path):
        logger.error(
            "%s is not a file: %s",
            file_type.capitalize(),
            file_path,
        )

        print(
            "Error: '{}' is not a valid file".format(
                file_path,
            )
        )

        return False

    if not os.access(file_path, os.R_OK):
        logger.error(
            "%s is not readable: %s",
            file_type.capitalize(),
            file_path,
        )

        print(
            "Error: Cannot read {} '{}'".format(
                file_type,
                file_path,
            )
        )

        return False

    return True


def print_image_shortcuts() -> None:
    """Print keyboard and mouse controls for image mode."""
    print("\n" + "=" * 65)
    print("IMAGE / PARKING BOX CONTROLS")
    print("=" * 65)
    print("  Left mouse drag   - Create one bounding box")
    print("  Right mouse click - Delete bounding box")
    print("  S                 - Save parking layout")
    print("  Z                 - Undo last change")
    print("  R                 - Clear all bounding boxes")
    print("  D                 - Detect vehicles and generate report")
    print("  Q                 - Quit image mode")
    print("=" * 65 + "\n")


def print_video_shortcuts() -> None:
    """Print keyboard controls for video mode."""
    print("\n" + "=" * 65)
    print("VIDEO CONTROLS")
    print("=" * 65)
    print("  Q                 - Quit video")
    print("=" * 65 + "\n")


def determine_mode(
    image_path: Optional[str],
    video_path: Optional[str],
    requested_mode: str,
) -> str:
    """
    Determine the application mode.

    Rules:
        image + video -> both
        video only    -> video
        image only    -> image
        no paths      -> requested mode
    """
    if image_path and video_path:
        return "both"

    if video_path:
        return "video"

    if image_path:
        return "image"

    return requested_mode


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------

def main() -> int:
    """Main entry point."""
    print_banner()

    parser = argparse.ArgumentParser(
        description=(
            "AI-powered car parking detection "
            "using OpenCV and YOLOv8"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  Create parking boxes:
      python run.py --image carParkImg.jpg

  Run a video using saved boxes:
      python run.py --video carPark.mp4

  Edit boxes and then run the video:
      python run.py --image carParkImg.jpg --video carPark.mp4

Keyboard controls:

  Left mouse drag   Create one bounding box
  Right click       Delete a bounding box
  S                 Save layout
  Z                 Undo
  R                 Clear all boxes
  D                 Detect and generate report
  Q                 Quit
""",
    )

    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help=(
            "Path to an image used to create "
            "or edit parking bounding boxes"
        ),
    )

    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default=None,
        help=(
            "Path to the video used for parking detection"
        ),
    )

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=[
            "image",
            "video",
            "both",
        ],
        default="image",
        help=(
            "Application mode used when no file path "
            "determines the mode"
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Car Parking Detection System v2.0",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Khi không truyền image hoặc video,
    # mặc định mở ảnh carParkImg.jpg.
    if (
        args.image is None
        and args.video is None
        and args.mode == "image"
    ):
        args.image = config.DEFAULT_IMAGE_PATH

    mode = determine_mode(
        image_path=args.image,
        video_path=args.video,
        requested_mode=args.mode,
    )

    logger.info(
        "Selected mode: %s",
        mode,
    )

    print(
        "\nSelected mode: {}".format(mode)
    )

    # -----------------------------------------------------------------
    # Validate input files
    # -----------------------------------------------------------------

    if mode in ["image", "both"]:
        if not args.image:
            print(
                "\nError: Image mode requires an image path."
            )

            print(
                "Example: python run.py "
                "--image carParkImg.jpg"
            )

            return 1

        if not validate_file(
            args.image,
            "image",
        ):
            return 1

        print(
            "Image file found: {}".format(
                args.image,
            )
        )

    if mode in ["video", "both"]:
        if not args.video:
            print(
                "\nError: Video mode requires a video path."
            )

            print(
                "Example: python run.py "
                "--video carPark.mp4"
            )

            return 1

        if not validate_file(
            args.video,
            "video",
        ):
            return 1

        print(
            "Video file found: {}".format(
                args.video,
            )
        )

    # -----------------------------------------------------------------
    # Initialize detector
    # -----------------------------------------------------------------

    try:
        logger.info(
            "Initializing parking detector..."
        )

        print(
            "\nInitializing parking detector..."
        )

        detector = EnhancedParkingDetector(
            image_path=(
                args.image
                if mode in ["image", "both"]
                else None
            ),
            video_path=(
                args.video
                if mode in ["video", "both"]
                else None
            ),
        )

        print(
            "Detector initialized successfully"
        )

        # -------------------------------------------------------------
        # Image mode
        # -------------------------------------------------------------

        if mode in ["image", "both"]:
            logger.info(
                "Starting image/layout mode: %s",
                args.image,
            )

            print(
                "\nOpening image: {}".format(
                    args.image,
                )
            )

            print_image_shortcuts()

            detector.process_image()

            logger.info(
                "Image/layout mode completed"
            )

        # -------------------------------------------------------------
        # Video mode
        # -------------------------------------------------------------

        if mode in ["video", "both"]:
            logger.info(
                "Starting video mode: %s",
                args.video,
            )

            print(
                "\nStarting video: {}".format(
                    args.video,
                )
            )

            print_video_shortcuts()

            # Đọc lại CarParkPos trong trường hợp người dùng
            # vừa chỉnh sửa và lưu layout ở image mode.
            detector.load_parking_positions()

            if not detector.posList:
                print(
                    "\nWarning: No parking bounding boxes found."
                )

                print(
                    "Create and save boxes first with:"
                )

                print(
                    "python run.py "
                    "--image carParkImg.jpg"
                )

                return 1

            detector.process_video()

            logger.info(
                "Video mode completed"
            )

        # -------------------------------------------------------------
        # Completion message
        # -------------------------------------------------------------

        print(
            "\nProcessing completed successfully!"
        )

        print(
            "Reports directory: {}/".format(
                config.REPORTS_DIR,
            )
        )

        print(
            "CSV data directory: {}/".format(
                config.DATA_DIR,
            )
        )

        return 0

    except KeyboardInterrupt:
        logger.info(
            "Application interrupted by user"
        )

        print(
            "\nApplication interrupted by user."
        )

        return 0

    except FileNotFoundError as error:
        logger.error(
            "File not found: %s",
            error,
        )

        print(
            "\nFile not found: {}".format(
                error,
            )
        )

        return 1

    except IOError as error:
        logger.error(
            "Input/output error: %s",
            error,
        )

        print(
            "\nInput/output error: {}".format(
                error,
            )
        )

        return 1

    except Exception as error:
        logger.exception(
            "Unexpected error"
        )

        print(
            "\nUnexpected error: {}".format(
                error,
            )
        )

        print(
            "Check parking_detection.log for details."
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())