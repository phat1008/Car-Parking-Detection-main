import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ultralytics import YOLO

import config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


ParkingBox = Tuple[int, int, int, int]


class CarDetector:
    def __init__(self) -> None:
        """Initialize YOLOv8 vehicle detector."""
        logger.info(
            "Loading YOLOv8 model from %s",
            config.MODEL_PATH,
        )

        try:
            self.model = YOLO(
                config.MODEL_PATH
            )

        except Exception as error:
            logger.error(
                "Could not load YOLO model: %s",
                error,
            )

            raise IOError(
                f"Failed to load YOLO model: {error}"
            )

        self.car_classes = config.CAR_CLASSES
        self.confidence_threshold = (
            config.CONFIDENCE_THRESHOLD
        )

    def detect_cars(
        self,
        image: np.ndarray,
    ) -> Any:
        """Run YOLOv8 inference."""
        try:
            results = self.model(
                image,
                conf=self.confidence_threshold,
            )

            return results[0]

        except Exception as error:
            logger.error(
                "Vehicle detection failed: %s",
                error,
            )
            raise

    def detect_special_parking(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> bool:
        """Detect yellow/special parking markings."""
        image_height, image_width = image.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + width)
        y2 = min(image_height, y + height)

        if x1 >= x2 or y1 >= y2:
            return False

        space_region = image[
            y1:y2,
            x1:x2,
        ]

        if space_region.size == 0:
            return False

        hsv = cv2.cvtColor(
            space_region,
            cv2.COLOR_BGR2HSV,
        )

        yellow_ranges = [
            (
                np.array([20, 100, 100]),
                np.array([30, 255, 255]),
            ),
            (
                np.array([15, 100, 100]),
                np.array([25, 255, 255]),
            ),
            (
                np.array([25, 100, 100]),
                np.array([35, 255, 255]),
            ),
        ]

        yellow_mask = np.zeros(
            hsv.shape[:2],
            dtype=np.uint8,
        )

        for lower, upper in yellow_ranges:
            mask = cv2.inRange(
                hsv,
                lower,
                upper,
            )

            yellow_mask = cv2.bitwise_or(
                yellow_mask,
                mask,
            )

        kernel = np.ones(
            (3, 3),
            np.uint8,
        )

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        region_area = max(
            1,
            yellow_mask.shape[0]
            * yellow_mask.shape[1],
        )

        yellow_percentage = (
            np.sum(yellow_mask > 0)
            / region_area
            * 100
        )

        horizontal_lines = cv2.HoughLinesP(
            yellow_mask,
            1,
            np.pi / 180,
            threshold=20,
            minLineLength=max(5, int(width * 0.3)),
            maxLineGap=20,
        )

        vertical_lines = cv2.HoughLinesP(
            yellow_mask,
            1,
            np.pi / 180,
            threshold=20,
            minLineLength=max(5, int(height * 0.3)),
            maxLineGap=20,
        )

        has_yellow_lines = (
            horizontal_lines is not None
            and len(horizontal_lines) > 0
        ) or (
            vertical_lines is not None
            and len(vertical_lines) > 0
        )

        return (
            yellow_percentage
            > config.YELLOW_PERCENTAGE_THRESHOLD
            or has_yellow_lines
        )

    def check_space_occupancy(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> bool:
        """Check occupancy using image thresholding."""
        image_height, image_width = image.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + width)
        y2 = min(image_height, y + height)

        if x1 >= x2 or y1 >= y2:
            return False

        space_region = image[
            y1:y2,
            x1:x2,
        ]

        if space_region.size == 0:
            return False

        gray = cv2.cvtColor(
            space_region,
            cv2.COLOR_BGR2GRAY,
        )

        _, threshold = cv2.threshold(
            gray,
            127,
            255,
            cv2.THRESH_BINARY,
        )

        region_area = max(
            1,
            threshold.shape[0]
            * threshold.shape[1],
        )

        occupancy_percentage = (
            np.sum(threshold > 0)
            / region_area
            * 100
        )

        return occupancy_percentage > 30

    def process_detections(
        self,
        results: Any,
        parking_spaces: List[ParkingBox],
        image: np.ndarray,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """Match YOLO detections with manually-created boxes."""
        detections: List[Dict[str, Any]] = []
        space_status: List[Dict[str, Any]] = []

        for parking_space in parking_spaces:
            # Tương thích dữ liệu cũ dạng (x, y)
            if len(parking_space) == 2:
                x, y = parking_space
                width = config.PARKING_WIDTH
                height = config.PARKING_HEIGHT
            else:
                x, y, width, height = parking_space

            is_special = self.detect_special_parking(
                image,
                x,
                y,
                width,
                height,
            )

            is_occupied = self.check_space_occupancy(
                image,
                x,
                y,
                width,
                height,
            )

            vehicle_detected = False
            vehicle_type = None
            confidence = 0.0

            for box in results.boxes:
                class_id = int(
                    box.cls[0]
                    if hasattr(box.cls, "__len__")
                    else box.cls
                )

                if class_id not in self.car_classes:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0],
                )

                overlaps = (
                    x1 < x + width
                    and x2 > x
                    and y1 < y + height
                    and y2 > y
                )

                if overlaps:
                    vehicle_detected = True
                    vehicle_type = class_id
                    confidence = float(
                        box.conf[0]
                    )
                    break

            status = {
                "position": (x, y),
                "width": width,
                "height": height,
                "is_special": is_special,
                "is_occupied": is_occupied,
                "vehicle_detected": vehicle_detected,
                "vehicle_type": vehicle_type,
                "confidence": confidence,
            }

            space_status.append(status)

            if vehicle_detected:
                detections.append(
                    {
                        "bbox": (
                            x,
                            y,
                            x + width,
                            y + height,
                        ),
                        "confidence": confidence,
                        "class_id": vehicle_type,
                        "is_special": is_special,
                    }
                )

        return detections, space_status

    def generate_report(
        self,
        image: np.ndarray,
        parking_spaces: List[ParkingBox],
        detections: List[Dict[str, Any]],
        space_status: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Generate visual and text reports."""
        if output_dir is None:
            output_dir = config.REPORTS_DIR

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        total_spaces = len(parking_spaces)

        special_spaces = sum(
            1
            for status in space_status
            if status["is_special"]
        )

        regular_spaces = (
            total_spaces - special_spaces
        )

        occupied_spaces = sum(
            1
            for status in space_status
            if status["is_occupied"]
        )

        available_spaces = (
            total_spaces - occupied_spaces
        )

        special_occupied = sum(
            1
            for status in space_status
            if status["is_special"]
            and status["is_occupied"]
        )

        regular_occupied = sum(
            1
            for status in space_status
            if not status["is_special"]
            and status["is_occupied"]
        )

        # --------------------------------------------------------------
        # Visual report
        # --------------------------------------------------------------

        figure = plt.figure(
            figsize=config.REPORT_FIGSIZE
        )

        report_image = image.copy()

        for status in space_status:
            x, y = status["position"]

            width = status.get(
                "width",
                config.PARKING_WIDTH,
            )

            height = status.get(
                "height",
                config.PARKING_HEIGHT,
            )

            if status["is_special"]:
                color = config.COLOR_SPECIAL
            else:
                color = config.COLOR_REGULAR

            cv2.rectangle(
                report_image,
                (x, y),
                (
                    x + width,
                    y + height,
                ),
                color,
                2,
            )

            label = ""

            if status["is_special"]:
                label += "Special "

            if status["is_occupied"]:
                label += "Occupied"
            else:
                label += "Empty"

            cv2.putText(
                report_image,
                label,
                (
                    x,
                    max(20, y - 5),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        plt.subplot(2, 2, 1)

        plt.imshow(
            cv2.cvtColor(
                report_image,
                cv2.COLOR_BGR2RGB,
            )
        )

        plt.title("Parking Lot Analysis")
        plt.axis("off")

        plt.subplot(2, 2, 2)

        statistics = {
            "Total": total_spaces,
            "Special": special_spaces,
            "Regular": regular_spaces,
            "Occupied": occupied_spaces,
            "Available": available_spaces,
        }

        plt.bar(
            statistics.keys(),
            statistics.values(),
            color=[
                "blue",
                "yellow",
                "magenta",
                "red",
                "green",
            ],
        )

        plt.title("Parking Status")
        plt.xticks(rotation=45)

        plt.subplot(2, 2, 3)

        if total_spaces > 0:
            plt.pie(
                [
                    regular_spaces,
                    special_spaces,
                ],
                labels=[
                    "Regular",
                    "Special",
                ],
                autopct="%1.1f%%",
                colors=[
                    "magenta",
                    "yellow",
                ],
            )

        plt.title("Parking Space Types")

        plt.subplot(2, 2, 4)

        plt.bar(
            ["Special", "Regular"],
            [
                special_occupied,
                regular_occupied,
            ],
            color="red",
            label="Occupied",
        )

        plt.title("Occupied Spaces by Type")
        plt.legend()

        plt.tight_layout()

        image_report_path = os.path.join(
            output_dir,
            f"parking_report_{timestamp}.png",
        )

        plt.savefig(
            image_report_path,
            dpi=config.REPORT_DPI,
            bbox_inches="tight",
        )

        plt.close(figure)

        # --------------------------------------------------------------
        # Text report
        # --------------------------------------------------------------

        text_report_path = os.path.join(
            output_dir,
            f"parking_report_{timestamp}.txt",
        )

        occupancy_rate = (
            occupied_spaces / total_spaces * 100
            if total_spaces > 0
            else 0
        )

        with open(
            text_report_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                "Parking Lot Analysis Report\n"
            )

            file.write(
                f"Generated on: "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            file.write(
                "Space Distribution:\n"
            )

            file.write(
                f"Total Parking Spaces: "
                f"{total_spaces}\n"
            )

            file.write(
                f"Special Parking Spaces: "
                f"{special_spaces}\n"
            )

            file.write(
                f"Regular Parking Spaces: "
                f"{regular_spaces}\n\n"
            )

            file.write(
                "Occupancy Status:\n"
            )

            file.write(
                f"Total Occupied Spaces: "
                f"{occupied_spaces}\n"
            )

            file.write(
                f"Total Available Spaces: "
                f"{available_spaces}\n"
            )

            file.write(
                f"Special Spaces Occupied: "
                f"{special_occupied}\n"
            )

            file.write(
                f"Regular Spaces Occupied: "
                f"{regular_occupied}\n\n"
            )

            file.write(
                "Occupancy Rates:\n"
            )

            file.write(
                f"Overall Occupancy: "
                f"{occupancy_rate:.1f}%\n"
            )

        logger.info(
            "Reports generated: %s, %s",
            image_report_path,
            text_report_path,
        )

        return (
            image_report_path,
            text_report_path,
        )