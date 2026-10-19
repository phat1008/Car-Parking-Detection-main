#!/usr/bin/env python3
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import cvzone
import numpy as np
import pandas as pd

import config
from car_detector import CarDetector
from setup_directories import setup_directories


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


ParkingBox = Tuple[int, int, int, int]


class EnhancedParkingDetector:
    def __init__(
        self,
        video_path: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> None:
        setup_directories()

        self.video_path = video_path
        self.image_path = image_path

        self.width = config.PARKING_WIDTH
        self.height = config.PARKING_HEIGHT

        # Mỗi phần tử có dạng:
        # (x, y, width, height)
        self.posList: List[ParkingBox] = []

        # Trạng thái vẽ box
        self.drawing = False
        self.start_point: Optional[Tuple[int, int]] = None
        self.end_point: Optional[Tuple[int, int]] = None

        # Lịch sử cho Undo
        self.history: List[List[ParkingBox]] = []

        self.car_detector = CarDetector()

        self.show_help = True
        self.is_reset = False

        self.original_image: Optional[np.ndarray] = None
        self.current_image: Optional[np.ndarray] = None

        # Zoom và pan
        self.zoom_scale = 1.0
        self.zoom_step = config.ZOOM_STEP
        self.min_zoom = config.MIN_ZOOM
        self.max_zoom = config.MAX_ZOOM
        self.pan_x = 0
        self.pan_y = 0
        self.pan_step = config.PAN_STEP
        self.is_panning = False
        self.last_mouse_pos: Optional[Tuple[int, int]] = None

        # Một số biến dùng cho GUI/statistics
        self.last_available_slots = 0
        self.last_occupied_slots = 0
        self.last_vehicle_types: Dict[str, int] = {}
        self.last_confidences: List[float] = []
        self.occupancy_history: List[int] = []

        self.load_parking_positions()

    # ------------------------------------------------------------------
    # Parking box persistence
    # ------------------------------------------------------------------

    def load_parking_positions(self) -> None:
        """
        Đọc các bounding box đã lưu.

        Hỗ trợ cả hai định dạng:

        Định dạng cũ:
            (x, y)

        Định dạng mới:
            (x, y, width, height)
        """
        try:
            with open(config.POSITION_FILE, "rb") as file:
                saved_positions = pickle.load(file)

            loaded_boxes: List[ParkingBox] = []

            for item in saved_positions:
                if len(item) == 2:
                    x, y = item
                    loaded_boxes.append(
                        (
                            int(x),
                            int(y),
                            self.width,
                            self.height,
                        )
                    )

                elif len(item) == 4:
                    x, y, box_width, box_height = item
                    loaded_boxes.append(
                        (
                            int(x),
                            int(y),
                            int(box_width),
                            int(box_height),
                        )
                    )

            self.posList = loaded_boxes
            self.is_reset = len(self.posList) == 0

            logger.info(
                "Loaded %d parking bounding boxes",
                len(self.posList),
            )

        except FileNotFoundError:
            logger.warning(
                "No parking position file found: %s",
                config.POSITION_FILE,
            )
            self.posList = []

        except Exception as error:
            logger.error(
                "Could not load parking positions: %s",
                error,
            )
            self.posList = []

    def save_parking_positions(self) -> None:
        """
        Lưu danh sách bounding box hiện tại.

        Hàm này không thêm dữ liệu vào history.
        """
        try:
            with open(config.POSITION_FILE, "wb") as file:
                pickle.dump(self.posList, file)

            logger.info(
                "Saved %d parking bounding boxes",
                len(self.posList),
            )

        except Exception as error:
            logger.error(
                "Could not save parking positions: %s",
                error,
            )

    # ------------------------------------------------------------------
    # Bounding box operations
    # ------------------------------------------------------------------

    def _save_history(self) -> None:
        """Lưu bản sao trạng thái hiện tại trước khi thay đổi."""
        self.history.append(
            [tuple(box) for box in self.posList]
        )

    def _normalize_box(
        self,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
    ) -> Optional[ParkingBox]:
        """
        Chuẩn hóa hai điểm kéo chuột thành:

            (left, top, width, height)

        Box quá nhỏ sẽ bị bỏ qua.
        """
        x1, y1 = start_point
        x2, y2 = end_point

        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)

        box_width = right - left
        box_height = bottom - top

        min_width = 10
        min_height = 10

        if box_width < min_width or box_height < min_height:
            return None

        return (
            int(left),
            int(top),
            int(box_width),
            int(box_height),
        )

    def _box_contains_point(
        self,
        box: ParkingBox,
        point: Tuple[int, int],
    ) -> bool:
        """Kiểm tra điểm có nằm trong bounding box hay không."""
        x, y, box_width, box_height = box
        px, py = point

        return (
            x <= px <= x + box_width
            and y <= py <= y + box_height
        )

    def _box_center(
        self,
        box: ParkingBox,
    ) -> Tuple[int, int]:
        """Lấy tâm của bounding box."""
        x, y, box_width, box_height = box

        return (
            x + box_width // 2,
            y + box_height // 2,
        )

    def add_parking_box(
        self,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
    ) -> None:
        """Thêm đúng một bounding box sau thao tác kéo chuột."""
        box = self._normalize_box(
            start_point,
            end_point,
        )

        if box is None:
            logger.warning(
                "Ignored an extremely small bounding box"
            )
            return

        if box in self.posList:
            logger.info(
                "This bounding box already exists"
            )
            return

        self._save_history()

        self.posList.append(box)
        self.is_reset = False

        logger.info(
            "Added parking bounding box: %s",
            box,
        )

    def remove_parking_box_at(
        self,
        point: Tuple[int, int],
    ) -> bool:
        """
        Xóa bounding box tại vị trí click.

        Ưu tiên box chứa điểm click.
        Nếu không có, tìm box gần nhất trong bán kính cho phép.
        """
        if not self.posList:
            return False

        # Ưu tiên box chứa điểm click
        for index, box in enumerate(self.posList):
            if self._box_contains_point(box, point):
                self._save_history()

                removed_box = self.posList.pop(index)

                logger.info(
                    "Removed parking bounding box: %s",
                    removed_box,
                )

                self.is_reset = len(self.posList) == 0
                return True

        # Nếu không click trực tiếp vào box,
        # tìm box gần tâm click nhất
        px, py = point
        distances: List[Tuple[int, int]] = []

        for index, box in enumerate(self.posList):
            center_x, center_y = self._box_center(box)

            distance_squared = (
                (center_x - px) ** 2
                + (center_y - py) ** 2
            )

            distances.append(
                (distance_squared, index)
            )

        distance_squared, nearest_index = min(distances)

        max_distance = max(
            self.width,
            self.height,
        ) * 2

        if distance_squared <= max_distance ** 2:
            self._save_history()

            removed_box = self.posList.pop(nearest_index)

            logger.info(
                "Removed nearest parking bounding box: %s",
                removed_box,
            )

            self.is_reset = len(self.posList) == 0
            return True

        logger.info(
            "No nearby parking bounding box found"
        )

        return False

    def clear_all_markings(self) -> None:
        """Xóa toàn bộ bounding box."""
        if self.posList:
            self._save_history()

        self.posList = []
        self.is_reset = True

        logger.info(
            "Cleared all parking bounding boxes"
        )

    def undo_last_selection(self) -> None:
        """Khôi phục layout trước đó."""
        if not self.history:
            logger.warning(
                "No layout history available"
            )
            return

        self.posList = self.history.pop()
        self.is_reset = len(self.posList) == 0

        logger.info(
            "Undo completed. Current box count: %d",
            len(self.posList),
        )

    # ------------------------------------------------------------------
    # Help and mouse handling
    # ------------------------------------------------------------------

    def add_help_text(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Hiển thị hướng dẫn thao tác với kích thước chữ to hơn."""
        image_height, image_width = image.shape[:2]

        help_text = [
            "=== Controls ===",
            "Left drag: Create one box",
            "Right click: Delete box",
            "Z: Undo",
            "S: Save",
            "R: Clear all boxes",
            "D: Detect and report",
            "Q: Quit",
        ]

        # Tăng lề ngang rộng hơn để phù hợp với font chữ lớn hơn (tăng từ 330 lên 420)
        start_x = max(
            10,
            image_width - 420,
        )

        y_offset = 40

        for text in help_text:
            if text.startswith("==="):
                scale = 2  # Tăng kích thước tiêu đề
                thickness = 5
            else:
                scale = 2  # Tăng kích thước nội dung hướng dẫn từ 1.0/0.5 cũ lên 0.85 cho rõ nét dễ nhìn
                thickness = 5

            cvzone.putTextRect(
                image,
                text,
                (start_x, y_offset),
                scale=scale,
                thickness=thickness,
                offset=8,
                colorR=(0, 0, 0),
            )

            # Tăng khoảng cách dòng dọc (từ 28 lên 42) để chữ to không bị đè lên nhau
            y_offset += 42

        return image

    def apply_zoom_and_pan(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Áp dụng zoom và pan cho ảnh hiển thị."""
        image_height, image_width = image.shape[:2]

        zoomed_width = max(
            1,
            int(image_width * self.zoom_scale),
        )
        zoomed_height = max(
            1,
            int(image_height * self.zoom_scale),
        )

        zoomed_image = cv2.resize(
            image,
            (zoomed_width, zoomed_height),
        )

        max_pan_x = max(
            0,
            zoomed_width - image_width,
        )
        max_pan_y = max(
            0,
            zoomed_height - image_height,
        )

        self.pan_x = max(
            0,
            min(self.pan_x, max_pan_x),
        )

        self.pan_y = max(
            0,
            min(self.pan_y, max_pan_y),
        )

        if zoomed_width <= image_width and zoomed_height <= image_height:
            return zoomed_image

        x1 = self.pan_x
        y1 = self.pan_y
        x2 = min(
            x1 + image_width,
            zoomed_width,
        )
        y2 = min(
            y1 + image_height,
            zoomed_height,
        )

        cropped_image = zoomed_image[
            y1:y2,
            x1:x2,
        ]

        if (
            cropped_image.shape[0] < image_height
            or cropped_image.shape[1] < image_width
        ):
            padded_image = np.zeros(
                (
                    image_height,
                    image_width,
                    3,
                ),
                dtype=np.uint8,
            )

            padded_image[
                : cropped_image.shape[0],
                : cropped_image.shape[1],
            ] = cropped_image

            return padded_image

        return cropped_image

    def mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: Any,
    ) -> None:
        """Xử lý kéo-thả và click chuột."""

        # Khi đang pan ảnh thì không tạo bounding box
        if self.is_panning:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.last_mouse_pos = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.last_mouse_pos is not None:
                    dx = x - self.last_mouse_pos[0]
                    dy = y - self.last_mouse_pos[1]

                    self.pan_x = max(
                        0,
                        self.pan_x - dx,
                    )

                    self.pan_y = max(
                        0,
                        self.pan_y - dy,
                    )

                    self.last_mouse_pos = (x, y)

            elif event == cv2.EVENT_LBUTTONUP:
                self.last_mouse_pos = None

            return

        border_size = config.BORDER_SIZE

        image_x = x - border_size
        image_y = y - border_size

        # Kiểm tra click có nằm bên trong ảnh không
        if self.current_image is not None:
            image_height, image_width = (
                self.current_image.shape[:2]
            )

            if (
                image_x < 0
                or image_y < 0
                or image_x >= image_width
                or image_y >= image_height
            ):
                return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (
                image_x,
                image_y,
            )
            self.end_point = (
                image_x,
                image_y,
            )

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (
                    image_x,
                    image_y,
                )

        elif event == cv2.EVENT_LBUTTONUP:
            if not self.drawing:
                return

            self.drawing = False

            self.end_point = (
                image_x,
                image_y,
            )

            if (
                self.start_point is not None
                and self.end_point is not None
            ):
                self.add_parking_box(
                    self.start_point,
                    self.end_point,
                )

            self.start_point = None
            self.end_point = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            removed = self.remove_parking_box_at(
                (
                    image_x,
                    image_y,
                )
            )

            if removed:
                self.is_reset = False

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Tiền xử lý frame bằng OpenCV."""
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        blur = cv2.GaussianBlur(
            gray,
            (3, 3),
            1,
        )

        threshold = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            16,
        )

        median = cv2.medianBlur(
            threshold,
            5,
        )

        kernel = np.ones(
            (3, 3),
            np.uint8,
        )

        dilated = cv2.dilate(
            median,
            kernel,
            iterations=1,
        )

        return dilated

    def check_parking_space(
        self,
        processed_image: np.ndarray,
        image: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Kiểm tra từng bounding box.

        Returns:
            available_slots, occupied_slots
        """
        available_slots = 0
        occupied_slots = 0

        for box in self.posList:
            x, y, box_width, box_height = box

            image_height, image_width = (
                processed_image.shape[:2]
            )

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(
                image_width,
                x + box_width,
            )
            y2 = min(
                image_height,
                y + box_height,
            )

            if x1 >= x2 or y1 >= y2:
                continue

            image_crop = processed_image[
                y1:y2,
                x1:x2,
            ]

            count = cv2.countNonZero(
                image_crop
            )

            if count < config.OCCUPANCY_THRESHOLD:
                color = config.COLOR_EMPTY
                thickness = 3
                available_slots += 1
            else:
                color = config.COLOR_OCCUPIED
                thickness = 2
                occupied_slots += 1

            cv2.rectangle(
                image,
                (x, y),
                (
                    x + box_width,
                    y + box_height,
                ),
                color,
                thickness,
            )

            cvzone.putTextRect(
                image,
                str(count),
                (
                    x,
                    y + box_height - 3,
                ),
                scale=1,
                thickness=2,
                offset=0,
                colorR=color,
            )

        cvzone.putTextRect(
            image,
            f"Free: {available_slots}/{len(self.posList)}",
            (100, 50),
            scale=3,
            thickness=5,
            offset=20,
            colorR=(0, 200, 0),
        )

        self.last_available_slots = available_slots
        self.last_occupied_slots = occupied_slots
        self.occupancy_history.append(
            occupied_slots
        )

        return available_slots, occupied_slots

    # ------------------------------------------------------------------
    # CSV and reports
    # ------------------------------------------------------------------

    def generate_csv_report(
        self,
        total_slots: int,
        occupied_slots: int,
        available_slots: int,
    ) -> None:
        """Ghi thống kê vào file CSV."""
        try:
            data = {
                "Total Slots": [total_slots],
                "Occupied Slots": [occupied_slots],
                "Available Slots": [available_slots],
                "Timestamp": [
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                ],
            }

            dataframe = pd.DataFrame(data)

            csv_path = os.path.join(
                config.DATA_DIR,
                config.CSV_FILE,
            )

            if (
                config.CSV_APPEND_MODE
                and os.path.exists(csv_path)
            ):
                dataframe.to_csv(
                    csv_path,
                    mode="a",
                    header=False,
                    index=False,
                )
            else:
                dataframe.to_csv(
                    csv_path,
                    index=False,
                )

            logger.info(
                "CSV report saved to %s",
                csv_path,
            )

        except Exception as error:
            logger.error(
                "Could not generate CSV report: %s",
                error,
            )

    def process_video(self) -> None:
        """Xử lý video."""
        if not self.video_path:
            raise ValueError(
                "Video path was not provided"
            )

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(
                self.video_path
            )

        capture = cv2.VideoCapture(
            self.video_path
        )

        if not capture.isOpened():
            raise IOError(
                f"Cannot open video: {self.video_path}"
            )

        frame_width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        frame_height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        cv2.namedWindow(
            config.WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            config.WINDOW_NAME,
            frame_width,
            frame_height,
        )

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    capture.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        0,
                    )
                    continue

                processed_image = self.process_frame(
                    frame
                )

                available_slots, occupied_slots = (
                    self.check_parking_space(
                        processed_image,
                        frame,
                    )
                )

                frame_number = int(
                    capture.get(
                        cv2.CAP_PROP_POS_FRAMES
                    )
                )

                if frame_number % 30 == 0:
                    self.generate_csv_report(
                        total_slots=len(self.posList),
                        occupied_slots=occupied_slots,
                        available_slots=available_slots,
                    )

                cv2.imshow(
                    config.WINDOW_NAME,
                    frame,
                )

                key = cv2.waitKey(10) & 0xFF

                if key == ord("q"):
                    break

        finally:
            capture.release()
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Image mode
    # ------------------------------------------------------------------

    def process_image(self) -> None:
        """Chọn bounding box và xử lý ảnh."""
        if not self.image_path:
            raise ValueError(
                "Image path was not provided"
            )

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(
                self.image_path
            )

        self.original_image = cv2.imread(
            self.image_path,
            cv2.IMREAD_COLOR,
        )

        if self.original_image is None:
            raise IOError(
                f"Could not load image: {self.image_path}"
            )

        self.current_image = (
            self.original_image.copy()
        )

        image_height, image_width = (
            self.current_image.shape[:2]
        )

        border_size = config.BORDER_SIZE

        window_width = image_width + (
            border_size * 2
        )

        window_height = image_height + (
            border_size * 2
        )

        cv2.namedWindow(
            config.WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            config.WINDOW_NAME,
            window_width,
            window_height,
        )

        cv2.setMouseCallback(
            config.WINDOW_NAME,
            self.mouse_callback,
        )

        try:
            while True:
                bordered_image = np.zeros(
                    (
                        window_height,
                        window_width,
                        3,
                    ),
                    dtype=np.uint8,
                )

                bordered_image[
                    border_size : border_size + image_height,
                    border_size : border_size + image_width,
                ] = self.current_image.copy()

                # Vẽ các box đã tạo
                for index, box in enumerate(
                    self.posList,
                    start=1,
                ):
                    x, y, box_width, box_height = box

                    top_left = (
                        x + border_size,
                        y + border_size,
                    )

                    bottom_right = (
                        x + box_width + border_size,
                        y + box_height + border_size,
                    )

                    cv2.rectangle(
                        bordered_image,
                        top_left,
                        bottom_right,
                        config.COLOR_REGULAR,
                        2,
                    )

                    cv2.putText(
                        bordered_image,
                        str(index),
                        (
                            x + border_size + 4,
                            y + border_size + 20,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        config.COLOR_REGULAR,
                        2,
                        cv2.LINE_AA,
                    )

                # Vẽ box tạm thời trong lúc kéo chuột
                if (
                    self.drawing
                    and self.start_point is not None
                    and self.end_point is not None
                ):
                    x1, y1 = self.start_point
                    x2, y2 = self.end_point

                    cv2.rectangle(
                        bordered_image,
                        (
                            min(x1, x2) + border_size,
                            min(y1, y2) + border_size,
                        ),
                        (
                            max(x1, x2) + border_size,
                            max(y1, y2) + border_size,
                        ),
                        config.COLOR_SELECTION,
                        2,
                    )

                # Hiển thị thông tin số lượng box
                cvzone.putTextRect(
                    bordered_image,
                    f"Boxes: {len(self.posList)}",
                    (
                        border_size + 10,
                        border_size + 30,
                    ),
                    scale=1.5,
                    thickness=2,
                    offset=8,
                    colorR=(0, 0, 0),
                )

                if self.show_help:
                    self.add_help_text(
                        bordered_image
                    )

                display_image = (
                    self.apply_zoom_and_pan(
                        bordered_image
                    )
                )

                cv2.imshow(
                    config.WINDOW_NAME,
                    display_image,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    logger.info(
                        "Quit requested by user"
                    )
                    break

                elif key == ord("r"):
                    self.clear_all_markings()

                elif key == ord("z"):
                    self.undo_last_selection()

                elif key == ord("s"):
                    self.save_parking_positions()
                    self.is_reset = False

                    cvzone.putTextRect(
                        bordered_image,
                        "Layout Saved!",
                        (
                            window_width // 2 - 100,
                            window_height - 50,
                        ),
                        scale=2,
                        thickness=2,
                        offset=10,
                        colorR=(0, 255, 0),
                    )

                    cv2.imshow(
                        config.WINDOW_NAME,
                        bordered_image,
                    )

                    cv2.waitKey(800)

                elif key == ord("d"):
                    if not self.posList:
                        logger.warning(
                            "Please create parking boxes first"
                        )
                        print(
                            "Please create parking boxes first"
                        )
                        continue

                    logger.info(
                        "Starting detection"
                    )

                    processed_image = (
                        self.process_frame(
                            self.current_image
                        )
                    )

                    available_slots, occupied_slots = (
                        self.check_parking_space(
                            processed_image,
                            self.current_image,
                        )
                    )

                    print(
                        "Running YOLO vehicle detection..."
                    )

                    detection_results = (
                        self.car_detector.detect_cars(
                            self.current_image
                        )
                    )

                    detections, space_status = (
                        self.car_detector.process_detections(
                            detection_results,
                            self.posList,
                            self.current_image,
                        )
                    )

                    report_path, text_report_path = (
                        self.car_detector.generate_report(
                            self.current_image,
                            self.posList,
                            detections,
                            space_status,
                        )
                    )

                    self.generate_csv_report(
                        total_slots=len(self.posList),
                        occupied_slots=occupied_slots,
                        available_slots=available_slots,
                    )

                    print(
                        "Visual report:",
                        report_path,
                    )

                    print(
                        "Text report:",
                        text_report_path,
                    )

                    cvzone.putTextRect(
                        bordered_image,
                        "Report Generated!",
                        (
                            window_width // 2 - 150,
                            window_height - 50,
                        ),
                        scale=2,
                        thickness=3,
                        offset=10,
                        colorR=(0, 255, 0),
                    )

                    cv2.imshow(
                        config.WINDOW_NAME,
                        bordered_image,
                    )

                    cv2.waitKey(1200)

        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = EnhancedParkingDetector(
        video_path=config.DEFAULT_VIDEO_PATH,
        image_path=config.DEFAULT_IMAGE_PATH,
    )

    detector.process_image()