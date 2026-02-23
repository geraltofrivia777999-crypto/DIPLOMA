"""Object detection module with YOLO-World open-vocabulary bag detection.

Uses YOLO11x for person detection and YOLO-World for bag detection.
YOLO-World can detect ANY type of bag by text prompt — backpacks, plastic bags,
shopping bags, tote bags, etc. — without fine-tuning.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
from loguru import logger
from ultralytics import YOLO

from .config import DetectionConfig, select_device


@dataclass
class DetectionResult:
    boxes: np.ndarray  # (N,4) xyxy
    scores: np.ndarray  # (N,)
    classes: np.ndarray  # (N,)
    is_bag: np.ndarray  # (N,) bool — True if bag, False if person


class Detector:
    """Hybrid detector: YOLO11x for persons + YOLO-World for bags.

    YOLO-World is an open-vocabulary detector that finds objects by text prompt.
    This means it can detect plastic bags, shopping bags, tote bags, etc. —
    not just the 3 COCO bag classes.
    """

    def __init__(self, cfg: DetectionConfig):
        device = select_device(cfg.device)
        self.cfg = cfg

        # Person detector (standard YOLO)
        logger.info(f"Loading person detector {cfg.model_path} on {device}")
        self.person_model = YOLO(cfg.model_path)
        self.person_model.to(device)

        # Bag detector (YOLO-World or fallback to standard YOLO)
        self.use_yolo_world = cfg.use_yolo_world
        if self.use_yolo_world:
            try:
                logger.info(f"Loading YOLO-World {cfg.yolo_world_model} for bag detection")
                self.bag_model = YOLO(cfg.yolo_world_model)
                self.bag_model.set_classes(list(cfg.yolo_world_prompts))
                self.bag_model.to(device)
                logger.info(f"YOLO-World ready with prompts: {cfg.yolo_world_prompts}")
            except Exception as e:
                logger.warning(f"YOLO-World failed to load: {e}. Falling back to COCO bags.")
                self.use_yolo_world = False

        self.person_conf = cfg.conf_threshold
        self.bag_conf = cfg.bag_conf_threshold if not self.use_yolo_world else cfg.yolo_world_bag_conf

        logger.info(f"Detection thresholds - Person: {self.person_conf}, Bag: {self.bag_conf}")

    def __call__(self, frame: np.ndarray) -> DetectionResult:
        """Detect persons and bags in frame."""

        all_boxes, all_scores, all_classes, all_is_bag = [], [], [], []

        # === Person detection (standard YOLO) ===
        person_results = self.person_model.predict(
            frame,
            conf=self.person_conf,
            iou=self.cfg.iou_threshold,
            classes=list(self.cfg.classes_person),
            device=self.person_model.device,
            verbose=False,
            imgsz=self.cfg.imgsz,
        )[0]

        if len(person_results.boxes) > 0:
            boxes = person_results.boxes.xyxy.cpu().numpy()
            scores = person_results.boxes.conf.cpu().numpy()
            classes = person_results.boxes.cls.cpu().numpy().astype(int)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)
            all_is_bag.append(np.zeros(len(boxes), dtype=bool))

        # === Bag detection ===
        if self.use_yolo_world:
            # YOLO-World: open-vocabulary detection
            bag_results = self.bag_model.predict(
                frame,
                conf=self.bag_conf,
                iou=self.cfg.iou_threshold,
                device=self.bag_model.device,
                verbose=False,
                imgsz=self.cfg.imgsz,
            )[0]

            if len(bag_results.boxes) > 0:
                boxes = bag_results.boxes.xyxy.cpu().numpy()
                scores = bag_results.boxes.conf.cpu().numpy()
                # YOLO-World classes are 0-indexed into our prompts list
                # Map all to a single "bag" class (24) for downstream compatibility
                classes = np.full(len(boxes), 24, dtype=int)
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(classes)
                all_is_bag.append(np.ones(len(boxes), dtype=bool))
        else:
            # Fallback: standard YOLO COCO bag classes
            bag_results = self.person_model.predict(
                frame,
                conf=self.bag_conf,
                iou=self.cfg.iou_threshold,
                classes=list(self.cfg.classes_bag),
                device=self.person_model.device,
                verbose=False,
                imgsz=self.cfg.imgsz,
                augment=self.cfg.augment,
            )[0]

            if len(bag_results.boxes) > 0:
                boxes = bag_results.boxes.xyxy.cpu().numpy()
                scores = bag_results.boxes.conf.cpu().numpy()
                classes = bag_results.boxes.cls.cpu().numpy().astype(int)
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(classes)
                all_is_bag.append(np.ones(len(boxes), dtype=bool))

        # Combine results
        if not all_boxes:
            return DetectionResult(
                boxes=np.array([]).reshape(0, 4),
                scores=np.array([]),
                classes=np.array([]),
                is_bag=np.array([], dtype=bool),
            )

        return DetectionResult(
            boxes=np.concatenate(all_boxes),
            scores=np.concatenate(all_scores),
            classes=np.concatenate(all_classes),
            is_bag=np.concatenate(all_is_bag),
        )
