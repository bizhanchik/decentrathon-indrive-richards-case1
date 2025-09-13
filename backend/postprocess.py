#!/usr/bin/env python3
"""
Post-processing functions for car defect detection.
Includes detection grouping and severity estimation.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
import cv2


@dataclass
class Detection:
    """Represents a single detection."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    area: float
    center: Tuple[float, float]  # x_center, y_center


@dataclass
class DefectGroup:
    """Represents a group of related detections."""
    defect_type: str
    detections: List[Detection]
    total_area: float
    bbox: Tuple[float, float, float, float]  # Combined bounding box
    severity: str
    confidence: float  # Average confidence


@dataclass
class CarAnalysis:
    """Complete analysis result for a car image."""
    image_path: str
    car_detected: bool
    car_bbox: Tuple[float, float, float, float]
    defect_groups: List[DefectGroup]
    overall_condition: str
    summary: str


class DefectPostProcessor:
    """Post-processor for car defect detection results."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize the post-processor.
        
        Args:
            class_names: List of class names corresponding to class IDs
        """
        self.class_names = class_names or ['dent', 'dirt', 'scratch']
        
        # Severity thresholds (can be tuned based on requirements)
        self.severity_thresholds = {
            'area_ratio': {
                'slight': 0.02,    # < 2% of car area
                'moderate': 0.05,  # 2-5% of car area
                'severe': 0.05     # > 5% of car area
            },
            'count': {
                'slight': 2,       # < 2 detections
                'moderate': 5,     # 2-5 detections
                'severe': 5        # > 5 detections
            }
        }
    
    def parse_yolo_results(self, results: Any, img_shape: Tuple[int, int]) -> List[Detection]:
        """
        Parse YOLO detection results into Detection objects.
        
        Args:
            results: YOLO model results
            img_shape: (height, width) of the image
        
        Returns:
            List of Detection objects
        """
        detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Extract box coordinates (xyxy format)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                
                # Extract other properties
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # Calculate derived properties
                area = (x2 - x1) * (y2 - y1)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    area=area,
                    center=center
                )
                
                detections.append(detection)
        
        return detections
    
    def calculate_iou(self, box1: Tuple[float, float, float, float], 
                     box2: Tuple[float, float, float, float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1, box2: Bounding boxes in format (x1, y1, x2, y2)
        
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def group_detections_by_proximity(self, detections: List[Detection], 
                                    iou_threshold: float = 0.3,
                                    distance_threshold: float = 100) -> Dict[str, List[List[Detection]]]:
        """
        Group detections by proximity using IoU and spatial distance.
        
        Args:
            detections: List of Detection objects
            iou_threshold: IoU threshold for grouping overlapping detections
            distance_threshold: Distance threshold for grouping nearby detections
        
        Returns:
            Dictionary mapping defect types to lists of detection groups
        """
        # Group detections by class first
        detections_by_class = {}
        for detection in detections:
            if detection.class_name not in detections_by_class:
                detections_by_class[detection.class_name] = []
            detections_by_class[detection.class_name].append(detection)
        
        grouped_detections = {}
        
        for defect_type, class_detections in detections_by_class.items():
            if not class_detections:
                continue
            
            # Use DBSCAN clustering based on center coordinates
            centers = np.array([det.center for det in class_detections])
            
            if len(centers) == 1:
                grouped_detections[defect_type] = [class_detections]
                continue
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(centers)
            labels = clustering.labels_
            
            # Group detections by cluster labels
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(class_detections[i])
            
            # Further refine groups using IoU
            refined_groups = []
            for cluster_detections in clusters.values():
                if len(cluster_detections) == 1:
                    refined_groups.append(cluster_detections)
                    continue
                
                # Check IoU within cluster and split if necessary
                subgroups = []
                remaining = cluster_detections.copy()
                
                while remaining:
                    current_group = [remaining.pop(0)]
                    
                    # Find all detections with significant IoU to current group
                    i = 0
                    while i < len(remaining):
                        max_iou = 0
                        for group_det in current_group:
                            iou = self.calculate_iou(group_det.bbox, remaining[i].bbox)
                            max_iou = max(max_iou, iou)
                        
                        if max_iou > iou_threshold:
                            current_group.append(remaining.pop(i))
                        else:
                            i += 1
                    
                    subgroups.append(current_group)
                
                refined_groups.extend(subgroups)
            
            grouped_detections[defect_type] = refined_groups
        
        return grouped_detections
    
    def estimate_severity(self, detections: List[Detection], 
                         car_area: float) -> str:
        """
        Estimate severity based on number of detections and total area.
        
        Args:
            detections: List of detections for a specific defect type
            car_area: Total area of the car bounding box
        
        Returns:
            Severity level: 'slight', 'moderate', or 'severe'
        """
        if not detections:
            return 'none'
        
        # Calculate metrics
        total_defect_area = sum(det.area for det in detections)
        area_ratio = total_defect_area / car_area if car_area > 0 else 0
        detection_count = len(detections)
        
        # Determine severity based on both area ratio and count
        area_severity = 'slight'
        if area_ratio > self.severity_thresholds['area_ratio']['severe']:
            area_severity = 'severe'
        elif area_ratio > self.severity_thresholds['area_ratio']['moderate']:
            area_severity = 'moderate'
        
        count_severity = 'slight'
        if detection_count > self.severity_thresholds['count']['severe']:
            count_severity = 'severe'
        elif detection_count > self.severity_thresholds['count']['moderate']:
            count_severity = 'moderate'
        
        # Take the maximum severity
        severity_levels = ['slight', 'moderate', 'severe']
        area_idx = severity_levels.index(area_severity)
        count_idx = severity_levels.index(count_severity)
        
        return severity_levels[max(area_idx, count_idx)]
    
    def create_defect_groups(self, grouped_detections: Dict[str, List[List[Detection]]], 
                           car_area: float) -> List[DefectGroup]:
        """
        Create DefectGroup objects from grouped detections.
        
        Args:
            grouped_detections: Dictionary of grouped detections by defect type
            car_area: Total area of the car bounding box
        
        Returns:
            List of DefectGroup objects
        """
        defect_groups = []
        
        for defect_type, detection_groups in grouped_detections.items():
            for group_detections in detection_groups:
                if not group_detections:
                    continue
                
                # Calculate group properties
                total_area = sum(det.area for det in group_detections)
                avg_confidence = sum(det.confidence for det in group_detections) / len(group_detections)
                
                # Calculate combined bounding box
                x1_min = min(det.bbox[0] for det in group_detections)
                y1_min = min(det.bbox[1] for det in group_detections)
                x2_max = max(det.bbox[2] for det in group_detections)
                y2_max = max(det.bbox[3] for det in group_detections)
                combined_bbox = (x1_min, y1_min, x2_max, y2_max)
                
                # Estimate severity
                severity = self.estimate_severity(group_detections, car_area)
                
                defect_group = DefectGroup(
                    defect_type=defect_type,
                    detections=group_detections,
                    total_area=total_area,
                    bbox=combined_bbox,
                    severity=severity,
                    confidence=avg_confidence
                )
                
                defect_groups.append(defect_group)
        
        return defect_groups
    
    def estimate_car_bbox(self, detections: List[Detection], 
                         img_shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """
        Estimate car bounding box from defect detections.
        If no detections, assume the whole image is the car.
        
        Args:
            detections: List of all detections
            img_shape: (height, width) of the image
        
        Returns:
            Estimated car bounding box (x1, y1, x2, y2)
        """
        if not detections:
            # If no detections, assume whole image is car
            return (0, 0, img_shape[1], img_shape[0])
        
        # Expand bounding box around all detections with some margin
        x1_min = min(det.bbox[0] for det in detections)
        y1_min = min(det.bbox[1] for det in detections)
        x2_max = max(det.bbox[2] for det in detections)
        y2_max = max(det.bbox[3] for det in detections)
        
        # Add margin (20% of detection area)
        width = x2_max - x1_min
        height = y2_max - y1_min
        margin_x = width * 0.2
        margin_y = height * 0.2
        
        x1_car = max(0, x1_min - margin_x)
        y1_car = max(0, y1_min - margin_y)
        x2_car = min(img_shape[1], x2_max + margin_x)
        y2_car = min(img_shape[0], y2_max + margin_y)
        
        return (x1_car, y1_car, x2_car, y2_car)
    
    def generate_summary(self, defect_groups: List[DefectGroup]) -> Tuple[str, str]:
        """
        Generate overall condition and summary text.
        
        Args:
            defect_groups: List of DefectGroup objects
        
        Returns:
            Tuple of (overall_condition, summary_text)
        """
        if not defect_groups:
            return "Clean", "Car detected → Clean, No defects detected"
        
        # Count defects by type and severity
        defect_summary = {}
        max_severity_level = 0
        severity_map = {'slight': 1, 'moderate': 2, 'severe': 3}
        
        for group in defect_groups:
            defect_type = group.defect_type
            severity = group.severity
            
            if defect_type not in defect_summary:
                defect_summary[defect_type] = {'slight': 0, 'moderate': 0, 'severe': 0}
            
            defect_summary[defect_type][severity] += 1
            max_severity_level = max(max_severity_level, severity_map[severity])
        
        # Determine overall condition
        if max_severity_level == 0:
            overall_condition = "Clean"
        elif max_severity_level == 1:
            overall_condition = "Good"
        elif max_severity_level == 2:
            overall_condition = "Fair"
        else:
            overall_condition = "Poor"
        
        # Generate summary text
        summary_parts = []
        for defect_type, severities in defect_summary.items():
            for severity, count in severities.items():
                if count > 0:
                    if count == 1:
                        summary_parts.append(f"{severity.capitalize()} {defect_type}")
                    else:
                        summary_parts.append(f"{severity.capitalize()} {defect_type} ({count} areas)")
        
        if summary_parts:
            summary = f"Car detected → {', '.join(summary_parts)}"
        else:
            summary = "Car detected → Clean, No defects detected"
        
        return overall_condition, summary
    
    def process_detections(self, results: Any, img_shape: Tuple[int, int], 
                          image_path: str = "") -> CarAnalysis:
        """
        Main processing function that combines all post-processing steps.
        
        Args:
            results: YOLO model results
            img_shape: (height, width) of the image
            image_path: Path to the processed image
        
        Returns:
            CarAnalysis object with complete analysis
        """
        # Parse YOLO results
        detections = self.parse_yolo_results(results, img_shape)
        
        # Estimate car bounding box
        car_bbox = self.estimate_car_bbox(detections, img_shape)
        car_area = (car_bbox[2] - car_bbox[0]) * (car_bbox[3] - car_bbox[1])
        
        # Group detections by proximity
        grouped_detections = self.group_detections_by_proximity(detections)
        
        # Create defect groups with severity estimation
        defect_groups = self.create_defect_groups(grouped_detections, car_area)
        
        # Generate overall summary
        overall_condition, summary = self.generate_summary(defect_groups)
        
        return CarAnalysis(
            image_path=image_path,
            car_detected=True,  # Assume car is always detected
            car_bbox=car_bbox,
            defect_groups=defect_groups,
            overall_condition=overall_condition,
            summary=summary
        )