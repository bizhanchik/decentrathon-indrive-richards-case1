from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

@dataclass
class UnifiedDetection:
    """
    Unified detection that combines detections from multiple models.
    """
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coordinates
    class_name: str    # Dominant class from the cluster
    confidence: float  # Aggregated confidence score
    source_models: List[str]  # List of models that contributed to this detection
    aggregated_confidence: float  # Combined confidence from all contributing models
    detection_count: int  # Number of individual detections that formed this unified detection
    severity: Optional[str] = None  # Estimated severity if applicable

class EnsembleLogic:
    """
    Ensemble logic for combining predictions from multiple car defect detection models.
    
    Key principles:
    1. Model 3's 'good_condition' class overrides all other classifications
    2. Other models use performance-based and confidence-based weights
    3. Smart voting system for final decision
    """
    
    def __init__(self):
        # Model performance weights based on test results and domain expertise
        # Model 4 has increased weight due to superior scratch detection capabilities
        self.model_weights = {
            'main_model': 0.30,    # YOLOv8s - Multi-class defect detection
            'model2': 0.20,        # Specialized defect detection
            'model3': 0.25,        # Has 'good_condition' class - special handling
            'model4': 0.25         # Advanced defect detection - increased for scratch detection
        }
        
        # Confidence thresholds for each model
        self.confidence_thresholds = {
            'main_model': 0.55,
            'model2': 0.55,
            'model3': 0.25,
            'model4': 0.55
        }
        
        # Model class mappings
        self.model_classes = {
            'main_model': ['Dent', 'Dislocation', 'Scratch', 'Shatter', 'damaged', 'severe damage'],
            'model2': ['crack', 'dent', 'glass shatter', 'lamp broken', 'scratch', 'tire flat'],
            'model3': ['good_condition', 'damaged-dent', 'damaged-scratch', 'severe damage'],
            'model4': ['damaged', 'good_condition']
        }
    
    def has_good_condition_detection(self, model3_analysis: Optional[Dict]) -> tuple[bool, float]:
        """
        Check if Model 3 detected 'good_condition' with sufficient confidence.
        
        Returns:
            tuple: (has_good_condition, max_confidence)
        """
        if not model3_analysis or not model3_analysis.get('detections'):
            return False, 0.0
        
        good_condition_detections = [
            det for det in model3_analysis['detections'] 
            if det.get('class', '').lower() in ['good_condition', 'good condition']
        ]
        
        if not good_condition_detections:
            return False, 0.0
        
        # Get the highest confidence good_condition detection
        max_confidence = max(det.get('confidence', 0.0) for det in good_condition_detections)
        
        # Require high confidence for good_condition override
        return max_confidence >= self.confidence_thresholds['model3'], max_confidence
    
    def resolve_model3_conflicts(self, model3_analysis: Optional[Dict]) -> Optional[Dict]:
        """
        Resolve Model 3 conflicts by choosing either good condition OR damage detections.
        When conflicts exist, choose the dominant detection type based on confidence and count.
        
        Returns:
            Modified model3_analysis with conflicts resolved, or None if no analysis
        """
        if not model3_analysis or not model3_analysis.get('detections'):
            return model3_analysis
        
        detections = model3_analysis['detections']
        
        # Separate good condition and damage detections
        good_condition_detections = [
            det for det in detections 
            if det.get('class', '').lower() in ['good_condition', 'good condition']
        ]
        
        damage_detections = [
            det for det in detections 
            if det.get('class', '').lower() in ['damaged-dent', 'damaged-scratch', 'severe damage']
        ]
        
        # If no conflict, return as is
        if not (good_condition_detections and damage_detections):
            return model3_analysis
        
        logger.info(f"Model 3 conflict detected: {len(good_condition_detections)} good condition vs {len(damage_detections)} damage detections")
        
        # Calculate scores for each type with bias toward damage detections
        good_score = sum(det.get('confidence', 0.0) for det in good_condition_detections)
        damage_score = sum(det.get('confidence', 0.0) for det in damage_detections)
        
        # Add count bonus (more detections = higher weight)
        good_score += len(good_condition_detections) * 0.1
        damage_score += len(damage_detections) * 0.1
        
        # Apply damage bias: if damage detections exist with reasonable confidence,
        # give them additional weight to prioritize damage over good condition
        if damage_detections:
            # Add bias based on highest damage confidence
            max_damage_confidence = max(det.get('confidence', 0.0) for det in damage_detections)
            if max_damage_confidence >= 0.3:  # Reasonable confidence threshold
                damage_score += 0.2  # Damage bias factor
                
        # Additional bias: if damage count is significant, add extra weight
        if len(damage_detections) >= 2:
            damage_score += 0.15  # Multiple damage detections bonus
        
        # Choose the dominant type
        if damage_score > good_score:
            # Keep only damage detections
            chosen_detections = damage_detections
            logger.info(f"Model 3 conflict resolved: choosing damage (score: {damage_score:.3f} vs {good_score:.3f}) - damage prioritized")
        else:
            # Keep only good condition detections
            chosen_detections = good_condition_detections
            logger.info(f"Model 3 conflict resolved: choosing good condition (score: {good_score:.3f} vs {damage_score:.3f}) - good condition wins despite bias")
        
        # Return modified analysis with only the chosen detections
        resolved_analysis = model3_analysis.copy()
        resolved_analysis['detections'] = chosen_detections
        resolved_analysis['conflict_resolved'] = True
        resolved_analysis['original_detection_count'] = len(detections)
        resolved_analysis['resolved_detection_count'] = len(chosen_detections)
        
        return resolved_analysis
    
    def calculate_damage_severity_score(self, detections: List[Dict]) -> float:
        """
        Calculate damage severity score based on detections.
        Higher score = more damage.
        """
        if not detections:
            return 0.0
        
        severity_weights = {
            'severe': 5.0,
            'severe damage': 5.0,
            'shatter': 4.0,
            'glass shatter': 4.0,
            'crack': 3.5,
            'dent': 3.0,
            'damaged-dent': 3.0,
            'dislocation': 2.5,
            'scratch': 2.0,
            'damaged-scratch': 2.0,
            'lamp broken': 3.5,
            'tire flat': 4.0,
            'damaged': 2.5
        }
        
        total_severity = 0.0
        for detection in detections:
            class_name = detection.get('class', '').lower()
            confidence = detection.get('confidence', 0.0)
            
            # Get severity weight for this class
            severity = severity_weights.get(class_name, 1.0)
            
            # Weight by confidence
            total_severity += severity * confidence
        
        return total_severity
    
    def normalize_class_names(self, class_name: str) -> str:
        """
        Normalize class names across different models for comparison.
        """
        class_name = class_name.lower().strip()
        
        # Mapping for similar classes across models
        class_mapping = {
            'dent': 'dent',
            'damaged-dent': 'dent',
            'scratch': 'scratch', 
            'damaged-scratch': 'scratch',
            'shatter': 'glass_damage',
            'glass shatter': 'glass_damage',
            'severe damage': 'severe_damage',
            'severe': 'severe_damage',
            'crack': 'crack',
            'lamp broken': 'lamp_damage',
            'tire flat': 'tire_damage',
            'dislocation': 'structural_damage',
            'damaged': 'general_damage'
        }
        
        return class_mapping.get(class_name, class_name)
    
    def aggregate_detections_by_class(self, all_detections: Dict[str, List[Dict]], model3_analysis: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Aggregate detections from all models by normalized class names.
        
        Returns:
            Dict with normalized class names as keys and aggregated info as values
        """
        aggregated = {}
        
        # Resolve Model 3 conflicts first
        resolved_model3_analysis = self.resolve_model3_conflicts(model3_analysis)
        
        for model_name, detections in all_detections.items():
            if not detections:
                continue
                
            # Use resolved detections for Model 3 if conflicts were resolved
            if model_name == 'model3' and resolved_model3_analysis and resolved_model3_analysis.get('conflict_resolved'):
                # Use the resolved detections instead of original ones
                detections = resolved_model3_analysis['detections']
                logger.info(f"Using resolved Model 3 detections: {resolved_model3_analysis['resolved_detection_count']} out of {resolved_model3_analysis['original_detection_count']} original detections")
                
            model_weight = self.model_weights.get(model_name, 0.1)
            
            for detection in detections:
                class_name = detection.get('class', '')
                confidence = detection.get('confidence', 0.0)
                
                # Skip low confidence detections
                if confidence < self.confidence_thresholds.get(model_name, 0.5):
                    continue
                
                normalized_class = self.normalize_class_names(class_name)
                
                if normalized_class not in aggregated:
                    aggregated[normalized_class] = {
                        'total_weighted_confidence': 0.0,
                        'total_weight': 0.0,
                        'detections': [],
                        'models': set()
                    }
                
                # Special handling for scratch detection - prioritize Model 4 over Model 1 and 2
                if normalized_class == 'scratch':
                    if model_name == 'model4':
                        # Give Model 4 higher weight for scratch detection
                        scratch_boost = 2.0
                        weighted_confidence = confidence * model_weight * scratch_boost
                        effective_weight = model_weight * scratch_boost
                    elif model_name in ['main_model', 'model2']:
                        # Reduce weight for Model 1 and 2 scratch detection when Model 4 is present
                        has_model4_scratch = any(
                            det.get('class', '').lower() in ['scratch', 'damaged-scratch'] 
                            for det in all_detections.get('model4', [])
                            if det.get('confidence', 0) >= self.confidence_thresholds.get('model4', 0.5)
                        )
                        if has_model4_scratch:
                            scratch_reduction = 0.5
                            weighted_confidence = confidence * model_weight * scratch_reduction
                            effective_weight = model_weight * scratch_reduction
                        else:
                            weighted_confidence = confidence * model_weight
                            effective_weight = model_weight
                    else:
                        # Normal weight for other models (model3)
                        weighted_confidence = confidence * model_weight
                        effective_weight = model_weight
                else:
                    # Normal weight for non-scratch detections
                    weighted_confidence = confidence * model_weight
                    effective_weight = model_weight
                
                aggregated[normalized_class]['total_weighted_confidence'] += weighted_confidence
                aggregated[normalized_class]['total_weight'] += effective_weight
                aggregated[normalized_class]['detections'].append({
                    'model': model_name,
                    'original_class': class_name,
                    'confidence': confidence,
                    'weighted_confidence': weighted_confidence
                })
                aggregated[normalized_class]['models'].add(model_name)
        
        # Calculate final confidence scores
        for class_info in aggregated.values():
            if class_info['total_weight'] > 0:
                class_info['final_confidence'] = class_info['total_weighted_confidence'] / class_info['total_weight']
            else:
                class_info['final_confidence'] = 0.0
        
        return aggregated
    
    def generate_ensemble_prediction(self, main_analysis, model2_analysis, model3_analysis, model4_analysis) -> Dict[str, Any]:
        """
        Generate ensemble prediction using all models with smart logic.
        
        Args:
            main_analysis: Main model analysis result
            model2_analysis: Model2 analysis result  
            model3_analysis: Model3 analysis result
            model4_analysis: Model4 analysis result
            
        Returns:
            Dict containing ensemble prediction results
        """
        logger.info("Starting ensemble prediction generation")
        
        # Check for Model 3's good_condition override with resolved conflicts
        resolved_model3_analysis = self.resolve_model3_conflicts(model3_analysis)
        has_good_condition, good_condition_confidence = self.has_good_condition_detection(resolved_model3_analysis)
        
        if has_good_condition:
            # Use standard confidence threshold since conflicts are already resolved
            required_confidence = self.confidence_thresholds['model3']
            
            if good_condition_confidence >= required_confidence:
                was_conflict_resolved = resolved_model3_analysis and resolved_model3_analysis.get('conflict_resolved', False)
                ensemble_score = 95 if not was_conflict_resolved else 85
                
                logger.info(f"Model 3 detected good_condition with confidence {good_condition_confidence:.3f} - overriding other predictions")
                return {
                    'prediction': 'good_condition',
                    'confidence': good_condition_confidence,
                    'reasoning': f"Model 3 detected good condition with high confidence ({good_condition_confidence:.3f}){' (conflicts resolved)' if was_conflict_resolved else ''}",
                    'override_applied': True,
                    'override_model': 'model3',
                    'ensemble_score': ensemble_score,
                    'damage_detected': False,
                    'damage_types': [],
                    'models_agreement': {
                        'model3_good_condition': True,
                        'other_models_overridden': True,
                        'model3_conflict_resolved': was_conflict_resolved
                    },
                    'unified_detections': []
                }
        
        # Collect all detections from available models
        all_detections = {}
        
        # Main model detections
        if main_analysis and hasattr(main_analysis, 'defect_groups'):
            main_detections = []
            for group in main_analysis.defect_groups:
                main_detections.append({
                    'class': group.defect_type,
                    'confidence': float(group.confidence),
                    'bbox': group.bbox
                })
            all_detections['main_model'] = main_detections
        
        # Model2 detections
        if model2_analysis and model2_analysis.get('detections'):
            all_detections['model2'] = model2_analysis['detections']
        
        # Model3 detections (excluding good_condition since we already checked)
        if model3_analysis and model3_analysis.get('detections'):
            model3_detections = [
                det for det in model3_analysis['detections']
                if det.get('class', '').lower() not in ['good_condition', 'good condition']
            ]
            all_detections['model3'] = model3_detections
        
        # Model4 detections (excluding good_condition)
        if model4_analysis and model4_analysis.get('detections'):
            model4_detections = [
                det for det in model4_analysis['detections']
                if det.get('class', '').lower() not in ['good_condition', 'good condition']
            ]
            all_detections['model4'] = model4_detections
        
        # If no damage detections from any model
        total_detections = sum(len(dets) for dets in all_detections.values())
        if total_detections == 0:
            return {
                'prediction': 'no_damage',
                'confidence': 0.85,
                'reasoning': 'No damage detected by any model',
                'override_applied': False,
                'ensemble_score': 90,
                'damage_detected': False,
                'damage_types': [],
                'models_agreement': {
                    'all_models_agree_no_damage': True
                },
                'unified_detections': []
            }
        
        # Aggregate detections by class (pass model3_analysis for conflict detection)
        aggregated_classes = self.aggregate_detections_by_class(all_detections, model3_analysis)
        
        # Calculate overall damage severity
        total_severity_score = 0.0
        detected_damage_types = []
        
        for class_name, class_info in aggregated_classes.items():
            if class_info['final_confidence'] >= 0.5:  # Minimum ensemble confidence
                detected_damage_types.append({
                    'type': class_name,
                    'confidence': class_info['final_confidence'],
                    'supporting_models': list(class_info['models'])
                })
                
                # Add to severity score
                severity_weights = {
                    'severe_damage': 5.0,
                    'glass_damage': 4.0,
                    'crack': 3.5,
                    'dent': 3.0,
                    'structural_damage': 2.5,
                    'scratch': 2.0,
                    'lamp_damage': 3.5,
                    'tire_damage': 4.0,
                    'general_damage': 2.5
                }
                
                severity = severity_weights.get(class_name, 2.0)
                total_severity_score += severity * class_info['final_confidence']
        
        # Determine final prediction
        if not detected_damage_types:
            prediction = 'no_damage'
            confidence = 0.75
            ensemble_score = 85
        else:
            prediction = 'damage_detected'
            # Confidence based on agreement between models
            num_supporting_models = len(set().union(*[dt['supporting_models'] for dt in detected_damage_types]))
            confidence = min(0.6 + (num_supporting_models * 0.1), 0.95)
            
            # Ensemble score based on severity with enhanced penalty for multiple damages
            base_score = max(100 - (total_severity_score * 8), 10)
            
            # Additional penalty for multiple damage types to prevent excellent ratings
            num_damage_types = len(detected_damage_types)
            if num_damage_types >= 3:
                # Significant penalty for 3+ damage types
                multiple_damage_penalty = min(num_damage_types * 8, 40)
                ensemble_score = max(base_score - multiple_damage_penalty, 15)
            elif num_damage_types == 2:
                # Moderate penalty for 2 damage types
                ensemble_score = max(base_score - 15, 20)
            else:
                ensemble_score = base_score
            
            # Cap excellent ratings when damage is detected
            if ensemble_score > 80 and num_damage_types > 0:
                ensemble_score = min(ensemble_score, 75)
        
        # Generate unified detections
        try:
            normalized_detections = self.normalize_detections_from_all_models(
                main_analysis, model2_analysis, model3_analysis, model4_analysis
            )
            
            if normalized_detections:
                clusters = self.cross_model_clustering(normalized_detections)
                
                # Extract image dimensions from main_analysis for coordinate conversion
                img_width, img_height = None, None
                if main_analysis and hasattr(main_analysis, 'car_bbox') and main_analysis.car_bbox:
                    # car_bbox format: (x1, y1, x2, y2) where x2, y2 represent image dimensions
                    img_width = main_analysis.car_bbox[2]
                    img_height = main_analysis.car_bbox[3]
                
                unified_detections = self.create_unified_bounding_boxes(
                    normalized_detections, clusters, img_width, img_height
                )
                
                # Convert UnifiedDetection objects to dictionaries for JSON serialization
                unified_detections_dict = []
                for unified_det in unified_detections:
                    unified_detections_dict.append({
                        'bbox': unified_det.bbox,
                        'class_name': unified_det.class_name,
                        'confidence': unified_det.confidence,
                        'source_models': unified_det.source_models,
                        'aggregated_confidence': unified_det.aggregated_confidence,
                        'detection_count': unified_det.detection_count,
                        'severity': unified_det.severity
                    })
            else:
                unified_detections_dict = []
                
            logger.info(f"Generated {len(unified_detections_dict)} unified detections from {len(normalized_detections)} individual detections")
            
        except Exception as e:
            logger.error(f"Error generating unified detections: {str(e)}")
            unified_detections_dict = []

        return {
            'prediction': prediction,
            'confidence': confidence,
            'reasoning': f"Ensemble decision based on {len(detected_damage_types)} damage types from {len(all_detections)} models",
            'override_applied': False,
            'ensemble_score': int(ensemble_score),
            'damage_detected': len(detected_damage_types) > 0,
            'damage_types': detected_damage_types,
            'severity_score': round(total_severity_score, 2),
            'models_agreement': {
                'total_models_used': len([k for k, v in all_detections.items() if v]),
                'damage_types_detected': len(detected_damage_types),
                'aggregated_classes': {k: v['final_confidence'] for k, v in aggregated_classes.items()}
            },
            'unified_detections': unified_detections_dict
        }
    
    def normalize_detections_from_all_models(self, main_analysis, model2_analysis, model3_analysis, model4_analysis) -> List[Dict[str, Any]]:
        """
        Extract and normalize detections from all model formats into a standardized format.
        
        Args:
            main_analysis: CarAnalysis object with defect_groups
            model2_analysis: Dict with detections list
            model3_analysis: Dict with detections list  
            model4_analysis: Dict with detections list
            
        Returns:
            List of normalized detections with format:
            {
                'bbox': [x1, y1, x2, y2],
                'class': str,
                'confidence': float,
                'source_model': str
            }
        """
        normalized_detections = []
        
        # Process main model (CarAnalysis with defect_groups)
        if main_analysis and hasattr(main_analysis, 'defect_groups'):
            for group in main_analysis.defect_groups:
                if hasattr(group, 'combined_bbox') and group.combined_bbox:
                    normalized_detections.append({
                        'bbox': group.combined_bbox,
                        'class': group.defect_type,
                        'confidence': getattr(group, 'confidence', 0.7),
                        'source_model': 'main_model'
                    })
        
        # Process model2, model3, model4 (raw detections format)
        models_data = [
            (model2_analysis, 'model2'),
            (model3_analysis, 'model3'), 
            (model4_analysis, 'model4')
        ]
        
        for model_analysis, model_name in models_data:
            if model_analysis and model_analysis.get('detections'):
                detections = model_analysis['detections']
                if isinstance(detections, list):
                    for detection in detections:
                        if isinstance(detection, dict):
                            bbox = detection.get('bbox', detection.get('box'))
                            class_name = detection.get('class', detection.get('class_name'))
                            confidence = detection.get('confidence', detection.get('conf', 0.5))
                            
                            if bbox and class_name:
                                # Ensure bbox is in [x1, y1, x2, y2] format
                                if len(bbox) >= 4:
                                    normalized_detections.append({
                                        'bbox': bbox[:4],
                                        'class': str(class_name),
                                        'confidence': float(confidence),
                                        'source_model': model_name
                                    })
        
        logger.info(f"Normalized {len(normalized_detections)} detections from all models")
        return normalized_detections
    
    def cross_model_clustering(self, normalized_detections: List[Dict[str, Any]], eps: float = 50.0, min_samples: int = 1) -> List[List[int]]:
        """
        Group overlapping detections from different models using DBSCAN clustering.
        
        Args:
            normalized_detections: List of normalized detections
            eps: Maximum distance between two samples for clustering
            min_samples: Minimum number of samples in a neighborhood
            
        Returns:
            List of clusters, where each cluster is a list of detection indices
        """
        if not normalized_detections:
            return []
        
        # Extract center points for clustering
        centers = []
        for detection in normalized_detections:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            centers.append([center_x, center_y])
        
        # Apply DBSCAN clustering
        centers_array = np.array(centers)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_array)
        
        # Group detections by cluster labels
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:  # Noise points become individual clusters
                clusters[f"noise_{idx}"] = [idx]
            else:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)
        
        # Refine clusters using IoU for better grouping
        refined_clusters = []
        for cluster_indices in clusters.values():
            if len(cluster_indices) == 1:
                refined_clusters.append(cluster_indices)
            else:
                # For multi-detection clusters, apply IoU refinement
                refined_cluster = self._refine_cluster_with_iou(normalized_detections, cluster_indices)
                refined_clusters.extend(refined_cluster)
        
        logger.info(f"Clustered {len(normalized_detections)} detections into {len(refined_clusters)} groups")
        return refined_clusters
    
    def _refine_cluster_with_iou(self, normalized_detections: List[Dict[str, Any]], cluster_indices: List[int], iou_threshold: float = 0.3) -> List[List[int]]:
        """
        Refine a cluster by splitting detections that don't have sufficient IoU overlap.
        
        Args:
            normalized_detections: List of normalized detections
            cluster_indices: Indices of detections in the cluster
            iou_threshold: Minimum IoU for detections to remain in same cluster
            
        Returns:
            List of refined sub-clusters
        """
        if len(cluster_indices) <= 1:
            return [cluster_indices]
        
        # Calculate IoU matrix for all pairs in the cluster
        refined_clusters = []
        remaining_indices = cluster_indices.copy()
        
        while remaining_indices:
            # Start a new sub-cluster with the first remaining detection
            current_cluster = [remaining_indices.pop(0)]
            
            # Find all detections that have sufficient IoU with any detection in current cluster
            i = 0
            while i < len(remaining_indices):
                candidate_idx = remaining_indices[i]
                candidate_bbox = normalized_detections[candidate_idx]['bbox']
                
                # Check IoU with any detection in current cluster
                has_overlap = False
                for cluster_idx in current_cluster:
                    cluster_bbox = normalized_detections[cluster_idx]['bbox']
                    iou = self._calculate_iou(candidate_bbox, cluster_bbox)
                    if iou >= iou_threshold:
                        has_overlap = True
                        break
                
                if has_overlap:
                    current_cluster.append(remaining_indices.pop(i))
                else:
                    i += 1
            
            refined_clusters.append(current_cluster)
        
        return refined_clusters
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Calculate intersection area
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Avoid division by zero
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area

    def create_unified_bounding_boxes(self, normalized_detections, clusters, img_width=None, img_height=None):
        """
        Create unified bounding boxes from clusters of detections.
        
        Args:
            normalized_detections: List of normalized detection dictionaries
            clusters: List of lists, each containing indices of detections in a cluster
            img_width: Image width for converting pixel coordinates to percentages
            img_height: Image height for converting pixel coordinates to percentages
            
        Returns:
            List of UnifiedDetection objects
        """
        unified_detections = []
        
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue
                
            # Get all detections in this cluster
            cluster_detections = [normalized_detections[i] for i in cluster]
            
            # Calculate unified bounding box (minimum bounding rectangle)
            min_x = min(det['bbox'][0] for det in cluster_detections)
            min_y = min(det['bbox'][1] for det in cluster_detections)
            max_x = max(det['bbox'][0] + det['bbox'][2] for det in cluster_detections)
            max_y = max(det['bbox'][1] + det['bbox'][3] for det in cluster_detections)
            
            # Convert to percentages if image dimensions are available
            if img_width and img_height:
                # Convert pixel coordinates to percentages and ensure they are Python floats
                unified_bbox = [
                    float((min_x / img_width) * 100),
                    float((min_y / img_height) * 100),
                    float(((max_x - min_x) / img_width) * 100),
                    float(((max_y - min_y) / img_height) * 100)
                ]
            else:
                # Keep pixel coordinates if no image dimensions and ensure they are Python floats
                unified_bbox = [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]
            
            # Aggregate class information (most common class)
            class_counts = {}
            for det in cluster_detections:
                class_name = det['class']  # Fixed: use 'class' instead of 'class_name'
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Get the most frequent class
            unified_class = max(class_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate aggregated confidence (weighted average by confidence)
            total_weight = sum(det['confidence'] for det in cluster_detections)
            if total_weight > 0:
                aggregated_confidence = sum(
                    det['confidence'] * det['confidence'] for det in cluster_detections
                ) / total_weight
            else:
                aggregated_confidence = 0.0
            
            # Get source models
            source_models = list(set(det['source_model'] for det in cluster_detections))
            
            # Calculate severity (highest severity in cluster)
            severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            max_severity_value = 0
            unified_severity = 'low'
            
            for det in cluster_detections:
                det_severity = det.get('severity', 'low')
                severity_value = severity_map.get(det_severity, 1)
                if severity_value > max_severity_value:
                    max_severity_value = severity_value
                    unified_severity = det_severity
            
            # Create unified detection
            unified_detection = UnifiedDetection(
                bbox=unified_bbox,
                class_name=unified_class,
                confidence=max(det['confidence'] for det in cluster_detections),
                source_models=source_models,
                aggregated_confidence=aggregated_confidence,
                detection_count=len(cluster_detections),
                severity=unified_severity
            )
            
            unified_detections.append(unified_detection)
        
        return unified_detections