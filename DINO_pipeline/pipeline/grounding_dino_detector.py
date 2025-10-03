"""
DINO Pipeline - SAM Merge Object Detection Module  
Uses SAM automatic segmentation with merge functionality for better object detection
"""

import cv2
import numpy as np
import torch
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Add the main project root to path to access existing SAM implementation
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import existing SAM implementation
try:
    from src.pipeline.object_detection_segmentation import ObjectDetectionSegmentation
    SAM_AVAILABLE = True
except ImportError:
    try:
        from pipeline.object_detection_segmentation import ObjectDetectionSegmentation
        SAM_AVAILABLE = True
    except ImportError:
        SAM_AVAILABLE = False

class SAMMergeDetector:
    """
    SAM-based object detector with automatic segmentation and merge functionality
    Uses the existing SAM implementation for better object detection
    """
    
    def __init__(self, device: str = "auto", confidence_threshold: float = 0.35, 
                 sam_checkpoint: str = None, model_type: str = "vit_b"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.confidence_threshold = confidence_threshold
        self.sam_detector = None
        
        # Initialize SAM detector or fallback
        if SAM_AVAILABLE:
            self._init_sam_detector(sam_checkpoint, model_type)
            self.model_type = "sam_merge"
        else:
            self._init_fallback()
            self.model_type = "fallback"
            
        print(f"✅ SAM Merge Detector initialized using: {self.model_type}")
    
    def _init_sam_detector(self, sam_checkpoint: str, model_type: str):
        """Initialize SAM detector using existing implementation"""
        try:
            # Try to find SAM checkpoint if not provided
            if sam_checkpoint is None:
                sam_checkpoint = self._find_sam_checkpoint()
            
            print(f"Loading SAM detector with checkpoint: {sam_checkpoint}")
            self.sam_detector = ObjectDetectionSegmentation(
                checkpoint_path=sam_checkpoint,
                model_type=model_type,
                device=self.device,
                use_yolo=False  # Use SAM-only mode for pure segmentation
            )
            print("✅ SAM detector initialized successfully")
            
        except Exception as e:
            print(f"⚠️ SAM initialization failed: {e}")
            print("Falling back to simulated detections")
            self._init_fallback()
            self.model_type = "fallback"
    
    def _find_sam_checkpoint(self):
        """Try to find SAM checkpoint in common locations"""
        possible_paths = [
            "sam_vit_b_01ec64.pth",
            "../sam_vit_b_01ec64.pth",
            "../../sam_vit_b_01ec64.pth",
            "../../../sam_vit_b_01ec64.pth",
            str(project_root / "sam_vit_b_01ec64.pth"),
            str(project_root / "models" / "sam_vit_b_01ec64.pth")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return str(path)
        
        # If no checkpoint found, provide instructions
        raise FileNotFoundError(
            "SAM checkpoint not found. Please download from: "
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )
    
    def _init_fallback(self):
        """Initialize simple fallback detector"""
        print("⚠️ Using fallback object detector (simulated detections)")
        # Create some realistic fallback detections for common objects
        self.fallback_objects = [
            {"label": "bottle", "confidence": 0.85},
            {"label": "cup", "confidence": 0.75},
            {"label": "laptop", "confidence": 0.90},
            {"label": "mouse", "confidence": 0.80},
            {"label": "keyboard", "confidence": 0.88},
            {"label": "book", "confidence": 0.70},
        ]
    
    def detect_objects(self, rgb_image: np.ndarray, text_prompt: str = None) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect objects in RGB image using SAM automatic segmentation with merge
        
        Args:
            rgb_image: RGB image (H, W, 3)
            text_prompt: Optional text prompt (not used in SAM mode)
            
        Returns:
            detections: List of detection dictionaries
            masks: List of segmentation masks
        """
        if self.model_type == "sam_merge":
            return self._detect_sam_merge(rgb_image)
        else:
            return self._detect_fallback(rgb_image)
    
    def _detect_sam_merge(self, rgb_image: np.ndarray) -> Tuple[List[Dict], List[np.ndarray]]:
        """Detect objects using SAM automatic segmentation with merge functionality - major objects only"""
        try:
            # Use SAM detector for automatic segmentation
            masks, boxes = self.sam_detector.detect_and_segment(rgb_image)
            
            # Filter for major objects only
            major_objects = self._filter_major_objects(masks, boxes, rgb_image)
            
            # Convert to detection format
            detections = []
            filtered_masks = []
            
            for mask, box in major_objects:
                # Estimate object type based on properties
                object_label = self._estimate_object_type(mask, box, rgb_image)
                
                # Calculate enhanced confidence for major objects
                confidence = self._calculate_major_object_confidence(mask, box, rgb_image)
                
                detection = {
                    "label": object_label,
                    "confidence": confidence,
                    "bbox": box,  # [x, y, w, h] 
                    "bbox_xyxy": [box[0], box[1], box[0]+box[2], box[1]+box[3]]
                }
                detections.append(detection)
                filtered_masks.append(mask)
            
            print(f"   ✅ SAM Merge detected {len(detections)} major objects (filtered from {len(masks)} total)")
            return detections, filtered_masks
            
        except Exception as e:
            print(f"   ⚠️ SAM detection failed: {e}")
            return self._detect_fallback(rgb_image)
    
    def _estimate_object_type(self, mask: np.ndarray, box: List[int], image: np.ndarray) -> str:
        """Estimate object type based on mask and bounding box properties"""
        x, y, w, h = box
        aspect_ratio = w / h if h > 0 else 1.0
        area_ratio = (w * h) / (image.shape[0] * image.shape[1])
        
        # Simple heuristics for object classification
        if aspect_ratio > 2.0:  # Wide objects
            return "keyboard" if area_ratio > 0.1 else "remote"
        elif aspect_ratio < 0.5:  # Tall objects
            return "bottle" if area_ratio < 0.1 else "monitor"
        elif area_ratio > 0.2:  # Large objects
            return "laptop"
        elif area_ratio > 0.05:  # Medium objects
            return "book"
        else:  # Small objects
            return "cup" if aspect_ratio < 1.2 else "mouse"
    
    def _calculate_mask_confidence(self, mask: np.ndarray) -> float:
        """Calculate confidence score based on mask properties"""
        # Use mask area and compactness as confidence indicators
        area = np.sum(mask > 0)
        if area == 0:
            return 0.1
        
        # Calculate compactness (area / perimeter^2)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.5
        
        perimeter = cv2.arcLength(contours[0], True)
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Normalize compactness to confidence score
        confidence = min(max(compactness * 2 + 0.3, 0.3), 0.95)
        return confidence
    
    def _filter_major_objects(self, masks: List[np.ndarray], boxes: List[List[int]], 
                             image: np.ndarray) -> List[Tuple[np.ndarray, List[int]]]:
        """Filter to keep only major/important objects"""
        if not masks or not boxes:
            return []
            
        image_area = image.shape[0] * image.shape[1]
        major_objects = []
        
        # Calculate properties for each object
        object_props = []
        for mask, box in zip(masks, boxes):
            x, y, w, h = box
            area = np.sum(mask > 0)
            area_ratio = area / image_area
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Calculate center position (prefer objects in center region)
            center_x = x + w / 2
            center_y = y + h / 2
            img_center_x = image.shape[1] / 2
            img_center_y = image.shape[0] / 2
            center_distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            max_distance = np.sqrt(img_center_x**2 + img_center_y**2)
            center_score = 1.0 - (center_distance / max_distance)
            
            object_props.append({
                'mask': mask,
                'box': box,
                'area': area,
                'area_ratio': area_ratio,
                'aspect_ratio': aspect_ratio,
                'center_score': center_score,
                'importance_score': area_ratio * 0.7 + center_score * 0.3
            })
        
        # Sort by importance score
        object_props.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Filter criteria for major objects
        for obj in object_props:
            # Skip very small objects (less than 0.5% of image)
            if obj['area_ratio'] < 0.005:
                continue
                
            # Skip very large objects that might be background (more than 40% of image)
            if obj['area_ratio'] > 0.4:
                continue
                
            # Skip objects with extreme aspect ratios that might be noise
            if obj['aspect_ratio'] > 10 or obj['aspect_ratio'] < 0.1:
                continue
                
            # Keep objects with decent size and importance
            if obj['area_ratio'] > 0.01 or obj['importance_score'] > 0.3:
                major_objects.append((obj['mask'], obj['box']))
                
            # Limit to top 15 most important objects to avoid clutter
            if len(major_objects) >= 15:
                break
        
        return major_objects
    
    def _calculate_major_object_confidence(self, mask: np.ndarray, box: List[int], image: np.ndarray) -> float:
        """Calculate enhanced confidence score for major objects"""
        area = np.sum(mask > 0)
        if area == 0:
            return 0.1
        
        x, y, w, h = box
        image_area = image.shape[0] * image.shape[1]
        area_ratio = area / image_area
        
        # Calculate compactness (area / perimeter^2)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.5
        
        perimeter = cv2.arcLength(contours[0], True)
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Size bonus for reasonably sized objects
        size_bonus = min(1.0, area_ratio * 20)  # Boost confidence for larger objects
        
        # Aspect ratio penalty for extreme shapes
        aspect_ratio = w / h if h > 0 else 1.0
        aspect_penalty = 1.0 if 0.2 <= aspect_ratio <= 5.0 else 0.7
        
        # Combined confidence score
        base_confidence = compactness * 0.5 + size_bonus * 0.3 + aspect_penalty * 0.2
        
        # Normalize confidence between 0.4 and 0.95 for major objects
        final_confidence = min(0.95, max(0.4, base_confidence))
        return final_confidence
    
    def _detect_fallback(self, rgb_image: np.ndarray) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """Generate simulated detections for testing"""
        h, w = rgb_image.shape[:2]
        
        detections = []
        # Generate some realistic bounding boxes
        boxes = [
            [int(w*0.1), int(h*0.2), int(w*0.15), int(h*0.3)],  # Small object
            [int(w*0.3), int(h*0.1), int(w*0.4), int(h*0.6)],   # Large object
            [int(w*0.6), int(h*0.4), int(w*0.2), int(h*0.25)],  # Medium object
        ]
        
        for i, (obj, box) in enumerate(zip(self.fallback_objects[:3], boxes)):
            detection = {
                "label": obj["label"],
                "confidence": obj["confidence"],
                "bbox": box,  # [x, y, w, h]
                "bbox_xyxy": [box[0], box[1], box[0]+box[2], box[1]+box[3]]
            }
            detections.append(detection)
        
        return detections, None
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict], 
                           save_path: str = None) -> np.ndarray:
        """
        Visualize detections on image
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            save_path: Optional path to save visualization
            
        Returns:
            Annotated image
        """
        vis_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection["bbox"]
            x, y, w, h = int(x), int(y), int(w), int(h)  # Ensure integer coordinates
            label = detection["label"]
            confidence = detection["confidence"]
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_image, (int(x), int(y - text_h - 10)), (int(x + text_w), int(y)), (0, 255, 0), -1)
            cv2.putText(vis_image, label_text, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"✅ Detection visualization saved to: {save_path}")
        
        return vis_image


def main():
    """Test the SAM Merge detector"""
    detector = SAMMergeDetector()
    
    # Test with sample image
    test_image_path = "../src/img.PNG"
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return
    
    # Load test image
    image = cv2.imread(test_image_path)
    print(f"Loaded image shape: {image.shape}")
    
    # Run detection
    detections, masks = detector.detect_objects(image)
    
    print(f"Found {len(detections)} objects:")
    for det in detections:
        print(f"  - {det['label']}: {det['confidence']:.2f}")
    
    # Visualize results
    vis_image = detector.visualize_detections(image, detections, "../outputs/sam_merge_detections.jpg")
    
    print("✅ SAM Merge detection test completed!")


if __name__ == "__main__":
    import os
    main()