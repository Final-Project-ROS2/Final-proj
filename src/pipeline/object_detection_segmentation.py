import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from typing import List, Dict, Tuple
import os

# Try to import YOLO - fallback if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO (ultralytics) not available - using SAM-only detection")

class ObjectDetectionSegmentation:
    """
    Stage 1: Object Detection and Segmentation using YOLO + SAM
    Combines YOLO detection with SAM segmentation and optional depth processing
    """
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str = "auto", 
                 yolo_model: str = "yolo11s.pt", use_yolo: bool = True):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize SAM
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(self.device)
        sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.predictor = SamPredictor(sam)  # For box-prompted segmentation
        
        # Initialize YOLO if available and requested
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.yolo_model = None
        if self.use_yolo:
            try:
                self.yolo_model = YOLO(yolo_model)
                print(f"✅ YOLO model loaded: {yolo_model}")
            except Exception as e:
                print(f"⚠️ Could not load YOLO model: {e}")
                self.use_yolo = False
        
        print(f"Detection mode: {'YOLO+SAM' if self.use_yolo else 'SAM-only'}")
        
    def detect_and_segment(self, image: np.ndarray, depth: np.ndarray = None) -> Tuple[List[np.ndarray], List[List[float]]]:
        """
        Use YOLO + SAM for object detection and segmentation
        
        Args:
            image: RGB image
            depth: Depth map (optional)
            
        Returns:
            masks: List of binary masks
            boxes: List of bounding boxes [x, y, w, h]
        """
        if self.use_yolo:
            return self._yolo_sam_detection(image, depth)
        else:
            return self._sam_only_detection(image, depth)
    
    def _yolo_sam_detection(self, image: np.ndarray, depth: np.ndarray = None) -> Tuple[List[np.ndarray], List[List[float]]]:
        """YOLO detection followed by SAM segmentation refinement"""
        # Step 1: YOLO Detection
        yolo_results = self.yolo_model(image)
        
        # Extract bounding boxes
        boxes = []
        confidences = []
        for result in yolo_results:
            if hasattr(result.boxes, 'xyxy'):
                for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                    x1, y1, x2, y2 = box[:4]
                    w, h = x2 - x1, y2 - y1
                    boxes.append([int(x1), int(y1), int(w), int(h)])
                    
                    # Get confidence if available
                    if hasattr(result.boxes, 'conf') and len(result.boxes.conf) > i:
                        confidences.append(float(result.boxes.conf[i].cpu().numpy()))
                    else:
                        confidences.append(1.0)
        
        # Step 2: SAM Segmentation using YOLO boxes as prompts
        masks = []
        if boxes:
            # Set image for SAM predictor
            self.predictor.set_image(image)
            
            for box in boxes:
                x, y, w, h = box
                # Convert to xyxy format for SAM
                input_box = np.array([x, y, x + w, y + h])
                
                # Generate mask using box prompt
                mask, _, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                masks.append(mask[0])  # Take first mask
        
        print(f"   ✅ YOLO+SAM: Detected {len(boxes)} objects with refined segmentation")
        return masks, boxes
    
    def _sam_only_detection(self, image: np.ndarray, depth: np.ndarray = None) -> Tuple[List[np.ndarray], List[List[float]]]:
        """SAM-only automatic segmentation (fallback method)"""
        # If depth is provided, use it for better segmentation
        if depth is not None:
            # Normalize depth for better fusion
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Create 3-channel depth for concatenation
            depth_3ch = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2RGB)
            # Fuse RGB and depth (simple concatenation - can be improved)
            fused_input = np.concatenate([image, depth_3ch], axis=2)[:, :, :3]  # Keep 3 channels for SAM
        else:
            fused_input = image
            
        with torch.no_grad():
            sam_masks = self.mask_generator.generate(fused_input)
            
        # Filter masks using depth information if available
        if depth is not None:
            sam_masks = self._filter_masks_with_depth(sam_masks, depth)
            
        # Sort by area and filter small masks
        sam_masks = sorted(sam_masks, key=lambda x: x['area'], reverse=True)
        min_area = image.shape[0] * image.shape[1] * 0.001
        filtered_masks = [mask for mask in sam_masks if mask['area'] > min_area]
        
        # Apply mask merging strategy for elongated objects
        merged_masks = self._merge_fragmented_masks(filtered_masks, image.shape[:2])
        
        # Extract masks and boxes
        masks = [mask['segmentation'] for mask in merged_masks]
        boxes = [mask['bbox'] for mask in merged_masks]
        
        return masks, boxes
    
    def segment_with_boxes(self, image: np.ndarray, boxes: List[List[float]]) -> List[np.ndarray]:
        """
        Use SAM with bounding box prompts from YOLO
        
        Args:
            image: RGB image
            boxes: List of bounding boxes [x, y, w, h] from YOLO
            
        Returns:
            masks: List of refined binary masks
        """
        self.predictor.set_image(image)
        refined_masks = []
        
        for bbox in boxes:
            x, y, w, h = bbox
            # Convert [x, y, w, h] to [x1, y1, x2, y2] for SAM
            input_box = np.array([x, y, x + w, y + h])
            
            with torch.no_grad():
                masks, scores, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
            
            # Take the best mask (first one when multimask_output=False)
            refined_masks.append(masks[0])
        
        return refined_masks
    
    def _filter_masks_with_depth(self, masks: List[Dict], depth: np.ndarray) -> List[Dict]:
        """Filter out masks in noisy/thin depth regions"""
        filtered_masks = []
        
        for mask in masks:
            segmentation = mask['segmentation']
            # Get depth values in the mask region
            mask_depth = depth[segmentation]
            
            # Filter out regions with too much depth noise or invalid depth
            valid_depth = mask_depth[mask_depth > 0]  # Remove invalid depth (0)
            
            if len(valid_depth) > 0:
                depth_std = np.std(valid_depth)
                depth_mean = np.mean(valid_depth)
                
                # Keep masks with reasonable depth consistency
                if depth_std < 0.1 * depth_mean:  # Less than 10% variation
                    filtered_masks.append(mask)
        
        return filtered_masks
    
    def _merge_fragmented_masks(self, masks: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Merge fragmented masks from elongated objects using bounding box overlap 
        and mask adjacency analysis.
        
        Args:
            masks: List of mask dictionaries from SAM
            image_shape: (height, width) of the original image
            
        Returns:
            List of merged mask dictionaries
        """
        if len(masks) <= 1:
            return masks
            
        merged_masks = []
        used_indices = set()
        
        for i, mask_a in enumerate(masks):
            if i in used_indices:
                continue
                
            # Start with current mask
            merged_mask = mask_a.copy()
            merge_candidates = [i]
            
            # Find masks to merge with current mask
            for j, mask_b in enumerate(masks[i+1:], start=i+1):
                if j in used_indices:
                    continue
                    
                # Check if masks should be merged
                if self._should_merge_masks(mask_a, mask_b, image_shape):
                    merge_candidates.append(j)
                    used_indices.add(j)
            
            # If we have masks to merge, combine them
            if len(merge_candidates) > 1:
                merged_mask = self._combine_masks([masks[idx] for idx in merge_candidates], image_shape)
            
            merged_masks.append(merged_mask)
            used_indices.add(i)
        
        return merged_masks
    
    def _should_merge_masks(self, mask_a: Dict, mask_b: Dict, image_shape: Tuple[int, int]) -> bool:
        """
        Determine if two masks should be merged based on:
        1. Bounding box overlap
        2. Mask adjacency
        3. Aspect ratio similarity (for elongated objects)
        """
        bbox_a = mask_a['bbox']  # [x, y, w, h]
        bbox_b = mask_b['bbox']
        
        # Calculate bounding box overlap (IoU)
        overlap_ratio = self._calculate_bbox_overlap(bbox_a, bbox_b)
        
        # Check mask adjacency
        adjacency_score = self._calculate_mask_adjacency(
            mask_a['segmentation'], mask_b['segmentation']
        )
        
        # Check aspect ratio similarity for elongated objects
        aspect_ratio_a = max(bbox_a[2], bbox_a[3]) / min(bbox_a[2], bbox_a[3])
        aspect_ratio_b = max(bbox_b[2], bbox_b[3]) / min(bbox_b[2], bbox_b[3])
        aspect_similarity = min(aspect_ratio_a, aspect_ratio_b) / max(aspect_ratio_a, aspect_ratio_b)
        
        # Merge criteria (tuned for elongated objects like pliers)
        merge_conditions = [
            overlap_ratio > 0.1,  # Some bounding box overlap
            adjacency_score > 0.05,  # Masks are close/adjacent
            aspect_ratio_a > 2.0 or aspect_ratio_b > 2.0,  # At least one is elongated
            aspect_similarity > 0.5,  # Similar elongation
            abs(mask_a['area'] - mask_b['area']) / max(mask_a['area'], mask_b['area']) < 0.8  # Similar sizes
        ]
        
        # Need at least 3 conditions for merging
        return sum(merge_conditions) >= 3
    
    def _calculate_bbox_overlap(self, bbox_a: List[float], bbox_b: List[float]) -> float:
        """Calculate intersection over union (IoU) of two bounding boxes"""
        x1_a, y1_a, w_a, h_a = bbox_a
        x2_a, y2_a = x1_a + w_a, y1_a + h_a
        
        x1_b, y1_b, w_b, h_b = bbox_b
        x2_b, y2_b = x1_b + w_b, y1_b + h_b
        
        # Calculate intersection
        x1_inter = max(x1_a, x1_b)
        y1_inter = max(y1_a, y1_b)
        x2_inter = min(x2_a, x2_b)
        y2_inter = min(y2_a, y2_b)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area_a = w_a * h_a
        area_b = w_b * h_b
        union = area_a + area_b - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_mask_adjacency(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Calculate how adjacent two masks are (higher score = more adjacent)"""
        # Dilate masks slightly to check for near-adjacency
        kernel = np.ones((5, 5), np.uint8)
        dilated_a = cv2.dilate(mask_a.astype(np.uint8), kernel, iterations=1)
        dilated_b = cv2.dilate(mask_b.astype(np.uint8), kernel, iterations=1)
        
        # Check overlap between dilated masks
        intersection = np.logical_and(dilated_a, mask_b).sum()
        intersection += np.logical_and(dilated_b, mask_a).sum()
        
        # Normalize by smaller mask size
        smaller_mask_size = min(mask_a.sum(), mask_b.sum())
        
        return intersection / smaller_mask_size if smaller_mask_size > 0 else 0.0
    
    def _combine_masks(self, masks: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """Combine multiple masks into a single mask"""
        # Combine segmentations
        combined_segmentation = np.zeros(image_shape, dtype=bool)
        total_area = 0
        
        for mask in masks:
            combined_segmentation = np.logical_or(combined_segmentation, mask['segmentation'])
            total_area += mask['area']
        
        # Calculate new bounding box
        coords = np.where(combined_segmentation)
        if len(coords[0]) == 0:
            # Fallback to first mask if combination failed
            return masks[0]
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        new_bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
        
        # Create combined mask dictionary
        combined_mask = {
            'segmentation': combined_segmentation,
            'bbox': new_bbox,
            'area': combined_segmentation.sum(),
            'predicted_iou': max(mask.get('predicted_iou', 0.5) for mask in masks),
            'stability_score': np.mean([mask.get('stability_score', 0.5) for mask in masks])
        }
        
        return combined_mask
    
    def visualize_detections(self, image: np.ndarray, masks: List[np.ndarray], boxes: List[List[float]], 
                           save_path: str = None) -> np.ndarray:
        """Visualize detected objects with masks and bounding boxes"""
        overlay = image.copy()
        
        # Define distinct colors for better visibility
        colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green  
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
        ]
        
        for i, (mask, bbox) in enumerate(zip(masks, boxes)):
            color = colors[i % len(colors)]
            
            # Create a lighter mask overlay (higher transparency)
            mask_colored = np.zeros_like(image, dtype=np.uint8)
            mask_colored[mask > 0] = color
            
            # Use lighter blending for better visibility
            overlay = cv2.addWeighted(overlay, 0.85, mask_colored, 0.15, 0)
            
            # Draw bounding box with thicker lines
            x, y, w, h = bbox
            cv2.rectangle(overlay, (int(x), int(y)), (int(x+w), int(y+h)), color, 3)
            
            # Add object label with background for better readability
            label = f"Obj_{i+1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(overlay, (int(x), int(y-label_size[1]-10)), 
                         (int(x+label_size[0]+5), int(y)), color, -1)
            cv2.putText(overlay, label, (int(x+2), int(y-5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 2)
        
        if save_path:
            cv2.imwrite(save_path, overlay)
            
        return overlay
