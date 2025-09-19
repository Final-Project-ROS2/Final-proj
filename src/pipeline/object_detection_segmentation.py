import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from typing import List, Dict, Tuple

class ObjectDetectionSegmentation:
    """
    Stage 1: Object Detection and Segmentation using Grounding-SAM or Mask R-CNN
    Combines RGB + depth for better segmentation
    """
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # Initialize SAM (can be replaced with Grounding-SAM)
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(self.device)
        sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.predictor = SamPredictor(sam)  # For box-prompted segmentation
        
    def detect_and_segment(self, image: np.ndarray, depth: np.ndarray = None) -> Tuple[List[np.ndarray], List[List[float]]]:
        """
        Use SAM / Grounding-SAM with RGB + depth fusion
        
        Args:
            image: RGB image
            depth: Depth map (optional)
            
        Returns:
            masks: List of binary masks
            boxes: List of bounding boxes [x, y, w, h]
        """
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
        
        # Extract masks and boxes
        masks = [mask['segmentation'] for mask in filtered_masks]
        boxes = [mask['bbox'] for mask in filtered_masks]
        
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
