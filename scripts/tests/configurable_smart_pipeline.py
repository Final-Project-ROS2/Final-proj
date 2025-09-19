"""
Configurable Smart Ensemble Pipeline: WBF + YOLOv11-Seg + SAM
Uses config.py for easy image and model switching.
"""

import cv2
import numpy as np
import json
from ultralytics import YOLO
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration and pipeline modules
from config import config
from src.pipeline.object_detection_segmentation import ObjectDetectionSegmentation

def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.01):
    """
    Simplified Weighted Boxes Fusion implementation
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    # Combine all detections
    for i, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        if len(boxes) > 0:
            # Apply weight to scores
            weighted_scores = scores * weights[i]
            all_boxes.extend(boxes)
            all_scores.extend(weighted_scores)
            all_labels.extend(labels)
    
    if not all_boxes:
        return np.array([]), np.array([]), np.array([])
    
    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Simple NMS-style fusion
    keep_indices = []
    sorted_indices = np.argsort(all_scores)[::-1]
    
    for i in sorted_indices:
        if all_scores[i] < skip_box_thr:
            continue
            
        keep = True
        for j in keep_indices:
            # Calculate IoU
            box1 = all_boxes[i]
            box2 = all_boxes[j]
            
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > iou_thr:
                    keep = False
                    break
        
        if keep:
            keep_indices.append(i)
    
    if keep_indices:
        return all_boxes[keep_indices], all_scores[keep_indices], all_labels[keep_indices]
    else:
        return np.array([]), np.array([]), np.array([])

def run_yolo_detection(model_name, image_path):
    """Run YOLO detection on an image"""
    try:
        model_path = config.get_model_path("yolo", model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"   ❌ Model {model_name} not found at {model_path}")
            return [], [], []
        
        model = YOLO(model_path)
        results = model(image_path, conf=config.YOLO_CONFIDENCE_THRESHOLD, iou=config.YOLO_IOU_THRESHOLD, verbose=False)
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            labels = results[0].boxes.cls.cpu().numpy()
            
            # Convert class indices to names
            class_names = [model.names[int(cls)] for cls in labels]
            
            print(f"   ✅ {model_name} detected {len(boxes)} objects with avg confidence: {scores.mean():.2f}")
            return boxes, scores, class_names
        else:
            print(f"   ✅ {model_name} detected 0 objects")
            return [], [], []
            
    except Exception as e:
        print(f"   ❌ Error with {model_name}: {str(e)}")
        return [], [], []

def run_yolo_segmentation(model_name, image_path):
    """Run YOLO segmentation on an image"""
    try:
        model_path = config.get_model_path("yolo", model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"   ❌ Segmentation model {model_name} not found")
            return [], [], []
        
        model = YOLO(model_path)
        results = model(image_path, conf=config.YOLO_CONFIDENCE_THRESHOLD, iou=config.YOLO_IOU_THRESHOLD, verbose=False)
        
        if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            labels = results[0].boxes.cls.cpu().numpy()
            
            class_names = [model.names[int(cls)] for cls in labels]
            
            print(f"   ✅ {model_name} segmented {len(masks)} objects")
            return masks, boxes, scores, class_names
        else:
            print(f"   ✅ {model_name} segmented 0 objects")
            return [], [], [], []
            
    except Exception as e:
        print(f"   ❌ Error with {model_name}: {str(e)}")
        return [], [], [], []

def merge_masks_by_iou(masks, threshold=0.7):
    """Merge masks that have high IoU overlap"""
    if len(masks) <= 1:
        return masks
    
    merged_masks = []
    used_indices = set()
    
    for i, mask1 in enumerate(masks):
        if i in used_indices:
            continue
        
        # Start with current mask
        combined_mask = mask1.copy()
        merge_group = [i]
        
        # Find masks to merge with this one
        for j, mask2 in enumerate(masks[i+1:], i+1):
            if j in used_indices:
                continue
            
            # Calculate IoU
            intersection = np.sum((mask1 > 0) & (mask2 > 0))
            union = np.sum((mask1 > 0) | (mask2 > 0))
            iou = intersection / union if union > 0 else 0
            
            if iou > threshold:
                combined_mask = np.maximum(combined_mask, mask2)
                merge_group.append(j)
                used_indices.add(j)
        
        merged_masks.append(combined_mask)
        used_indices.add(i)
    
    print(f"   📊 Result: {len(merged_masks)} masks (from {len(masks)} original)")
    return merged_masks

def create_visualization(image, fused_boxes, fused_scores, fused_labels, masks, output_path):
    """Create enhanced visualization with masks and boxes"""
    vis_image = image.copy()
    
    # Draw masks first (underneath)
    if len(masks) > 0:
        mask_overlay = np.zeros_like(vis_image, dtype=np.uint8)
        
        for i, mask in enumerate(masks):
            if mask.size > 0:
                # Resize mask to image size if needed
                if mask.shape != image.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))
                
                # Create colored mask
                color = config.COLORS["ensemble"]
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask > 0] = color
                mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 0.7, 0)
        
        # Blend with original image
        vis_image = cv2.addWeighted(vis_image, 1.0, mask_overlay, config.MASK_ALPHA, 0)
    
    # Draw boxes and labels
    for i, (box, score, label) in enumerate(zip(fused_boxes, fused_scores, fused_labels)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), config.COLORS["ensemble"], config.BOX_THICKNESS)
        
        # Draw label and confidence
        if config.SHOW_LABELS and config.SHOW_CONFIDENCE:
            label_text = f"{label}: {score:.2f}"
        elif config.SHOW_LABELS:
            label_text = label
        elif config.SHOW_CONFIDENCE:
            label_text = f"{score:.2f}"
        else:
            label_text = ""
        
        if label_text:
            # Get text size for background
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALE, 1)
            
            # Draw background rectangle
            cv2.rectangle(vis_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), config.COLORS["ensemble"], -1)
            
            # Draw text
            cv2.putText(vis_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALE, (255, 255, 255), 1)
    
    # Save image
    cv2.imwrite(output_path, vis_image)
    return vis_image

def main():
    """Main pipeline function"""
    print("🚀 Configurable Smart Ensemble Pipeline")
    print("=" * 60)
    
    # Validate configuration
    if not config.validate_config():
        print("❌ Configuration validation failed. Please check config.py")
        return
    
    # Get current image path
    image_path = config.get_current_image_path()
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"📸 Processing image: {config.CURRENT_IMAGE} ({image.shape[1]}x{image.shape[0]})")
    print()
    
    # Step 1: Multi-Model Detection
    print("📊 STEP 1: Multi-Model Detection")
    print()
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for model_name in config.CURRENT_YOLO_MODELS:
        print(f"🔍 Running {model_name} detection...")
        boxes, scores, labels = run_yolo_detection(model_name, image_path)
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
    
    # Step 2: Weighted Boxes Fusion
    print()
    print("🔗 STEP 2: Weighted Boxes Fusion")
    
    # Ensure we have the right number of weights
    weights = config.MODEL_WEIGHTS[:len(config.CURRENT_YOLO_MODELS)]
    if len(weights) < len(config.CURRENT_YOLO_MODELS):
        weights.extend([1.0] * (len(config.CURRENT_YOLO_MODELS) - len(weights)))
    
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels, 
        weights=weights,
        iou_thr=config.WBF_IOU_THRESHOLD,
        skip_box_thr=config.WBF_SKIP_BOX_THRESHOLD
    )
    
    total_detections = sum(len(boxes) for boxes in all_boxes)
    print(f"   📊 WBF Result: {len(fused_boxes)} fused boxes (from {total_detections} total detections)")
    
    # Step 3: YOLOv11-Seg Direct Segmentation
    print()
    print("🎯 STEP 3: YOLOv11-Seg Direct Segmentation")
    print()
    
    seg_masks = []
    if "yolo11s_seg" in config.YOLO_MODELS:
        print("🎯 Running YOLO11s-Seg segmentation...")
        masks, seg_boxes, seg_scores, seg_labels = run_yolo_segmentation("yolo11s_seg", image_path)
        if len(masks) > 0:
            seg_masks = masks
    
    # Step 4: SAM Box-Prompt Refinement
    print()
    print("🔧 STEP 4: SAM Box-Prompt Refinement")
    
    sam_masks = []
    if len(fused_boxes) > 0:
        try:
            sam_model_path = config.get_model_path("sam", config.CURRENT_SAM_MODEL)
            # Determine SAM model type from filename
            model_type = "vit_l" if "vit_l" in sam_model_path else "vit_b"
            detector = ObjectDetectionSegmentation(checkpoint_path=sam_model_path, model_type=model_type)
            
            # Load the image
            image = cv2.imread(image_path)
            
            # Convert boxes to the format expected by SAM (x1, y1, x2, y2)
            boxes_for_sam = []
            for box in fused_boxes:
                boxes_for_sam.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
            
            refined_masks = detector.segment_with_boxes(image, boxes_for_sam)
            if refined_masks:
                sam_masks = refined_masks
                print(f"   ✅ SAM refined {len(sam_masks)} masks from WBF boxes")
            else:
                print("   ⚠️ SAM refinement produced no masks")
                
        except Exception as e:
            print(f"   ❌ SAM refinement failed: {str(e)}")
    
    # Combine segmentation masks (prefer YOLO-Seg, fallback to SAM)
    final_masks = seg_masks if len(seg_masks) > 0 else sam_masks
    
    # Step 5: Post-Processing Mask Merge
    print()
    print("🧹 STEP 5: Post-Processing Mask Merge")
    print()
    
    if len(final_masks) > 1:
        print(f"🔄 Merging similar masks (IoU > {config.MASK_MERGE_IOU_THRESHOLD})...")
        final_masks = merge_masks_by_iou(final_masks, config.MASK_MERGE_IOU_THRESHOLD)
    else:
        print("   📊 No mask merging needed (≤1 mask)")
    
    # Create results summary
    results = {
        "pipeline": "Configurable Smart Ensemble: WBF + YOLOv11-Seg + SAM",
        "image": config.CURRENT_IMAGE,
        "sam_model": config.CURRENT_SAM_MODEL,
        "yolo_models": config.CURRENT_YOLO_MODELS,
        "model_weights": weights,
        "initial_detections": total_detections,
        "wbf_fused": len(fused_boxes),
        "yolo_seg_masks": len(seg_masks),
        "sam_refined": len(sam_masks),
        "final_objects": len(fused_boxes),
        "final_masks": len(final_masks),
        "fused_labels": fused_labels.tolist() if len(fused_labels) > 0 else [],
        "confidence_scores": fused_scores.tolist() if len(fused_scores) > 0 else [],
        "config": {
            "yolo_conf_threshold": config.YOLO_CONFIDENCE_THRESHOLD,
            "wbf_iou_threshold": config.WBF_IOU_THRESHOLD,
            "mask_merge_iou_threshold": config.MASK_MERGE_IOU_THRESHOLD
        }
    }
    
    # Print final results
    print()
    print("📋 FINAL SMART PIPELINE RESULTS:")
    print(f"   📸 Image: {config.CURRENT_IMAGE}")
    print(f"   🔍 Initial detections: {total_detections}")
    print(f"   🔗 WBF fused: {len(fused_boxes)}")
    print(f"   🎯 YOLO-Seg masks: {len(seg_masks)}")
    print(f"   🔧 SAM refined: {len(sam_masks)}")
    print(f"   🧹 Final merged: {len(final_masks)}")
    
    # Create visualizations
    print()
    print("🎨 Creating enhanced visualizations...")
    
    # WBF visualization
    wbf_vis_path = config.get_output_path("wbf_fused.jpg")
    if len(fused_boxes) > 0:
        create_visualization(image, fused_boxes, fused_scores, fused_labels, [], wbf_vis_path)
    
    # Final result with masks
    final_vis_path = config.get_output_path("smart_final.jpg")
    if len(fused_boxes) > 0:
        create_visualization(image, fused_boxes, fused_scores, fused_labels, final_masks, final_vis_path)
    
    # Save results JSON
    results_path = config.get_output_path("smart_pipeline_results.json").replace("visualizations", "metrics")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("📊 Smart Pipeline Output Files:")
    print(f"   📸 {wbf_vis_path} - Weighted Boxes Fusion results")
    print(f"   📸 {final_vis_path} - Final smart ensemble result") 
    print(f"   📄 {results_path} - Comprehensive results")
    
    print()
    print("✨ Configurable Smart Ensemble Pipeline completed successfully!")
    print(f"   🎯 Reduced {total_detections} detections → {len(fused_boxes)} high-quality objects")
    print(f"   🧠 Used: {' + '.join(config.CURRENT_YOLO_MODELS)} + {config.CURRENT_SAM_MODEL.upper()} + WBF + Mask Merging")

if __name__ == "__main__":
    main()