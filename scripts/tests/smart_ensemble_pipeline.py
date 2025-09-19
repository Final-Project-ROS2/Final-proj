"""
Smart Ensemble Pipeline: WBF + YOLOv11-Seg + SAM Refinement
This advanced pipeline implements:
1. Weighted Boxes Fusion (WBF) for smarter ensemble
2. YOLOv11-Seg for direct segmentation
3. SAM as box-prompt refinement only
4. Post-processing mask merge for clean results
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import json

# --- Robustly add project root to sys.path ---
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import modules ---
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics package not found. Install with 'pip install ultralytics'.")

from pipeline.object_detection_segmentation import ObjectDetectionSegmentation

def weighted_boxes_fusion(boxes_list, scores_list, labels_list, model_weights=None, iou_thr=0.5, skip_box_thr=0.01):
    """
    Weighted Boxes Fusion implementation
    Args:
        boxes_list: list of boxes for each model [[x1,y1,x2,y2], ...]
        scores_list: list of scores for each model
        labels_list: list of labels for each model  
        model_weights: weights for each model
        iou_thr: IoU threshold for fusion
        skip_box_thr: skip boxes with confidence below this
    """
    if model_weights is None:
        model_weights = [1.0] * len(boxes_list)
    
    all_boxes = []
    all_scores = []
    all_labels = []
    all_weights = []
    
    # Collect all boxes with model weights
    for model_idx, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        for box, score, label in zip(boxes, scores, labels):
            if score > skip_box_thr:
                # Convert [x,y,w,h] to [x1,y1,x2,y2] if needed
                if len(box) == 4 and box[2] < 1:  # If width/height are normalized
                    x, y, w, h = box
                    all_boxes.append([x, y, x+w, y+h])
                else:
                    x, y, w, h = box
                    all_boxes.append([x, y, x+w, y+h])
                all_scores.append(score * model_weights[model_idx])
                all_labels.append(label)
                all_weights.append(model_weights[model_idx])
    
    if not all_boxes:
        return [], [], []
    
    # Simple WBF implementation
    fused_boxes = []
    fused_scores = []
    fused_labels = []
    used = [False] * len(all_boxes)
    
    for i, (box1, score1, label1) in enumerate(zip(all_boxes, all_scores, all_labels)):
        if used[i]:
            continue
            
        cluster_boxes = [box1]
        cluster_scores = [score1]
        cluster_labels = [label1]
        used[i] = True
        
        # Find overlapping boxes
        for j, (box2, score2, label2) in enumerate(zip(all_boxes, all_scores, all_labels)):
            if used[j] or i == j:
                continue
                
            iou = calculate_iou_xyxy(box1, box2)
            if iou > iou_thr:
                cluster_boxes.append(box2)
                cluster_scores.append(score2)
                cluster_labels.append(label2)
                used[j] = True
        
        # Weighted average fusion
        weights = np.array(cluster_scores)
        weights = weights / weights.sum()
        
        avg_box = np.average(cluster_boxes, axis=0, weights=weights)
        avg_score = np.average(cluster_scores, weights=weights)
        
        # Most common label
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        fused_label = unique_labels[np.argmax(counts)]
        
        # Convert back to [x,y,w,h]
        x1, y1, x2, y2 = avg_box
        fused_boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
        fused_scores.append(avg_score)
        fused_labels.append(fused_label)
    
    return fused_boxes, fused_scores, fused_labels

def calculate_iou_xyxy(box1, box2):
    """Calculate IoU for boxes in [x1,y1,x2,y2] format"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def calculate_iou(box1, box2):
    """Calculate IoU for boxes in [x,y,w,h] format"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to [x1,y1,x2,y2]
    return calculate_iou_xyxy([x1, y1, x1+w1, y1+h1], [x2, y2, x2+w2, y2+h2])

def run_yolo_detection_with_scores(model_path, image_path, model_name):
    """Run YOLO detection and return boxes, scores, and labels"""
    print(f"\n🔍 Running {model_name} detection...")
    yolo = YOLO(model_path)
    results = yolo(str(image_path))
    
    boxes = []
    scores = []
    labels = []
    
    for result in results:
        if hasattr(result.boxes, 'conf'):  # Check if confidence scores are available
            for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), 
                                    result.boxes.conf.cpu().numpy(),
                                    result.boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = box[:4]
                w, h = x2 - x1, y2 - y1
                boxes.append([int(x1), int(y1), int(w), int(h)])
                scores.append(float(conf))
                labels.append(result.names[int(cls)])
    
    print(f"   ✅ {model_name} detected {len(boxes)} objects with avg confidence: {np.mean(scores):.2f}")
    return boxes, scores, labels

def run_yolo_segmentation(model_path, image_path, model_name):
    """Run YOLOv11-Seg for direct segmentation"""
    print(f"\n🎯 Running {model_name} segmentation...")
    yolo = YOLO(model_path)
    results = yolo(str(image_path))
    
    boxes = []
    masks = []
    scores = []
    labels = []
    
    for result in results:
        if hasattr(result, 'masks') and result.masks is not None:
            for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),
                                                   result.boxes.conf.cpu().numpy(),
                                                   result.boxes.cls.cpu().numpy())):
                x1, y1, x2, y2 = box[:4]
                w, h = x2 - x1, y2 - y1
                boxes.append([int(x1), int(y1), int(w), int(h)])
                scores.append(float(conf))
                labels.append(result.names[int(cls)])
                
                # Extract mask
                mask = result.masks.data[i].cpu().numpy()
                masks.append(mask)
    
    print(f"   ✅ {model_name} segmented {len(masks)} objects")
    return boxes, masks, scores, labels

def merge_similar_masks(masks, boxes, labels, iou_threshold=0.7):
    """Merge similar masks to reduce fragmentation"""
    if not masks:
        return masks, boxes, labels
    
    merged_masks = []
    merged_boxes = []
    merged_labels = []
    used = [False] * len(masks)
    
    print(f"\n🔄 Merging similar masks (IoU > {iou_threshold})...")
    
    for i, (mask1, box1, label1) in enumerate(zip(masks, boxes, labels)):
        if used[i]:
            continue
            
        cluster_masks = [mask1]
        cluster_boxes = [box1]
        cluster_labels = [label1]
        used[i] = True
        
        # Find similar masks
        for j, (mask2, box2, label2) in enumerate(zip(masks, boxes, labels)):
            if used[j] or i == j:
                continue
                
            iou = calculate_iou(box1, box2)
            if iou > iou_threshold and label1 == label2:
                cluster_masks.append(mask2)
                cluster_boxes.append(box2)
                cluster_labels.append(label2)
                used[j] = True
        
        # Merge masks and boxes
        if len(cluster_masks) > 1:
            # Union of masks
            merged_mask = np.logical_or.reduce(cluster_masks)
            
            # Average box
            avg_box = np.mean(cluster_boxes, axis=0).astype(int)
            
            merged_masks.append(merged_mask.astype(np.uint8))
            merged_boxes.append(avg_box.tolist())
            merged_labels.append(f"{label1}_merged")
            
            print(f"   🔗 Merged {len(cluster_masks)} {label1} masks")
        else:
            merged_masks.append(mask1)
            merged_boxes.append(box1)
            merged_labels.append(label1)
    
    print(f"   📊 Result: {len(merged_masks)} masks (from {len(masks)} original)")
    return merged_masks, merged_boxes, merged_labels

# --- Main Smart Pipeline ---
def main():
    # Paths
    image_path = Path(project_root) / "src" / "tools.png"
    sam_checkpoint = Path(project_root) / "sam_vit_l_0b3195.pth"
    output_dir = Path(project_root) / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    print("🚀 Smart Ensemble Pipeline: WBF + YOLOv11-Seg + SAM")
    print("=" * 60)
    
    # Step 1: Multi-YOLO Detection with Confidence Scores
    print("\n📊 STEP 1: Multi-Model Detection")
    yolo8_boxes, yolo8_scores, yolo8_labels = run_yolo_detection_with_scores("yolov8n.pt", image_path, "YOLOv8n")
    yolo11s_boxes, yolo11s_scores, yolo11s_labels = run_yolo_detection_with_scores("yolo11s.pt", image_path, "YOLO11s") 
    yolo11m_boxes, yolo11m_scores, yolo11m_labels = run_yolo_detection_with_scores("yolo11m.pt", image_path, "YOLO11m")
    
    # Step 2: Weighted Boxes Fusion
    print("\n🔗 STEP 2: Weighted Boxes Fusion")
    model_weights = [1.0, 1.2, 1.5]  # Give more weight to larger/newer models
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        [yolo8_boxes, yolo11s_boxes, yolo11m_boxes],
        [yolo8_scores, yolo11s_scores, yolo11m_scores], 
        [yolo8_labels, yolo11s_labels, yolo11m_labels],
        model_weights=model_weights,
        iou_thr=0.5
    )
    
    total_detections = len(yolo8_boxes) + len(yolo11s_boxes) + len(yolo11m_boxes)
    print(f"   📊 WBF Result: {len(fused_boxes)} fused boxes (from {total_detections} total detections)")
    
    # Step 3: YOLOv11-Seg Direct Segmentation
    print("\n🎯 STEP 3: YOLOv11-Seg Direct Segmentation")
    try:
        seg_boxes, seg_masks, seg_scores, seg_labels = run_yolo_segmentation("yolo11s-seg.pt", image_path, "YOLO11s-Seg")
    except Exception as e:
        print(f"   ⚠️  YOLOv11-Seg not available: {e}")
        print(f"   📋 Using detection-only mode")
        seg_boxes, seg_masks, seg_scores, seg_labels = [], [], [], []
    
    # Step 4: SAM Refinement for Fused Boxes
    print("\n🔧 STEP 4: SAM Box-Prompt Refinement")
    try:
        sam = ObjectDetectionSegmentation(
            checkpoint_path=str(sam_checkpoint),
            model_type="vit_l",
            device="auto"
        )
        
        sam_refined_masks = sam.segment_with_boxes(image, fused_boxes)
        print(f"   ✅ SAM refined {len(sam_refined_masks)} masks from WBF boxes")
        
        # Step 5: Post-Processing Mask Merge
        print("\n🧹 STEP 5: Post-Processing Mask Merge")
        final_masks, final_boxes, final_labels = merge_similar_masks(
            sam_refined_masks, fused_boxes, fused_labels, iou_threshold=0.7
        )
        
    except Exception as e:
        print(f"   ⚠️  SAM refinement failed: {e}")
        print(f"   📋 Using WBF results only")
        final_masks = []
        final_boxes = fused_boxes
        final_labels = fused_labels
    
    # Results Summary
    print(f"\n📋 FINAL SMART PIPELINE RESULTS:")
    print(f"   🔍 Initial detections: {total_detections}")
    print(f"   🔗 WBF fused: {len(fused_boxes)}")
    print(f"   🎯 YOLO-Seg masks: {len(seg_masks)}")
    print(f"   🔧 SAM refined: {len(sam_refined_masks) if 'sam_refined_masks' in locals() else 0}")
    print(f"   🧹 Final merged: {len(final_masks) if final_masks else len(final_boxes)}")
    
    # Create Enhanced Visualizations
    print(f"\n🎨 Creating enhanced visualizations...")
    
    # Individual model results
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
    
    # WBF results
    wbf_vis = image.copy()
    for i, (box, score, label) in enumerate(zip(fused_boxes, fused_scores, fused_labels)):
        color = colors[i % len(colors)]
        x, y, w, h = box
        cv2.rectangle(wbf_vis, (x, y), (x+w, y+h), color, 3)
        cv2.putText(wbf_vis, f"WBF:{label}({score:.2f})", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(output_dir / "1_wbf_fused.jpg"), wbf_vis)
    
    # Final results with masks
    if final_masks:
        final_vis = sam.visualize_detections(image, final_masks, final_boxes, 
                                           str(output_dir / "2_smart_final.jpg"))
    else:
        final_vis = wbf_vis.copy()
        cv2.imwrite(str(output_dir / "2_smart_final.jpg"), final_vis)
    
    # Save comprehensive results
    smart_results = {
        "pipeline": "Smart Ensemble: WBF + YOLOv11-Seg + SAM",
        "initial_detections": total_detections,
        "wbf_fused": len(fused_boxes),
        "yolo_seg_masks": len(seg_masks),
        "sam_refined": len(sam_refined_masks) if 'sam_refined_masks' in locals() else 0,
        "final_objects": len(final_masks) if final_masks else len(final_boxes),
        "model_weights": model_weights,
        "fused_labels": fused_labels,
        "final_labels": final_labels,
        "confidence_scores": fused_scores
    }
    
    with open(output_dir / "smart_pipeline_results.json", "w") as f:
        json.dump(smart_results, f, indent=2)
    
    print(f"\n📊 Smart Pipeline Output Files:")
    print(f"   📸 1_wbf_fused.jpg - Weighted Boxes Fusion results")
    print(f"   📸 2_smart_final.jpg - Final smart ensemble result")
    print(f"   📄 smart_pipeline_results.json - Comprehensive results")
    
    print(f"\n✨ Smart Ensemble Pipeline completed successfully!")
    print(f"   🎯 Reduced {total_detections} detections → {len(final_boxes)} high-quality objects")
    print(f"   🧠 Used: WBF + {'YOLO-Seg + ' if seg_masks else ''}SAM + Mask Merging")

if __name__ == "__main__":
    main()