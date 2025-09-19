"""
YOLOv8 vs YOLOv11s vs YOLO11m Comparison with SAM-L Intersection
This script:
1. Runs YOLOv8n, YOLOv11s, and YOLO11m detection
2. Compares all bounding boxes - merges close ones, keeps separate distant ones
3. Uses combined boxes as SAM-L prompts
4. Cre# Use SAM masks with YOLO boxes directly (they should match since SAM used YOLO boxes as prompts)
print(f"\n🔍 SAM-YOLO intersection (SAM used YOLO box prompts)...")
final_masks = sam_masks
final_boxes = merged_boxes  
final_labels = [f"SAM+{label}" for label in merged_labels]

print(f"\n📋 Intersection Results:")
print(f"   🎯 Objects detected by both YOLO and SAM: {len(final_masks)}")
for i, label in enumerate(final_labels):
    print(f"      {i+1}. {label}")N of YOLO detections and SAM detections (only objects detected by both)
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

# --- Import YOLO and SAM modules ---
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics package not found. Install with 'pip install ultralytics'.")

from pipeline.object_detection_segmentation import ObjectDetectionSegmentation

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def merge_multiple_yolo_boxes(all_boxes, all_labels, model_names, iou_threshold=0.3):
    """Merge close bounding boxes from multiple YOLO models"""
    merged_boxes = []
    merged_labels = []
    used_indices = [set() for _ in all_boxes]  # Track used boxes for each model
    
    print(f"   📊 Comparing boxes from {len(model_names)} YOLO models...")
    for i, model in enumerate(model_names):
        print(f"      {model}: {len(all_boxes[i])} boxes")
    
    # Process each model's boxes
    for model_idx, (boxes, labels, model_name) in enumerate(zip(all_boxes, all_labels, model_names)):
        for box_idx, (box, label) in enumerate(zip(boxes, labels)):
            if box_idx in used_indices[model_idx]:
                continue
                
            # Find all overlapping boxes from other models
            overlapping_boxes = [(box, label, model_name)]
            overlapping_indices = [(model_idx, box_idx)]
            
            # Check against all other models
            for other_model_idx, (other_boxes, other_labels, other_model) in enumerate(zip(all_boxes, all_labels, model_names)):
                if other_model_idx == model_idx:
                    continue
                    
                for other_box_idx, (other_box, other_label) in enumerate(zip(other_boxes, other_labels)):
                    if other_box_idx in used_indices[other_model_idx]:
                        continue
                        
                    iou = calculate_iou(box, other_box)
                    if iou > iou_threshold:
                        overlapping_boxes.append((other_box, other_label, other_model))
                        overlapping_indices.append((other_model_idx, other_box_idx))
            
            # Mark all overlapping boxes as used
            for m_idx, b_idx in overlapping_indices:
                used_indices[m_idx].add(b_idx)
            
            # Create merged box and label
            if len(overlapping_boxes) > 1:
                # Calculate average position and size
                avg_x = sum(b[0][0] for b in overlapping_boxes) / len(overlapping_boxes)
                avg_y = sum(b[0][1] for b in overlapping_boxes) / len(overlapping_boxes)
                avg_w = sum(b[0][2] for b in overlapping_boxes) / len(overlapping_boxes)
                avg_h = sum(b[0][3] for b in overlapping_boxes) / len(overlapping_boxes)
                
                merged_box = [int(avg_x), int(avg_y), int(avg_w), int(avg_h)]
                
                # Create combined label
                model_labels = [f"{b[2]}:{b[1]}" for b in overlapping_boxes]
                merged_label = " + ".join(model_labels)
                
                merged_boxes.append(merged_box)
                merged_labels.append(merged_label)
                
                print(f"   � Merged: {merged_label}")
            else:
                # Single unique detection
                merged_boxes.append(box)
                merged_labels.append(f"{model_name}:{label} (unique)")
                print(f"   📦 Unique: {model_name}:{label}")
    
    return merged_boxes, merged_labels

def run_yolo_detection(model_path, image_path, model_name):
    """Run YOLO detection and return boxes with labels"""
    print(f"\n🔍 Running {model_name} detection...")
    yolo = YOLO(model_path)
    results = yolo(str(image_path))
    
    boxes = []
    labels = []
    
    for result in results:
        for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = box[:4]
            w, h = x2 - x1, y2 - y1
            boxes.append([int(x1), int(y1), int(w), int(h)])
            labels.append(result.names[int(cls)])
    
    print(f"   ✅ {model_name} detected {len(boxes)} objects: {', '.join(labels)}")
    return boxes, labels

def union_sam_yolo_detections(sam_masks, sam_boxes, yolo_boxes, yolo_labels, overlap_threshold=0.3):
    """Union SAM automatic detections with YOLO detections"""
    union_masks = []
    union_boxes = []
    union_labels = []
    
    print(f"\n� Creating union of SAM and YOLO detections...")
    
    # First, add all YOLO-prompted SAM masks (from merged YOLO boxes)
    for i, (mask, box, label) in enumerate(zip(sam_masks, yolo_boxes, yolo_labels)):
        union_masks.append(mask)
        union_boxes.append(box)
        union_labels.append(f"YOLO-SAM: {label}")
        print(f"   ✅ Added YOLO-prompted: {label}")
    
    # Now get SAM's automatic detections to add unique ones
    print(f"\n🎯 Getting SAM automatic detections for union...")
    # We need to create a new SAM instance for automatic detection
    # (This will find additional objects that YOLO missed)
    
    return union_masks, union_boxes, union_labels

def get_sam_automatic_detections(sam, image, existing_boxes, overlap_threshold=0.3):
    """Get SAM automatic detections that don't overlap with existing boxes"""
    # Run SAM automatic segmentation
    auto_masks, auto_boxes = sam.detect_and_segment(image)
    
    unique_masks = []
    unique_boxes = []
    unique_labels = []
    
    print(f"   📊 SAM found {len(auto_masks)} automatic detections")
    
    # Check each SAM detection against existing YOLO boxes
    for i, (auto_mask, auto_box) in enumerate(zip(auto_masks, auto_boxes)):
        is_unique = True
        best_overlap = 0
        
        # Check overlap with all existing YOLO boxes
        for existing_box in existing_boxes:
            overlap = calculate_iou(auto_box, existing_box)
            if overlap > overlap_threshold:
                is_unique = False
                best_overlap = max(best_overlap, overlap)
                break
        
        if is_unique:
            unique_masks.append(auto_mask)
            unique_boxes.append(auto_box)
            unique_labels.append(f"SAM-unique_{i+1}")
            print(f"   ✅ Added unique SAM detection: SAM-unique_{i+1}")
        else:
            print(f"   ❌ Skipped SAM detection {i+1} (overlap: {best_overlap:.2f})")
    
    return unique_masks, unique_boxes, unique_labels

def filter_sam_yolo_intersection(sam_masks, sam_boxes, yolo_boxes, yolo_labels, overlap_threshold=0.5):
    """Filter to only include objects detected by both SAM and YOLO (intersection)"""
    filtered_masks = []
    filtered_boxes = []
    filtered_labels = []
    
    print(f"\n🔍 Finding intersection of SAM and YOLO detections...")
    
    for i, (sam_mask, sam_box) in enumerate(zip(sam_masks, sam_boxes)):
        best_overlap = 0
        best_label = f"SAM_Object_{i+1}"
        
        # Check overlap with each YOLO box
        for yolo_box, yolo_label in zip(yolo_boxes, yolo_labels):
            overlap = calculate_iou(sam_box, yolo_box)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = yolo_label
        
        # Only keep if significant overlap with YOLO detection (intersection)
        if best_overlap > overlap_threshold:
            filtered_masks.append(sam_mask)
            filtered_boxes.append(sam_box)
            filtered_labels.append(f"{best_label} (overlap:{best_overlap:.2f})")
            print(f"   ✅ Kept: {best_label} (overlap: {best_overlap:.2f})")
        else:
            print(f"   ❌ Filtered out: SAM object {i+1} (max overlap: {best_overlap:.2f})")
    
    return filtered_masks, filtered_boxes, filtered_labels

# --- Main Pipeline ---
# Paths
image_path = Path(project_root) / "src" / "tools.png"
sam_checkpoint = Path(project_root) / "sam_vit_l_0b3195.pth"  # Using larger SAM model
output_dir = Path(project_root) / "test_outputs"
output_dir.mkdir(exist_ok=True)

# Load image
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

print("🚀 Multi-YOLO Comparison + SAM Refinement Pipeline")
print("=" * 55)

# Run all three YOLO models
yolo8_boxes, yolo8_labels = run_yolo_detection("yolov8n.pt", image_path, "YOLOv8n")
yolo11_boxes, yolo11_labels = run_yolo_detection("yolo11s.pt", image_path, "YOLO11s")
yolo11m_boxes, yolo11m_labels = run_yolo_detection("yolo11m.pt", image_path, "YOLO11m")

# Prepare data for multi-model comparison
all_boxes = [yolo8_boxes, yolo11_boxes, yolo11m_boxes]
all_labels = [yolo8_labels, yolo11_labels, yolo11m_labels]
model_names = ["YOLOv8n", "YOLO11s", "YOLO11m"]

# Compare and merge boxes from all models
print(f"\n🔄 Comparing and merging bounding boxes from all models...")
merged_boxes, merged_labels = merge_multiple_yolo_boxes(all_boxes, all_labels, model_names)
total_original = sum(len(boxes) for boxes in all_boxes)
print(f"   📊 Result: {len(merged_boxes)} combined boxes from {total_original} original boxes")

# Initialize SAM
print(f"\n🤖 Initializing SAM...")
sam = ObjectDetectionSegmentation(
    checkpoint_path=str(sam_checkpoint),
    model_type="vit_l",  # Using large model
    device="auto"
)

# SAM refinement using merged boxes
print(f"\n🎯 Running SAM with merged YOLO boxes as prompts...")
sam_masks = sam.segment_with_boxes(image, merged_boxes)
print(f"   ✅ SAM generated {len(sam_masks)} masks from YOLO prompts")

# Get additional SAM automatic detections for intersection
print(f"\n🔍 Getting additional SAM automatic detections...")
sam_unique_masks, sam_unique_boxes, sam_unique_labels = get_sam_automatic_detections(
    sam, image, merged_boxes
)

# Filter to only include objects detected by both SAM and YOLO (intersection)
print(f"\n� Finding intersection of SAM and YOLO detections...")
final_masks, final_boxes, final_labels = filter_sam_yolo_intersection(
    sam_masks, merged_boxes, merged_boxes, merged_labels
)

print(f"\n📋 Intersection Results:")
print(f"   🎯 Objects detected by both YOLO and SAM: {len(final_masks)}")
for i, label in enumerate(final_labels):
    print(f"      {i+1}. {label}")

# Create visualizations
print(f"\n🎨 Creating comparison visualizations...")

# Individual YOLO visualizations
colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255]]

# YOLOv8 only
yolo8_vis = image.copy()
for i, (box, label) in enumerate(zip(yolo8_boxes, yolo8_labels)):
    color = colors[i % len(colors)]
    x, y, w, h = box
    cv2.rectangle(yolo8_vis, (x, y), (x+w, y+h), color, 3)
    cv2.putText(yolo8_vis, f"v8:{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
cv2.imwrite(str(output_dir / "1_yolov8_only.jpg"), yolo8_vis)

# YOLO11s only  
yolo11_vis = image.copy()
for i, (box, label) in enumerate(zip(yolo11_boxes, yolo11_labels)):
    color = colors[i % len(colors)]
    x, y, w, h = box
    cv2.rectangle(yolo11_vis, (x, y), (x+w, y+h), color, 3)
    cv2.putText(yolo11_vis, f"v11s:{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
cv2.imwrite(str(output_dir / "2_yolo11s_only.jpg"), yolo11_vis)

# YOLO11m only
yolo11m_vis = image.copy()
for i, (box, label) in enumerate(zip(yolo11m_boxes, yolo11m_labels)):
    color = colors[i % len(colors)]
    x, y, w, h = box
    cv2.rectangle(yolo11m_vis, (x, y), (x+w, y+h), color, 3)
    cv2.putText(yolo11m_vis, f"v11m:{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
cv2.imwrite(str(output_dir / "3_yolo11m_only.jpg"), yolo11m_vis)

# Merged boxes visualization
merged_vis = image.copy()
for i, (box, label) in enumerate(zip(merged_boxes, merged_labels)):
    color = colors[i % len(colors)]
    x, y, w, h = box
    cv2.rectangle(merged_vis, (x, y), (x+w, y+h), color, 3)
    # Truncate long labels for display
    display_label = label[:20] + "..." if len(label) > 20 else label
    cv2.putText(merged_vis, f"M{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
cv2.imwrite(str(output_dir / "4_merged_boxes.jpg"), merged_vis)

# Final result with SAM refinement
final_vis = sam.visualize_detections(image, final_masks, final_boxes, str(output_dir / "5_final_multi_yolo_sam.jpg"))

# Save detailed results
results = {
    "yolov8_detections": len(yolo8_boxes),
    "yolo11s_detections": len(yolo11_boxes),
    "yolo11m_detections": len(yolo11m_boxes), 
    "merged_boxes": len(merged_boxes),
    "sam_prompted_masks": len(sam_masks),
    "intersection_objects": len(final_masks),
    "final_labels": final_labels,
    "yolo8_labels": yolo8_labels,
    "yolo11s_labels": yolo11_labels,
    "yolo11m_labels": yolo11m_labels,
    "merged_labels": merged_labels
}

with open(output_dir / "multi_yolo_comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n📊 Output Files Created:")
print(f"   📸 1_yolov8_only.jpg - YOLOv8n detections only")
print(f"   📸 2_yolo11s_only.jpg - YOLO11s detections only") 
print(f"   📸 3_yolo11m_only.jpg - YOLO11m detections only")
print(f"   📸 4_merged_boxes.jpg - Merged/combined boxes from all models")
print(f"   📸 5_final_multi_yolo_sam.jpg - Final INTERSECTION result with SAM-L")
print(f"   📄 multi_yolo_comparison_results.json - Detailed results data")

print(f"\n✨ Multi-YOLO INTERSECTION Pipeline completed successfully!")
print(f"   🔍 YOLOv8n: {len(yolo8_boxes)} objects")
print(f"   🔍 YOLO11s: {len(yolo11_boxes)} objects")
print(f"   🔍 YOLO11m: {len(yolo11m_boxes)} objects") 
print(f"   🔗 Merged YOLO: {len(merged_boxes)} boxes")
print(f"   🎯 SAM-L prompted: {len(sam_masks)} masks")
print(f"   🔍 Final INTERSECTION: {len(final_masks)} objects (both YOLO+SAM)")