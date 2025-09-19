"""
YOLOv11s Detection followed by SAM Segmentation Refinement
This script detects objects using YOLOv11s, then refines segmentation with SAM using YOLOv11's bounding boxes as prompts.
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# --- Robustly add project root to sys.path ---
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import YOLOv11 and SAM modules ---
try:
    from ultralytics import YOLO  # YOLOv11
except ImportError:
    raise ImportError("ultralytics package (YOLOv11) not found. Install with 'pip install ultralytics'.")

from pipeline.object_detection_segmentation import ObjectDetectionSegmentation

# --- Paths ---
image_path = Path(project_root) / "src" / "tools.png"
sam_checkpoint = Path(project_root) / "sam_vit_b_01ec64.pth"
yolo_model_path = "yolo11s.pt"  # Using YOLOv11s model
output_dir = Path(project_root) / "test_outputs"
output_dir.mkdir(exist_ok=True)

# --- Load image ---
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

# --- YOLOv11 Detection ---
print("\n🔍 Running YOLOv11s detection...")
yolo = YOLO(yolo_model_path)
yolo_results = yolo(str(image_path))

boxes = []
for result in yolo_results:
    for box in result.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box[:4]
        w, h = x2 - x1, y2 - y1
        boxes.append([int(x1), int(y1), int(w), int(h)])

print(f"   ✅ YOLOv11s detected {len(boxes)} objects")

# --- Initialize SAM ---
print("\n🤖 Initializing SAM...")
sam = ObjectDetectionSegmentation(
    checkpoint_path=str(sam_checkpoint),
    model_type="vit_b",
    device="auto"
)

# --- SAM Segmentation Refinement ---
print("\n🎯 Refining segmentation with SAM using YOLOv11 boxes as prompts...")

# Use YOLOv11 bounding boxes as prompts for SAM
refined_masks = sam.segment_with_boxes(image, boxes)

print(f"   ✅ SAM refined {len(refined_masks)} masks using YOLOv11 box prompts")

# Use YOLOv11 boxes and SAM refined masks for visualization
masks = refined_masks
# Keep original YOLO boxes

# --- Visualization ---
print("\n🎨 Creating visualizations...")

# Create YOLOv11-only visualization for comparison
yolo_only_vis = image.copy()
colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255]]
for i, bbox in enumerate(boxes):
    color = colors[i % len(colors)]
    x, y, w, h = bbox
    cv2.rectangle(yolo_only_vis, (int(x), int(y)), (int(x+w), int(y+h)), color, 3)
    cv2.putText(yolo_only_vis, f"YOLOv11_{i+1}", (int(x), int(y-10)), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imwrite(str(output_dir / "1_yolov11_only.jpg"), yolo_only_vis)

# Create YOLOv11 + SAM refined visualization
vis_image = sam.visualize_detections(
    image, masks, boxes, str(output_dir / "2_yolov11_sam_refined.jpg")
)

print(f"   🖼️  YOLOv11-only output: {output_dir / '1_yolov11_only.jpg'}")
print(f"   🖼️  YOLOv11+SAM refined: {output_dir / '2_yolov11_sam_refined.jpg'}")
print(f"\n✨ Pipeline completed successfully!")
print(f"   🔍 YOLOv11s detected {len(boxes)} objects")
print(f"   🎯 SAM refined {len(masks)} masks")
print(f"   📊 Check the output directory for results")
