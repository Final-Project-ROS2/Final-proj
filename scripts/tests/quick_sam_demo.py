"""
Quick SAM Demo for eng_tool.jpg
Uses grid-based prompting to demonstrate SAM capabilities faster than automatic segmentation.
"""

import cv2
import numpy as np
import json
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import config
from src.pipeline.object_detection_segmentation import ObjectDetectionSegmentation

def create_grid_boxes(image_shape, grid_size=4):
    """Create a grid of boxes for SAM prompting"""
    h, w = image_shape[:2]
    
    box_h = h // grid_size
    box_w = w // grid_size
    
    boxes = []
    for i in range(grid_size):
        for j in range(grid_size):
            x1 = j * box_w
            y1 = i * box_h
            x2 = min((j + 1) * box_w, w)
            y2 = min((i + 1) * box_h, h)
            
            # Add some overlap and margin
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            boxes.append([x1, y1, x2, y2])
    
    return boxes

def visualize_sam_grid_results(image, masks, boxes, output_path):
    """Visualize SAM results from grid prompting"""
    vis_image = image.copy()
    
    if len(masks) == 0:
        cv2.putText(vis_image, "No masks generated", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(output_path, vis_image)
        return vis_image
    
    # Create mask overlay
    mask_overlay = np.zeros_like(vis_image, dtype=np.uint8)
    
    print(f"   🎨 Visualizing {len(masks)} SAM masks from grid prompting")
    
    for i, mask in enumerate(masks):
        if mask is not None and mask.size > 0:
            # Resize mask if needed
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))
            
            # Generate color for this mask
            np.random.seed(i)
            color = np.random.randint(50, 255, 3).tolist()
            
            # Create colored mask
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask > 0] = color
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 0.6, 0)
    
    # Draw grid boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (100, 100, 100), 1)
        cv2.putText(vis_image, str(i+1), (x1+5, y1+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Blend masks with image
    final_image = cv2.addWeighted(vis_image, 0.6, mask_overlay, 0.4, 0)
    
    # Add title
    cv2.putText(final_image, f"SAM Grid Results: {len([m for m in masks if m is not None])} masks", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, final_image)
    return final_image

def run_quick_sam_demo():
    """Run quick SAM demonstration using grid prompting"""
    print("⚡ Quick SAM Demo for eng_tool.jpg")
    print("=" * 45)
    
    # Set configuration
    config.set_image('eng_tool')
    config.set_sam_model('sam_vit_b')  # Use faster model
    
    # Load image
    image_path = config.get_current_image_path()
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"📸 Image: {config.CURRENT_IMAGE} ({image.shape[1]}x{image.shape[0]})")
    print(f"🤖 SAM Model: {config.CURRENT_SAM_MODEL}")
    print()
    
    # Initialize SAM
    print("🚀 Initializing SAM...")
    try:
        sam_model_path = config.get_model_path("sam", config.CURRENT_SAM_MODEL)
        model_type = "vit_b" if "vit_b" in sam_model_path else "vit_l"
        
        detector = ObjectDetectionSegmentation(checkpoint_path=sam_model_path, model_type=model_type)
        print(f"   ✅ SAM-{model_type.upper()} ready")
        
    except Exception as e:
        print(f"   ❌ SAM initialization failed: {str(e)}")
        return
    
    # Create grid boxes for prompting
    print()
    print("📐 Creating Grid Prompts...")
    grid_boxes = create_grid_boxes(image.shape, grid_size=3)  # 3x3 grid = 9 regions
    print(f"   📊 Generated {len(grid_boxes)} grid prompts")
    
    # Run SAM with grid prompts
    print()
    print("🎯 Running SAM with Grid Prompts...")
    print("   ⏳ Processing grid regions...")
    
    try:
        sam_masks = detector.segment_with_boxes(image, grid_boxes)
        valid_masks = [mask for mask in sam_masks if mask is not None and mask.size > 0]
        
        print(f"   ✅ SAM generated {len(valid_masks)} masks from {len(grid_boxes)} prompts")
        
    except Exception as e:
        print(f"   ❌ SAM grid prompting failed: {str(e)}")
        return
    
    # Analyze results
    print()
    print("📊 Analyzing SAM Results...")
    
    mask_areas = []
    mask_coverage = []
    
    for mask in valid_masks:
        if mask is not None and mask.size > 0:
            area = np.sum(mask > 0)
            coverage = area / (image.shape[0] * image.shape[1]) * 100
            mask_areas.append(area)
            mask_coverage.append(coverage)
    
    if mask_areas:
        print(f"   🔢 Valid masks: {len(mask_areas)}")
        print(f"   📏 Area range: {min(mask_areas)} - {max(mask_areas)} pixels")
        print(f"   📊 Coverage range: {min(mask_coverage):.1f}% - {max(mask_coverage):.1f}%")
        print(f"   🎯 Average coverage: {np.mean(mask_coverage):.1f}% per mask")
    
    # Create visualization
    print()
    print("🎨 Creating Visualization...")
    
    output_path = config.get_output_path("sam_grid_demo.jpg")
    visualize_sam_grid_results(image, sam_masks, grid_boxes, output_path)
    
    # Save results
    results = {
        "demo": "Quick SAM Grid Prompting",
        "image": config.CURRENT_IMAGE,
        "sam_model": config.CURRENT_SAM_MODEL,
        "grid_prompts": len(grid_boxes),
        "valid_masks": len(valid_masks),
        "mask_statistics": {
            "areas": [int(area) for area in mask_areas],  # Convert to regular int
            "coverage_percentages": [float(cov) for cov in mask_coverage],  # Convert to regular float
            "mean_area": float(np.mean(mask_areas)) if mask_areas else 0,
            "mean_coverage": float(np.mean(mask_coverage)) if mask_coverage else 0
        },
        "grid_boxes": [[float(coord) for coord in box] for box in grid_boxes]  # Convert to regular float
    }
    
    results_path = config.get_output_path("sam_grid_demo_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   📸 Visualization: {output_path}")
    print(f"   📄 Results: {results_path}")
    
    # Summary
    print()
    print("📋 Quick SAM Demo Summary:")
    print(f"   🎯 Grid prompts: {len(grid_boxes)}")
    print(f"   ✅ Valid masks: {len(valid_masks)}")
    print(f"   📊 Success rate: {len(valid_masks)/len(grid_boxes)*100:.1f}%")
    if mask_coverage:
        print(f"   🔍 Average coverage: {np.mean(mask_coverage):.1f}% per mask")
    
    print()
    print("✨ Quick SAM Demo completed!")
    print("   🔍 This shows SAM's box-prompted segmentation capabilities")
    print("   ⚡ Much faster than automatic mask generation")
    print("   📊 Compare with YOLO+SAM ensemble for targeted detection")

if __name__ == "__main__":
    run_quick_sam_demo()