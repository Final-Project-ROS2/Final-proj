"""
SAM-Only Pipeline for eng_tool.jpg
This script uses only SAM (Segment Anything Model) for automatic segmentation
without any YOLO detection preprocessing.
"""

import cv2
import numpy as np
import json
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import config
from src.pipeline.object_detection_segmentation import ObjectDetectionSegmentation

def create_sam_visualization(image, masks, output_path, show_all_masks=True):
    """Create visualization showing SAM masks"""
    vis_image = image.copy()
    
    if len(masks) == 0:
        cv2.putText(vis_image, "No masks detected", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(output_path, vis_image)
        return vis_image
    
    # Create colored overlay for masks
    mask_overlay = np.zeros_like(vis_image, dtype=np.uint8)
    
    # Sort masks by area (largest first) for better visualization
    if show_all_masks:
        masks_to_show = masks
    else:
        # Show only top 20 largest masks to avoid clutter
        mask_areas = [np.sum(mask['segmentation']) for mask in masks]
        sorted_indices = np.argsort(mask_areas)[::-1]
        masks_to_show = [masks[i] for i in sorted_indices[:20]]
    
    print(f"   🎨 Visualizing {len(masks_to_show)} masks (of {len(masks)} total)")
    
    for i, mask_data in enumerate(masks_to_show):
        mask = mask_data['segmentation']
        
        # Generate a unique color for each mask
        np.random.seed(i)  # Consistent colors
        color = np.random.randint(0, 255, 3).tolist()
        
        # Create colored mask
        colored_mask = np.zeros_like(vis_image)
        colored_mask[mask] = color
        mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 0.7, 0)
        
        # Add bounding box if available
        if 'bbox' in mask_data:
            x, y, w, h = mask_data['bbox']
            cv2.rectangle(vis_image, (int(x), int(y)), 
                         (int(x + w), int(y + h)), color, 2)
            
            # Add mask index
            cv2.putText(vis_image, str(i+1), (int(x), int(y-5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Blend with original image
    final_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)
    
    # Add info text
    info_text = f"SAM Masks: {len(masks)} total, showing {len(masks_to_show)}"
    cv2.putText(final_image, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, final_image)
    return final_image

def analyze_sam_masks(masks):
    """Analyze SAM mask properties"""
    if not masks:
        return {}
    
    areas = [np.sum(mask['segmentation']) for mask in masks]
    stability_scores = [mask.get('stability_score', 0) for mask in masks]
    predicted_ious = [mask.get('predicted_iou', 0) for mask in masks]
    
    analysis = {
        "total_masks": len(masks),
        "area_stats": {
            "min": int(np.min(areas)),
            "max": int(np.max(areas)),
            "mean": float(np.mean(areas)),
            "median": float(np.median(areas))
        },
        "stability_score_stats": {
            "min": float(np.min(stability_scores)),
            "max": float(np.max(stability_scores)),
            "mean": float(np.mean(stability_scores))
        },
        "predicted_iou_stats": {
            "min": float(np.min(predicted_ious)),
            "max": float(np.max(predicted_ious)),
            "mean": float(np.mean(predicted_ious))
        }
    }
    
    return analysis

def run_sam_only_pipeline():
    """Run SAM-only segmentation pipeline"""
    print("🤖 SAM-Only Pipeline for eng_tool.jpg")
    print("=" * 50)
    
    # Set image to eng_tool and use faster SAM model
    config.set_image('eng_tool')
    config.set_sam_model('sam_vit_b')  # Use faster model for demo
    
    # Get image path and load image
    image_path = config.get_current_image_path()
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"📸 Processing image: {config.CURRENT_IMAGE} ({image.shape[1]}x{image.shape[0]})")
    print(f"🤖 SAM Model: {config.CURRENT_SAM_MODEL}")
    print()
    
    # Initialize SAM
    print("🚀 Initializing SAM...")
    try:
        sam_model_path = config.get_model_path("sam", config.CURRENT_SAM_MODEL)
        model_type = "vit_l" if "vit_l" in sam_model_path else "vit_b"
        
        print(f"   📁 Model path: {sam_model_path}")
        print(f"   🔧 Model type: {model_type}")
        
        detector = ObjectDetectionSegmentation(checkpoint_path=sam_model_path, model_type=model_type)
        print("   ✅ SAM initialized successfully")
        
    except Exception as e:
        print(f"   ❌ SAM initialization failed: {str(e)}")
        return
    
    # Run SAM automatic mask generation
    print()
    print("🎯 Running SAM Automatic Mask Generation...")
    print("   ⏳ This may take a moment...")
    
    try:
        masks, boxes = detector.detect_and_segment(image)
        print(f"   ✅ SAM generated {len(masks)} masks")
        
        if len(masks) == 0:
            print("   ⚠️ No masks were generated")
            return
        
    except Exception as e:
        print(f"   ❌ SAM segmentation failed: {str(e)}")
        return
    
    # Analyze masks
    print()
    print("📊 Analyzing SAM Results...")
    analysis = analyze_sam_masks(masks)
    
    print(f"   🔢 Total masks: {analysis['total_masks']}")
    print(f"   📏 Area range: {analysis['area_stats']['min']} - {analysis['area_stats']['max']} pixels")
    print(f"   📈 Mean area: {analysis['area_stats']['mean']:.0f} pixels")
    print(f"   🎯 Stability scores: {analysis['stability_score_stats']['min']:.3f} - {analysis['stability_score_stats']['max']:.3f}")
    print(f"   📊 Mean stability: {analysis['stability_score_stats']['mean']:.3f}")
    
    # Filter high-quality masks
    print()
    print("🔍 Filtering High-Quality Masks...")
    
    # Filter by stability score and predicted IoU
    min_stability = 0.85
    min_predicted_iou = 0.85
    min_area = 500  # Minimum area to avoid tiny fragments
    
    high_quality_masks = []
    for mask in masks:
        stability = mask.get('stability_score', 0)
        pred_iou = mask.get('predicted_iou', 0)
        area = np.sum(mask['segmentation'])
        
        if stability >= min_stability and pred_iou >= min_predicted_iou and area >= min_area:
            high_quality_masks.append(mask)
    
    print(f"   🎯 High-quality masks: {len(high_quality_masks)} (from {len(masks)} total)")
    print(f"   ⚙️ Filters: stability≥{min_stability}, IoU≥{min_predicted_iou}, area≥{min_area}")
    
    # Create visualizations
    print()
    print("🎨 Creating Visualizations...")
    
    # All masks visualization
    all_masks_path = config.get_output_path("sam_all_masks.jpg")
    create_sam_visualization(image, masks, all_masks_path, show_all_masks=False)
    print(f"   📸 All masks: {all_masks_path}")
    
    # High-quality masks visualization
    if high_quality_masks:
        hq_masks_path = config.get_output_path("sam_high_quality.jpg")
        create_sam_visualization(image, high_quality_masks, hq_masks_path, show_all_masks=True)
        print(f"   📸 High-quality masks: {hq_masks_path}")
    
    # Save detailed results
    results = {
        "pipeline": "SAM-Only Automatic Segmentation",
        "image": config.CURRENT_IMAGE,
        "sam_model": config.CURRENT_SAM_MODEL,
        "total_masks": len(masks),
        "high_quality_masks": len(high_quality_masks),
        "analysis": analysis,
        "filter_criteria": {
            "min_stability_score": min_stability,
            "min_predicted_iou": min_predicted_iou,
            "min_area_pixels": min_area
        },
        "config": {
            "sam_points_per_side": config.SAM_POINTS_PER_SIDE,
            "sam_pred_iou_thresh": config.SAM_PRED_IOU_THRESH,
            "sam_stability_score_thresh": config.SAM_STABILITY_SCORE_THRESH
        }
    }
    
    results_path = config.get_output_path("sam_only_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   📄 Results JSON: {results_path}")
    
    # Summary
    print()
    print("📋 SAM-Only Pipeline Summary:")
    print(f"   🎯 Total masks generated: {len(masks)}")
    print(f"   ⭐ High-quality masks: {len(high_quality_masks)}")
    print(f"   📊 Mean stability score: {analysis['stability_score_stats']['mean']:.3f}")
    print(f"   🔍 Area range: {analysis['area_stats']['min']}-{analysis['area_stats']['max']} pixels")
    
    print()
    print("✨ SAM-Only Pipeline completed!")
    print("   🔍 This shows SAM's automatic segmentation capabilities")
    print("   📊 Compare with YOLO+SAM ensemble results for accuracy")

if __name__ == "__main__":
    run_sam_only_pipeline()