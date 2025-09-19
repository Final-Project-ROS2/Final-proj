"""
Unit Test for Object Classification Module
Tests if the classification module can properly identify tools from segmented objects.

Expected Output Format:
[
  { "id": 1, "label": "wrench", "confidence": 0.94 },
  { "id": 2, "label": "screwdriver", "confidence": 0.89 },
  { "id": 3, "label": "box cutter", "confidence": 0.81 }
]
"""

import cv2
import numpy as np
import json
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.object_classification import ObjectClassification


def load_detection_results():
    """Load detection results from previous segmentation test"""
    results_path = Path("test_outputs/detection_results.json")
    
    if not results_path.exists():
        print("❌ Detection results not found. Please run object detection test first.")
        print("   Expected file: test_outputs/detection_results.json")
        return None, None, None
    
    with open(results_path, 'r') as f:
        detection_data = json.load(f)
    
    # Load original image
    image_path = Path("../src/tools.png")
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None, None, None
    
    # Extract boxes from detection data
    boxes = detection_data.get("bounding_boxes", [])
    
    # Create dummy masks based on bounding boxes (since masks weren't saved)
    masks = []
    for box in boxes:
        x, y, w, h = box
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # Create a simple rectangular mask for each bounding box
        mask[int(y):int(y+h), int(x):int(x+w)] = 255
        masks.append(mask)
    
    print(f"✅ Loaded {len(masks)} objects from detection results")
    print(f"   Total detected objects: {detection_data.get('num_objects', len(boxes))}")
    
    return image, masks, boxes


def test_object_classification():
    """
    Test Object Classification Module
    
    Expected behavior:
    - Load segmented objects from detection results
    - Classify each object into tool categories
    - Return structured results with id, label, and confidence
    """
    print("🏷️  Testing Object Classification Module")
    print("=" * 50)
    
    # Load detection results
    image, masks, boxes = load_detection_results()
    if image is None:
        return
    
    try:
        # Initialize classifier
        print("Initializing Object Classification module...")
        classifier = ObjectClassification()
        
        # Run classification
        print("Classifying detected objects...")
        labels, confidences = classifier.classify_objects(image, masks)
        
        # Format results in expected structure
        classification_results = []
        for i, (label, confidence) in enumerate(zip(labels, confidences)):
            result = {
                "id": i + 1,
                "label": label,
                "confidence": round(confidence, 2)
            }
            classification_results.append(result)
        
        # Print results
        print(f"\n✅ Classification completed for {len(classification_results)} objects")
        print("\n📊 CLASSIFICATION RESULTS:")
        print("-" * 30)
        
        for result in classification_results:
            print(f"Object {result['id']}: {result['label']} (confidence: {result['confidence']})")
        
        # Print as JSON format (expected output)
        print(f"\n📋 JSON OUTPUT:")
        print(json.dumps(classification_results, indent=2))
        
        # Save results
        output_file = "test_outputs/classification_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                "classification_results": classification_results,
                "summary": {
                    "total_objects": len(classification_results),
                    "tool_types_detected": list(set([r["label"] for r in classification_results])),
                    "average_confidence": round(np.mean([r["confidence"] for r in classification_results]), 2)
                }
            }, f, indent=2)
        
        print(f"✅ Results saved to: {output_file}")
        
        # Create visualization
        create_classification_visualization(image, masks, boxes, classification_results)
        
        # Validate results
        validate_classification_results(classification_results)
        
        return classification_results
        
    except Exception as e:
        print(f"❌ Classification test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_classification_visualization(image, masks, boxes, results):
    """Create visualization showing classified objects with original image background"""
    print("\n🎨 Creating classification visualization...")
    
    # Start with the original image as background
    vis_image = image.copy()
    
    # Color map for different tool types (BGR format for OpenCV)
    color_map = {
        "hammer": (0, 0, 255),          # Red
        "screwdriver": (0, 255, 0),     # Green  
        "wrench": (255, 0, 0),          # Blue
        "pliers": (255, 255, 0),        # Cyan
        "drill": (255, 0, 255),         # Magenta
        "saw": (0, 255, 255),           # Yellow
        "clamp": (255, 165, 0),         # Orange
        "file": (128, 0, 128),          # Purple
        "chisel": (0, 128, 255),        # Light Blue
        "measuring_tape": (255, 192, 203), # Pink
        "allen_key": (165, 42, 42),     # Brown
        "socket": (255, 20, 147),       # Deep Pink
        "scissors": (50, 205, 50),      # Lime Green
        "knife": (220, 20, 60),         # Crimson
        "box_cutter": (128, 128, 128),  # Gray
        "unknown": (169, 169, 169)      # Dark Gray
    }
    
    # Create multiple visualization versions
    # 1. Original image with bounding boxes and labels only
    bbox_image = vis_image.copy()
    
    # 2. Image with semi-transparent mask overlays
    overlay_image = vis_image.copy()
    
    for i, result in enumerate(results):
        if i >= len(masks) or i >= len(boxes):
            continue
            
        mask = masks[i]
        box = boxes[i]
        label = result["label"]
        confidence = result["confidence"]
        
        # Get color for this tool type
        color = color_map.get(label, (128, 128, 128))
        
        # Draw bounding box on both images
        x, y, w, h = box
        
        # Ensure coordinates are valid
        x, y, w, h = int(x), int(y), int(w), int(h)
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            continue
            
        # Draw bounding box
        cv2.rectangle(bbox_image, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(overlay_image, (x, y), (x+w, y+h), color, 2)
        
        # Create mask overlay for the overlay image
        if mask.shape[:2] == vis_image.shape[:2]:  # Ensure mask dimensions match
            mask_colored = np.zeros_like(vis_image, dtype=np.uint8)
            mask_colored[mask > 0] = color
            # Apply semi-transparent overlay (70% original, 30% mask color)
            overlay_image = cv2.addWeighted(overlay_image, 0.85, mask_colored, 0.15, 0)
        
        # Draw label and confidence
        label_text = f"{result['id']}: {label} ({confidence})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Draw text background (make sure it fits within image bounds)
        text_bg_y1 = max(0, y - text_height - 10)
        text_bg_y2 = max(text_height + 10, y)
        text_bg_x1 = x
        text_bg_x2 = min(vis_image.shape[1], x + text_width + 10)
        
        # Draw background rectangles for text
        cv2.rectangle(bbox_image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
        cv2.rectangle(overlay_image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
        
        # Draw text
        text_y = max(text_height, y - 5)
        cv2.putText(bbox_image, label_text, (x + 5, text_y), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(overlay_image, label_text, (x + 5, text_y), font, font_scale, (255, 255, 255), thickness)
    
    # Save both versions
    bbox_output_path = "test_outputs/classification_bbox_only.jpg"
    overlay_output_path = "test_outputs/classification_with_masks.jpg"
    
    cv2.imwrite(bbox_output_path, bbox_image)
    cv2.imwrite(overlay_output_path, overlay_image)
    
    print(f"✅ Classification visualizations saved:")
    print(f"   - Bounding boxes only: {bbox_output_path}")
    print(f"   - With mask overlays: {overlay_output_path}")
    
    # Also save the main visualization (with masks) as the original name for compatibility
    cv2.imwrite("test_outputs/classification_visualization.jpg", overlay_image)


def validate_classification_results(results):
    """Validate the classification results"""
    print("\n🔍 Validating classification results...")
    
    # Expected tool classes
    valid_tool_classes = [
        "hammer", "screwdriver", "pliers", "wrench", "drill", 
        "saw", "chisel", "file", "clamp", "measuring_tape",
        "scissors", "knife", "allen_key", "socket", "box_cutter", "unknown"
    ]
    
    # Validation checks
    validation_results = {
        "total_objects": len(results),
        "valid_labels": True,
        "confidence_range": True,
        "tool_diversity": 0,
        "issues": []
    }
    
    detected_tools = set()
    
    for result in results:
        # Check label validity
        if result["label"] not in valid_tool_classes:
            validation_results["valid_labels"] = False
            validation_results["issues"].append(f"Invalid label: {result['label']}")
        
        # Check confidence range
        if not (0.0 <= result["confidence"] <= 1.0):
            validation_results["confidence_range"] = False
            validation_results["issues"].append(f"Invalid confidence: {result['confidence']}")
        
        detected_tools.add(result["label"])
    
    validation_results["tool_diversity"] = len(detected_tools)
    
    # Print validation results
    print(f"✅ Total objects classified: {validation_results['total_objects']}")
    print(f"✅ Valid labels: {validation_results['valid_labels']}")
    print(f"✅ Valid confidence range: {validation_results['confidence_range']}")
    print(f"✅ Tool diversity: {validation_results['tool_diversity']} unique types")
    print(f"✅ Detected tool types: {', '.join(detected_tools)}")
    
    if validation_results["issues"]:
        print("⚠️  Issues found:")
        for issue in validation_results["issues"]:
            print(f"   - {issue}")
    else:
        print("✅ All validation checks passed!")
    
    # Test status
    if (validation_results["valid_labels"] and 
        validation_results["confidence_range"] and 
        validation_results["total_objects"] > 0):
        print("\n🎯 CLASSIFICATION TEST: PASS")
    else:
        print("\n❌ CLASSIFICATION TEST: FAIL")


def main():
    """Run the object classification unit test"""
    print("🚀 Starting Object Classification Unit Test")
    print("=" * 60)
    
    # Change to tests directory
    test_dir = Path(__file__).parent
    os.chdir(test_dir)
    
    # Run test
    results = test_object_classification()
    
    if results:
        print("\n🎉 Object Classification test completed successfully!")
        print(f"📁 Check test_outputs/ folder for:")
        print("   - classification_results.json")
        print("   - classification_visualization.jpg")
    else:
        print("\n💥 Object Classification test failed!")
        print("Please check the error messages above.")


if __name__ == "__main__":
    main()
