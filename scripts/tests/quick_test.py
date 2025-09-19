"""
Quick Test Runner - Minimal setup for testing individual modules
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quick_detection_test(image_path: str, checkpoint_path: str):
    """Quick test for object detection only"""
    from pipeline.object_detection_segmentation import ObjectDetectionSegmentation
    
    print("🔍 Quick Detection Test")
    print("-" * 30)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"✅ Loaded image: {image.shape}")
    
    try:
        # Test detection
        detector = ObjectDetectionSegmentation(checkpoint_path)
        masks, boxes = detector.detect_and_segment(image)
        
        print(f"✅ Detected {len(masks)} objects")
        
        # Save visualization
        vis = detector.visualize_detections(image, masks, boxes, "quick_detection_test.jpg")
        print("✅ Saved: quick_detection_test.jpg")
        
        return masks, boxes
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


def quick_classification_test(image: np.ndarray, masks):
    """Quick test for classification only"""
    from pipeline.object_classification import ObjectClassification
    
    print("\n🏷️  Quick Classification Test")
    print("-" * 30)
    
    if masks is None:
        print("❌ No masks provided")
        return
    
    try:
        classifier = ObjectClassification()
        labels, confidences = classifier.classify_objects(image, masks)
        
        print(f"✅ Classified {len(labels)} objects:")
        for i, (label, conf) in enumerate(zip(labels, confidences)):
            print(f"   {i}: {label} ({conf:.2f})")
            
        return labels, confidences
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


def quick_grasp_test(image: np.ndarray, masks):
    """Quick test for grasp synthesis only"""
    from pipeline.grasp_synthesis import GraspSynthesis
    
    print("\n🤖 Quick Grasp Test")
    print("-" * 30)
    
    if masks is None:
        print("❌ No masks provided")
        return
    
    try:
        grasp_synth = GraspSynthesis()
        
        # Dummy depth
        dummy_depth = np.ones((image.shape[0], image.shape[1]), dtype=np.float32) * 0.5
        
        grasps = grasp_synth.generate_grasps(image, dummy_depth, masks)
        
        print(f"✅ Generated {len(grasps)} grasps")
        
        # Save visualization
        vis = grasp_synth.visualize_grasps(image, grasps, "quick_grasp_test.jpg")
        print("✅ Saved: quick_grasp_test.jpg")
        
        return grasps
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def quick_scene_test(image: np.ndarray, labels, boxes):
    """Quick test for scene understanding only"""
    from pipeline.scene_understanding_vlm import SceneUnderstandingVLM
    
    print("\n🧠 Quick Scene Test")
    print("-" * 30)
    
    if labels is None or boxes is None:
        print("❌ No labels/boxes provided")
        return
    
    try:
        scene_vlm = SceneUnderstandingVLM()
        scene_analysis = scene_vlm.understand_scene(image, labels, boxes)
        
        print(f"✅ Scene: {scene_analysis['scene_summary']}")
        print(f"✅ Relations: {len(scene_analysis['spatial_graph'])}")
        
        return scene_analysis
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Run quick tests"""
    # Paths - use absolute paths relative to project root
    project_root = Path(__file__).parent.parent
    image_path = project_root / "src" / "tools.png"
    checkpoint_path = project_root / "sam_vit_b_01ec64.pth"
    
    # Check files
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        print("Create src/tools.png with test image")
        return
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Load image once
    image = cv2.imread(str(image_path))
    
    # Run quick tests
    masks, boxes = quick_detection_test(str(image_path), str(checkpoint_path))
    labels, confidences = quick_classification_test(image, masks)
    grasps = quick_grasp_test(image, masks)
    scene = quick_scene_test(image, labels, boxes)
    
    print("\n🎉 Quick tests completed!")
    print("Check generated files:")
    print("  - quick_detection_test.jpg")
    print("  - quick_grasp_test.jpg")


if __name__ == "__main__":
    main()
