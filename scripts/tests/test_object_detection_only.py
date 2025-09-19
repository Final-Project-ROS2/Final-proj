"""
Unit Test for Object Detection & Segmentation Module
Tests SAM's ability to segment tools.png and outputs bounding boxes + visualization
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
import os
import json
# --- Robustly add project root to sys.path ---
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipeline.object_detection_segmentation import ObjectDetectionSegmentation


class ObjectDetectionTest:
    """Focused test for object detection and segmentation"""
    
    def __init__(self):
        # Use absolute paths from project root
        self.project_root = Path(__file__).parent.parent
        self.image_path = self.project_root / "src" / "tools.png"
        self.checkpoint_path = self.project_root / "sam_vit_b_01ec64.pth"
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_test_image(self):
        """Load and validate test image"""
        print("📷 Loading test image...")
        
        if not self.image_path.exists():
            raise FileNotFoundError(f"Test image not found: {self.image_path}")
        
        image = cv2.imread(str(self.image_path))
        if image is None:
            raise ValueError(f"Could not read image: {self.image_path}")
        
        print(f"   ✅ Image loaded successfully")
        print(f"   📏 Image shape: {image.shape}")
        print(f"   📁 Image path: {self.image_path}")
        
        return image
    
    def initialize_detector(self):
        """Initialize SAM detector"""
        print("\n🤖 Initializing SAM detector...")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {self.checkpoint_path}")
        
        try:
            detector = ObjectDetectionSegmentation(
                checkpoint_path=str(self.checkpoint_path),
                model_type="vit_b",  # Match the checkpoint
                device="auto"
            )
            print(f"   ✅ SAM detector initialized successfully")
            print(f"   🔗 Checkpoint: {self.checkpoint_path}")
            return detector
            
        except Exception as e:
            print(f"   ❌ Failed to initialize detector: {e}")
            raise
    
    def run_segmentation(self, detector, image):
        """Run segmentation and get results"""
        print("\n🔍 Running object detection and segmentation...")
        
        try:
            # Run detection (no depth available)
            masks, boxes = detector.detect_and_segment(image, depth=None)
            
            print(f"   ✅ Segmentation completed")
            print(f"   📊 Objects detected: {len(masks)}")
            print(f"   📦 Bounding boxes: {len(boxes)}")
            
            return masks, boxes
            
        except Exception as e:
            print(f"   ❌ Segmentation failed: {e}")
            raise
    
    def analyze_results(self, masks, boxes):
        """Analyze and print segmentation results"""
        print("\n📊 Analyzing segmentation results...")
        
        if not masks:
            print("   ⚠️  No objects detected!")
            return
        
        # Calculate statistics
        total_pixels = sum(np.sum(mask) for mask in masks)
        avg_area = total_pixels / len(masks)
        
        print(f"   📈 Statistics:")
        print(f"      - Total objects: {len(masks)}")
        print(f"      - Total segmented pixels: {total_pixels:,.0f}")
        print(f"      - Average object area: {avg_area:,.0f} pixels")
        print(f"      - Mask merging: Applied for elongated objects (pliers, etc.)")
        
        # Analyze object shapes for elongated detection
        elongated_objects = 0
        for i, bbox in enumerate(boxes):
            x, y, w, h = bbox
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 2.0:
                elongated_objects += 1
        
        print(f"      - Elongated objects detected: {elongated_objects}")
        
        # Print bounding box details
        print(f"\n   📦 Bounding Boxes:")
        for i, bbox in enumerate(boxes):
            x, y, w, h = bbox
            area = w * h
            aspect_ratio = max(w, h) / min(w, h)
            shape_type = "elongated" if aspect_ratio > 2.0 else "compact"
            print(f"      Object {i+1}: x={x:>4.0f}, y={y:>4.0f}, w={w:>4.0f}, h={h:>4.0f}, area={area:>7.0f}, {shape_type}")
        
        return {
            "num_objects": len(masks),
            "total_pixels": int(total_pixels),
            "average_area": float(avg_area),
            "elongated_objects": elongated_objects,
            "mask_merging_applied": True,
            "bounding_boxes": boxes
        }
    
    def create_visualizations(self, detector, image, masks, boxes):
        """Create and save visualization images"""
        print("\n🎨 Creating visualizations...")
        
        # 1. Original image
        cv2.imwrite(str(self.output_dir / "1_original.jpg"), image)
        
        # 2. Detection visualization (masks + boxes)
        vis_image = detector.visualize_detections(
            image, masks, boxes, 
            str(self.output_dir / "2_detection_overlay.jpg")
        )
        
        # 3. Individual masks
        self._save_individual_masks(image, masks, boxes)
        
        # 4. Create matplotlib visualization
        self._create_matplotlib_plot(image, masks, boxes)
        
        print(f"   ✅ Visualizations saved to: {self.output_dir}")
        print(f"      - 1_original.jpg")
        print(f"      - 2_detection_overlay.jpg") 
        print(f"      - 3_individual_masks.jpg")
        print(f"      - 4_matplotlib_plot.png")
        
        return vis_image
    
    def _save_individual_masks(self, image, masks, boxes):
        """Save individual mask visualizations"""
        if not masks:
            return
            
        # Create a grid of individual masks
        num_objects = len(masks)
        cols = min(4, num_objects)
        rows = (num_objects + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(num_objects):
            # Create colored mask
            mask = masks[i]
            colored_mask = np.zeros_like(image)
            color = np.random.randint(0, 255, 3)
            colored_mask[mask > 0] = color
            
            # Overlay on original
            overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
            
            # Add bounding box
            x, y, w, h = boxes[i]
            cv2.rectangle(overlay, (int(x), int(y)), (int(x+w), int(y+h)), color.tolist(), 2)
            
            # Plot
            axes[i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f'Object {i+1}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for j in range(num_objects, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "3_individual_masks.jpg", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_matplotlib_plot(self, image, masks, boxes):
        """Create detailed matplotlib visualization with improved mask and box visibility"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # All masks overlay - improved visibility
        overlay = image.copy().astype(np.float32)
        # Make the background a bit lighter for better contrast
        overlay = overlay * 0.8 + 50
        overlay = np.clip(overlay, 0, 255)

        # Use a visually distinct color palette
        color_palette = plt.cm.get_cmap('tab20', max(10, len(masks)))

        mask_alpha = 0.35  # More transparent mask overlay
        for i, mask in enumerate(masks):
            color = color_palette(i)[:3]  # RGB tuple, 0-1
            color_bgr = np.array([color[2], color[1], color[0]]) * 255
            mask_overlay = np.zeros_like(image, dtype=np.float32)
            mask_overlay[mask > 0] = color_bgr
            # Blend mask with higher transparency
            overlay[mask > 0] = overlay[mask > 0] * (1 - mask_alpha) + mask_overlay[mask > 0] * mask_alpha

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'All Masks ({len(masks)} objects)')
        axes[1].axis('off')

        # Bounding boxes only (thinner, more distinct)
        bbox_image = image.copy()
        for i, bbox in enumerate(boxes):
            x, y, w, h = bbox
            color = color_palette(i)[:3]
            color_bgr = [int(color[2] * 255), int(color[1] * 255), int(color[0] * 255)]
            cv2.rectangle(bbox_image, (int(x), int(y)), (int(x+w), int(y+h)), color_bgr, 2)
            cv2.putText(bbox_image, f'{i+1}', (int(x+5), int(y+25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)

        axes[2].imshow(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Bounding Boxes')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / "4_matplotlib_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results_json(self, results):
        """Save results to JSON file"""
        print("\n💾 Saving results to JSON...")
        
        results_file = self.output_dir / "detection_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   ✅ Results saved to: {results_file}")
    
    def run_test(self):
        """Run the complete test"""
        print("🚀 SAM Object Detection & Segmentation Test")
        print("=" * 50)
        
        try:
            # Load image
            image = self.load_test_image()
            
            # Initialize detector
            detector = self.initialize_detector()
            
            # Run segmentation
            masks, boxes = self.run_segmentation(detector, image)
            
            # Analyze results
            results = self.analyze_results(masks, boxes)
            
            # Create visualizations
            if masks:
                self.create_visualizations(detector, image, masks, boxes)
            
            # Save results
            if results:
                self.save_results_json(results)
            
            # Test verdict
            self.print_verdict(results)
            
            return results
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_verdict(self, results):
        """Print final test verdict"""
        print("\n" + "=" * 50)
        print("🎯 TEST VERDICT")
        print("=" * 50)
        
        if not results:
            print("❌ FAIL: No segmentation results")
            return
        
        num_objects = results["num_objects"]
        
        if num_objects == 0:
            print("❌ FAIL: No objects detected")
            print("   Possible issues:")
            print("   - Image quality too low")
            print("   - No clear objects in image")
            print("   - SAM model issues")
        elif num_objects >= 1:
            print(f"✅ PASS: Successfully detected {num_objects} objects")
            print("   Test criteria met:")
            print(f"   - Objects detected: {num_objects} ≥ 1 ✓")
            print(f"   - Bounding boxes generated: {len(results['bounding_boxes'])} ✓")
            print(f"   - Visual output created ✓")
        
        print(f"\n📁 Output files in '{self.output_dir}':")
        for file in sorted(self.output_dir.glob("*")):
            print(f"   - {file.name}")


def main():
    """Run the object detection test"""
    tester = ObjectDetectionTest()
    results = tester.run_test()
    
    if results and results["num_objects"] > 0:
        print("\n🎉 Test completed successfully!")
        print("Check the 'test_outputs' folder for visual results.")
    else:
        print("\n🔧 Test needs troubleshooting.")


if __name__ == "__main__":
    main()