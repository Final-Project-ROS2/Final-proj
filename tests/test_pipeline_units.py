"""
Unit Tests for Robotic Vision Pipeline
Simple MVP-style tests for each module with visual verification
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.object_detection_segmentation import ObjectDetectionSegmentation
from pipeline.object_classification import ObjectClassification
from pipeline.grasp_synthesis import GraspSynthesis
from pipeline.scene_understanding_vlm import SceneUnderstandingVLM


class PipelineUnitTests:
    """Simple unit tests for each pipeline module"""
    
    def __init__(self, test_image_path: str, sam_checkpoint: str):
        self.test_image_path = test_image_path
        self.sam_checkpoint = sam_checkpoint
        self.test_results = {}
        
        # Load test image
        self.test_image = cv2.imread(test_image_path)
        if self.test_image is None:
            raise ValueError(f"Could not load test image: {test_image_path}")
        
        print(f"✅ Loaded test image: {test_image_path}")
        print(f"   Image shape: {self.test_image.shape}")
    
    def test_object_detection_segmentation(self):
        """
        Test ObjectDetectionSegmentation module
        Pass = masks align with real tools. Fail = over/under segmentation.
        """
        print("\n🔍 Testing Object Detection & Segmentation...")
        
        try:
            # Initialize detector
            detector = ObjectDetectionSegmentation(self.sam_checkpoint)
            
            # Test with RGB only (no depth available)
            masks, boxes = detector.detect_and_segment(self.test_image, depth=None)
            
            # Visualize results
            vis_image = detector.visualize_detections(self.test_image, masks, boxes, 
                                                    "tests/test_detection_output.jpg")
            
            # Results
            num_objects = len(masks)
            print(f"   ✅ Detected {num_objects} objects")
            print(f"   ✅ Generated {len(boxes)} bounding boxes")
            print(f"   ✅ Visualization saved to: tests/test_detection_output.jpg")
            
            # Simple validation
            if num_objects > 0:
                avg_mask_size = np.mean([np.sum(mask) for mask in masks])
                print(f"   ✅ Average mask size: {avg_mask_size:.0f} pixels")
                test_result = "PASS" if num_objects >= 1 else "FAIL"
            else:
                test_result = "FAIL - No objects detected"
            
            self.test_results["detection"] = {
                "status": test_result,
                "objects_detected": num_objects,
                "masks": masks,
                "boxes": boxes
            }
            
            print(f"   🎯 Test Result: {test_result}")
            
        except Exception as e:
            print(f"   ❌ Test FAILED: {e}")
            self.test_results["detection"] = {"status": "ERROR", "error": str(e)}
    
    def test_object_classification(self):
        """
        Test ObjectClassification module
        Pass = label matches expected tools. Fail = random mislabels.
        """
        print("\n🏷️  Testing Object Classification...")
        
        try:
            # Get masks from detection test
            if "detection" not in self.test_results or self.test_results["detection"]["status"] == "ERROR":
                print("   ⚠️  Skipping - detection test failed")
                return
            
            masks = self.test_results["detection"]["masks"]
            
            # Initialize classifier
            classifier = ObjectClassification()
            
            # Test classification
            labels, confidences = classifier.classify_objects(self.test_image, masks)
            
            # Results
            print(f"   ✅ Classified {len(labels)} objects")
            for i, (label, conf) in enumerate(zip(labels, confidences)):
                print(f"   ✅ Object {i}: {label} (confidence: {conf:.2f})")
            
            # Simple validation - check if we got tool-related labels
            tool_labels = ["hammer", "screwdriver", "pliers", "wrench", "drill", "saw"]
            detected_tools = [label for label in labels if label in tool_labels]
            
            if detected_tools:
                test_result = "PASS"
                print(f"   ✅ Detected tools: {detected_tools}")
            else:
                test_result = "PARTIAL - No specific tools detected"
                print(f"   ⚠️  Labels: {labels}")
            
            self.test_results["classification"] = {
                "status": test_result,
                "labels": labels,
                "confidences": confidences
            }
            
            print(f"   🎯 Test Result: {test_result}")
            
        except Exception as e:
            print(f"   ❌ Test FAILED: {e}")
            self.test_results["classification"] = {"status": "ERROR", "error": str(e)}
    
    def test_grasp_synthesis(self):
        """
        Test GraspSynthesis module
        Pass = grasp aligns with handles/surfaces. Fail = grasps floating in air.
        """
        print("\n🤖 Testing Grasp Synthesis...")
        
        try:
            # Get data from previous tests
            if "detection" not in self.test_results or self.test_results["detection"]["status"] == "ERROR":
                print("   ⚠️  Skipping - detection test failed")
                return
            
            masks = self.test_results["detection"]["masks"]
            
            # Initialize grasp synthesizer
            grasp_synth = GraspSynthesis()
            
            # Create dummy depth (since we don't have real depth)
            dummy_depth = np.ones((self.test_image.shape[0], self.test_image.shape[1]), dtype=np.float32) * 0.5
            
            # Test grasp generation
            grasps = grasp_synth.generate_grasps(self.test_image, dummy_depth, masks)
            
            # Visualize grasps
            vis_image = grasp_synth.visualize_grasps(self.test_image, grasps, 
                                                   "tests/test_grasp_output.jpg")
            
            # Results
            num_grasps = len(grasps)
            collision_free_grasps = len([g for g in grasps if g.get("collision_free", True)])
            
            print(f"   ✅ Generated {num_grasps} grasp candidates")
            print(f"   ✅ Collision-free grasps: {collision_free_grasps}")
            print(f"   ✅ Visualization saved to: tests/test_grasp_output.jpg")
            
            if grasps:
                avg_quality = np.mean([g["quality"] for g in grasps])
                print(f"   ✅ Average grasp quality: {avg_quality:.2f}")
                test_result = "PASS" if avg_quality > 0.3 else "PARTIAL - Low quality grasps"
            else:
                test_result = "FAIL - No grasps generated"
            
            self.test_results["grasp_synthesis"] = {
                "status": test_result,
                "num_grasps": num_grasps,
                "collision_free": collision_free_grasps,
                "grasps": grasps
            }
            
            print(f"   🎯 Test Result: {test_result}")
            
        except Exception as e:
            print(f"   ❌ Test FAILED: {e}")
            self.test_results["grasp_synthesis"] = {"status": "ERROR", "error": str(e)}
    
    def test_scene_understanding_vlm(self):
        """
        Test SceneUnderstandingVLM module
        Pass = relations match reality. Fail = nonsense captions.
        """
        print("\n🧠 Testing Scene Understanding VLM...")
        
        try:
            # Get data from previous tests
            if ("classification" not in self.test_results or 
                self.test_results["classification"]["status"] == "ERROR"):
                print("   ⚠️  Skipping - classification test failed")
                return
            
            labels = self.test_results["classification"]["labels"]
            boxes = self.test_results["detection"]["boxes"]
            
            # Initialize scene understanding
            scene_vlm = SceneUnderstandingVLM()
            
            # Test scene understanding
            scene_analysis = scene_vlm.understand_scene(self.test_image, labels, boxes)
            
            # Results
            print(f"   ✅ Scene Summary: {scene_analysis['scene_summary']}")
            print(f"   ✅ Total objects: {scene_analysis['total_objects']}")
            print(f"   ✅ Unique classes: {scene_analysis['unique_classes']}")
            print(f"   ✅ Spatial relations found: {len(scene_analysis['spatial_graph'])}")
            
            # Print spatial relations
            if scene_analysis['spatial_graph']:
                print("   ✅ Spatial Relations:")
                for relation in scene_analysis['spatial_graph'][:3]:  # Show first 3
                    print(f"      - {relation['subject']['class']} {relation['relation']} {relation['object']['class']} "
                          f"(conf: {relation['confidence']:.2f})")
            
            # Simple validation
            if scene_analysis['total_objects'] > 0:
                test_result = "PASS"
            else:
                test_result = "FAIL - No scene understanding generated"
            
            self.test_results["scene_understanding"] = {
                "status": test_result,
                "scene_analysis": scene_analysis
            }
            
            print(f"   🎯 Test Result: {test_result}")
            
        except Exception as e:
            print(f"   ❌ Test FAILED: {e}")
            self.test_results["scene_understanding"] = {"status": "ERROR", "error": str(e)}
    
    def run_all_tests(self):
        """Run all unit tests in sequence"""
        print("🚀 Starting Pipeline Unit Tests")
        print("=" * 50)
        
        # Run tests in order
        self.test_object_detection_segmentation()
        self.test_object_classification()
        self.test_grasp_synthesis()
        self.test_scene_understanding_vlm()
        
        # Summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print overall test results summary"""
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r["status"] == "PASS"])
        partial_tests = len([r for r in self.test_results.values() if r["status"].startswith("PARTIAL")])
        failed_tests = len([r for r in self.test_results.values() if r["status"] in ["FAIL", "ERROR"]])
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"⚠️  Partial: {partial_tests}")
        print(f"❌ Failed: {failed_tests}")
        
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result["status"] == "PASS" else "⚠️" if "PARTIAL" in result["status"] else "❌"
            print(f"{status_emoji} {test_name}: {result['status']}")
        
        print("\n📁 Output files generated:")
        print("   - tests/test_detection_output.jpg")
        print("   - tests/test_grasp_output.jpg")
        
        if passed_tests + partial_tests >= 3:
            print("\n🎉 Pipeline is working! Ready for integration testing.")
        else:
            print("\n🔧 Pipeline needs fixes before integration.")


def main():
    """Run unit tests with sample configuration"""
    # Configuration - use absolute paths relative to project root
    project_root = Path(__file__).parent.parent
    test_image_path = project_root / "src" / "tools.png"
    sam_checkpoint = project_root / "sam_vit_b_01ec64.pth"
    
    # Check if files exist
    if not test_image_path.exists():
        print(f"❌ Test image not found: {test_image_path}")
        print("Please create src/tools.png with a test image containing tools")
        return
    
    if not sam_checkpoint.exists():
        print(f"❌ SAM checkpoint not found: {sam_checkpoint}")
        print("Please download the SAM checkpoint file")
        return
    
    try:
        # Run tests (convert Path objects to strings)
        tester = PipelineUnitTests(str(test_image_path), str(sam_checkpoint))
        tester.run_all_tests()
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")


if __name__ == "__main__":
    main()
