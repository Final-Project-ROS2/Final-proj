"""
Main Robotic Vision Pipeline - Research-Oriented Approach
Integrates all 4 pipeline stages with RGB + depth processing
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Import pipeline modules
from .object_detection_segmentation import ObjectDetectionSegmentation
from .object_classification import ObjectClassification  
from .grasp_synthesis import GraspSynthesis
from .scene_understanding_vlm import SceneUnderstandingVLM


class RoboticVisionPipeline:
    """
    Research-oriented robotic vision pipeline for tool manipulation
    """
    
    def __init__(self, sam_checkpoint: str = None, model_type: str = "vit_b", device: str = "auto", 
                 yolo_model: str = "yolo11s.pt", use_yolo: bool = True):
        """Initialize the complete pipeline"""
        # Try to find SAM checkpoint if not provided
        if sam_checkpoint is None:
            sam_checkpoint = self._find_sam_checkpoint()
            
        self.detection = ObjectDetectionSegmentation(sam_checkpoint, model_type, device, yolo_model, use_yolo)
        self.classification = ObjectClassification(device)
        self.grasp_synthesis = GraspSynthesis()
        self.scene_understanding = SceneUnderstandingVLM()
        
    def _find_sam_checkpoint(self):
        """Try to find SAM checkpoint in common locations"""
        possible_paths = [
            "sam_vit_b_01ec64.pth",
            "models/sam_vit_b_01ec64.pth", 
            "../sam_vit_b_01ec64.pth",
            "../../sam_vit_b_01ec64.pth"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return str(path)
        
        # If no checkpoint found, provide instructions
        raise FileNotFoundError(
            "SAM checkpoint not found. Please download from: "
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )
        
    def run(self, image: np.ndarray, depth: np.ndarray = None) -> Dict:
        """
        Process RGB + depth image through complete pipeline
        
        Args:
            image: RGB image
            depth: Depth map (optional)
            
        Returns:
            Complete pipeline results with labels, grasps, and scene analysis
        """
        # Stage 1: Object Detection and Segmentation
        print("Stage 1: Object Detection and Segmentation...")
        masks, boxes = self.detection.detect_and_segment(image, depth)
        
        # Stage 2: Object Classification using CLIP embeddings
        print("Stage 2: Object Classification...")
        classification_result = self.classification.classify_objects(image, masks, boxes)
        if len(classification_result) == 3:
            labels, confidences, details = classification_result
        else:
            labels, confidences = classification_result
            details = []
        
        # Stage 3: Grasp Synthesis with collision checking
        print("Stage 3: Grasp Synthesis...")
        grasps = self.grasp_synthesis.generate_grasps(image, depth, masks)
        
        # Stage 4: Scene Understanding - extract spatial relations
        print("Stage 4: Scene Understanding...")
        scene_desc = self.scene_understanding.understand_scene(image, labels, boxes)
        
        return {
            "labels": labels,
            "confidences": confidences,
            "classification_details": details if 'details' in locals() else [],
            "boxes": boxes,
            "masks": masks,
            "grasps": grasps,
            "scene": scene_desc
        }
    
    def process_image_file(self, image_path: str, depth_path: str = None, 
                          output_dir: str = "pipeline_output") -> Dict:
        """
        Process image file through complete pipeline
        
        Args:
            image_path: Path to RGB image
            depth_path: Path to depth image (optional)
            output_dir: Directory to save outputs
            
        Returns:
            Complete pipeline results
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load images
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        depth = None
        if depth_path:
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth is None:
                print(f"Warning: Could not load depth image: {depth_path}")
        
        # Run pipeline
        results = self.run(image, depth)
        
        # Save visualizations
        self._save_visualizations(image, results, output_path)
        
        # Save results to JSON
        json_results = self._prepare_for_json(results)
        with open(output_path / "pipeline_results.json", "w") as f:
            json.dump(json_results, f, indent=2)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _save_visualizations(self, image: np.ndarray, results: Dict, output_path: Path):
        """Save visualization images"""
        # Detection visualization
        detection_vis = self.detection.visualize_detections(
            image, results["masks"], results["boxes"], 
            str(output_path / "1_detection.jpg")
        )
        
        # Grasp visualization
        if results["grasps"]:
            grasp_vis = self.grasp_synthesis.visualize_grasps(
                image, results["grasps"], 
                str(output_path / "3_grasps.jpg")
            )
        
        print(f"Visualizations saved to: {output_path}")
    
    def _prepare_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _print_summary(self, results: Dict):
        """Print pipeline results summary"""
        print("\n=== Pipeline Results Summary ===")
        print(f"Detected objects: {len(results['labels'])}")
        print(f"Object classes: {', '.join(set(results['labels']))}")
        print(f"Generated grasps: {len(results['grasps'])}")
        print(f"Collision-free grasps: {len([g for g in results['grasps'] if g.get('collision_free', True)])}")
        print(f"Scene summary: {results['scene']['scene_summary']}")
        
        if results['scene']['spatial_graph']:
            print(f"Spatial relations found: {len(results['scene']['spatial_graph'])}")
            for relation in results['scene']['spatial_graph'][:3]:  # Show first 3
                print(f"  - {relation['subject']['class']} {relation['relation']} {relation['object']['class']}")


def main():
    """Example usage of the research-oriented robotic vision pipeline"""
    # Configuration
    sam_checkpoint = "sam_vit_b_01ec64.pth"  # Update with your SAM checkpoint path
    image_path = "test_image.jpg"  # Update with your test image
    depth_path = "test_depth.png"  # Optional depth image
    
    try:
        # Initialize pipeline
        print("Initializing Robotic Vision Pipeline...")
        pipeline = RoboticVisionPipeline(sam_checkpoint)
        
        # Process image
        print(f"Processing image: {image_path}")
        results = pipeline.process_image_file(image_path, depth_path)
        
        print("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have:")
        print("  - SAM checkpoint file (sam_vit_b_01ec64.pth)")
        print("  - Test image file")
        print("  - All required dependencies installed")


if __name__ == "__main__":
    main()
