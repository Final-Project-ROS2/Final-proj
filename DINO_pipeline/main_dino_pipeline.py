"""
DINO Pipeline - Main Pipeline Runner
Integrates all components: DINO Detection → CLIP Tagging → GraspNet → Scene Graph
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import pipeline components
from pipeline.grounding_dino_detector import SAMMergeDetector
from pipeline.clip_semantic_tagger import CLIPSemanticTagger  
from pipeline.graspnet_predictor import GraspNetPredictor
from pipeline.scene_graph_builder import SceneGraphBuilder

class DINOPipeline:
    """
    Complete DINO-based robotic vision pipeline
    RGB-D → DINO → CLIP → GraspNet → Scene Graph
    """
    
    def __init__(self, device: str = "auto"):
        self.device = device
        
        # Initialize all pipeline components
        print("🚀 Initializing SAM-Enhanced DINO Pipeline Components...")
        
        self.detector = SAMMergeDetector(device=device)
        self.semantic_tagger = CLIPSemanticTagger(device=device) 
        self.grasp_predictor = GraspNetPredictor(device=device)
        self.scene_builder = SceneGraphBuilder()
        
        print("✅ SAM-Enhanced DINO Pipeline fully initialized!")
    
    def process_rgbd(self, rgb_path: str, depth_path: str, 
                     output_dir: str = "outputs") -> Dict:
        """
        Process RGB-D image pair through complete pipeline
        
        Args:
            rgb_path: Path to RGB image
            depth_path: Path to depth image  
            output_dir: Directory to save all outputs
            
        Returns:
            Complete pipeline results
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = output_path / f"dino_pipeline_{timestamp}"
        run_output_dir.mkdir(exist_ok=True)
        
        print(f"\n📁 Output directory: {run_output_dir}")
        
        # Load images
        print("\n📸 Loading RGB-D images...")
        rgb_image, depth_image = self._load_rgbd_images(rgb_path, depth_path)
        
        # Save input visualization
        self._save_input_visualization(rgb_image, depth_image, run_output_dir / "1_input_rgbd.jpg")
        
        # Stage 1: DINO Object Detection
        print("\n🔍 Stage 1: DINO Object Detection...")
        detections, masks = self.detector.detect_objects(rgb_image)
        print(f"   Detected {len(detections)} objects")
        
        # Save DINO detection visualization
        dino_vis = self.detector.visualize_detections(
            rgb_image, detections, str(run_output_dir / "2_dino_detections.jpg")
        )
        
        # Stage 2: CLIP Semantic Tagging
        print("\n🏷️ Stage 2: CLIP Semantic Tagging...")
        enhanced_detections = self.semantic_tagger.tag_objects(rgb_image, detections)
        print(f"   Enhanced {len(enhanced_detections)} objects with semantic tags")
        
        # Save CLIP tagging visualization
        clip_vis = self.semantic_tagger.visualize_semantic_tags(
            rgb_image, enhanced_detections, str(run_output_dir / "3_clip_semantic_tags.jpg")
        )
        
        # Stage 3: GraspNet 6D Grasp Prediction
        print("\n🤏 Stage 3: GraspNet 6D Grasp Prediction...")
        grasps = self.grasp_predictor.predict_grasps(rgb_image, depth_image, enhanced_detections)
        print(f"   Generated {len(grasps)} 6D grasps")
        
        # Save grasp visualization
        grasp_vis = self.grasp_predictor.visualize_grasps(
            rgb_image, grasps, str(run_output_dir / "4_graspnet_6d.jpg")
        )
        
        # Stage 4: Scene Graph Construction
        print("\n🌐 Stage 4: Scene Graph Construction...")
        scene_graph = self.scene_builder.build_scene_graph(
            rgb_image, depth_image, enhanced_detections, grasps
        )
        print(f"   Built scene graph with {len(scene_graph['nodes'])} nodes and {len(scene_graph['edges'])} relations")
        
        # Save scene graph visualization  
        scene_vis = self.scene_builder.visualize_scene_graph(
            rgb_image, scene_graph, str(run_output_dir / "5_scene_graph.jpg")
        )
        
        # Save scene graph data
        self.scene_builder.save_scene_graph(scene_graph, str(run_output_dir / "scene_graph.json"))
        
        # Create final combined visualization
        self._create_pipeline_summary_visualization(
            [rgb_image, dino_vis, clip_vis, grasp_vis, scene_vis],
            str(run_output_dir / "6_pipeline_summary.jpg")
        )
        
        # Compile complete pipeline results
        pipeline_results = {
            "metadata": {
                "timestamp": timestamp,
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "output_directory": str(run_output_dir)
            },
            "stage_1_detection": {
                "num_detections": len(detections),
                "detections": detections
            },
            "stage_2_semantic": {
                "num_enhanced": len(enhanced_detections),
                "enhanced_detections": enhanced_detections
            },
            "stage_3_grasps": {
                "num_grasps": len(grasps),
                "grasps": grasps
            },
            "stage_4_scene_graph": scene_graph,
            "output_files": {
                "input_visualization": str(run_output_dir / "1_input_rgbd.jpg"),
                "dino_detections": str(run_output_dir / "2_dino_detections.jpg"),
                "clip_semantic": str(run_output_dir / "3_clip_semantic_tags.jpg"),
                "graspnet_6d": str(run_output_dir / "4_graspnet_6d.jpg"),
                "scene_graph_vis": str(run_output_dir / "5_scene_graph.jpg"),
                "pipeline_summary": str(run_output_dir / "6_pipeline_summary.jpg"),
                "scene_graph_json": str(run_output_dir / "scene_graph.json"),
                "pipeline_results": str(run_output_dir / "pipeline_results.json")
            }
        }
        
        # Save complete results
        with open(run_output_dir / "pipeline_results.json", 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        # Print pipeline summary
        self._print_pipeline_summary(pipeline_results)
        
        return pipeline_results
    
    def _load_rgbd_images(self, rgb_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and validate RGB-D image pair"""
        # Load RGB image
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        
        rgb_image = cv2.imread(rgb_path)
        if rgb_image is None:
            raise ValueError(f"Could not load RGB image: {rgb_path}")
        
        # Load depth image
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        
        depth_image = cv2.imread(depth_path, -1)  # Load as-is (16-bit if available)
        if depth_image is None:
            raise ValueError(f"Could not load depth image: {depth_path}")
        
        # Convert depth to single channel if needed
        if len(depth_image.shape) == 3:
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        
        print(f"   RGB shape: {rgb_image.shape}, Depth shape: {depth_image.shape}")
        
        return rgb_image, depth_image
    
    def _save_input_visualization(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
                                save_path: str):
        """Create side-by-side RGB-D input visualization"""
        # Resize images to same height if needed
        if rgb_image.shape[0] != depth_image.shape[0]:
            target_height = min(rgb_image.shape[0], depth_image.shape[0])
            rgb_image = cv2.resize(rgb_image, (rgb_image.shape[1], target_height))
            depth_image = cv2.resize(depth_image, (depth_image.shape[1], target_height))
        
        # Convert depth to 3-channel for concatenation
        if len(depth_image.shape) == 2:
            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=1), cv2.COLORMAP_JET)
        else:
            depth_vis = depth_image
        
        # Create side-by-side visualization
        combined = np.hstack([rgb_image, depth_vis])
        
        # Add labels
        cv2.putText(combined, "RGB", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Depth", (rgb_image.shape[1] + 20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(save_path, combined)
        print(f"   ✅ Input visualization saved: {save_path}")
    
    def _create_pipeline_summary_visualization(self, images: List[np.ndarray], save_path: str):
        """Create grid visualization of all pipeline stages"""
        if not images:
            return
        
        # Resize all images to same size
        target_size = (320, 240)  # Smaller for grid layout
        resized_images = []
        
        for img in images:
            if img is not None:
                resized = cv2.resize(img, target_size)
                resized_images.append(resized)
        
        if len(resized_images) < 2:
            return
        
        # Create grid layout (2x3 or 3x2 depending on number of images)
        if len(resized_images) <= 3:
            # Single row
            grid = np.hstack(resized_images)
        else:
            # Two rows
            top_row = np.hstack(resized_images[:3])
            bottom_row = np.hstack(resized_images[3:])
            
            # Pad bottom row if needed
            if bottom_row.shape[1] < top_row.shape[1]:
                padding = np.zeros((bottom_row.shape[0], 
                                  top_row.shape[1] - bottom_row.shape[1], 3), dtype=np.uint8)
                bottom_row = np.hstack([bottom_row, padding])
            
            grid = np.vstack([top_row, bottom_row])
        
        # Add stage labels
        stage_labels = ["RGB Input", "DINO Detection", "CLIP Semantic", "GraspNet 6D", "Scene Graph"]
        label_positions = [
            (20, 30), (340, 30), (660, 30), (20, 270), (340, 270)
        ]
        
        for i, (label, pos) in enumerate(zip(stage_labels, label_positions)):
            if i < len(resized_images):
                cv2.putText(grid, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(save_path, grid)
        print(f"   ✅ Pipeline summary visualization saved: {save_path}")
    
    def _print_pipeline_summary(self, results: Dict):
        """Print comprehensive pipeline summary"""
        print("\n" + "=" * 80)
        print("🎉 DINO PIPELINE EXECUTION COMPLETE!")
        print("=" * 80)
        
        # Stage summaries
        print(f"\n📊 PIPELINE RESULTS SUMMARY:")
        print(f"   🔍 Stage 1 - DINO Detection: {results['stage_1_detection']['num_detections']} objects")
        print(f"   🏷️ Stage 2 - CLIP Semantic: {results['stage_2_semantic']['num_enhanced']} enhanced objects")
        print(f"   🤏 Stage 3 - GraspNet: {results['stage_3_grasps']['num_grasps']} 6D grasps")
        print(f"   🌐 Stage 4 - Scene Graph: {len(results['stage_4_scene_graph']['nodes'])} nodes, {len(results['stage_4_scene_graph']['edges'])} relations")
        
        # Object details
        print(f"\n🏷️ DETECTED OBJECTS:")
        for detection in results['stage_2_semantic']['enhanced_detections']:
            print(f"   • {detection['label']} → {detection['semantic_category']} (conf: {detection['confidence']:.2f})")
            print(f"     Affordances: {', '.join(detection['affordances'][:3])}")
        
        # Grasp summary
        scene_graph = results['stage_4_scene_graph']
        grasp_stats = scene_graph['grasp_summary']
        print(f"\n🤏 GRASP SUMMARY:")
        print(f"   • Total grasps: {grasp_stats['total_grasps']}")
        print(f"   • Average quality: {grasp_stats.get('avg_quality', 0):.2f}")
        print(f"   • High quality grasps: {grasp_stats.get('high_quality_grasps', 0)}")
        
        # Task insights
        task_insights = scene_graph['task_insights']
        print(f"\n🎯 TASK INSIGHTS:")
        print(f"   • Manipulation candidates: {len(task_insights['manipulation_candidates'])}")
        print(f"   • Spatial constraints: {len(task_insights['spatial_constraints'])}")
        print(f"   • Task recommendations: {len(task_insights['task_recommendations'])}")
        
        # Output files
        print(f"\n📁 OUTPUT FILES:")
        for file_type, file_path in results['output_files'].items():
            status = "✅" if os.path.exists(file_path) else "❌"
            print(f"   {status} {file_type}: {file_path}")
        
        print(f"\n🎯 MAIN OUTPUT FILE: {results['output_files']['pipeline_results']}")
        print("=" * 80)


def main():
    """Main function to run DINO pipeline with test images"""
    # Test image paths
    rgb_path = "src/img.PNG"
    depth_path = "src/img-d.PNG"
    
    # Check if test images exist
    if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
        print("❌ Test images not found!")
        print(f"   Expected RGB: {rgb_path}")
        print(f"   Expected Depth: {depth_path}")
        return
    
    try:
        # Initialize and run pipeline
        pipeline = DINOPipeline(device="auto")
        
        # Process RGB-D images
        results = pipeline.process_rgbd(rgb_path, depth_path, "outputs")
        
        print("\n🚀 DINO Pipeline execution completed successfully!")
        print(f"📄 Results saved to: {results['output_files']['pipeline_results']}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()