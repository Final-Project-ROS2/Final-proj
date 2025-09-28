#!/usr/bin/env python3
"""
Complete SAM+YOLO+CLIP Pipeline Runner
Runs object detection (YOLO), segmentation (SAM), and classification (CLIP) pipeline
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from config.config import get_current_image_path, AVAILABLE_IMAGES, TEST_OUTPUTS_DIR
from src.pipeline.main_pipeline import RoboticVisionPipeline

def run_complete_pipeline():
    """Run the complete SAM+YOLO+CLIP pipeline"""
    
    print("🚀 Starting Complete SAM+YOLO+CLIP Pipeline")
    print("=" * 60)
    
    # Configuration
    output_base = TEST_OUTPUTS_DIR / "complete_pipeline"
    output_base.mkdir(exist_ok=True)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / f"run_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    # Model paths
    yolo_model = "yolo11s.pt"  # Will download automatically if not present
    
    try:
        # Initialize pipeline
        print("\n📦 Initializing Pipeline Components...")
        pipeline = RoboticVisionPipeline(
            sam_checkpoint=None,  # Will auto-find or provide instructions
            yolo_model=yolo_model,
            use_yolo=True,
            device="auto"
        )
        print("✅ Pipeline initialized successfully!")
        
        # Process all available test images
        results_summary = {}
        
        for image_name, image_path in AVAILABLE_IMAGES.items():
            if not image_path.exists():
                print(f"⚠️ Skipping {image_name}: File not found at {image_path}")
                continue
                
            print(f"\n🖼️ Processing image: {image_name}")
            print(f"   Path: {image_path}")
            
            # Create image-specific output directory
            image_output_dir = output_dir / image_name
            image_output_dir.mkdir(exist_ok=True)
            
            try:
                # Process the image
                results = pipeline.process_image_file(
                    str(image_path),
                    depth_path=None,  # No depth for now
                    output_dir=str(image_output_dir)
                )
                
                # Store results summary
                results_summary[image_name] = {
                    "num_objects": len(results["labels"]),
                    "object_classes": list(set(results["labels"])),
                    "avg_confidence": float(np.mean(results["confidences"])) if results["confidences"] else 0.0,
                    "num_grasps": len(results["grasps"]),
                    "output_files": {
                        "detection_viz": str(image_output_dir / "1_detection.jpg"),
                        "grasp_viz": str(image_output_dir / "3_grasps.jpg"),
                        "results_json": str(image_output_dir / "pipeline_results.json")
                    }
                }
                
                print(f"✅ {image_name}: Detected {len(results['labels'])} objects")
                print(f"   Classes: {', '.join(set(results['labels']))}")
                
            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")
                results_summary[image_name] = {"error": str(e)}
        
        # Save overall summary
        summary_file = output_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "pipeline_config": {
                    "yolo_model": yolo_model,
                    "use_sam": True,
                    "use_clip": True
                },
                "results": results_summary
            }, f, indent=2)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🎉 Pipeline Execution Complete!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Summary file: {summary_file}")
        
        successful_runs = [k for k, v in results_summary.items() if "error" not in v]
        print(f"✅ Successfully processed: {len(successful_runs)} images")
        
        if successful_runs:
            print("\n📊 Results Summary:")
            for image_name in successful_runs:
                result = results_summary[image_name]
                print(f"   {image_name}: {result['num_objects']} objects, {result['num_grasps']} grasps")
        
        # Return output file paths for next pipeline
        output_files = []
        for image_name, result in results_summary.items():
            if "output_files" in result:
                output_files.extend(result["output_files"].values())
        
        print(f"\n🔗 Output files ready for next pipeline stage:")
        for file_path in output_files:
            if Path(file_path).exists():
                print(f"   ✅ {file_path}")
            else:
                print(f"   ⚠️ {file_path} (not found)")
        
        return str(summary_file), output_files
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure YOLO is installed: pip install ultralytics")
        print("2. Download SAM checkpoint: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        print("3. Check that test images exist in data/test_images/")
        raise

def main():
    """Main entry point"""
    try:
        summary_file, output_files = run_complete_pipeline()
        print(f"\n🎯 MAIN OUTPUT FILE: {summary_file}")
        return summary_file, output_files
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()