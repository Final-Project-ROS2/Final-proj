"""
Example script showing how to easily switch images and models using config.py
This demonstrates the dependency system for training/testing with different images.
"""

import config
import subprocess
import sys
from pathlib import Path

def run_pipeline_with_config():
    """Run the pipeline with current configuration"""
    print(f"🚀 Running pipeline with:")
    print(f"   📸 Image: {config.CURRENT_IMAGE}")
    print(f"   🤖 SAM Model: {config.CURRENT_SAM_MODEL}")
    print(f"   🎯 YOLO Models: {config.CURRENT_YOLO_MODELS}")
    print("=" * 50)
    
    # Get the path to the Python executable and the script
    python_exe = sys.executable
    script_path = Path(__file__).parent / "configurable_smart_pipeline.py"
    
    # Run the pipeline
    result = subprocess.run([python_exe, str(script_path)], 
                          capture_output=False, 
                          text=True)
    
    return result.returncode == 0

def demo_different_configurations():
    """Demonstrate running pipeline with different configurations"""
    
    print("🎯 DEMO: Testing Different Image Configurations")
    print("=" * 60)
    
    # Test 1: Tools image with SAM-L (current default)
    print("\n📋 TEST 1: Tools image with SAM-L")
    config.set_image("tools")
    config.set_sam_model("sam_vit_l")
    if not run_pipeline_with_config():
        print("❌ Test 1 failed")
        return
    
    print("\n✅ Test 1 completed - check test_outputs/tools_* files")
    
    # Test 2: Tools image with SAM-B (faster)
    print("\n📋 TEST 2: Same image with faster SAM-B model")
    config.set_sam_model("sam_vit_b")
    if not run_pipeline_with_config():
        print("❌ Test 2 failed")
        return
    
    print("\n✅ Test 2 completed - compare with Test 1 results")
    
    # Test 3: Different image if available
    if "clip_test" in config.AVAILABLE_IMAGES:
        print("\n📋 TEST 3: Different image (clip_test)")
        config.set_image("clip_test")
        config.set_sam_model("sam_vit_l")  # Back to high-quality model
        if not run_pipeline_with_config():
            print("❌ Test 3 failed")
            return
        
        print("\n✅ Test 3 completed - check test_outputs/clip_test_* files")
    
    # Test 4: Reduced model set for faster processing
    print("\n📋 TEST 4: Faster processing with fewer models")
    config.set_image("tools")  # Back to tools
    config.set_yolo_models(["yolo11s", "yolo11m"])  # Skip YOLOv8n
    config.set_sam_model("sam_vit_b")  # Faster SAM
    if not run_pipeline_with_config():
        print("❌ Test 4 failed")
        return
    
    print("\n✅ Test 4 completed - faster processing demo")
    
    print("\n🎉 All configuration tests completed!")
    print("\n📁 Check the test_outputs/ folder for all generated results:")
    print("   - Different image prefixes show results from each test")
    print("   - Compare quality vs speed tradeoffs")
    print("   - JSON files contain detailed metrics for analysis")

def add_custom_image_example():
    """Show how to add a custom image for testing"""
    print("\n💡 HOW TO ADD YOUR OWN IMAGES:")
    print("=" * 40)
    print("1. Copy your image to the src/ folder")
    print("2. Edit config.py and add to AVAILABLE_IMAGES:")
    print("   'my_image': SRC_DIR / 'my_custom_image.jpg',")
    print("3. Use config.set_image('my_image') to switch to it")
    print("4. Run the pipeline normally!")
    print()
    print("Example config.py addition:")
    print("```python")
    print("AVAILABLE_IMAGES = {")
    print('    "tools": SRC_DIR / "tools.png",')
    print('    "clip_test": SRC_DIR / "clip_test.png",')
    print('    "my_custom_scene": SRC_DIR / "my_scene.jpg",  # <- Add this')
    print('    "robotics_test": SRC_DIR / "robot_workspace.png",')
    print("}")
    print("```")

def show_current_config():
    """Display current configuration"""
    print("🔧 CURRENT CONFIGURATION:")
    print("=" * 30)
    config.list_available_images()
    print()
    print(f"🤖 SAM Model: {config.CURRENT_SAM_MODEL}")
    print(f"🎯 YOLO Models: {config.CURRENT_YOLO_MODELS}")
    print(f"⚙️ Confidence Threshold: {config.YOLO_CONFIDENCE_THRESHOLD}")
    print(f"📊 WBF IoU Threshold: {config.WBF_IOU_THRESHOLD}")
    print()

if __name__ == "__main__":
    print("🎯 Smart Pipeline Configuration Demo")
    print("=" * 50)
    
    # Show menu
    while True:
        print("\nChoose an option:")
        print("1. 📋 Show current configuration")
        print("2. 🚀 Run pipeline with current settings")
        print("3. 🔄 Demo different configurations")
        print("4. 💡 Show how to add custom images")
        print("5. ⚙️ Quick config changes")
        print("6. ❌ Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            show_current_config()
            
        elif choice == "2":
            if config.validate_config():
                run_pipeline_with_config()
            else:
                print("❌ Configuration validation failed")
                
        elif choice == "3":
            demo_different_configurations()
            
        elif choice == "4":
            add_custom_image_example()
            
        elif choice == "5":
            print("\n⚙️ Quick Configuration Changes:")
            print("Available images:", list(config.AVAILABLE_IMAGES.keys()))
            img_choice = input("Enter image name (or press Enter to skip): ").strip()
            if img_choice:
                config.set_image(img_choice)
                
            print("Available SAM models:", list(config.SAM_MODELS.keys()))
            sam_choice = input("Enter SAM model (or press Enter to skip): ").strip()
            if sam_choice:
                config.set_sam_model(sam_choice)
                
            print("Configuration updated!")
            
        elif choice == "6":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-6.")