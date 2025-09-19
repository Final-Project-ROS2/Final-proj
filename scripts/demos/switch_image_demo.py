"""
Simple demo showing how to switch images using the config system
"""

import config

# Show current configuration
print("🔧 Current Configuration:")
config.list_available_images()
print()

# Switch to clip_test image
print("🔄 Switching to clip_test image...")
config.set_image('clip_test')
print()

# Show updated configuration
print("✅ Updated Configuration:")
config.list_available_images()
print()

# Switch to faster SAM model for demo
print("⚡ Switching to faster SAM model...")
config.set_sam_model('sam_vit_b')
print()

# Show final config
print("📋 Final Configuration:")
print(f"   📸 Image: {config.CURRENT_IMAGE}")
print(f"   🤖 SAM Model: {config.CURRENT_SAM_MODEL}")
print(f"   🎯 YOLO Models: {config.CURRENT_YOLO_MODELS}")
print()

print("💡 Now you can run the pipeline with:")
print("   python tests/configurable_smart_pipeline.py")
print()
print("📁 Output files will be prefixed with 'clip_test_'")