"""
Configuration file for YOLO+SAM Object Detection Pipeline
This file serves as a dependency to easily change settings and test different images.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TEST_OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

# Ensure output directory exists
TEST_OUTPUTS_DIR.mkdir(exist_ok=True)
(TEST_OUTPUTS_DIR / "visualizations").mkdir(exist_ok=True)
(TEST_OUTPUTS_DIR / "metrics").mkdir(exist_ok=True)
(TEST_OUTPUTS_DIR / "logs").mkdir(exist_ok=True)

# =============================================================================
# IMAGE CONFIGURATION
# =============================================================================

# Available test images - add your own images here
AVAILABLE_IMAGES = {
    "tools": DATA_DIR / "test_images" / "tools.png",
    "clip_test": DATA_DIR / "test_images" / "clip_test.png",
    "eng_tool": DATA_DIR / "test_images" / "eng_tool.jpg",
    "ht": DATA_DIR / "test_images" / "HT.jpg",
    # Add more images here as needed
    # "custom_image": DATA_DIR / "test_images" / "your_image.jpg",
    # "another_test": DATA_DIR / "test_images" / "another_image.png",
}

# Current image to process (change this to test different images)
CURRENT_IMAGE = "ht"  # Change this key to switch images

# Get the current image path
def get_current_image_path():
    """Get the path to the currently selected image."""
    if CURRENT_IMAGE not in AVAILABLE_IMAGES:
        raise ValueError(f"Image '{CURRENT_IMAGE}' not found. Available images: {list(AVAILABLE_IMAGES.keys())}")
    
    image_path = AVAILABLE_IMAGES[CURRENT_IMAGE]
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    return str(image_path)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# YOLO Models
YOLO_MODELS = {
    "yolov8n": MODELS_DIR / "yolo" / "yolov8n.pt",
    "yolo11s": MODELS_DIR / "yolo" / "yolo11s.pt", 
    "yolo11m": MODELS_DIR / "yolo" / "yolo11m.pt",
    "yolo11s_seg": MODELS_DIR / "yolo" / "yolo11s-seg.pt",
}

# SAM Models
SAM_MODELS = {
    "sam_vit_b": MODELS_DIR / "sam" / "sam_vit_b_01ec64.pth",
    "sam_vit_l": MODELS_DIR / "sam" / "sam_vit_l_0b3195.pth",
}

# Current model selection
CURRENT_SAM_MODEL = "sam_vit_l"  # Change to "sam_vit_b" for faster processing
CURRENT_YOLO_MODELS = ["yolov8n", "yolo11s", "yolo11m"]  # Models to use in ensemble

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

# Detection thresholds
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45

# SAM configuration
SAM_POINTS_PER_SIDE = 32
SAM_PRED_IOU_THRESH = 0.88
SAM_STABILITY_SCORE_THRESH = 0.95

# Weighted Boxes Fusion (WBF) settings
WBF_IOU_THRESHOLD = 0.55
WBF_SKIP_BOX_THRESHOLD = 0.01
WBF_SIGMA = 0.1

# Model weights for WBF (order: yolov8n, yolo11s, yolo11m)
MODEL_WEIGHTS = [1.0, 1.2, 1.5]

# Mask merging settings
MASK_MERGE_IOU_THRESHOLD = 0.7

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Output file naming
OUTPUT_PREFIX = CURRENT_IMAGE  # Prefix for output files

# Visualization settings
SHOW_CONFIDENCE = True
SHOW_LABELS = True
BOX_THICKNESS = 2
TEXT_SCALE = 0.6
MASK_ALPHA = 0.3

# Colors for different models (BGR format)
COLORS = {
    "yolov8n": (255, 0, 0),      # Blue
    "yolo11s": (0, 255, 0),      # Green  
    "yolo11m": (0, 0, 255),      # Red
    "sam": (255, 255, 0),        # Cyan
    "ensemble": (255, 0, 255),   # Magenta
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_path(model_type, model_name):
    """Get the path to a specific model."""
    if model_type == "yolo":
        return str(YOLO_MODELS.get(model_name, ""))
    elif model_type == "sam":
        return str(SAM_MODELS.get(model_name, ""))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_output_path(filename):
    """Get the full path for an output file."""
    return str(TEST_OUTPUTS_DIR / "visualizations" / f"{OUTPUT_PREFIX}_{filename}")

def list_available_images():
    """List all available images."""
    print("Available images:")
    for name, path in AVAILABLE_IMAGES.items():
        exists = "✅" if path.exists() else "❌"
        current = "👈 CURRENT" if name == CURRENT_IMAGE else ""
        print(f"  {exists} {name}: {path} {current}")

def validate_config():
    """Validate the current configuration."""
    errors = []
    
    # Check current image
    try:
        get_current_image_path()
    except (ValueError, FileNotFoundError) as e:
        errors.append(f"Image error: {e}")
    
    # Check SAM model
    sam_path = SAM_MODELS.get(CURRENT_SAM_MODEL)
    if not sam_path or not sam_path.exists():
        errors.append(f"SAM model not found: {sam_path}")
    
    # Check YOLO models
    for model_name in CURRENT_YOLO_MODELS:
        model_path = YOLO_MODELS.get(model_name)
        if not model_path or not model_path.exists():
            errors.append(f"YOLO model not found: {model_name} -> {model_path}")
    
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Configuration validated successfully!")
        return True

# =============================================================================
# QUICK SETUP FUNCTIONS
# =============================================================================

def set_image(image_name):
    """Quickly change the current image."""
    global CURRENT_IMAGE, OUTPUT_PREFIX
    if image_name not in AVAILABLE_IMAGES:
        print(f"❌ Image '{image_name}' not available. Use list_available_images() to see options.")
        return False
    
    CURRENT_IMAGE = image_name
    OUTPUT_PREFIX = image_name
    print(f"✅ Switched to image: {image_name}")
    return True

def set_sam_model(model_name):
    """Quickly change the SAM model."""
    global CURRENT_SAM_MODEL
    if model_name not in SAM_MODELS:
        print(f"❌ SAM model '{model_name}' not available. Options: {list(SAM_MODELS.keys())}")
        return False
    
    CURRENT_SAM_MODEL = model_name
    print(f"✅ Switched to SAM model: {model_name}")
    return True

def set_yolo_models(model_list):
    """Quickly change the YOLO models list."""
    global CURRENT_YOLO_MODELS
    invalid_models = [m for m in model_list if m not in YOLO_MODELS]
    if invalid_models:
        print(f"❌ Invalid YOLO models: {invalid_models}. Options: {list(YOLO_MODELS.keys())}")
        return False
    
    CURRENT_YOLO_MODELS = model_list
    print(f"✅ Switched to YOLO models: {model_list}")
    return True

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("🔧 YOLO+SAM Pipeline Configuration")
    print("=" * 50)
    
    # Show current configuration
    print(f"📸 Current Image: {CURRENT_IMAGE}")
    print(f"🤖 SAM Model: {CURRENT_SAM_MODEL}")
    print(f"🎯 YOLO Models: {CURRENT_YOLO_MODELS}")
    print()
    
    # List available images
    list_available_images()
    print()
    
    # Validate configuration
    validate_config()
    print()
    
    print("💡 Quick Usage Examples:")
    print("  import config")
    print("  config.set_image('tools')  # Switch to tools.png")
    print("  config.set_sam_model('sam_vit_b')  # Use smaller SAM model")
    print("  config.set_yolo_models(['yolo11s', 'yolo11m'])  # Use only these models")
    print("  config.validate_config()  # Check if everything is ready")