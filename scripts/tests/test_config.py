"""
Test Configuration and Setup
"""
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Test image paths
TEST_IMAGE_PATH = PROJECT_ROOT / "src" / "tools.png"
SAM_CHECKPOINT_PATH = PROJECT_ROOT / "sam_vit_b_01ec64.pth"

# Expected test results (ground truth for validation)
EXPECTED_RESULTS = {
    "min_objects": 1,      # Minimum objects to detect
    "max_objects": 10,     # Maximum reasonable objects
    "tool_classes": [      # Expected tool classes
        "hammer", "screwdriver", "pliers", "wrench", 
        "drill", "saw", "chisel", "file"
    ],
    "min_grasp_quality": 0.3,  # Minimum acceptable grasp quality
    "min_relations": 0,        # Minimum spatial relations
}

# Test output paths
OUTPUT_PATHS = {
    "detection_vis": "tests/detection_output.jpg",
    "grasp_vis": "tests/grasp_output.jpg",
    "results_json": "tests/test_results.json"
}

def setup_test_environment():
    """Setup test environment and check prerequisites"""
    import os
    from pathlib import Path
    
    # Create test output directory
    Path("tests").mkdir(exist_ok=True)
    Path("src").mkdir(exist_ok=True)
    
    # Check if test image exists
    if not TEST_IMAGE_PATH.exists():
        print(f"⚠️  Test image not found: {TEST_IMAGE_PATH}")
        print("   Please add a test image with tools to src/tools.png")
        return False
    
    # Check if SAM checkpoint exists
    if not SAM_CHECKPOINT_PATH.exists():
        print(f"⚠️  SAM checkpoint not found: {SAM_CHECKPOINT_PATH}")
        print("   Please download sam_vit_b_01ec64.pth")
        return False
    
    print("✅ Test environment ready")
    return True
