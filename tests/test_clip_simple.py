"""
Simple test to verify CLIP integration is working
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_clip_import():
    """Test if we can import and initialize CLIP"""
    print("🧪 Testing CLIP Import and Initialization")
    print("=" * 50)
    
    try:
        # Test transformers import
        print("📦 Testing transformers import...")
        from transformers import CLIPProcessor, CLIPModel
        print("✅ transformers imported successfully")
        
        # Test CLIP model loading
        print("🤖 Loading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ CLIP model loaded successfully")
        
        # Test basic inference
        print("🔍 Testing basic CLIP inference...")
        import torch
        from PIL import Image
        import numpy as np
        
        # Create a dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        texts = ["a hammer", "a screwdriver", "a wrench"]
        
        inputs = processor(text=texts, images=dummy_image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        print(f"✅ CLIP inference successful - probabilities: {probs.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLIP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dino_import():
    """Test if we can import and initialize DINOv2"""
    print("\n🧪 Testing DINOv2 Import and Initialization")
    print("=" * 50)
    
    try:
        # Test DINOv2 import
        print("📦 Testing DINOv2 import...")
        from transformers import AutoImageProcessor, AutoModel
        print("✅ AutoModel imported successfully")
        
        # Test DINOv2 model loading
        print("🤖 Loading DINOv2 model...")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        print("✅ DINOv2 model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ DINOv2 test failed: {e}")
        print("This is optional - proceeding with CLIP only...")
        return False

def main():
    print("🚀 CLIP + DINOv2 Integration Test")
    print("=" * 60)
    
    clip_ok = test_clip_import()
    dino_ok = test_dino_import()
    
    print("\n🎯 TEST SUMMARY:")
    print("=" * 30)
    print(f"CLIP Integration: {'✅ PASS' if clip_ok else '❌ FAIL'}")
    print(f"DINOv2 Integration: {'✅ PASS' if dino_ok else '⚠️ OPTIONAL'}")
    
    if clip_ok:
        print("\n🎉 Ready to run enhanced object classification!")
    else:
        print("\n🔧 Please install transformers library and check CLIP setup")

if __name__ == "__main__":
    main()