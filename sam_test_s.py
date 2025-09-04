import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import time

# Configuration
CHECKPOINT = "sam_vit_b_01ec64.pth"
VIDEO_PATH = "sam_test.mp4"

# FIX: Your checkpoint is actually ViT-B, not ViT-H despite the filename!
MODEL_TYPE = "vit_b"  # Match the actual checkpoint you have
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If you want to use ViT-H, download the correct checkpoint:
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

print(f"Using device: {DEVICE}")
print(f"Loading model: {MODEL_TYPE}")

# Load SAM model
start_time = time.time()
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(DEVICE)
sam.eval()

# Optimization 2: Configure mask generator for speed
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,        # Reduce from default 32 for speed
    pred_iou_thresh=0.8,       # Higher threshold = fewer masks
    stability_score_thresh=0.9, # Higher threshold = fewer masks
    crop_n_layers=0,           # Disable crop processing for speed
    crop_n_points_downscale_factor=1,
    min_mask_region_area=1000, # Ignore very small masks
)

load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open video: {VIDEO_PATH}")
    exit(1)

# Get video info
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {total_frames} frames at {fps} FPS")

ret, frame = cap.read()
if not ret:
    print("ERROR: Could not read first frame.")
    exit(2)

# Optimization 3: Resize frame for faster processing
original_height, original_width = frame.shape[:2]
print(f"Original frame size: {original_width}x{original_height}")

# Resize to smaller resolution for faster processing
max_dimension = 128  # Adjust this value
scale = min(max_dimension / original_width, max_dimension / original_height)
if scale < 1:
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    frame_resized = cv2.resize(frame, (new_width, new_height))
    print(f"Resized to: {new_width}x{new_height}")
else:
    frame_resized = frame
    print("No resizing needed")

# Generate masks with timing
print("Generating masks...")
mask_start = time.time()

with torch.no_grad():
    # CRITICAL: Clear all GPU memory before processing
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        # Check available GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - allocated_memory
        print(f"GPU Memory: {free_memory / 1024**3:.2f}GB free of {total_memory / 1024**3:.2f}GB total")
        
        # If low memory, force garbage collection
        if free_memory < 2 * 1024**3:  # Less than 2GB free
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    
    try:
        masks = mask_generator.generate(frame_resized)
    except RuntimeError as e:
        if "out of memory" in str(e) or "not enough memory" in str(e):
            print("MEMORY ERROR: Try reducing max_dimension to 256 or 128")
            print("Or use CPU processing (slower but uses system RAM)")
            raise e
        else:
            raise e

mask_time = time.time() - mask_start
print(f"Mask generation completed in {mask_time:.2f} seconds")

print(f"Number of masks found: {len(masks)}")
if masks:
    # Sort masks by area (largest first)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    print(f"Largest mask area: {masks[0]['area']}")
    print(f"Smallest mask area: {masks[-1]['area']}")
    
    # Show mask statistics
    areas = [m['area'] for m in masks]
    print(f"Average mask area: {sum(areas)/len(areas):.0f}")
else:
    print("No masks found.")

cap.release()

# Memory cleanup
if DEVICE == "cuda":
    torch.cuda.empty_cache()

print(f"Total processing time: {time.time() - start_time:.2f} seconds")