import torch
import torchvision
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import time
import os, time, gc, urllib.request
import torch, torchvision, numpy as np, cv2


CHECKPOINT = "sam_vit_b_01ec64.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"  # official
IMAGE_PATH = "clip_teste.png"  # put your image in the Colab working dir or update path
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def redownload_checkpoint(path=CHECKPOINT, url=CHECKPOINT_URL):
    print(f"Downloading SAM checkpoint from:\n  {url}")
    urllib.request.urlretrieve(url, path)
    size_mb = os.path.getsize(path) / (1024**2)
    print(f"Downloaded: {path} ({size_mb:.1f} MB)")

def ensure_checkpoint_ok(path=CHECKPOINT, url=CHECKPOINT_URL, retries=1):
    # If missing or suspiciously small, (re)download.
    if (not os.path.exists(path)) or (os.path.getsize(path) < 50 * 1024 * 1024):  # <50MB => definitely wrong
        redownload_checkpoint(path, url)
    # Try to instantiate the model to validate the file. If it fails with zip/stream error, re-download once.
    try:
        _ = sam_model_registry[MODEL_TYPE](checkpoint=path)  # constructor touches the file via torch.load
    except Exception as e:
        msg = str(e).lower()
        if ("pytorchstreamreader" in msg) or ("failed reading zip archive" in msg) or ("magic number" in msg):
            if retries > 0:
                print("Checkpoint looks corrupted. Re-downloading once...")
                redownload_checkpoint(path, url)
                return ensure_checkpoint_ok(path, url, retries=retries-1)
        raise

# =========================
# 3) Validate / load model
# =========================
print(f"Using device: {DEVICE}")
print(f"Loading model: {MODEL_TYPE}")

start_time = time.time()
ensure_checkpoint_ok(CHECKPOINT, CHECKPOINT_URL)

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(DEVICE).eval()

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=8,                # faster
    pred_iou_thresh=0.95,             # fewer masks
    stability_score_thresh=0.95,      # fewer masks
    crop_n_layers=0,                  # no multi-scale crops
    crop_n_points_downscale_factor=1,
    min_mask_region_area=2000,        # ignore small masks
)

print(f"Model ready in {time.time() - start_time:.2f} s")

# =========================
# 4) Load & prep image
# =========================
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(
        f"ERROR: Cannot read image: {IMAGE_PATH}\n"
        "➡️ Upload your image to the Colab working directory, or set IMAGE_PATH to the correct path."
    )

# SAM expects RGB np.uint8 (H,W,3)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

H, W = image_rgb.shape[:2]
print(f"Image size: {W}x{H}")

# Aggressive resize for speed (adjust if you want better quality)
max_dimension = 1024
scale = min(max_dimension / W, max_dimension / H)
if scale < 1:
    new_w, new_h = int(W * scale), int(H * scale)
    image_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"Resized to: {new_w}x{new_h}")
else:
    image_resized = image_rgb.copy()
    print("No resizing needed")

# =========================
# 5) Generate masks
# =========================
print("Generating masks...")
mask_start = time.time()
with torch.no_grad():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
    try:
        masks = mask_generator.generate(image_resized)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("MEMORY ERROR: Try reducing max_dimension to 32 or run on CPU (slower).")
        raise

print(f"Mask generation completed in {time.time() - mask_start:.2f} s")
print(f"Number of masks found: {len(masks)}")

if masks:
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    areas = [m['area'] for m in masks]
    print(f"Largest mask area: {masks[0]['area']}")
    print(f"Smallest mask area: {masks[-1]['area']}")
    print(f"Average mask area: {sum(areas)/len(areas):.0f}")
else:
    print("No masks found.")

# =========================
# 6) Visualize & save
# =========================
overlay = image_resized.copy()
max_masks_to_show = min(5, len(masks))
fixed_colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255)]

for i, m in enumerate(masks[:max_masks_to_show]):
    color = np.array(fixed_colors[i % len(fixed_colors)], dtype=np.uint8)
    overlay[m['segmentation']] = color

# Convert back to BGR for OpenCV save
overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
out_path = "Capture_masked.png"
cv2.imwrite(out_path, overlay_bgr)
print(f"Masked image saved to {out_path}")
print("Skipping display window; check the saved file.")

# =========================
# 7) Cleanup
# =========================
del sam, mask_generator, masks, overlay, overlay_bgr
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    gc.collect()
print(f"Total processing time: {time.time() - start_time:.2f} s")
