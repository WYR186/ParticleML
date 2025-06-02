# This notebook is intended to run on the Kaggle platform

# %% [code]
# Cell 1: Install all required packages (including imagecodecs for LZW support)
# ------------------------------------------------------------------------------

# We need the following libraries:
#  - imagecodecs and tifffile    → to read/write LZW‐compressed TIFFs
#  - scikit-image, dask, zarr    → for image preprocessing (optional but often useful)
#  - napari, stardist, csbdeep    → for StarDist inference
#  - tensorflow (or pytorch)      → backend for StarDist
#  - pandas                        → to save centroids as a CSV
#  - matplotlib                    → for visualization

!pip install --quiet \
    imagecodecs \
    tifffile \
    scikit-image \
    dask \
    zarr \
    napari \
    stardist \
    csbdeep \
    tensorflow \
    pandas \
    matplotlib

print("✅ All packages installed.")


# %% [code]
# Cell 2: Convert every .tif under /kaggle/input/particle-segmentation to normalized uint16 grayscale
# ----------------------------------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import tifffile               # tifffile + imagecodecs handles LZW‐compressed TIFF
from skimage.color import rgb2gray
from skimage import exposure

# 1) Define the INPUT_ROOT (where the original colored TIFFs live)
#    and the output folder (where grayscale files will be saved)
INPUT_ROOT = Path("/kaggle/input/particle-segmentation/Particle Segmentation")
GRAY_ROOT  = Path("/kaggle/working/gray-images")

def convert_to_grayscale_and_save(src_path: Path, dst_root: Path):
    """
    Read a TIFF (possibly RGB, RGBA, or single‐channel), convert it to a float in [0..1],
    optionally apply a 2–98% contrast stretch, then save as uint16 under dst_root,
    preserving the same subfolder structure.
    """
    # Recreate the same subfolder structure under dst_root
    relative_path = src_path.relative_to(INPUT_ROOT)
    dst_path = dst_root / relative_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Read the original TIFF with tifffile
    image = tifffile.imread(str(src_path))

    # Step 2: Convert to float32 grayscale in [0..1]
    if image.ndim == 3 and image.shape[2] in (3, 4):
        # If the image has 3 or 4 channels (RGB or RGBA), drop the alpha channel if present
        rgb = image[..., :3] if image.shape[2] == 4 else image
        gray = rgb2gray(rgb)           # returns float64 in [0..1]
        gray = gray.astype(np.float32)
    else:
        # Single channel: normalize to [0..1]
        arr = image.astype(np.float32)
        gray = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # Step 3: Optional 2–98% contrast stretch
    p2, p98 = np.percentile(gray, (2, 98))
    gray_rescaled = exposure.rescale_intensity(gray, in_range=(p2, p98))
    gray_rescaled = gray_rescaled.astype(np.float32)

    # Step 4: Save as uint16 (scale [0..1] → [0..65535])
    uint16_image = (gray_rescaled * 65535.0).round().astype(np.uint16)
    tifffile.imwrite(str(dst_path), uint16_image)

# 2) Gather all .tif files under INPUT_ROOT
all_tif_paths = list(INPUT_ROOT.rglob("*.tif"))
print(f"Found {len(all_tif_paths)} TIFF files under INPUT_ROOT.")

# 3) Convert all of them, printing progress every 50 files
for idx, source_path in enumerate(all_tif_paths, start=1):
    convert_to_grayscale_and_save(source_path, GRAY_ROOT)
    if idx % 50 == 0 or idx == len(all_tif_paths):
        print(f"  → Converted {idx}/{len(all_tif_paths)}")

print("✅ All TIFF images have been converted to grayscale and saved.")


# %% [code]
# Cell 2.5: Create a ZIP archive of /kaggle/working/gray-images
# ----------------------------------------------------------------

import shutil
from pathlib import Path

# 1) Ensure the gray-images folder exists
GRAY_ROOT = Path("/kaggle/working/gray-images")
if not GRAY_ROOT.exists():
    raise FileNotFoundError(f"Could not find the grayscale directory at {GRAY_ROOT}")

# 2) Define the destination ZIP filename (placed under /kaggle/working/)
zip_output_path = Path("/kaggle/working/gray-images.zip")

# 3) Create the archive – this includes every file & subfolder under gray-images
print(f"🗜️  Creating ZIP archive at {zip_output_path} ...")
shutil.make_archive(
    base_name=str(zip_output_path.with_suffix("")),  # omit “.zip”: make_archive adds it
    format="zip",
    root_dir=str(GRAY_ROOT.parent),                 # “/kaggle/working”
    base_dir=GRAY_ROOT.name                          # just “gray-images”
)

print("✅ Created ZIP archive of all grayscale images:")
print(f"   → {zip_output_path}")


# %% [code]
# Cell 3.0: Check GPU availability
# ----------------------------------

import tensorflow as tf

gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("✔ TensorFlow detects the following GPU devices:")
    for device in gpu_devices:
        print("   ", device)
else:
    print("⚠ No GPU found; TensorFlow will run on CPU.")

# Uncomment the following line to see every operation’s device placement:
# tf.debugging.set_log_device_placement(True)


# %% [code]
# Cell 3.1: Load StarDist model and run inference on a single example (195 image)
# --------------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from stardist.models import Config2D, StarDist2D

# 0) Verify GPU availability
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("✔ TensorFlow detects the following GPU devices:")
    for device in gpu_devices:
        print("   ", device)
else:
    print("⚠ No GPU found; inference will run on CPU.")

# 1) Recreate the exact same Config2D used during training
cfg = Config2D(
    axes="YX",    # same axes ordering as training
    n_rays=32,    # same number of rays
    grid=(1, 1)   # same grid
)

# 2) Instantiate a new StarDist2D model (without reading config.json from disk)
model = StarDist2D(cfg, name="stardist_run1", basedir=None)

# 3) Load your uploaded H5 weights
WEIGHTS_PATH = Path("/kaggle/input/stardist-run1/stardist_run1_weights_manual.h5")
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Could not find weights at {WEIGHTS_PATH}")
model.load_weights(str(WEIGHTS_PATH))
print(f"✔ Loaded weights from: {WEIGHTS_PATH.name}")

# 4) Specify the “195” TIFF under gray-images
EXAMPLE_TIF = Path("/kaggle/working/gray-images/Cement/20 min/IMG_20240614-113421-195.tif")
if not EXAMPLE_TIF.exists():
    raise FileNotFoundError(f"Could not find example TIFF at {EXAMPLE_TIF}")
print("Example image:", EXAMPLE_TIF.name)

# 5) Read and normalize to [0,1]
image_uint16 = tifffile.imread(str(EXAMPLE_TIF)).astype(np.float32)
image_norm   = image_uint16 / 65535.0  # float32 in [0,1]

# 6) Perform “big” (tiled) inference with custom thresholds
#    - prob_thresh=0.50  → only pixels with ≥50% object probability considered
#    - nms_thresh=0.25   → merge/suppress overlapping small detections
#    - block_size=(512,512), min_overlap=(48,48)

PROB_THRESH  = 0.50
NMS_THRESH   = 0.25
BLOCK_SIZE   = (512, 512)
MIN_OVERLAP  = (48, 48)

labels, details = model.predict_instances_big(
    image_norm,
    axes="YX",               # REQUIRED
    prob_thresh=PROB_THRESH,
    nms_thresh=NMS_THRESH,
    block_size=BLOCK_SIZE,
    n_tiles=None,            # let StarDist compute tile layout automatically
    min_overlap=MIN_OVERLAP
)

num_detected = len(details["coord"])
print(f"✔ Detected {num_detected} objects (prob ≥ {PROB_THRESH}, nms ≥ {NMS_THRESH}).")

# 7) Save: label mask, centroid CSV, and preview PNG
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(exist_ok=True)

# 7a) Save label mask (each region gets a unique uint16 label)
label_tiff = OUTPUT_DIR / f"{EXAMPLE_TIF.stem}_labels.tif"
tifffile.imwrite(str(label_tiff), labels.astype(np.uint16))
print(f"✔ Saved label mask to: {label_tiff.name}")

# 7b) Extract centroids from “ray coords” array and save to CSV
#    details["coord"] has shape (N, 2, n_rays). We take the mean over axis=2.
raw_coords = np.array(details["coord"], dtype=np.float32)  # shape = (N, 2, 32)
centroids = raw_coords.mean(axis=2).astype(np.int32)       # shape = (N, 2)
df = pd.DataFrame(centroids, columns=["y", "x"])
csv_path = OUTPUT_DIR / f"{EXAMPLE_TIF.stem}_coords.csv"
df.to_csv(str(csv_path), index=False)
print(f"✔ Saved centroids to: {csv_path.name}  (shape={centroids.shape})")

# 7c) Save a preview PNG with red contours overlaid on the grayscale image
plt.figure(figsize=(8, 8))
plt.imshow(image_norm, cmap="gray")
plt.contour(labels > 0, levels=[0.5], colors="r", linewidths=0.5)
plt.title(f"Contours on {EXAMPLE_TIF.name}  (prob={PROB_THRESH}, nms={NMS_THRESH})")
plt.axis("off")
preview_png = OUTPUT_DIR / f"{EXAMPLE_TIF.stem}_preview.png"
plt.savefig(str(preview_png), bbox_inches="tight", pad_inches=0.02, dpi=150)
print(f"✔ Saved preview PNG to: {preview_png.name}")
plt.show()


# %% [code]
# Cell 3.5: Refined grid search around prob=[0.40–0.50] and nms=[0.20–0.35]
# ---------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import tifffile
import tensorflow as tf

from stardist.models import Config2D, StarDist2D

# 0) Check GPU
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("✔ TensorFlow detects the following GPU devices:")
    for device in gpu_devices:
        print("   ", device)
else:
    print("⚠ No GPU detected; inference will run on CPU.")

# 1) Rebuild Config2D and reload the StarDist model
cfg = Config2D(axes="YX", n_rays=32, grid=(1, 1))
model = StarDist2D(cfg, name="stardist_run1", basedir=None)

WEIGHTS_PATH = Path("/kaggle/input/stardist-run1/stardist_run1_weights_manual.h5")
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Could not find weights at {WEIGHTS_PATH}")
model.load_weights(str(WEIGHTS_PATH))
print(f"✔ Loaded weights from: {WEIGHTS_PATH.name}\n")

# 2) Load & normalize the “195” image
EXAMPLE_TIF = Path("/kaggle/working/gray-images/Cement/20 min/IMG_20240614-113421-195.tif")
if not EXAMPLE_TIF.exists():
    raise FileNotFoundError(f"Could not find example TIFF at {EXAMPLE_TIF}")
print("Example image:", EXAMPLE_TIF.name)

image_uint16 = tifffile.imread(str(EXAMPLE_TIF)).astype(np.float32)
image_norm   = image_uint16 / 65535.0  # float32 in [0,1]
H, W         = image_norm.shape
full_area    = H * W

# 3) Define target & tiling parameters
TARGET_COUNT = 420    # approximate “ground truth” count
TOLERANCE    = 5      # ±5 is acceptable
MAX_COUNT    = 1000   # skip any estimate >1000 immediately

# Narrow search grid around likely thresholds
prob_list = [0.40, 0.45, 0.50]
nms_list  = [0.20, 0.25, 0.30, 0.35]

best_params = None
best_diff   = np.inf
best_count  = None

print(f"\n→ Starting refined grid search:")
print(f"  prob_thresh ∈ {prob_list},  nms_thresh ∈ {nms_list}\n")

# 4) Prepare a central 512×512 crop for fast estimation
crop_size = 512
y0 = max(0, (H // 2) - (crop_size // 2))
x0 = max(0, (W // 2) - (crop_size // 2))
crop_patch = image_norm[y0 : y0 + crop_size, x0 : x0 + crop_size]
crop_area  = crop_size * crop_size
area_ratio = full_area / crop_area

# 5) Perform grid search
for p in prob_list:
    if best_diff <= TOLERANCE:
        break

    for n in nms_list:
        print(f"Testing prob={p:.2f}, nms={n:.2f} …", end=" ")

        # 5a) Quick patch inference
        labels_crop, details_crop = model.predict_instances(
            crop_patch,
            prob_thresh=float(p),
            nms_thresh=float(n),
            axes="YX"
        )
        count_crop     = len(details_crop["coord"])
        est_full_count = int(count_crop * area_ratio)

        print(f"(crop={count_crop}, est_full={est_full_count})", end=" ")

        # 5b) Skip if estimated full-image count > MAX_COUNT
        if est_full_count > MAX_COUNT:
            print("→ skip (est_full > 1000).")
            continue

        # 5c) Full tiled inference
        labels_full, details_full = model.predict_instances_big(
            image_norm,
            axes="YX",
            prob_thresh=float(p),
            nms_thresh=float(n),
            block_size=(512, 512),
            n_tiles=None,
            min_overlap=(48, 48)
        )
        count_full = len(details_full["coord"])
        print(f"→ full={count_full}", end=" ")

        # 5d) Skip if full count > MAX_COUNT
        if count_full > MAX_COUNT:
            print("→ skip (full > 1000).")
            continue

        # 5e) Update best parameters if closer to target
        diff = abs(count_full - TARGET_COUNT)
        if diff < best_diff:
            best_diff   = diff
            best_params = (p, n)
            best_count  = count_full

        # 5f) Stop search if within tolerance
        if diff <= TOLERANCE:
            print(f"→ acceptable!  prob={p:.2f}, nms={n:.2f}, count={count_full}\n")
            break

        print("")  # newline for readability

# 6) Report final results
if best_params is None:
    print(
        "\n❌ None of the combinations produced ≤ 1000 detections on the full image. "
        "You may need to lower prob_thresh below 0.40 or adjust the crop strategy."
    )
else:
    print(
        f"\n✔ Best parameters found:  prob_thresh = {best_params[0]:.2f}, "
        f"nms_thresh = {best_params[1]:.2f},  detected_count = {best_count}, "
        f"diff = {best_diff}"
    )


# %% [code]
# Cell 4: Load a StarDist model from a ZIP file or use a pretrained model
# ------------------------------------------------------------------------

from stardist.models import StarDist2D
import zipfile
from pathlib import Path

# Option A: Load your own exported model ZIP
MODEL_ZIP = Path("/kaggle/input/particle-segmentation/your_model.zip")  # Update to your actual ZIP filename
MODEL_DIR = Path("/kaggle/working/my_stardist_model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(str(MODEL_ZIP), 'r') as zf:
    zf.extractall(path=str(MODEL_DIR))
print("Model ZIP extracted to:", MODEL_DIR)

# Create the StarDist2D instance by pointing 'basedir' to the extracted folder
model = StarDist2D(None, name=None, basedir=str(MODEL_DIR))
print("Loaded StarDist model from ZIP:", model.name)

# Option B (optional): If you do not have your own model, use an official pretrained model
# model = StarDist2D.from_pretrained('2D_versatile_fluo')
# print("Using pretrained model:", model.name)


# %% [code]
# Cell 5: Run inference on all grayscale images and save masks
# -------------------------------------------------------------

from csbdeep.utils import normalize
from skimage import io

def run_stardist_inference(src_gray: Path, stardist_model: StarDist2D, dst_root: Path):
    """
    Run StarDist inference on a single grayscale TIFF (src_gray),
    then save the instance label map under dst_root, preserving subfolders.
    """
    # Read the image as float32
    img = io.imread(str(src_gray)).astype(np.float32)
    # Normalize using percentiles [1..99.8], same approach used in training pipeline
    img_norm = normalize(img, 1, 99.8, axis=(0, 1))

    labels, _ = stardist_model.predict_instances(
        img_norm,
        n_tiles=None,
        show_tile_progress=False
    )

    # Preserve relative path under GRAY_ROOT
    relative_path = src_gray.relative_to(GRAY_ROOT)
    dst_path = dst_root / relative_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the label map as uint16 TIFF
    io.imsave(str(dst_path), labels.astype(np.uint16))
    return dst_path

# List all grayscale TIFFs produced earlier
all_gray_images = list(GRAY_ROOT.rglob("*.tif"))
print(f"Found {len(all_gray_images)} grayscale images. Starting inference...")

# Define where to save the predicted masks
MASK_ROOT = Path("/kaggle/working/predicted-masks")

for idx, gray_path in enumerate(all_gray_images, start=1):
    mask_path = run_stardist_inference(gray_path, model, MASK_ROOT)
    if idx % 50 == 0 or idx == len(all_gray_images):
        print(f"  Completed {idx}/{len(all_gray_images)}: {mask_path.name}")

print("✅ All inference done. Masks are saved under:", MASK_ROOT)


# %% [code]
# Cell 6: Visualize a sample result by overlaying mask on grayscale image
# -------------------------------------------------------------------------

import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage import io

# Select an example grayscale image and its corresponding mask
test_gray = all_gray_images[0]
test_mask = MASK_ROOT / test_gray.relative_to(GRAY_ROOT)

# Read the grayscale image and the mask
img_gray = io.imread(str(test_gray))
mask = io.imread(str(test_mask))

# Create an overlay of mask contours on the grayscale image
overlay = label2rgb(mask, image=img_gray, bg_label=0, alpha=0.3, kind='overlay')

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Grayscale")
plt.imshow(img_gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Instance Mask")
plt.imshow(mask, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis('off')

plt.tight_layout()
plt.show()
