import os
import time
import glob
import torch
import numpy as np
from PIL import Image
import yaml # For loading the YML config file (pip install pyyaml)
from collections import OrderedDict

# --- BasicSR Specific Imports ---
# Assuming 'basicsr' is installed and accessible (e.g., via python setup.py develop)
try:
    from basicsr.archs.span_arch import SPAN # Import the specific model architecture
    from basicsr.utils.img_util import img2tensor, tensor2img # BasicSR image/tensor conversion utils
except ImportError as e:
    print(f"Error importing BasicSR components: {e}")
    print("Please ensure BasicSR is installed correctly (e.g., run 'python setup.py develop')")
    print("And that the SPAN architecture file exists (e.g., basicsr/models/archs/span_arch.py)")
    exit()

# --- Configuration ---

# Path to your SPAN configuration file
CONFIG_PATH = 'options/test/EDSR/test_SPAN_48x4.yml' #<--- ADJUST THIS PATH if needed

# Overrides or variables not in YML
OUTPUT_BASE_DIR = 'results_SPAN' # <--- Base directory for SPAN results
MODEL_NAME_FOR_OUTPUT = 'SPANX4_CH48' # <--- Name for the subfolder in OUTPUT_BASE_DIR
WARMUP_COUNT = 5 # Increase warm-up slightly for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- Load Configuration from YML ---
try:
    with open(CONFIG_PATH, 'r') as f:
        opt = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ERROR: Configuration file not found at '{CONFIG_PATH}'")
    exit()
except Exception as e:
    print(f"ERROR: Could not read or parse YML file '{CONFIG_PATH}': {e}")
    exit()

# Extract necessary parameters from loaded options (opt)
# Datasets
INPUT_DIR = opt['datasets']['test_1'].get('dataroot_lq', None)
if INPUT_DIR is None:
    print("ERROR: 'dataroot_lq' not found in the configuration file under datasets -> test_1.")
    exit()

# Network settings (ensure defaults match SPAN if not in YML)
SCALE = opt['network_g'].get('upscale', 4)
NUM_IN_CH = opt['network_g'].get('num_in_ch', 3)
NUM_OUT_CH = opt['network_g'].get('num_out_ch', 3)
FEATURE_CHANNELS = opt['network_g'].get('feature_channels', 52) # Get feature channels from YML
IMG_RANGE = opt['network_g'].get('img_range', 255.)
RGB_MEAN = opt['network_g'].get('rgb_mean', [0.4488, 0.4371, 0.4040])

# Path settings
CHECKPOINT_PATH = opt['path'].get('pretrain_network_g', None)
STRICT_LOAD = opt['path'].get('strict_load_g', True)
if CHECKPOINT_PATH is None:
    print("ERROR: 'pretrain_network_g' path not found in the configuration file.")
    exit()

# Validation settings
SAVE_IMG = opt['val'].get('save_img', True)
CROP_BORDER = opt['val'].get('crop_border', SCALE) # Use scale as default crop if not specified in val block

print("--- Configuration Loaded ---")
print(f"Input Directory: {INPUT_DIR}")
print(f"Output Base Directory: {OUTPUT_BASE_DIR}")
print(f"Model Checkpoint: {CHECKPOINT_PATH}")
print(f"Scale Factor: {SCALE}")
print(f"Feature Channels: {FEATURE_CHANNELS}")
print(f"Device: {DEVICE}")
print(f"IMG_RANGE: {IMG_RANGE}")
print(f"RGB_MEAN: {RGB_MEAN}")
print(f"Save Images: {SAVE_IMG}")
print("-----------------------------")


# --- Helper Functions ---

def load_pretrained_model(model, checkpoint_path, strict=True, device='cpu'):
    """Loads pretrained weights into a model."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    load_net = torch.load(checkpoint_path, map_location=device)

    # Determine the key for the state dictionary
    param_key = 'params_ema' # Often used in BasicSR testing
    if param_key not in load_net:
        param_key = 'params' # Fallback for regular training checkpoints
        if param_key not in load_net:
             print("Warning: Neither 'params_ema' nor 'params' found in checkpoint. Trying to load root.")
             param_key = None # Load the whole dict

    if param_key is not None:
        load_net = load_net[param_key]
        print(f"Using state dict key: '{param_key}'")
    else:
        print("Using root of the checkpoint as state dict.")


    # Remove 'module.' prefix if it exists (from DDP training)
    cleaned_load_net = OrderedDict()
    has_module_prefix = False
    for k, v in load_net.items():
        if k.startswith('module.'):
            has_module_prefix = True
            cleaned_load_net[k[7:]] = v
        else:
            cleaned_load_net[k] = v
    if has_module_prefix:
        print("Removed 'module.' prefix from state dict keys.")
    load_net = cleaned_load_net


    model.load_state_dict(load_net, strict=strict)
    print("Checkpoint loaded successfully.")
    return model

# --- Main Processing Logic ---

# Sanity check input directory
if not os.path.isdir(INPUT_DIR):
    print(f"Error: Input directory '{INPUT_DIR}' not found.")
    exit()

# Find input images
image_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.jpg'))) + \
              sorted(glob.glob(os.path.join(INPUT_DIR, '*.png'))) # Add png support

if not image_files:
    print(f"Error: No '.jpg' or '.png' images found in '{INPUT_DIR}'.")
    exit()

print(f"Found {len(image_files)} images in '{INPUT_DIR}'.")

# --- Load Model ---
print("Building SPAN model...")
try:
    # Pass config values to the constructor to ensure consistency
    model = SPAN(
        upscale=SCALE,
        num_in_ch=NUM_IN_CH,
        num_out_ch=NUM_OUT_CH,
        feature_channels=FEATURE_CHANNELS,
        img_range=IMG_RANGE,         # Pass loaded config value
        rgb_mean=RGB_MEAN            # Pass loaded config value
    )
    model.eval() # Set model to evaluation mode
    model = load_pretrained_model(model, CHECKPOINT_PATH, strict=STRICT_LOAD, device='cpu') # Load to CPU first
    model = model.to(DEVICE) # Move model to target device
    print("Model built and loaded successfully.")
except FileNotFoundError:
    print(f"Error: Checkpoint file not found at '{CHECKPOINT_PATH}'.")
    exit()
except Exception as e:
    print(f"Error building or loading model: {e}")
    exit()


# --- Prepare Output Directory ---
output_dir = os.path.join(OUTPUT_BASE_DIR, MODEL_NAME_FOR_OUTPUT, f"scale_{SCALE}")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory for this run: '{output_dir}'")


# --- Prepare Normalization Tensors ---
# Create mean tensor on the correct device and shape (1, C, 1, 1) for broadcasting
mean_tensor = torch.tensor(RGB_MEAN).view(1, NUM_IN_CH, 1, 1).to(DEVICE)

# --- Reset Timers and Process Images ---
processing_times = []
total_start_time_script = time.time()

for i, img_path in enumerate(image_files):
    img_filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, img_filename)
    print(f"\n[{MODEL_NAME_FOR_OUTPUT} | Scale {SCALE}x] Processing image {i+1}/{len(image_files)}: '{img_filename}'")

    # --- Load Input Image ---
    try:
        # Load with PIL (RGB)
        img_pil = Image.open(img_path).convert('RGB')
        # Convert to NumPy HWC, uint8
        img_np = np.array(img_pil)
    except FileNotFoundError:
        print(f"  Error: Input image file not found. Skipping.")
        continue
    except Exception as e:
        print(f"  Error loading image: {e}. Skipping.")
        continue

    # --- Preprocess Image ---
    # Use BasicSR's img2tensor: Converts HWC uint8 [0,255] (RGB) -> NCHW float32 [0,1] (RGB)
    input_tensor = img2tensor(img_np, bgr2rgb=False, float32=True).unsqueeze(0).to(DEVICE) / 255.0
    # --- Run Super-Resolution (Inference) ---
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize(DEVICE) # Ensure previous GPU work is done
    start_time_img = time.time()

    try:
        with torch.no_grad(): # Disable gradient calculation for efficiency
            output_tensor_raw = model(input_tensor) # Get the raw model output
        # >>> Apply Inverse Normalization/Scaling <<<
        # This reverses the (x - mean) * img_range done inside the model's forward pass

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize(DEVICE) # IMPORTANT: Wait for GPU work to finish before stopping timer

    except Exception as e:
        print(f"  Error during model inference: {e}")
        if "CUDA out of memory" in str(e):
             print("  Suggestion: CUDA OOM Error likely occurred. Try processing smaller images or using a GPU with more VRAM.")
        continue # Skip timing and saving this image

    end_time_img = time.time()
    duration_img = end_time_img - start_time_img

    # --- Timing and Warm-up ---
    if i >= WARMUP_COUNT:
        processing_times.append(duration_img)
        print(f"  Processing time: {duration_img:.4f} seconds")
    else:
        print(f"  Warm-up run {i+1}/{WARMUP_COUNT}. Time: {duration_img:.4f} seconds (not counted)")

    # --- Postprocess Image ---
    # Use BasicSR's tensor2img: Converts NCHW float32 [0,1] (RGB) -> HWC uint8 [0,255] (RGB)
    # Pass the DENORMALIZED tensor [0, 1] to tensor2img
    try:
        # Ensure the denormalized tensor is clamped to [0, 1] before converting to uint8
        # output_tensor_denorm_clamped = torch.clamp(output_tensor_denorm, 0.0, 1.0)
        output_img_np = tensor2img(output_tensor_raw, rgb2bgr=False, out_type=np.uint8, min_max=(0, 1))
    except Exception as e:
        print(f"  Error during post-processing (tensor to image): {e}")
        continue

    # --- Save Output Image ---
    if SAVE_IMG:
        try:
            output_image_pil = Image.fromarray(output_img_np) # Expects HWC uint8 RGB
            output_image_pil.save(output_path)
            # print(f"  Output image saved to '{output_path}'") # Less verbose
        except Exception as e:
            print(f"  Error saving image '{output_path}': {e}")

# --- Print Final Summary ---
total_end_time_script = time.time()
total_duration_script = total_end_time_script - total_start_time_script
print("-" * 60)
print(f"Finished processing for Model: {MODEL_NAME_FOR_OUTPUT}, Scale: {SCALE}x")
print(f"Total script time (incl. warm-up, loading): {total_duration_script:.2f} seconds")

if processing_times:
    avg_time = sum(processing_times) / len(processing_times)
    print(f"Average processing time per image (after {WARMUP_COUNT} warm-ups): {avg_time:.4f} seconds")
    images_per_sec = 1.0 / avg_time if avg_time > 0 else float('inf')
    print(f"Average throughput: {images_per_sec:.2f} images/second")
else:
    if len(image_files) > WARMUP_COUNT:
         print("No images were processed successfully after the warm-up phase to calculate average time.")
    else:
         print(f"Not enough images ({len(image_files)}) processed for warm-up ({WARMUP_COUNT}) and timing.")
print("-" * 60)

print("\nStandalone BasicSR/SPAN inference complete.")