import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Allow very large whole-slide images without PIL raising a DecompressionBombWarning.
# This is common for whole-slide pathology images, which can be gigapixels.
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


def preprocess_image(img, size=256):
    """Resize to a fixed square and normalize to [-1, 1] for GAN training."""
    # Keep spatial shape consistent for the model input.
    img = img.resize((size, size), Image.BICUBIC)
    # Convert to float for normalization math.
    img = np.array(img).astype(np.float32)
    # Map [0, 255] to [-1, 1] (CycleGAN-style convention).
    img = (img / 127.5) - 1.0   # normalize to [-1, 1]
    return img


def extract_patches_pil(img, patch_size=256, stride=256):
    """Extract non-overlapping (or strided) patches from a PIL image."""
    # PIL uses (width, height) ordering.
    img_w, img_h = img.size
    patches = []

    # Slide a window across the image to collect tiles.
    # Using stride == patch_size yields non-overlapping patches.
    for top in range(0, img_h - patch_size + 1, stride):
        for left in range(0, img_w - patch_size + 1, stride):
            patch = img.crop((left, top, left + patch_size, top + patch_size))
            patches.append(patch)

    return patches


def save_patches(image_path, save_dir, patch_size=256):
    """Load an image, extract patches, and write them as PNGs to save_dir."""
    # Convert to RGB to guarantee 3-channel patches for CycleGAN-style datasets.
    img = Image.open(image_path).convert("RGB")
    # Extract tiles directly from the full-resolution slide.
    patches = extract_patches_pil(img, patch_size)

    base = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing Patch for image {base}")

    for i, patch in enumerate(patches):
        # Normalize then denormalize so saved patches are standard 8-bit images.
        patch = preprocess_image(patch, patch_size)
        # Convert back to 0-255 uint8 for saving.
        patch = ((patch + 1) * 127.5).astype(np.uint8)
        patch = Image.fromarray(patch)
        patch.save(os.path.join(save_dir, f"{base}_{i}.png"))


def main():
    # Dataset root directory.
    DATASET_DIR = "data\\E_Staining_DermaRepo\\H_E-Staining_dataset"
    print(f'Dataset Dir "{DATASET_DIR}"')

    # Source folders for unstained and stained images.
    UNSTAINED_DIR = DATASET_DIR + "\\Un_Stained"
    print(f'Unstained Images Dir "{UNSTAINED_DIR}"')

    STAINED_DIR = DATASET_DIR + "\\C_Stained"
    print(f'Stained Images Dir "{STAINED_DIR}"')

    # Inspect one sample unstained image for sanity check.
    # This visualization is optional and can be removed for headless runs.
    img_path = os.path.join(UNSTAINED_DIR, os.listdir(UNSTAINED_DIR)[0])

    with Image.open(img_path) as img:
        plt.figure(figsize=(10, 20))
        plt.imshow(img)
        plt.title("Sample Unstained Tissue Image")
        plt.axis("off")

    # CycleGAN-style output folders.
    os.makedirs(f"{DATASET_DIR}\\trainA", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}\\trainB", exist_ok=True)

    print(f"Saving Unstained Images Patch")
    for img_name in os.listdir(UNSTAINED_DIR):
        # Unstained images become domain A.
        save_patches(
            os.path.join(UNSTAINED_DIR, img_name),
            f"{DATASET_DIR}\\trainA"
        )
    print(f"Saved Unstained Images Patch")

    print(f"Saving Stained Images Patch")
    for img_name in os.listdir(STAINED_DIR):
        # Stained images should be saved to domain B (verify path if needed).
        save_patches(
            os.path.join(STAINED_DIR, img_name),
            f"{DATASET_DIR}\\trainA"
        )
    print(f"Saved Stained Images Patch")


if __name__ == "__main__":
    main()
