"""
Inference script for the CycleGAN histology stain / unstain translator.

This repository contains one model family: a ResNet-based CycleGAN
generator pair saved in the checkpoint as ``G_AB`` and ``G_BA``. The script
loads that checkpoint, splits large images into 256x256 patches, runs each
patch through the appropriate generator, and blends overlapping patches back
into a full-resolution image.
"""

import math
import os

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torchvision.utils import save_image

from generator import ResnetGenerator

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _ensure_parent_dir(path: str):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _load_checkpoint(checkpoint_path: str, device: str):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_model(checkpoint_path: str, device: str = "cpu"):
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required")

    checkpoint = _load_checkpoint(checkpoint_path, device)
    if "G_AB" not in checkpoint or "G_BA" not in checkpoint:
        raise KeyError("Checkpoint must contain 'G_AB' and 'G_BA' state dicts.")

    G_AB = ResnetGenerator().to(device)
    G_BA = ResnetGenerator().to(device)
    G_AB.load_state_dict(checkpoint["G_AB"])
    G_BA.load_state_dict(checkpoint["G_BA"])
    G_AB.eval()
    G_BA.eval()
    return G_AB, G_BA


def stain_image(image, model, device="cpu"):
    with torch.inference_mode():
        return model(image.to(device)).cpu()


def unstain_image(image, model, device="cpu"):
    with torch.inference_mode():
        return model(image.to(device)).cpu()


def pad_to_patch_multiple(image, patch_size=256):
    width, height = image.size
    padded_width = math.ceil(width / patch_size) * patch_size
    padded_height = math.ceil(height / patch_size) * patch_size

    if (padded_width, padded_height) == (width, height):
        return image, (width, height)

    padded = Image.new("RGB", (padded_width, padded_height), color=(255, 255, 255))
    padded.paste(image, (0, 0))
    return padded, (width, height)


def extract_patches_with_coords(pil_image, patch_size=256, stride=256):
    width, height = pil_image.size
    top_positions = list(range(0, height - patch_size + 1, stride))
    left_positions = list(range(0, width - patch_size + 1, stride))

    if top_positions[-1] != height - patch_size:
        top_positions.append(height - patch_size)
    if left_positions[-1] != width - patch_size:
        left_positions.append(width - patch_size)

    patches = []
    positions = []
    for top in top_positions:
        for left in left_positions:
            patch = pil_image.crop((left, top, left + patch_size, top + patch_size))
            patches.append(patch)
            positions.append((top, left))

    return patches, positions


def _blend_window(patch_size, device, dtype, eps=0.05):
    if patch_size <= 1:
        return torch.ones(1, 1, device=device, dtype=dtype)

    window_1d = (
        torch.sin(torch.linspace(0, math.pi, patch_size, device=device, dtype=dtype))
        ** 2
    )
    window_1d = window_1d * (1.0 - eps) + eps
    return window_1d[:, None] * window_1d[None, :]


def reconstruct_tensor_from_patches(
    patches, positions, image_size, patch_size=256, stride=256
):
    width, height = image_size
    dtype = patches[0].dtype
    device = patches[0].device

    reconstructed = torch.zeros(3, height, width, dtype=dtype, device=device)
    weight_map = torch.zeros(1, height, width, dtype=dtype, device=device)

    if stride < patch_size:
        window = _blend_window(patch_size, device=device, dtype=dtype).unsqueeze(0)
    else:
        window = torch.ones(1, patch_size, patch_size, dtype=dtype, device=device)

    for patch, (top, left) in zip(patches, positions):
        reconstructed[:, top : top + patch_size, left : left + patch_size] += (
            patch * window
        )
        weight_map[:, top : top + patch_size, left : left + patch_size] += window

    return reconstructed / weight_map.clamp_min(1e-6)


def translate_image_from_patches(
    input_image_path,
    model,
    transform,
    output_path,
    patch_size=256,
    stride=256,
    device="cpu",
    log_progress=True,
    log_interval=100,
):
    input_image = Image.open(input_image_path).convert("RGB")
    original_size = input_image.size
    padded_image, _ = pad_to_patch_multiple(input_image, patch_size=patch_size)

    input_patches, positions = extract_patches_with_coords(
        padded_image, patch_size=patch_size, stride=stride
    )
    translated_patches = []

    with torch.inference_mode():
        for patch in input_patches:
            patch_tensor = transform(patch).unsqueeze(0).to(device)
            translated_patch = model(patch_tensor).cpu().squeeze(0)
            translated_patches.append(translated_patch)

            if log_progress and (
                len(translated_patches) % log_interval == 0
                or len(translated_patches) == len(input_patches)
            ):
                print(
                    f"Processed {len(translated_patches)} / {len(input_patches)} patches"
                )

    reconstructed_padded = reconstruct_tensor_from_patches(
        translated_patches,
        positions,
        padded_image.size,
        patch_size=patch_size,
        stride=stride,
    )
    reconstructed = reconstructed_padded[:, : original_size[1], : original_size[0]]

    _ensure_parent_dir(output_path)
    save_image(
        reconstructed.unsqueeze(0),
        output_path,
        normalize=True,
        value_range=(-1, 1),
    )

    return original_size, padded_image.size, len(input_patches), output_path


def _build_transform(patch_size: int):
    return transforms.Compose(
        [
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_path = input("Enter checkpoint path: ").strip()
    if not checkpoint_path:
        checkpoint_path = os.path.join(
            "data",
            "E_Staining_DermaRepo",
            "H_E-Staining_dataset",
            "models_2026_02_24_18_30_51",
            "final_checkpoint_epoch_200.pth",
        )

    run_full_unstained_to_stained_translation = (
        input("Run full unstained to stained translation? (y/n): ").strip().lower()
        == "y"
    )

    dataset_root = os.path.join("data", "E_Staining_DermaRepo", "H_E-Staining_dataset")
    model_dir_name = os.path.basename(os.path.dirname(checkpoint_path))
    stained_output_dir = os.path.join(dataset_root, model_dir_name, "V_Stained")
    os.makedirs(stained_output_dir, exist_ok=True)

    G_AB, G_BA = load_model(checkpoint_path=checkpoint_path, device=device)

    patch_size = 256
    stride = patch_size // 2
    transform = _build_transform(patch_size)

    if run_full_unstained_to_stained_translation:
        print("Running full dataset translation...")
        unstained_dir = os.path.join(dataset_root, "Un_Stained")
        if not os.path.exists(unstained_dir):
            raise FileNotFoundError(f"Unstained directory not found: {unstained_dir}")

        unstained_images = [
            f
            for f in os.listdir(unstained_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        unstained_images.sort()

        for i, img_name in enumerate(unstained_images):
            print(f"Processing image {i + 1}/{len(unstained_images)}: {img_name}")
            unstained_path = os.path.join(unstained_dir, img_name)
            output_path = os.path.join(stained_output_dir, img_name)

            try:
                _, _, _, stained_output_path = translate_image_from_patches(
                    input_image_path=unstained_path,
                    model=G_AB,
                    transform=transform,
                    output_path=output_path,
                    patch_size=patch_size,
                    stride=stride,
                    device=device,
                    log_progress=True,
                    log_interval=1000,
                )
                print(
                    f"[Stain] Saved reconstructed stained image at: {stained_output_path}"
                )
            except Exception as e:
                print(f"[Error] Failed to process {img_name}: {e}")
                continue
    else:
        unstained_image_path = input("Provide Path to Unstained Image: ").strip()
        stained_image_path = input("Provide Path to Stained Image: ").strip()

        if not unstained_image_path:
            unstained_image_path = os.path.join(
                dataset_root,
                "Un_Stained",
                "HC21-01338(A3-1).10X unstained.jpg",
            )
        if not stained_image_path:
            stained_image_path = os.path.join(
                dataset_root,
                "C_Stained",
                "HC21-01338(A3-2).10X unstained.jpg",
            )

        print(f"Unstained Image Path: {unstained_image_path}")
        print(f"Stained Image Path: {stained_image_path}")

        stained_output_path = os.path.join(
            stained_output_dir, os.path.basename(unstained_image_path)
        )
        original_size, padded_size, num_patches, stained_output_path = (
            translate_image_from_patches(
                input_image_path=unstained_image_path,
                model=G_AB,
                transform=transform,
                output_path=stained_output_path,
                patch_size=patch_size,
                stride=stride,
                device=device,
            )
        )
        print(f"[Stain] Original Image size: {original_size}")
        print(f"[Stain] Padded Image size: {padded_size}")
        print(f"[Stain] Num patches: {num_patches}")
        print(f"[Stain] Saved reconstructed stained image at: {stained_output_path}")
        print(f"[Stain] Patch stride: {stride}")

        original_size, padded_size, num_patches, unstained_output_path = (
            translate_image_from_patches(
                input_image_path=stained_image_path,
                model=G_BA,
                transform=transform,
                output_path=os.path.join("data", "reconstructed_unstained_output.png"),
                patch_size=patch_size,
                stride=stride,
                device=device,
            )
        )
        print(f"[Unstain] Original Image size: {original_size}")
        print(f"[Unstain] Padded Image size: {padded_size}")
        print(f"[Unstain] Num patches: {num_patches}")
        print(f"[Unstain] Saved reconstructed unstained image at: {unstained_output_path}")
        print(f"[Unstain] Patch stride: {stride}")
