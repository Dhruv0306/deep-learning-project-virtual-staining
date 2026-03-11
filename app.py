import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import math
from generator import ResnetGenerator
from preprocess_data import extract_patches_pil


def load_model(checkpoint_path=None, device="cpu"):
    if checkpoint_path is None:
        raise ValueError("Checkpoint_path is required")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    G_AB = ResnetGenerator().to(device)
    G_BA = ResnetGenerator().to(device)

    G_AB.load_state_dict(checkpoint["G_AB"])
    G_BA.load_state_dict(checkpoint["G_BA"])

    G_AB.eval()
    G_BA.eval()

    return G_AB, G_BA


def stain_image(image, model, device="cpu"):
    with torch.no_grad():
        image = image.to(device)
        generated_image = model(image)
        return generated_image.cpu()


def unstain_image(image, model, device="cpu"):
    with torch.no_grad():
        image = image.to(device)
        generated_image = model(image)
        return generated_image.cpu()


def pad_to_patch_multiple(image, patch_size=256):
    width, height = image.size
    padded_width = math.ceil(width / patch_size) * patch_size
    padded_height = math.ceil(height / patch_size) * patch_size

    if (padded_width, padded_height) == (width, height):
        return image, (width, height)

    padded = Image.new("RGB", (padded_width, padded_height), color=(255, 255, 255))
    padded.paste(image, (0, 0))
    return padded, (width, height)


def reconstruct_tensor_from_patches(patches, image_size, patch_size=256):
    width, height = image_size
    # Keep output in normalized [-1, 1] tensor space to avoid patch-wise quantization.
    reconstructed = torch.ones(3, height, width, dtype=patches[0].dtype)

    idx = 0
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            reconstructed[:, top : top + patch_size, left : left + patch_size] = (
                patches[idx]
            )
            idx += 1

    return reconstructed


def translate_image_from_patches(
    input_image_path, model, transform, output_path, patch_size=256, device="cpu"
):
    input_image = Image.open(input_image_path).convert("RGB")
    original_size = input_image.size
    padded_image, _ = pad_to_patch_multiple(input_image, patch_size=patch_size)

    input_patches = extract_patches_pil(
        padded_image, patch_size=patch_size, stride=patch_size
    )
    translated_patches = []

    with torch.inference_mode():
        for patch in input_patches:
            patch_tensor = transform(patch).unsqueeze(0).to(device)  # [1, 3, H, W]
            translated_patch = model(patch_tensor).cpu().squeeze(0)
            translated_patches.append(translated_patch)

    reconstructed_padded = reconstruct_tensor_from_patches(
        translated_patches, padded_image.size, patch_size=patch_size
    )
    reconstructed = reconstructed_padded[:, : original_size[1], : original_size[0]]

    save_image(
        reconstructed.unsqueeze(0),
        output_path,
        normalize=True,
        value_range=(-1, 1),
    )

    return original_size, padded_image.size, len(input_patches), output_path


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model
    G_AB, G_BA = load_model(
        checkpoint_path="data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models_2026_02_24_18_30_51\\final_checkpoint_epoch_200.pth",
        device=device,
    )

    # Create transform
    patch_size = 256
    transform = transforms.Compose(
        [
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Image paths
    unstained_image_path = input("Provide Path to Unstained Image: ")
    stained_image_path = input("Provide Path to Stained Image: ")
    unstained_image_path = (
        "data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\Un_Stained\\HC21-01338(A3-1).10X unstained.jpg"
        if unstained_image_path is None or unstained_image_path == ""
        else unstained_image_path.replace("\\", "\\\\" )
    )
    stained_image_path = (
        "data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\C_Stained\\HC21-01338(A3-2).10X unstained.jpg"
        if stained_image_path is None or stained_image_path == ""
        else stained_image_path.replace("\\", "\\\\")
    )

    print(f"Unstained Image Path: {unstained_image_path}")
    print(f"Stained Image Path: {stained_image_path}")

    # A -> B (unstained -> stained)
    original_size, padded_size, num_patches, stained_output_path = (
        translate_image_from_patches(
            input_image_path=unstained_image_path,
            model=G_AB,
            transform=transform,
            output_path="data\\reconstructed_stained_output.png",
            patch_size=patch_size,
            device=device,
        )
    )

    print(f"[Stain] Original Image size: {original_size}")
    print(f"[Stain] Padded Image size: {padded_size}")
    print(f"[Stain] Num patches: {num_patches}")
    print(f"[Stain] Saved reconstructed stained image at: {stained_output_path}")

    # B -> A (stained -> unstained)
    original_size, padded_size, num_patches, unstained_output_path = (
        translate_image_from_patches(
            input_image_path=stained_image_path,
            model=G_BA,
            transform=transform,
            output_path="data\\reconstructed_unstained_output.png",
            patch_size=patch_size,
            device=device,
        )
    )
    print(f"[Unstain] Original Image size: {original_size}")
    print(f"[Unstain] Padded Image size: {padded_size}")
    print(f"[Unstain] Num patches: {num_patches}")
    print(f"[Unstain] Saved reconstructed unstained image at: {unstained_output_path}")
