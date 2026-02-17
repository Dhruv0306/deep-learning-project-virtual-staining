# Imports
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Function to denormalize tensors from [-1,1] back to [0,1] range for visualization
def denormalize(t):
    """
    Denormalize tensors from [-1,1] back to [0,1] range for visualization.

    This function reverses the normalization applied during preprocessing to make
    images suitable for display with matplotlib or PIL.

    Args:
        t (torch.Tensor): Input tensor with values in [-1,1] range

    Returns:
        torch.Tensor: Denormalized tensor with values clamped to [0,1] range
    """
    # Reverse the normalization: (x - 0.5) / 0.5 becomes x * 0.5 + 0.5
    # Clamp ensures values stay within valid [0,1] range for image display
    return (t * 0.5 + 0.5).clamp(0, 1)


# Custom dataset class for handling unpaired image datasets used in CycleGAN training
# Unlike paired datasets, this allows training with images from two domains that don't correspond 1:1
class UnpairedImageDataset(Dataset):
    def __init__(self, dir_A, dir_B, transform=None, epoch_size=None):
        """
        Initialize the UnpairedImageDataset.

        This dataset is specifically designed for CycleGAN which requires unpaired data
        from two different domains (e.g., unstained vs stained images).

        Args:
            dir_A (str): Path to directory containing images from domain A (e.g., unstained)
            dir_B (str): Path to directory containing images from domain B (e.g., stained)
            transform (callable, optional): Optional transform to be applied on images
            epoch_size (int, optional): Fixed epoch size to control training iterations
        """
        # Store directory paths for both image domains
        self.dir_A = dir_A
        self.dir_B = dir_B

        # Get sorted list of image filenames from both directories
        # Sorting ensures consistent ordering across runs
        self.images_A = sorted(os.listdir(dir_A))
        self.images_B = sorted(os.listdir(dir_B))

        # Store transformation pipeline and optional epoch size
        self.transform = transform
        self.epoch_size = epoch_size

    def __len__(self):
        """
        Return the size of the dataset.

        For unpaired datasets, we need to ensure both domains are adequately sampled.
        Using max length prevents premature epoch termination.

        Returns:
            int: Maximum length between the two domains to ensure both are fully utilized
        """
        # If epoch_size is specified, use it to control training iterations
        if self.epoch_size is not None:
            return self.epoch_size

        # CycleGAN uses max length to keep sampling both domains effectively
        # This ensures the smaller domain doesn't limit training on the larger domain
        return max(len(self.images_A), len(self.images_B))

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        This method implements the core unpaired sampling strategy:
        - Domain A: Sequential sampling with wraparound using modulo
        - Domain B: Random sampling to increase diversity

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            dict: Dictionary containing images from both domains with keys "A" and "B"
        """
        # Import random here to avoid global import overhead
        import random

        # Sequential sampling for domain A with wraparound using modulo
        # This ensures all images in domain A are eventually used
        img_A = self.images_A[idx % len(self.images_A)]

        # Random sampling for domain B to increase pairing diversity
        # This prevents the model from learning spurious correlations between specific pairs
        img_B = random.choice(self.images_B)

        # Construct full file paths by joining directory and filename
        path_A = os.path.join(self.dir_A, img_A)
        path_B = os.path.join(self.dir_B, img_B)

        # Load images using PIL and ensure RGB format (handles grayscale conversion)
        image_A = Image.open(path_A).convert("RGB")
        image_B = Image.open(path_B).convert("RGB")

        # Apply transformation pipeline if provided (resize, normalize, augment, etc.)
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        # Return dictionary format expected by CycleGAN training loop
        return {"A": image_A, "B": image_B}


# Data Loader Main Function
# Comprehensive function to set up the entire data loading pipeline for CycleGAN training
def getDataLoader(epoch_size=None):
    """
    Create and return train and test data loaders for CycleGAN training.

    This function orchestrates the complete data loading pipeline including:
    - System diagnostics and GPU availability checking
    - Dataset path configuration for train/test splits
    - Image transformation pipeline setup with CycleGAN standards
    - Training and testing dataset instantiation
    - DataLoader initialization with optimized performance settings
    - Comprehensive sanity checks to verify data loading functionality

    Args:
        epoch_size (int, optional): Fixed number of samples per epoch for training control

    Returns:
        tuple: A tuple containing (train_loader, test_loader)
            - train_loader (DataLoader): Optimized DataLoader for training with shuffling
            - test_loader (DataLoader): DataLoader for testing without shuffling
    """
    # System Diagnostics - Print comprehensive system information for debugging
    print(f"torch version: {torch.__version__}")
    print(f"Checking GPU available: ")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Dataset Path Configuration
    # Define paths using Windows-style backslashes (consider using os.path.join for cross-platform compatibility)

    # Training dataset paths - larger datasets for model learning
    trainA = f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\trainA"  # Domain A: Unstained images
    trainB = f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\trainB"  # Domain B: Stained images

    # Testing dataset paths - held-out data for evaluation
    testA = f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\testA"  # Domain A: Unstained test images
    testB = f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\testB"  # Domain B: Stained test images

    # Image Transformation Pipeline (CycleGAN Standard)
    # Comprehensive preprocessing pipeline following CycleGAN best practices
    transform = transforms.Compose(
        [
            # Resize all images to standard 256x256 resolution using bilinear interpolation
            # This ensures consistent input dimensions and reasonable computational requirements
            transforms.Resize(
                (256, 256), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            # Convert PIL Image to PyTorch tensor and automatically scale from [0,255] to [0,1]
            transforms.ToTensor(),
            # Normalize to [-1,1] range as expected by CycleGAN generators and discriminators
            # This normalization improves training stability and convergence
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    # Training Dataset and DataLoader Creation
    # Initialize training dataset with unpaired images and optional epoch size control
    train_dataset = UnpairedImageDataset(
        dir_A=trainA,
        dir_B=trainB,
        transform=transform,
        epoch_size=epoch_size,  # Allows controlling training iterations per epoch
    )

    # Create optimized training data loader with performance-focused settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=6,  # CycleGAN standard batch size - balance between memory and gradient quality
        shuffle=True,  # Randomize sample order each epoch for better generalization
        num_workers=4,  # Parallel data loading workers for improved throughput
        pin_memory=True,  # Pin memory for faster CPU-to-GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs to reduce startup overhead
        prefetch_factor=2,  # Number of batches to prefetch per worker for pipeline efficiency
        drop_last=True,  # Ensure consistent batch sizes by dropping incomplete final batch
    )

    # Testing Dataset and DataLoader Creation
    # Initialize testing dataset without epoch size limitation for complete evaluation
    test_dataset = UnpairedImageDataset(
        dir_A=testA,
        dir_B=testB,
        transform=transform,  # Apply same transforms as training for consistency
    )

    # Create testing data loader with evaluation-focused settings
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Single image processing for detailed evaluation and visualization
        shuffle=False,  # Maintain consistent order for reproducible evaluation results
        # Note: Fewer workers and no performance optimizations needed for testing
    )

    # Comprehensive Sanity Checks
    # Verify that data loading pipeline works correctly before training begins

    # Training dataloader verification - ensure batch loading and tensor shapes are correct
    batchTrain = next(iter(train_loader))

    print(f"Train dataloader batch shape test: ")
    print(
        f"BatchA or trainA : {batchTrain["A"].shape}"
    )  # Expected: [batch_size, 3, 256, 256]
    print(
        f"BatchB or trainB : {batchTrain["B"].shape}"
    )  # Expected: [batch_size, 3, 256, 256]

    # Testing dataloader verification - ensure test data loading works correctly
    batchTest = next(iter(test_loader))

    print(f"Test dataloader batch shape test: ")
    print(f"BatchA or testA : {batchTest["A"].shape}")  # Expected: [1, 3, 256, 256]
    print(f"BatchB or testB : {batchTest["B"].shape}")  # Expected: [1, 3, 256, 256]

    # Return configured data loaders ready for CycleGAN training and evaluation
    return train_loader, test_loader


# Main execution block - demonstration and testing when script is run directly
if __name__ == "__main__":
    # Initialize data loaders with specified epoch size for controlled training
    train_loader, test_loader = getDataLoader(epoch_size=3000)

    # Training Data Visualization
    # Load and display a sample batch from training data to verify data quality
    batchTrain = next(iter(train_loader))

    # Denormalize first images in batch and convert to numpy format for matplotlib
    # Permute from CHW (Channel-Height-Width) to HWC (Height-Width-Channel) format
    A = denormalize(batchTrain["A"][0]).permute(1, 2, 0).cpu().numpy()
    B = denormalize(batchTrain["B"][0]).permute(1, 2, 0).cpu().numpy()

    # Create side-by-side visualization of training sample pair
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(A)
    plt.title(f"Un-Stained (A)")  # Domain A visualization
    plt.axis("off")  # Remove axis for cleaner display

    plt.subplot(1, 2, 2)
    plt.imshow(B)
    plt.title("C-Stained (B)")  # Domain B visualization
    plt.axis("off")  # Remove axis for cleaner display
    plt.show()

    # Testing Data Visualization
    # Load and display a sample from test data to verify test pipeline
    batchTest = next(iter(test_loader))

    # Denormalize test images and convert to display format
    A = denormalize(batchTest["A"][0]).permute(1, 2, 0).cpu().numpy()
    B = denormalize(batchTest["B"][0]).permute(1, 2, 0).cpu().numpy()

    # Create side-by-side visualization of test sample pair
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(A)
    plt.title(f"Un-Stained (A)")  # Test domain A visualization
    plt.axis("off")  # Remove axis for cleaner display

    plt.subplot(1, 2, 2)
    plt.imshow(B)
    plt.title("C-Stained (B)")  # Test domain B visualization
    plt.axis("off")  # Remove axis for cleaner display
    plt.show()
