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

    Args:
        t (torch.Tensor): Input tensor with values in [-1,1] range

    Returns:
        torch.Tensor: Denormalized tensor with values clamped to [0,1] range
    """
    return (t * 0.5 + 0.5).clamp(0, 1)


# Unpaired Dataset Class (This Is the Core)
# Custom dataset class for handling unpaired image datasets used in CycleGAN training
class UnpairedImageDataset(Dataset):
    def __init__(self, dir_A, dir_B, transform=None, epoch_size=None):
        """
        Initialize the UnpairedImageDataset.

        Args:
            dir_A (str): Path to directory containing images from domain A
            dir_B (str): Path to directory containing images from domain B
            transform (callable, optional): Optional transform to be applied on images
        """
        # Initialize dataset with paths to two image directories and optional transforms
        self.dir_A = dir_A
        self.dir_B = dir_B
        # Get sorted list of image filenames from both directories
        self.images_A = sorted(os.listdir(dir_A))
        self.images_B = sorted(os.listdir(dir_B))
        self.transform = transform
        self.epoch_size = epoch_size

    def __len__(self):
        """
        Return the size of the dataset.

        Returns:
            int: Maximum length between the two domains to ensure both are fully utilized
        """
        # CycleGAN uses max length to keep sampling both domains
        # Return the maximum length to ensure both domains are fully utilized
        if self.epoch_size is not None:
            return self.epoch_size
        return max(len(self.images_A), len(self.images_B))

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            dict: Dictionary containing images from both domains with keys "A" and "B"
        """
        # Get image filenames using modulo to handle different domain sizes
        import random

        img_A = self.images_A[idx % len(self.images_A)]
        img_B = random.choice(self.images_B)

        # Construct full file paths
        path_A = os.path.join(self.dir_A, img_A)
        path_B = os.path.join(self.dir_B, img_B)

        # Load images and convert to RGB format
        image_A = Image.open(path_A).convert("RGB")
        image_B = Image.open(path_B).convert("RGB")

        # Apply transforms if provided (resize, normalize, etc.)
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        # Return dictionary with both images
        return {"A": image_A, "B": image_B}


# Data Loader Main Function
# Main function to create and return train and test data loaders
def getDataLoader(epoch_size=None):
    """
    Create and return train and test data loaders for CycleGAN training.

    This function sets up the complete data loading pipeline including:
    - System information printing for debugging
    - Dataset path configuration
    - Image transformation pipeline setup
    - Training and testing dataset creation
    - DataLoader initialization with appropriate settings
    - Sanity checks to verify data loading

    Returns:
        tuple: A tuple containing (train_loader, test_loader)
            - train_loader (DataLoader): DataLoader for training data
            - test_loader (DataLoader): DataLoader for testing data
    """
    # Print system information for debugging
    print(f"torch version: {torch.__version__}")
    print(f"Checking GPU available: ")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Define paths
    # Train dataset paths
    trainA = f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\trainA"  # Unstained
    trainB = f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\trainB"  # Stained

    # Test dataset paths
    testA = f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\testA"  # Unstained
    testB = f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\testB"  # Stained

    # Define Image Transforms (CycleGAN Standard)
    # Standard preprocessing pipeline for CycleGAN: resize, convert to tensor, normalize to [-1,1]
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Resize images to 256x256 pixels
            transforms.ToTensor(),  # Convert PIL image to tensor and scale to [0,1]
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),  # Normalize to [-1,1] range
        ]
    )

    # Create Training Dataset + DataLoader
    # Initialize training dataset with unpaired images
    train_dataset = UnpairedImageDataset(
        dir_A=trainA,
        dir_B=trainB,
        transform=transform,
        epoch_size=epoch_size,
    )

    # Create training data loader with CycleGAN standard settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # IMPORTANT: CycleGAN standard batch size
        shuffle=True,  # Shuffle training data for better learning
        num_workers=4,  # Use 4 worker processes for data loading
        pin_memory=True,  # Pin memory for faster GPU transfer
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Create Testing Dataset + DataLoader
    # Initialize testing dataset with unpaired images
    test_dataset = UnpairedImageDataset(
        dir_A=testA,
        dir_B=testB,
        transform=transform,
    )

    # Create testing data loader (no shuffling needed for testing)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # IMPORTANT: CycleGAN standard batch size
        shuffle=False,  # Don't shuffle test data
    )

    # Sanity Check
    # Train dataloader test - verify data loading works correctly
    batchTrain = next(iter(train_loader))

    print(f"Train dataloader batch shape test: ")
    print(f"BatchA or trainA : {batchTrain["A"].shape}")
    print(f"BatchB or trainB : {batchTrain["B"].shape}")

    # Test dataloader test - verify test data loading works correctly
    batchTest = next(iter(test_loader))

    print(f"Test dataloader batch shape test: ")
    print(f"BatchA or testA : {batchTest["A"].shape}")
    print(f"BatchB or testB : {batchTest["B"].shape}")

    # Return both data loaders for use in training and testing
    return train_loader, test_loader


# Main execution block - runs when script is executed directly
if __name__ == "__main__":
    # Get the data loaders
    train_loader, test_loader = getDataLoader(epoch_size=3000)

    # Check train data loader
    # Load a batch from training data for visualization
    batchTrain = next(iter(train_loader))
    # Denormalize and convert tensors to numpy arrays for matplotlib
    A = denormalize(batchTrain["A"][0]).permute(1, 2, 0).cpu().numpy()
    B = denormalize(batchTrain["B"][0]).permute(1, 2, 0).cpu().numpy()

    # Display training images side by side
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(A)
    plt.title(f"Un-Stained (A)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(B)
    plt.title("C-Stained (B)")
    plt.axis("off")
    plt.show()

    # Check test data loader
    # Load a batch from test data for visualization
    batchTest = next(iter(test_loader))
    # Denormalize and convert tensors to numpy arrays for matplotlib
    A = denormalize(batchTest["A"][0]).permute(1, 2, 0).cpu().numpy()
    B = denormalize(batchTest["B"][0]).permute(1, 2, 0).cpu().numpy()

    # Display test images side by side
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(A)
    plt.title(f"Un-Stained (A)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(B)
    plt.title("C-Stained (B)")
    plt.axis("off")
    plt.show()
